/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 *Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/numeric_types.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/function.h>
#include <torch/library.h> // @manual
#include <torch/nn/functional.h>
#include "flash_common.h"

extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_C", /* name of module */
      NULL, /* module documentation, may be NULL */
      -1, /* size of per-interpreter state of the module,
              or -1 if the module keeps state in global variables. */
      NULL, /* methods */
  };
  return PyModule_Create(&module_def);
}
}

namespace hstu {

class HSTUFlashAttentionFunctionGPU
    : public torch::autograd::Function<HSTUFlashAttentionFunctionGPU> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      int64_t max_seq_len,
      double alpha,
      at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
      at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
      at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
      const std::optional<at::Tensor>& seq_offsets,
      bool causal,
      const std::optional<at::Tensor>& num_targets,
      const std::optional<at::Tensor>& attn_scale,
      int64_t max_attn_len,
      int64_t min_full_attn_seq_len,
      int64_t contextual_seq_len,
      const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
      const std::optional<at::Tensor>& k_descale, // (b, h_k)
      const std::optional<at::Tensor>& v_descale, // (b, h_k)
      bool sort_by_length,
      bool deterministic,
      const int64_t sm_margin,
      int64_t max_q_len,
      const std::optional<at::Tensor>& seq_offsets_q,
      int64_t num_softmax_heads,
      bool training,
      const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
      const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
      const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
      const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
          std::nullopt,
      int64_t num_groups = 1) {
    ctx->saved_data["max_seq_len"] = max_seq_len;
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["causal"] = causal;
    ctx->saved_data["max_attn_len"] = max_attn_len;
    ctx->saved_data["min_full_attn_seq_len"] = min_full_attn_seq_len;
    ctx->saved_data["contextual_seq_len"] = contextual_seq_len;
    ctx->saved_data["deterministic"] = deterministic;
    ctx->saved_data["sort_by_length"] = sort_by_length;
    ctx->saved_data["sm_margin"] = sm_margin;
    ctx->saved_data["max_q_len"] = max_q_len;
    ctx->saved_data["num_softmax_heads"] = num_softmax_heads;
    ctx->saved_data["num_groups"] = num_groups;
    auto fwd_out = hstu::hstu_mha_fwd(
        max_seq_len, // max_seq_len
        alpha, // alpha
        q, // q
        k, // k
        v, // v
        seq_offsets, // seq_offsets
        causal, // causal
        num_targets, // num_targets
        attn_scale, // attn_scale
        max_attn_len, // max_attn_len
        min_full_attn_seq_len, // min_full_attn_seq_len
        contextual_seq_len, // contextual_seq_len
        q_descale, // q_descale
        k_descale, // k_descale
        v_descale, // v_descale
        sm_margin, // sm_margin
        max_q_len, // max_q_len
        seq_offsets_q, // seq_offsets_q
        num_softmax_heads, // num_softmax_heads
        training,
        max_seq_len_tensor,
        contextual_seq_len_tensor,
        max_attn_len_tensor,
        min_full_attn_seq_len_tensor,
        num_groups);
    auto out = get<0>(fwd_out);
    auto softmax_lse = get<1>(fwd_out);
    ctx->save_for_backward(
        {q,
         k,
         v,
         out,
         seq_offsets.value_or(at::Tensor()),
         num_targets.value_or(at::Tensor()),
         attn_scale.value_or(at::Tensor()),
         seq_offsets_q.value_or(at::Tensor()),
         softmax_lse.value_or(at::Tensor()),
         max_seq_len_tensor.value_or(at::Tensor()),
         contextual_seq_len_tensor.value_or(at::Tensor()),
         max_attn_len_tensor.value_or(at::Tensor()),
         min_full_attn_seq_len_tensor.value_or(at::Tensor())});
    return out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved_tensors = ctx->get_saved_variables();
    auto saved_data = ctx->saved_data;
    auto q = saved_tensors[0];
    auto k = saved_tensors[1];
    auto v = saved_tensors[2];
    auto out = saved_tensors[3];
    auto seq_offsets = saved_tensors[4];
    auto num_targets = saved_tensors[5];
    auto attn_scale = saved_tensors[6];
    auto seq_offsets_q = saved_tensors[7];
    auto softmax_lse = saved_tensors[8];
    auto max_seq_len_tensor = saved_tensors[9];
    auto contextual_seq_len_tensor = saved_tensors[10];
    auto max_attn_len_tensor = saved_tensors[11];
    auto min_full_attn_seq_len_tensor = saved_tensors[12];
    auto seq_offsets_opt =
        seq_offsets.defined() ? std::optional(seq_offsets) : std::nullopt;
    auto num_targets_opt =
        num_targets.defined() ? std::optional(num_targets) : std::nullopt;
    auto attn_scale_opt =
        attn_scale.defined() ? std::optional(attn_scale) : std::nullopt;
    auto seq_offsets_q_opt =
        seq_offsets_q.defined() ? std::optional(seq_offsets_q) : std::nullopt;
    auto softmax_lse_opt =
        softmax_lse.defined() ? std::optional(softmax_lse) : std::nullopt;
    auto max_seq_len_tensor_opt = max_seq_len_tensor.defined()
        ? std::optional(max_seq_len_tensor)
        : std::nullopt;
    auto contextual_seq_len_tensor_opt = contextual_seq_len_tensor.defined()
        ? std::optional(contextual_seq_len_tensor)
        : std::nullopt;
    auto max_attn_len_tensor_opt = max_attn_len_tensor.defined()
        ? std::optional(max_attn_len_tensor)
        : std::nullopt;
    auto min_full_attn_seq_len_tensor_opt =
        min_full_attn_seq_len_tensor.defined()
        ? std::optional(min_full_attn_seq_len_tensor)
        : std::nullopt;

    auto dq = at::empty_like(q);
    auto dk = at::empty_like(k);
    auto dv = at::empty_like(v);

    auto bwd_res = hstu::hstu_mha_bwd(
        saved_data["max_seq_len"].toInt(), // max_seq_len
        saved_data["alpha"].toDouble(), // alpha
        grad_outputs[0], // dout
        q, // q
        k, // k
        v, // v
        dq, // dq
        dk, // dk
        dv, // dv
        out, // out
        seq_offsets_opt, // seq_offsets
        saved_data["causal"].toBool(), // causal
        num_targets_opt, // num_targets
        attn_scale_opt, // attn_scale
        saved_data["max_attn_len"].toInt(), // max_attn_len
        saved_data["min_full_attn_seq_len"].toInt(), // min_full_attn_seq_len
        saved_data["contextual_seq_len"].toInt(), // contextual_seq_len
        saved_data["sort_by_length"].toBool(), // sort_by_length
        saved_data["deterministic"].toBool(), // deterministic
        saved_data["sm_margin"].toInt(), // sm_margin
        saved_data["max_q_len"].toInt(), // max_q_len
        seq_offsets_q_opt, // seq_offsets_q
        saved_data["num_softmax_heads"].toInt(), // num_softmax_heads
        softmax_lse_opt,
        max_seq_len_tensor_opt,
        contextual_seq_len_tensor_opt,
        max_attn_len_tensor_opt,
        min_full_attn_seq_len_tensor_opt,
        saved_data["num_groups"].toInt());

    return {
        torch::autograd::Variable(), // max_seq_len
        torch::autograd::Variable(), // alpha
        bwd_res[0], // dq
        bwd_res[1], // dk
        bwd_res[2], // dv
        torch::autograd::Variable(), // seq_offsets
        torch::autograd::Variable(), // causal
        torch::autograd::Variable(), // num_targets
        torch::autograd::Variable(), // attn_scale
        torch::autograd::Variable(), // max_attn_len
        torch::autograd::Variable(), // min_full_attn_seq_len
        torch::autograd::Variable(), // contextual_seq_len
        torch::autograd::Variable(), // q_descale
        torch::autograd::Variable(), // k_descale
        torch::autograd::Variable(), // v_descale
        torch::autograd::Variable(), // sort_by_length
        torch::autograd::Variable(), // deterministic
        torch::autograd::Variable(), // sm_margin
        torch::autograd::Variable(), // max_q_len
        torch::autograd::Variable(), // seq_offsets_q
        torch::autograd::Variable(), // num_softmax_heads
        torch::autograd::Variable(), // training
        torch::autograd::Variable(), // max_seq_len_tensor
        torch::autograd::Variable(), // contextual_seq_len_tensor
        torch::autograd::Variable(), // max_attn_len_tensor
        torch::autograd::Variable(), // min_full_attn_seq_len_tensor
        torch::autograd::Variable(), // num_groups
    };
  }
};

at::Tensor cuda_hstu_mha(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
    const std::optional<at::Tensor>& k_descale, // (b, h_k)
    const std::optional<at::Tensor>& v_descale, // (b, h_k)
    bool sort_by_length,
    bool deterministic,
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true,
    const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
        std::nullopt,
    int64_t num_groups = 1) {
  return hstu::HSTUFlashAttentionFunctionGPU::apply(
      max_seq_len,
      alpha,
      q,
      k,
      v,
      seq_offsets,
      causal,
      num_targets,
      attn_scale,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      q_descale,
      k_descale,
      v_descale,
      sort_by_length,
      deterministic,
      sm_margin,
      max_q_len,
      seq_offsets_q,
      num_softmax_heads,
      training,
      max_seq_len_tensor,
      contextual_seq_len_tensor,
      max_attn_len_tensor,
      min_full_attn_seq_len_tensor,
      num_groups);
}

at::Tensor hstu_mha_cpu(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
    const std::optional<at::Tensor>& k_descale, // (b, h_k)
    const std::optional<at::Tensor>& v_descale, // (b, h_k)
    bool sort_by_length,
    bool deterministic,
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true,
    const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
        std::nullopt,
    int64_t num_groups = 1) {
  auto fwd_out = hstu::hstu_mha_fwd_dummy(
      max_seq_len,
      alpha,
      q,
      k,
      v,
      seq_offsets,
      causal,
      num_targets,
      attn_scale,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      q_descale,
      k_descale,
      v_descale,
      sm_margin,
      max_q_len,
      seq_offsets_q,
      num_softmax_heads,
      training);
  return get<0>(fwd_out);
}

at::Tensor hstu_mha_meta(
    const at::SymInt max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
    const std::optional<at::Tensor>& k_descale, // (b, h_k)
    const std::optional<at::Tensor>& v_descale, // (b, h_k)
    bool sort_by_length,
    bool deterministic,
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true,
    const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
        std::nullopt,
    int64_t num_groups = 1) {
  auto fwd_out = hstu::hstu_mha_fwd_meta(
      max_seq_len,
      alpha,
      q,
      k,
      v,
      seq_offsets,
      causal,
      num_targets,
      attn_scale,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      q_descale,
      k_descale,
      v_descale,
      sm_margin,
      max_q_len,
      seq_offsets_q,
      num_softmax_heads,
      training);
  return get<0>(fwd_out);
}

TORCH_LIBRARY_FRAGMENT(hstu, m) {
  m.def(
      "hstu_mha_fwd("
      "SymInt max_seq_len, "
      "float alpha, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor? seq_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "Tensor? attn_scale, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "Tensor? q_descale, "
      "Tensor? k_descale, "
      "Tensor? v_descale, "
      "int sm_margin = 0,"
      "int max_q_len = 0,"
      "Tensor? seq_offsets_q = None,"
      "int num_softmax_heads = 0,"
      "bool training = True,"
      "Tensor? max_seq_len_tensor = None,"
      "Tensor? contextual_seq_len_tensor = None,"
      "Tensor? max_attn_len_tensor = None,"
      "Tensor? min_full_attn_seq_len_tensor = None,"
      "int num_groups = 1"
      ") -> (Tensor, Tensor?)");

  m.def(
      "hstu_mha_bwd("
      "int max_seq_len, "
      "float alpha, "
      "Tensor dout, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor dq, "
      "Tensor dk, "
      "Tensor dv, "
      "Tensor out, "
      "Tensor? seq_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "Tensor? attn_scale, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "bool sort_by_length,"
      "bool deterministic,"
      "int sm_margin = 0,"
      "int max_q_len = 0,"
      "Tensor? seq_offsets_q = None,"
      "int num_softmax_heads = 0,"
      "Tensor? softmax_lse = None,"
      "Tensor? max_seq_len_tensor = None,"
      "Tensor? contextual_seq_len_tensor = None,"
      "Tensor? max_attn_len_tensor = None,"
      "Tensor? min_full_attn_seq_len_tensor = None,"
      "int num_groups = 1"
      ") -> Tensor[]");

  m.def(
      "hstu_mha("
      "SymInt max_seq_len, "
      "float alpha, "
      "Tensor q, "
      "Tensor k, "
      "Tensor v, "
      "Tensor? seq_offsets, "
      "bool causal, "
      "Tensor? num_targets, "
      "Tensor? attn_scale, "
      "int max_attn_len, "
      "int min_full_attn_seq_len, "
      "int contextual_seq_len, "
      "Tensor? q_descale, "
      "Tensor? k_descale, "
      "Tensor? v_descale, "
      "bool sort_by_length, "
      "bool deterministic, "
      "int sm_margin = 0,"
      "int max_q_len = 0,"
      "Tensor? seq_offsets_q = None,"
      "int num_softmax_heads = 0,"
      "bool training = True,"
      "Tensor? max_seq_len_tensor = None,"
      "Tensor? contextual_seq_len_tensor = None,"
      "Tensor? max_attn_len_tensor = None,"
      "Tensor? min_full_attn_seq_len_tensor = None,"
      "int num_groups = 1"
      ") -> Tensor");

  m.impl(
      "hstu_mha",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(cuda_hstu_mha)));
  m.impl(
      "hstu_mha",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(hstu_mha_cpu)));
  m.impl(
      "hstu_mha",
      torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(hstu_mha_meta)));

  m.impl(
      "hstu_mha_fwd",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(hstu::hstu_mha_fwd)));
  m.impl(
      "hstu_mha_fwd",
      torch::dispatch(
          c10::DispatchKey::CPU, TORCH_FN(hstu::hstu_mha_fwd_dummy)));
  m.impl(
      "hstu_mha_fwd",
      torch::dispatch(
          c10::DispatchKey::Meta, TORCH_FN(hstu::hstu_mha_fwd_meta)));

  m.impl(
      "hstu_mha_bwd",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(hstu::hstu_mha_bwd)));
  m.impl(
      "hstu_mha_bwd",
      torch::dispatch(
          c10::DispatchKey::CPU, TORCH_FN(hstu::hstu_mha_bwd_dummy)));
}
} // namespace hstu
