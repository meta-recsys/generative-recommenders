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

#include <torch/library.h> // @manual
#include <torch/nn/functional.h>
#include "flash_common_cpu.h"

namespace hstu {

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

// CPU-only implementation that registers under main hstu namespace
// This provides fallback implementations when GPU code is not compiled
TORCH_LIBRARY_FRAGMENT(hstu, m) {
  // Only register operators if they haven't been registered by GPU code
  // This allows CPU-only builds to work while GPU builds use GPU
  // implementations

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

  // Register CPU implementations
  m.impl(
      "hstu_mha",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(hstu_mha_cpu)));
  m.impl(
      "hstu_mha",
      torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(hstu_mha_meta)));

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
      torch::dispatch(
          c10::DispatchKey::CPU, TORCH_FN(hstu::hstu_mha_bwd_dummy)));
}

} // namespace hstu
