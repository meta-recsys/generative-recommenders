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

/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <limits>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h> // @manual
#include <c10/hip/HIPGuard.h>
#include <torch/nn/functional.h>
#include <torch/version.h> // For TORCH_VERSION* macros

#include <torch/torch.h> // @manual

#include "hstu_attention_api.hpp"
#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_params.hpp"

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Split-kv workspace from the caching allocator: the default hipMallocAsync pool releases at a
// threshold of 0 here, costing ~3.5 ms per call. The MasqueradingAsCUDA spelling compiles both
// in PyTorch's ROCm extension build (which hipifies c10::cuda names) and in builds that do not.
void* hstu_workspace_alloc(size_t bytes, hipStream_t stream) {
  return c10::hip::HIPCachingAllocatorMasqueradingAsCUDA::raw_alloc_with_stream(bytes, stream);
}

void hstu_workspace_free(void* ptr, hipStream_t /*stream*/) {
  c10::hip::HIPCachingAllocatorMasqueradingAsCUDA::raw_delete(ptr);
}

at::Tensor switch_to_contiguous_if_needed(const at::Tensor& x) {
  if (x.stride(x.dim() - 1) == 1) {
    return x;
  }
  return x.contiguous();
}

void set_fwd_params(
    HstuAttentionNoGroupFwdParams& params,
    const int32_t batch_size,
    const at::Tensor& q_dev,
    const at::Tensor& k_dev,
    const at::Tensor& v_dev,
    const at::Tensor& o_dev,
    const at::Tensor& lse_dev,
    const at::Tensor& seq_q_offsets,
    const at::Tensor& seq_kv_offsets,
    const at::Tensor& num_targets,
    float alpha,
    const int32_t max_seq_len,
    const int32_t max_seq_len_kv,
    const int32_t min_seq_len_q,
    const int32_t min_seq_len_kv,
    const int32_t hdim_qk,
    const int32_t hdim_v,
    const int32_t num_head,
    bool causal,
    const int32_t max_attn_len,
    const int32_t contextual_seq_len,
    const int32_t min_full_attn_seq_len,
    bool is_jagged,
    bool has_multiple_targets,
    float attn_scale,
    bool use_softmax,
    bool is_cross_attention,
    bool is_training,
    const int32_t kv_truncate_len = 0,
    bool kv_keep_last = false) {
  params.is_cross_attention = is_cross_attention;
  params.is_jagged = is_jagged;
  params.num_batch = batch_size;
  params.seq_q_offsets_ptr = is_jagged ? seq_q_offsets.data_ptr() : nullptr;
  params.seq_kv_offsets_ptr = is_jagged ? seq_kv_offsets.data_ptr() : nullptr;
  params.num_targets_ptr =
      has_multiple_targets ? num_targets.data_ptr() : nullptr;
  params.max_seqlen_q = max_seq_len; // jagged mode; harmless in batched mode
  // jagged/group mode: KV extent drives the cross-attention mtile / split-kv
  // dispatch rules. Fall back to max_seq_len for self-attention (KV == Q).
  params.max_seqlen_kv = (max_seq_len_kv > 0) ? max_seq_len_kv : max_seq_len;
  // jagged mode: the min/max seqlen ratio selects the grid schedule via
  // is_almost_invariant_seqlen(). Fall back to max (ratio 1 -> "invariant",
  // i.e. batch-leading schedule) when the caller does not provide the min.
  params.min_seqlen_q = (min_seq_len_q > 0) ? min_seq_len_q : params.max_seqlen_q;
  params.min_seqlen_kv = (min_seq_len_kv > 0) ? min_seq_len_kv : params.max_seqlen_kv;
  // GQA broadcasting: when K/V has fewer heads than Q, set nhead stride to 0
  // so all Q head blocks read from K/V head 0.
  int32_t num_head_kv = k_dev.size(-2);
  if (!is_jagged) {
    params.seqlen_q = q_dev.size(-3);
    params.seqlen_kv = k_dev.size(-3);
    params.batch_stride_q = q_dev.stride(-4);
    params.batch_stride_k = k_dev.stride(-4);
    params.batch_stride_v = v_dev.stride(-4);
    params.batch_stride_bias = 0;
    params.batch_stride_o = o_dev.stride(-4);
    params.batch_stride_lse = lse_dev.defined() ? lse_dev.stride(-3) : 0;
  }
  if (is_jagged) {
    params.total_seqlen_q = q_dev.size(0);
    params.total_seqlen_kv = k_dev.size(0);
    if (kv_truncate_len > 0) {
      // The kernels see at most kv_truncate_len keys per sequence; size the
      // dispatch (mtile, split-kv) for the truncated extent.
      params.kv_truncate_len = kv_truncate_len;
      params.kv_keep_last = kv_keep_last;
      params.max_seqlen_kv = std::min<int32_t>(params.max_seqlen_kv, kv_truncate_len);
      params.min_seqlen_kv = std::min<int32_t>(params.min_seqlen_kv, kv_truncate_len);
      params.total_seqlen_kv = static_cast<ck_tile::index_t>(std::min<int64_t>(
          params.total_seqlen_kv, static_cast<int64_t>(batch_size) * kv_truncate_len));
    }
  }
  params.q_ptr = q_dev.data_ptr();
  params.k_ptr = k_dev.data_ptr();
  params.v_ptr = v_dev.data_ptr();
  params.bias_ptr = nullptr; // bias is not supported at present
  params.o_ptr = o_dev.data_ptr();
  params.lse_ptr = lse_dev.defined() ? lse_dev.data_ptr() : nullptr;
  params.hdim_qk = hdim_qk;
  params.hdim_v = hdim_v;
  params.num_head = num_head;
  params.scale_s = alpha;
  params.attn_scale = attn_scale;
  params.seq_stride_q = q_dev.stride(-3);
  params.seq_stride_k = k_dev.stride(-3);
  params.seq_stride_v = v_dev.stride(-3);
  params.seq_stride_bias = 0;
  params.seq_stride_o = o_dev.stride(-3);
  params.seq_stride_lse = lse_dev.defined() ? lse_dev.stride(-2) : 0;
  params.nhead_stride_q = q_dev.stride(-2);
  params.nhead_stride_k = (num_head_kv == 1 && num_head > 1) ? 0 : k_dev.stride(-2);
  params.nhead_stride_v = (num_head_kv == 1 && num_head > 1) ? 0 : v_dev.stride(-2);
  params.nhead_stride_bias = 0;
  params.nhead_stride_o = o_dev.stride(-2);
  params.nhead_stride_lse = lse_dev.defined() ? lse_dev.stride(-1) : 0;
  params.use_causal = causal;
  params.window_size = max_attn_len;
  params.contextual_seqlen = contextual_seq_len;
  params.min_full_attn_seqlen = min_full_attn_seq_len;
  params.p_drop = 0.0f; // dropout is not supported at present
  params.philox_seed = 0UL;
  params.philox_offset = 0UL;
  params.use_softmax = use_softmax;
  params.is_training = is_training;
}

std::tuple<at::Tensor, at::Tensor> ck_hstu_mha_fwd_impl(
    const int64_t max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_q_offsets,
    const std::optional<at::Tensor>& seq_kv_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const int64_t max_attn_len,
    const int64_t min_full_attn_seq_len,
    const int64_t contextual_seq_len,
    const std::optional<at::Tensor>&
        attn_scale, // scale factor of SiLu(Q@K), nullopt/0 means using
                    // 1/max_seqlen for scaling
    bool use_softmax,
    bool is_cross_attention,
    const int64_t max_seq_len_kv, // KV extent for cross-attention dispatch; 0 -> max_seq_len
    const int64_t min_seq_len_q, // min Q seqlen for grid-schedule choice; 0 -> max_seq_len
    const int64_t min_seq_len_kv, // min KV seqlen for grid-schedule choice; 0 -> max_seq_len_kv
    bool is_training,
    const int64_t kv_truncate_len = 0, // > 0: keep kv_truncate_len keys per sequence (jagged only)
    bool kv_keep_last = false // with kv_truncate_len: keep the last keys, else the first
) {
  // attn_scale arrives as a Tensor so the value stays out of the traced graph
  // (it must not surface as a data-dependent SymFloat). For a device tensor the
  // kernel reads an fp32 copy through attn_scale_ptr: item<float>() would be a
  // blocking D2H per call. Graph lowering passes can turn a caller-side
  // .to(torch.float32) into another dtype, so any floating dtype is converted
  // here.
  float attn_scale_val = 0.0f;
  at::Tensor attn_scale_fp32; // alive until the kernels below are enqueued
  if (attn_scale.has_value()) {
    TORCH_CHECK(
        attn_scale->is_floating_point(),
        "attn_scale must be a floating-point tensor");
    attn_scale_fp32 = attn_scale->to(at::kFloat);
    if (!attn_scale_fp32.is_cuda()) {
      attn_scale_val = attn_scale_fp32.item<float>();
    }
  }
  q = switch_to_contiguous_if_needed(q);
  k = switch_to_contiguous_if_needed(k);
  v = switch_to_contiguous_if_needed(v);

  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
      "CK HSTU Attn only supports fp16 and bf16 data type");
  TORCH_CHECK(
      k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(
      v.scalar_type() == q_type, "query and value must have the same dtype");

  TORCH_CHECK(
      kv_truncate_len >= 0, "kv_truncate_len must be non-negative");
  TORCH_CHECK(
      kv_truncate_len == 0 || seq_q_offsets.has_value(),
      "kv_truncate_len is only supported for jagged inputs");
  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  at::Tensor seq_q_offsets_;
  at::Tensor seq_kv_offsets_;
  bool const is_jagged = seq_q_offsets.has_value();
  if (is_jagged) {
    seq_q_offsets_ = seq_q_offsets.value();
    seq_kv_offsets_ = seq_kv_offsets.value();
    CHECK_CONTIGUOUS(seq_q_offsets_);
    CHECK_CONTIGUOUS(seq_kv_offsets_);
    TORCH_CHECK(
        seq_q_offsets_.dtype() == torch::kInt32,
        "seq_q_offsets_ must have dtype torch.int32");
    TORCH_CHECK(
        seq_kv_offsets_.dtype() == torch::kInt32,
        "seq_kv_offsets_ must have dtype torch.int32");
  }

  at::Tensor num_targets_;
  bool const has_multiple_targets = num_targets.has_value();
  if (has_multiple_targets) {
    num_targets_ = num_targets.value();
    CHECK_CONTIGUOUS(num_targets_);
    TORCH_CHECK(
        num_targets_.dtype() == torch::kInt32,
        "num_targets_ must have dtype torch.int32");
  }

  auto const sizes = q.sizes();
  const int batch_size = !is_jagged ? sizes[0] : seq_q_offsets_.size(0) - 1;
  int32_t num_head = q.size(-2);
  int32_t num_head_kv = k.size(-2);
  TORCH_CHECK(
      num_head_kv == num_head || num_head_kv == 1,
      "K/V num_heads must equal Q num_heads or be 1 (GQA), got Q=",
      num_head, " K/V=", num_head_kv);
  TORCH_CHECK(
      v.size(-2) == num_head_kv,
      "K and V must have the same number of heads, got K=",
      num_head_kv, " V=", v.size(-2));
  TORCH_CHECK(
      q.size(-1) == k.size(-1),
      "Q and K must have the same head dimension, got Q=",
      q.size(-1), " K=", k.size(-1));
  int32_t hdim_qk = q.size(-1);
  int32_t hdim_v = v.size(-1);
  HstuAttentionNoGroupFwdParams params{};

  auto out = torch::empty_like(q);
  at::Tensor lse;
  if (use_softmax && is_training) {
    auto lse_sizes = q.sizes().vec();
    lse_sizes.pop_back();
    lse = torch::empty(lse_sizes, q.options().dtype(at::ScalarType::Float));
  }
  at::DeviceGuard device_guard{q.device()};

  set_fwd_params(
    params, //HstuAttentionFwdParams
    batch_size,
    q,
    k,
    v,
    out,
    lse,
    !is_jagged ? at::Tensor() : seq_q_offsets_,
    !is_jagged ? at::Tensor() : seq_kv_offsets_,
    !has_multiple_targets ? at::Tensor() : num_targets_,
    alpha,
    max_seq_len,
    max_seq_len_kv,
    min_seq_len_q,
    min_seq_len_kv,
    hdim_qk,
    hdim_v,
    num_head,
    causal,
    max_attn_len,
    contextual_seq_len,
    min_full_attn_seq_len,
    is_jagged,
    has_multiple_targets,
    attn_scale_val,
    use_softmax,
    is_cross_attention,
    is_training,
    kv_truncate_len,
    kv_keep_last
    );
  if (attn_scale_fp32.defined() && attn_scale_fp32.is_cuda()) {
    params.attn_scale_ptr = attn_scale_fp32.data_ptr();
  }
  params.workspace_allocator = {hstu_workspace_alloc, hstu_workspace_free};
  auto stream = at::hip::getCurrentHIPStream().stream();

  // Empty input handling: skip the launch but still write the result. With no
  // keys/values every query row attends to nothing, which the kernel writes as
  // zero output and -inf lse; an unwritten `out` would be uninitialized memory.
  if (q.numel() == 0 || k.numel() == 0 || v.numel() == 0) {
    out.zero_();
    if (lse.defined()) {
      lse.fill_(-std::numeric_limits<float>::infinity());
    }
    return {out, lse};
  }

  // no_group forward branches internally on params.is_jagged
  if (q_type == at::ScalarType::BFloat16) {
    hstu_attention_no_group_forward_bf16(params, stream);
  } else {
    hstu_attention_no_group_forward_fp16(params, stream);
  }
  return {out, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ck_hstu_mha_bwd(
    const int64_t max_seq_len,
    double alpha,
    const at::Tensor& grad_out,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const std::optional<at::Tensor>& softmax_lse,
    const std::optional<at::Tensor>& seq_q_offsets,
    const std::optional<at::Tensor>& seq_kv_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const int64_t max_attn_len,
    const int64_t min_full_attn_seq_len,
    const int64_t contextual_seq_len,
    const std::optional<at::Tensor>& attn_scale_tensor,
    bool use_softmax,
    bool is_cross_attention,
    const int64_t max_seq_len_kv,
    const int64_t min_seq_len_q,
    const int64_t min_seq_len_kv) {
  TORCH_CHECK(
      !use_softmax || (softmax_lse.has_value() && softmax_lse->defined()),
      "ck_hstu_mha_bwd: lse is required with use_softmax");
  const at::Tensor lse = use_softmax ? *softmax_lse : at::Tensor();
  const double attn_scale = attn_scale_tensor.has_value()
      ? static_cast<double>(attn_scale_tensor->item<float>())
      : 0.0;
  auto grad_out_contiguous = switch_to_contiguous_if_needed(grad_out);
  auto dq = torch::zeros_like(q);
  auto dk = torch::zeros_like(k);
  auto dv = torch::zeros_like(v);

  if (q.numel() == 0 || k.numel() == 0 || v.numel() == 0) {
    return {dq, dk, dv};
  }

  TORCH_CHECK(
      k.size(-2) == q.size(-2),
      "CK HSTU backward does not support GQA");

  at::Tensor seq_q_offsets_;
  at::Tensor seq_kv_offsets_;
  const bool is_jagged = seq_q_offsets.has_value();
  if (is_jagged) {
    seq_q_offsets_ = seq_q_offsets.value();
    seq_kv_offsets_ = seq_kv_offsets.value();
  }

  at::Tensor num_targets_;
  const bool has_multiple_targets = num_targets.has_value();
  if (has_multiple_targets) {
    num_targets_ = num_targets.value();
  }

  HstuAttentionNoGroupBwdParams params{};
  params.is_cross_attention = is_cross_attention;
  params.is_jagged = is_jagged;
  params.use_softmax = use_softmax;
  params.num_batch =
      !is_jagged ? q.size(0) : seq_q_offsets_.size(0) - 1;
  params.seq_q_offsets_ptr =
      is_jagged ? seq_q_offsets_.data_ptr() : nullptr;
  params.seq_kv_offsets_ptr =
      is_jagged ? seq_kv_offsets_.data_ptr() : nullptr;
  params.max_seqlen_q = max_seq_len;
  params.max_seqlen_kv =
      max_seq_len_kv > 0 ? max_seq_len_kv : max_seq_len;
  params.min_seqlen_q =
      min_seq_len_q > 0 ? min_seq_len_q : params.max_seqlen_q;
  params.min_seqlen_kv =
      min_seq_len_kv > 0 ? min_seq_len_kv : params.max_seqlen_kv;
  if (!is_jagged) {
    params.seqlen_q = q.size(-3);
    params.seqlen_kv = k.size(-3);
    params.batch_stride_q = q.stride(-4);
    params.batch_stride_k = k.stride(-4);
    params.batch_stride_v = v.stride(-4);
    params.batch_stride_o = out.stride(-4);
    params.batch_stride_do = grad_out_contiguous.stride(-4);
    params.batch_stride_lse = lse.defined() ? lse.stride(-3) : 0;
    params.batch_stride_dq = dq.stride(-4);
    params.batch_stride_dk = dk.stride(-4);
    params.batch_stride_dv = dv.stride(-4);
  }

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.bias_ptr = nullptr;
  params.o_ptr = out.data_ptr();
  params.do_ptr = grad_out_contiguous.data_ptr();
  params.lse_ptr = lse.defined() ? lse.data_ptr() : nullptr;
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.hdim_qk = q.size(-1);
  params.hdim_v = v.size(-1);
  params.num_head = q.size(-2);
  params.scale_s = alpha;
  params.attn_scale = attn_scale;
  params.seq_stride_q = q.stride(-3);
  params.seq_stride_k = k.stride(-3);
  params.seq_stride_v = v.stride(-3);
  params.seq_stride_o = out.stride(-3);
  params.seq_stride_do = grad_out_contiguous.stride(-3);
  params.seq_stride_lse = lse.defined() ? lse.stride(-2) : 0;
  params.seq_stride_dq = dq.stride(-3);
  params.seq_stride_dk = dk.stride(-3);
  params.seq_stride_dv = dv.stride(-3);
  params.nhead_stride_q = q.stride(-2);
  params.nhead_stride_k = k.stride(-2);
  params.nhead_stride_v = v.stride(-2);
  params.nhead_stride_o = out.stride(-2);
  params.nhead_stride_do = grad_out_contiguous.stride(-2);
  params.nhead_stride_lse = lse.defined() ? lse.stride(-1) : 0;
  params.nhead_stride_dq = dq.stride(-2);
  params.nhead_stride_dk = dk.stride(-2);
  params.nhead_stride_dv = dv.stride(-2);
  params.num_targets_ptr =
      has_multiple_targets ? num_targets_.data_ptr() : nullptr;
  params.use_causal = causal;
  params.window_size = max_attn_len;
  params.contextual_seqlen = contextual_seq_len;
  params.min_full_attn_seqlen = min_full_attn_seq_len;
  params.p_drop = 0.0f;
  params.philox_seed = 0UL;
  params.philox_offset = 0UL;

  at::DeviceGuard device_guard{q.device()};
#if defined(HSTU_DISABLE_BWD)
  TORCH_CHECK(false, "CK HSTU backward is not built for this GPU arch");
#else
  auto stream = at::hip::getCurrentHIPStream().stream();
  if (q.scalar_type() == at::ScalarType::BFloat16) {
    hstu_attention_no_group_backward_bf16(params, stream);
  } else {
    hstu_attention_no_group_backward_fp16(params, stream);
  }
#endif
  return {dq, dk, dv};
}

class HSTUFlashAttentionFunctionCK
    : public torch::autograd::Function<HSTUFlashAttentionFunctionCK> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      int64_t max_seq_len,
      double alpha,
      at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
      at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
      at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
      const std::optional<at::Tensor>& seq_q_offsets,
      const std::optional<at::Tensor>& seq_kv_offsets,
      bool causal,
      const std::optional<at::Tensor>& num_targets,
      int64_t max_attn_len,
      int64_t min_full_attn_seq_len,
      int64_t contextual_seq_len,
      const std::optional<at::Tensor>& attn_scale,
      bool use_softmax,
      bool is_cross_attention,
      int64_t max_seq_len_kv,
      int64_t min_seq_len_q,
      int64_t min_seq_len_kv,
      bool is_training,
      int64_t kv_truncate_len,
      bool kv_keep_last) {
    auto [out, lse] = ck_hstu_mha_fwd_impl(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_q_offsets,
        seq_kv_offsets,
        causal,
        num_targets,
        max_attn_len,
        min_full_attn_seq_len,
        contextual_seq_len,
        attn_scale,
        use_softmax,
        is_cross_attention,
        max_seq_len_kv,
        min_seq_len_q,
        min_seq_len_kv,
        is_training,
        kv_truncate_len,
        kv_keep_last);

    auto tensor_placeholder = torch::empty({0}, q.options());
    auto index_placeholder =
        torch::empty({0}, q.options().dtype(at::ScalarType::Int));
    ctx->save_for_backward({
        q,
        k,
        v,
        out,
        lse.defined() ? lse : tensor_placeholder,
        seq_q_offsets.has_value() ? seq_q_offsets.value() : index_placeholder,
        seq_kv_offsets.has_value() ? seq_kv_offsets.value() : index_placeholder,
        num_targets.has_value() ? num_targets.value() : index_placeholder,
    });
    ctx->saved_data["max_seq_len"] = max_seq_len;
    ctx->saved_data["alpha"] = alpha;
    ctx->saved_data["has_seq_offsets"] = seq_q_offsets.has_value();
    ctx->saved_data["causal"] = causal;
    ctx->saved_data["has_num_targets"] = num_targets.has_value();
    ctx->saved_data["max_attn_len"] = max_attn_len;
    ctx->saved_data["min_full_attn_seq_len"] = min_full_attn_seq_len;
    ctx->saved_data["contextual_seq_len"] = contextual_seq_len;
    // Kept as a Tensor: converting here would be a blocking D2H on every
    // forward, including inference, where backward never runs.
    ctx->saved_data["attn_scale"] =
        attn_scale.has_value() ? c10::IValue(*attn_scale) : c10::IValue();
    ctx->saved_data["use_softmax"] = use_softmax;
    ctx->saved_data["is_cross_attention"] = is_cross_attention;
    ctx->saved_data["max_seq_len_kv"] = max_seq_len_kv;
    ctx->saved_data["min_seq_len_q"] = min_seq_len_q;
    ctx->saved_data["min_seq_len_kv"] = min_seq_len_kv;
    return out;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    const bool has_seq_offsets =
        ctx->saved_data["has_seq_offsets"].toBool();
    const bool has_num_targets =
        ctx->saved_data["has_num_targets"].toBool();
    const bool use_softmax = ctx->saved_data["use_softmax"].toBool();

    auto [dq, dk, dv] = ck_hstu_mha_bwd(
        ctx->saved_data["max_seq_len"].toInt(),
        ctx->saved_data["alpha"].toDouble(),
        grad_outputs[0],
        saved[0],
        saved[1],
        saved[2],
        saved[3],
        use_softmax ? std::optional<at::Tensor>(saved[4]) : std::nullopt,
        has_seq_offsets ? std::optional<at::Tensor>(saved[5]) : std::nullopt,
        has_seq_offsets ? std::optional<at::Tensor>(saved[6]) : std::nullopt,
        ctx->saved_data["causal"].toBool(),
        has_num_targets ? std::optional<at::Tensor>(saved[7]) : std::nullopt,
        ctx->saved_data["max_attn_len"].toInt(),
        ctx->saved_data["min_full_attn_seq_len"].toInt(),
        ctx->saved_data["contextual_seq_len"].toInt(),
        ctx->saved_data["attn_scale"].isTensor()
            ? std::optional<at::Tensor>(
                  ctx->saved_data["attn_scale"].toTensor())
            : std::nullopt,
        use_softmax,
        ctx->saved_data["is_cross_attention"].toBool(),
        ctx->saved_data["max_seq_len_kv"].toInt(),
        ctx->saved_data["min_seq_len_q"].toInt(),
        ctx->saved_data["min_seq_len_kv"].toInt());

    return {
      at::Tensor(), // max_seq_len
      at::Tensor(), // alpha
      dq,            // q
      dk,            // k
      dv,            // v
      at::Tensor(), // seq_q_offsets
      at::Tensor(), // seq_kv_offsets
      at::Tensor(), // causal
      at::Tensor(), // num_targets
      at::Tensor(), // max_attn_len
      at::Tensor(), // min_full_attn_seq_len
      at::Tensor(), // contextual_seq_len
      at::Tensor(), // attn_scale
      at::Tensor(), // use_softmax
      at::Tensor(), // is_cross_attention
      at::Tensor(), // max_seq_len_kv
      at::Tensor(), // min_seq_len_q
      at::Tensor(), // min_seq_len_kv
      at::Tensor(), // is_training
      at::Tensor(), // kv_truncate_len
      at::Tensor() // kv_keep_last
    };
  }
};


at::Tensor ck_hstu_mha(
    const int64_t max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_q_offsets,
    const std::optional<at::Tensor>& seq_kv_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const int64_t max_attn_len,
    const int64_t min_full_attn_seq_len,
    const int64_t contextual_seq_len,
    const std::optional<at::Tensor>& attn_scale,
    bool use_softmax,
    bool is_cross_attention,
    const int64_t max_seq_len_kv,
    const int64_t min_seq_len_q,
    const int64_t min_seq_len_kv,
    const int64_t kv_truncate_len,
    bool kv_keep_last) {
  // The softmax LSE is only read by the backward, so skip storing it when autograd will not
  // record this call (forward() itself always runs with grad mode off).
  const bool is_training = at::GradMode::is_enabled() &&
      (q.requires_grad() || k.requires_grad() || v.requires_grad());
  // The backward kernels have no KV truncation, so their gradients would cover
  // the whole KV sequence.
  TORCH_CHECK(
      !is_training || kv_truncate_len == 0,
      "ck_hstu_mha: kv_truncate_len is forward-only");
  return HSTUFlashAttentionFunctionCK::apply(
      max_seq_len,
      alpha,
      q,
      k,
      v,
      seq_q_offsets,
      seq_kv_offsets,
      causal,
      num_targets,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      attn_scale,
      use_softmax,
      is_cross_attention,
      max_seq_len_kv,
      min_seq_len_q,
      min_seq_len_kv,
      is_training,
      kv_truncate_len,
      kv_keep_last);
}


TORCH_LIBRARY_FRAGMENT(ck_hstu_experimental, m) {
  m.impl(
      "ck_hstu_mha",
      torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(ck_hstu_mha)));
}


std::tuple<at::Tensor, at::Tensor> ck_hstu_mha_fwd(
    const int64_t max_seq_len,
    double alpha,
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    const std::optional<at::Tensor>& seq_q_offsets,
    const std::optional<at::Tensor>& seq_kv_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const int64_t max_attn_len,
    const int64_t min_full_attn_seq_len,
    const int64_t contextual_seq_len,
    const std::optional<at::Tensor>& attn_scale,
    bool use_softmax,
    bool is_cross_attention,
    const int64_t max_seq_len_kv,
    const int64_t min_seq_len_q,
    const int64_t min_seq_len_kv) {
  auto [out, lse] = ck_hstu_mha_fwd_impl(
      max_seq_len,
      alpha,
      q,
      k,
      v,
      seq_q_offsets,
      seq_kv_offsets,
      causal,
      num_targets,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      attn_scale,
      use_softmax,
      is_cross_attention,
      max_seq_len_kv,
      min_seq_len_q,
      min_seq_len_kv,
      /*is_training=*/true);
  // lse only exists with softmax; keep the second output defined so it
  // matches the Meta-dispatch (fake) kernel and can be saved for backward.
  if (!lse.defined()) {
    lse = at::empty({0}, q.options().dtype(at::ScalarType::Float));
  }
  return {out, lse};
}

TORCH_LIBRARY_IMPL(ck_hstu_experimental, CUDA, m) {
  m.impl("ck_hstu_mha_fwd", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(ck_hstu_mha_fwd)));
  m.impl("ck_hstu_mha_bwd", TORCH_FN(ck_hstu_mha_bwd));
}
