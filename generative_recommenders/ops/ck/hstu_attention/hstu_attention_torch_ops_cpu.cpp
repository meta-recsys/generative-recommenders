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
#include <torch/library.h> // @manual
#include <torch/nn/functional.h>
#include <torch/torch.h> // @manual

at::Tensor ck_hstu_mha_fwd_dummy(
    const int64_t max_seq_len,
    double /*alpha*/,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& /*k*/, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_q_offsets,
    const std::optional<at::Tensor>& /*seq_kv_offsets*/,
    bool /*causal*/,
    const std::optional<at::Tensor>& /*num_targets*/,
    const int64_t /*max_attn_len*/,
    const int64_t /*min_full_attn_seq_len*/,
    const int64_t /*contextual_seq_len*/,
    const std::optional<at::Tensor>& /*attn_scale*/,
    bool /*use_softmax*/,
    bool /*is_cross_attention*/,
    const int64_t /*max_seq_len_kv*/) {
  auto const sizes = q.sizes();
  at::Tensor seq_q_offsets_;
  bool const is_jagged = seq_q_offsets.has_value();
  if (is_jagged) {
    seq_q_offsets_ = seq_q_offsets.value();
  }
  const int batch_size = !is_jagged ? sizes[0] : seq_q_offsets_.size(0) - 1;
  int total_seq_len = !is_jagged ? batch_size * max_seq_len : sizes[0];
  int num_heads = q.size(-2);
  //   int const qk_head_size = q.size(-1);
  int const v_head_size = v.size(-1);
  //   int const max_headdim = get_max_headdim();
  auto out_type = at::ScalarType::BFloat16;
  auto opts = q.options();

  at::Tensor out;
  if (!is_jagged) {
    out = torch::empty(
        {batch_size, max_seq_len, num_heads, v_head_size},
        opts.dtype(out_type));
  } else {
    out = torch::empty(
        {total_seq_len, num_heads, v_head_size}, opts.dtype(out_type));
  }
  return out;
}

at::Tensor ck_hstu_mha_cpu(
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
    const int64_t /*kv_truncate_len*/,
    bool /*kv_keep_last*/) {
  return ck_hstu_mha_fwd_dummy(
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
      max_seq_len_kv);
}

at::Tensor ck_hstu_mha_fwd_meta(
    const at::SymInt max_seq_len,
    double /*alpha*/,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& /*k*/, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_q_offsets,
    const std::optional<at::Tensor>& /*seq_kv_offsets*/,
    bool /*causal*/,
    const std::optional<at::Tensor>& /*num_targets*/,
    const int64_t /*max_attn_len*/,
    const int64_t /*min_full_attn_seq_len*/,
    const int64_t /*contextual_seq_len*/,
    const std::optional<at::Tensor>& /*attn_scale*/,
    bool /*use_softmax*/,
    bool /*is_cross_attention*/,
    const at::SymInt& /*max_seq_len_kv*/) {
  auto q_type = q.scalar_type();
  auto const sizes = q.sym_sizes();
  at::Tensor seq_q_offsets_;
  bool const is_jagged = seq_q_offsets.has_value();
  if (is_jagged) {
    seq_q_offsets_ = seq_q_offsets.value();
  }
  const c10::SymInt batch_size =
      !is_jagged ? sizes[0] : seq_q_offsets_.sym_sizes()[0] - 1;
  auto total_seq_len = !is_jagged ? batch_size * max_seq_len : sizes[0];
  const auto& num_heads = sizes[sizes.size() - 2];
  auto v_head_size = v.sym_sizes()[v.sym_sizes().size() - 1];
  auto out_type = q_type == at::ScalarType::Float8_e4m3fn
      ? at::ScalarType::BFloat16
      : q_type;
  auto opts = q.options();

  at::Tensor out;
  if (!is_jagged) {
    out = at::empty_symint(
        {batch_size, max_seq_len, num_heads, v_head_size},
        opts.dtype(out_type));
  } else {
    out = at::empty_symint(
        {total_seq_len, num_heads, v_head_size}, opts.dtype(out_type));
  }
  return out;
}

at::Tensor ck_hstu_mha_meta(
    const at::SymInt max_seq_len,
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
    const at::SymInt max_seq_len_kv,
    const at::SymInt min_seq_len_q,
    const at::SymInt min_seq_len_kv,
    const int64_t /*kv_truncate_len*/,
    bool /*kv_keep_last*/) {
  return ck_hstu_mha_fwd_meta(
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
      max_seq_len_kv);
}

std::tuple<at::Tensor, at::Tensor> ck_hstu_mha_fwd_op_meta(
    const at::SymInt max_seq_len,
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
    const at::SymInt max_seq_len_kv,
    const at::SymInt /*min_seq_len_q*/,
    const at::SymInt /*min_seq_len_kv*/) {
  auto out = ck_hstu_mha_fwd_meta(
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
      max_seq_len_kv);
  auto lse_opts = q.options().dtype(at::ScalarType::Float);
  auto lse = use_softmax
      ? at::empty_symint(q.sym_sizes().slice(0, q.dim() - 1), lse_opts)
      : at::empty({0}, lse_opts);
  return {out, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> ck_hstu_mha_bwd_meta(
    const at::SymInt /*max_seq_len*/,
    double /*alpha*/,
    const at::Tensor& /*grad_out*/,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& /*out*/,
    const std::optional<at::Tensor>& /*lse*/,
    const std::optional<at::Tensor>& /*seq_q_offsets*/,
    const std::optional<at::Tensor>& /*seq_kv_offsets*/,
    bool /*causal*/,
    const std::optional<at::Tensor>& /*num_targets*/,
    const int64_t /*max_attn_len*/,
    const int64_t /*min_full_attn_seq_len*/,
    const int64_t /*contextual_seq_len*/,
    const std::optional<at::Tensor>& /*attn_scale*/,
    bool /*use_softmax*/,
    bool /*is_cross_attention*/,
    const at::SymInt /*max_seq_len_kv*/,
    const at::SymInt /*min_seq_len_q*/,
    const at::SymInt /*min_seq_len_kv*/) {
  return {at::empty_like(q), at::empty_like(k), at::empty_like(v)};
}

TORCH_LIBRARY_FRAGMENT(ck_hstu_experimental, m) {
  m.impl(
      "ck_hstu_mha",
      torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(ck_hstu_mha_cpu)));
  m.impl(
      "ck_hstu_mha",
      torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(ck_hstu_mha_meta)));
  m.impl(
      "ck_hstu_mha_fwd",
      torch::dispatch(
          c10::DispatchKey::Meta, TORCH_FN(ck_hstu_mha_fwd_op_meta)));
  m.impl(
      "ck_hstu_mha_bwd",
      torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(ck_hstu_mha_bwd_meta)));
}
