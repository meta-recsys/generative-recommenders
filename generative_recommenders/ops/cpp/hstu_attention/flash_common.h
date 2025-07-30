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

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>

#include <torch/nn/functional.h>

#include <torch/library.h> // @manual

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                           \
  TORCH_CHECK(                                        \
      x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

inline int round_up_headdim(int head_size) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
  if (head_size <= 64) {
    return 64;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
  if (head_size <= 96) {
    return 96;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  if (head_size <= 128) {
    return 128;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  if (head_size <= 192) {
    return 192;
  }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
  if (head_size <= 256) {
    return 256;
  }
#endif
  return 256;
}

inline int get_max_headdim() {
#ifndef FLASHATTENTION_DISABLE_HDIM256
  return 256;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
  return 192;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
  return 128;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
  return 96;
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM64
  return 64;
#endif
  return 0;
}

namespace hstu {

std::tuple<at::Tensor, std::optional<at::Tensor>> hstu_mha_fwd(
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
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true);

std::vector<at::Tensor> hstu_mha_bwd(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& dout,
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& out,
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    bool sort_by_length,
    bool const deterministic,
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    const std::optional<at::Tensor>& softmax_lse = std::nullopt);

std::tuple<at::Tensor, std::optional<at::Tensor>> hstu_mha_fwd_dummy(
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
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true);

std::vector<at::Tensor> hstu_mha_bwd_dummy(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& dout,
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& out,
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    bool sort_by_length,
    bool const deterministic,
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    const std::optional<at::Tensor>& softmax_lse = std::nullopt);

std::tuple<at::Tensor, std::optional<at::Tensor>> hstu_mha_fwd_meta(
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
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true);
} // namespace hstu
