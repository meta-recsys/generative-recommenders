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

#include <cutlass/numeric_types.h>
#include <torch/nn/functional.h>
#include <torch/torch.h> // @manual
#include <torch/version.h> // For TORCH_VERSION* macros

#include "flash_common.h"

namespace hstu {

at::Tensor hstu_mha_fwd_meta(
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
    const int64_t sm_margin) {
  auto q_type = q.scalar_type();
  auto const sizes = q.sym_sizes();
  at::Tensor seq_offsets_;
  bool const is_jagged = seq_offsets.has_value();
  if (is_jagged) {
    seq_offsets_ = seq_offsets.value();
  }
  const c10::SymInt batch_size =
      !is_jagged ? sizes[0] : seq_offsets_.sym_sizes()[0] - 1;
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
};

} // namespace hstu
