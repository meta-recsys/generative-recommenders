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

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "fbgemm_gpu/sparse_ops.h" // @manual

namespace hstu {

template <typename index_t, typename val_t>
void _jagged_transpose_1d_cpu_kernel(
    int32_t size1,
    int32_t size2,
    int32_t max_len,
    const at::TensorAccessor<index_t, 1>& offsets,
    const at::TensorAccessor<val_t, 1>& values,
    const at::TensorAccessor<index_t, 1>& lengths,
    const at::TensorAccessor<index_t, 1>& trans_offsets,
    at::TensorAccessor<val_t, 1> trans_values) {
  for (auto i : c10::irange(size1)) {
    for (auto j : c10::irange(size2)) {
      auto src_idx = i * size2 + j;
      auto dst_idx = j * size1 + i;
      auto src_offset = offsets[src_idx];
      auto src_length = lengths[src_idx];
      auto dst_offset = trans_offsets[dst_idx];

      for (auto k = 0; k < src_length; ++k) {
        trans_values[dst_offset + k] = values[src_offset + k];
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> jagged_transpose_1d_cpu(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const at::Tensor& lengths,
    const int64_t max_len,
    const int64_t size1,
    const int64_t size2) {
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(offsets.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(lengths.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(offsets.size(0) == size1 * size2 + 1);
  TORCH_CHECK(lengths.size(0) == size1 * size2);

  auto trans_lengths =
      lengths.view({size1, size2}).transpose(0, 1).contiguous().view({-1});
  auto trans_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(trans_lengths);
  auto L_out = trans_offsets[-1].item<int64_t>();
  auto trans_values = at::empty({L_out}, values.options());

  if (L_out == 0) {
    return std::make_tuple(trans_values, trans_offsets, trans_lengths);
  }

  AT_DISPATCH_INTEGRAL_TYPES(
      lengths.scalar_type(), "jagged_transpose_1d_cpu_kernel_input1", [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values.scalar_type(),
            "jagged_transpose_1d_cpu_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _jagged_transpose_1d_cpu_kernel<index_t, val_t>(
                  size1,
                  size2,
                  max_len,
                  offsets.accessor<index_t, 1>(),
                  values.accessor<val_t, 1>(),
                  lengths.accessor<index_t, 1>(),
                  trans_offsets.accessor<index_t, 1>(),
                  trans_values.accessor<val_t, 1>());
            });
      });

  return std::make_tuple(trans_values, trans_offsets, trans_lengths);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> jagged_transpose_1d_meta(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const at::Tensor& lengths,
    const int64_t max_len,
    const int64_t size1,
    const int64_t size2) {
  auto trans_lengths =
      lengths.view({size1, size2}).transpose(0, 1).contiguous().view({-1});
  auto L_out = trans_lengths.sum().item<int64_t>();

  auto trans_values = at::native::empty_meta_symint(
      {L_out},
      /*dtype=*/::std::make_optional(values.scalar_type()),
      /*layout=*/::std::make_optional(values.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);

  auto trans_offsets = at::native::empty_meta_symint(
      {size1 * size2 + 1},
      /*dtype=*/::std::make_optional(lengths.scalar_type()),
      /*layout=*/::std::make_optional(lengths.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);

  return std::make_tuple(trans_values, trans_offsets, trans_lengths);
}
} // namespace hstu
