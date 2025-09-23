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
void _split_1d_jagged_jagged_cpu_kernel(
    int32_t B,
    const at::TensorAccessor<index_t, 1>& combined_offsets,
    const at::TensorAccessor<val_t, 1>& combined_values,
    const at::TensorAccessor<index_t, 1>& lengths_left,
    const at::TensorAccessor<index_t, 1>& offsets_left,
    const at::TensorAccessor<index_t, 1>& offsets_right,
    at::TensorAccessor<val_t, 1> values_left,
    at::TensorAccessor<val_t, 1> values_right) {
  for (auto b : c10::irange(B)) {
    auto combined_start = combined_offsets[b];
    auto left_len = lengths_left[b];
    auto left_start = offsets_left[b];
    auto right_start = offsets_right[b];

    for (auto i = 0; i < left_len; ++i) {
      values_left[left_start + i] = combined_values[combined_start + i];
    }

    auto right_len = combined_offsets[b + 1] - combined_offsets[b] - left_len;
    for (auto i = 0; i < right_len; ++i) {
      values_right[right_start + i] =
          combined_values[combined_start + left_len + i];
    }
  }
}

std::tuple<at::Tensor, at::Tensor> split_1d_jagged_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& lengths_right,
    const at::Tensor& combined_values) {
  TORCH_INTERNAL_ASSERT(lengths_left.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(lengths_right.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(combined_values.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(lengths_left.size(0) == lengths_right.size(0));
  auto B = lengths_left.size(0);

  auto L_left = lengths_left.sum().item<int64_t>();
  auto L_right = lengths_right.sum().item<int64_t>();
  TORCH_CHECK(L_left + L_right == combined_values.numel());

  auto values_left = at::empty({L_left}, combined_values.options());
  auto values_right = at::empty({L_right}, combined_values.options());

  if (L_left == 0 && L_right == 0) {
    return std::make_tuple(values_left, values_right);
  }

  const auto combined_lengths = lengths_left + lengths_right;
  const auto combined_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(combined_lengths.view({-1}));
  const auto offsets_left =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(lengths_left.view({-1}));
  const auto offsets_right =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(lengths_right.view({-1}));

  AT_DISPATCH_INTEGRAL_TYPES(
      lengths_left.scalar_type(),
      "split_1d_jagged_jagged_values_cpu_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            combined_values.scalar_type(),
            "split_1d_jagged_jagged_values_cpu_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _split_1d_jagged_jagged_cpu_kernel<index_t, val_t>(
                  B,
                  combined_offsets.accessor<index_t, 1>(),
                  combined_values.accessor<val_t, 1>(),
                  lengths_left.accessor<index_t, 1>(),
                  offsets_left.accessor<index_t, 1>(),
                  offsets_right.accessor<index_t, 1>(),
                  values_left.accessor<val_t, 1>(),
                  values_right.accessor<val_t, 1>());
            });
      });

  return std::make_tuple(values_left, values_right);
}

std::tuple<at::Tensor, at::Tensor> split_1d_jagged_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& lengths_right,
    const at::Tensor& combined_values) {
  auto L_left = lengths_left.sum().item<int64_t>();
  auto L_right = lengths_right.sum().item<int64_t>();

  auto values_left = at::native::empty_meta_symint(
      {L_left},
      /*dtype=*/::std::make_optional(combined_values.scalar_type()),
      /*layout=*/::std::make_optional(combined_values.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);

  auto values_right = at::native::empty_meta_symint(
      {L_right},
      /*dtype=*/::std::make_optional(combined_values.scalar_type()),
      /*layout=*/::std::make_optional(combined_values.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);

  return std::make_tuple(values_left, values_right);
}
} // namespace hstu
