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
#include <c10/cuda/CUDAGuard.h>

#include "fbgemm_gpu/sparse_ops.h" // @manual

namespace hstu {

template <typename index_t, typename val_t>
void _concat_1d_jagged_jagged_cpu_kernel(
    int32_t B,
    const at::TensorAccessor<index_t, 1>& offsets_left,
    const at::TensorAccessor<val_t, 1>& values_left,
    const at::TensorAccessor<index_t, 1>& offsets_right,
    const at::TensorAccessor<val_t, 1>& values_right,
    at::TensorAccessor<val_t, 1> combined_values) {
  for (auto b : c10::irange(B)) {
    auto left_start = offsets_left[b];
    auto left_len = offsets_left[b + 1] - left_start;
    auto right_start = offsets_right[b];
    auto right_len = offsets_right[b + 1] - right_start;
    auto combined_start = left_start + right_start;
    for (auto i = 0; i < left_len; ++i) {
      combined_values[combined_start + i] = values_left[left_start + i];
    }
    for (auto i = 0; i < right_len; ++i) {
      combined_values[left_len + combined_start + i] =
          values_right[right_start + i];
    }
  }
}

at::Tensor concat_1d_jagged_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right) {
  TORCH_INTERNAL_ASSERT(lengths_left.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(values_left.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(lengths_right.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(values_right.device().type() == at::DeviceType::CPU);
  auto L = values_left.numel() + values_right.numel();
  TORCH_CHECK(L < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(lengths_left.size(0) == lengths_right.size(0));
  auto B = lengths_left.size(0);
  auto combined_values = at::empty({L}, values_left.options());
  if (L == 0) {
    return combined_values;
  }
  const auto offsets_left =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(lengths_left.view({-1}));
  const auto offsets_right =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(lengths_right.view({-1}));
  AT_DISPATCH_INTEGRAL_TYPES(
      lengths_left.scalar_type(),
      "concat_1d_jagged_jagged_values_cpu_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values_left.scalar_type(),
            "concat_1d_jagged_jagged_values_cpu_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _concat_1d_jagged_jagged_cpu_kernel<index_t, val_t>(
                  B,
                  offsets_left.accessor<index_t, 1>(),
                  values_left.accessor<val_t, 1>(),
                  offsets_right.accessor<index_t, 1>(),
                  values_right.accessor<val_t, 1>(),
                  combined_values.accessor<val_t, 1>());
            });
      });
  return combined_values;
}

at::Tensor concat_1d_jagged_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right) {
  auto L = values_left.numel() + values_right.numel();
  return at::native::empty_meta_symint(
      {L},
      /*dtype=*/::std::make_optional(values_left.scalar_type()),
      /*layout=*/::std::make_optional(values_left.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);
}
} // namespace hstu
