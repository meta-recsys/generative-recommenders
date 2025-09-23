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
#include <torch/library.h>

#include "common.h"
#include "fbgemm_gpu/sparse_ops.h" // @manual
#include "fbgemm_gpu/utils/fixed_divisor.cuh" // @manual

namespace hstu {

static constexpr int32_t kMaxThreads = 1024;

template <typename index_t, typename val_t>
__global__
__launch_bounds__(kMaxThreads) void _split_1d_jagged_jagged_cuda_kernel(
    int32_t B,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        combined_offsets,
    const at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits>
        combined_values,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        lengths_left,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets_left,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets_right,
    at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits> values_left,
    at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits> values_right) {
  for (auto b = blockIdx.x * blockDim.y + threadIdx.y;
       b < static_cast<uint32_t>(B);
       b += gridDim.x * blockDim.y) {
    auto combined_start = combined_offsets[b];
    auto left_len = lengths_left[b];
    auto right_len = combined_offsets[b + 1] - combined_offsets[b] - left_len;
    auto left_start = offsets_left[b];
    auto right_start = offsets_right[b];

    for (auto i = threadIdx.x; i < static_cast<uint32_t>(left_len + right_len);
         i += blockDim.x) {
      if (i < static_cast<uint32_t>(left_len)) {
        values_left[left_start + i] = combined_values[combined_start + i];
      } else {
        values_right[right_start + i - left_len] =
            combined_values[combined_start + i];
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> split_1d_jagged_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& lengths_right,
    const at::Tensor& combined_values) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(combined_values.get_device());
  TORCH_INTERNAL_ASSERT(lengths_left.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(lengths_right.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(
      combined_values.device().type() == at::DeviceType::CUDA);
  TORCH_CHECK(lengths_left.size(0) == lengths_right.size(0));

  auto B = lengths_left.size(0);
  auto L_left = lengths_left.sum().item<int64_t>();
  auto L_right = lengths_right.sum().item<int64_t>();
  TORCH_CHECK(L_left + L_right == combined_values.numel());
  TORCH_CHECK(L_left < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(L_right < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(combined_values.get_device() == lengths_left.get_device());
  TORCH_CHECK(combined_values.get_device() == lengths_right.get_device());

  auto values_left = at::empty({L_left}, combined_values.options());
  auto values_right = at::empty({L_right}, combined_values.options());

  if (L_left == 0 && L_right == 0) {
    return std::make_tuple(values_left, values_right);
  }

  const auto combined_lengths = lengths_left + lengths_right;
  const auto combined_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(combined_lengths.view({-1}));
  const auto offsets_left =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(lengths_left.view({-1}));
  const auto offsets_right =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(lengths_right.view({-1}));

  // Optimized thread block configuration based on benchmark results
  uint32_t B_blocks = 4;
  dim3 threads(256, B_blocks);
  auto blocks = div_round_up(B, B_blocks);

  AT_DISPATCH_INTEGRAL_TYPES(
      lengths_left.scalar_type(),
      "split_1d_jagged_jagged_values_cuda_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            combined_values.scalar_type(),
            "split_1d_jagged_jagged_values_cuda_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _split_1d_jagged_jagged_cuda_kernel<index_t, val_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  B,
                  combined_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  combined_values
                      .packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                  lengths_left
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  offsets_left
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  offsets_right
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  values_left
                      .packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                  values_right
                      .packed_accessor32<val_t, 1, at::RestrictPtrTraits>());
            });
      });

  return std::make_tuple(values_left, values_right);
}
} // namespace hstu
