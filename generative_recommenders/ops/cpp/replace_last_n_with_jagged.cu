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
__launch_bounds__(kMaxThreads) void _replace_last_n_with_jagged_cuda_kernel(
    int32_t B,
    int32_t D,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        lengths_left,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets_left,
    const at::PackedTensorAccessor32<val_t, 2, at::RestrictPtrTraits>
        values_left,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        lengths_right,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        offsets_right,
    const at::PackedTensorAccessor32<val_t, 2, at::RestrictPtrTraits>
        values_right,
    at::PackedTensorAccessor32<val_t, 2, at::RestrictPtrTraits> output) {
  for (auto b = blockIdx.x * blockDim.y + threadIdx.y;
       b < static_cast<uint32_t>(B);
       b += gridDim.x * blockDim.y) {
    auto left_start = offsets_left[b];
    auto left_len = lengths_left[b];
    auto right_start = offsets_right[b];
    auto right_len = lengths_right[b];
    auto output_start = offsets_left[b];
    auto keep_len = left_len - right_len;

    for (auto i = threadIdx.x; i < static_cast<uint32_t>(left_len * D);
         i += blockDim.x) {
      auto seq_pos = i / D;
      auto dim_pos = i % D;
      if (seq_pos < static_cast<uint32_t>(keep_len)) {
        output[output_start + seq_pos][dim_pos] =
            values_left[left_start + seq_pos][dim_pos];
      } else {
        auto right_idx = seq_pos - keep_len;
        if (right_idx < static_cast<uint32_t>(right_len)) {
          output[output_start + seq_pos][dim_pos] =
              values_right[right_start + right_idx][dim_pos];
        }
      }
    }
  }
}

at::Tensor replace_last_n_with_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values_left.get_device());
  TORCH_INTERNAL_ASSERT(lengths_left.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(lengths_right.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(values_left.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(values_right.device().type() == at::DeviceType::CUDA);
  TORCH_CHECK(lengths_left.size(0) == lengths_right.size(0));
  TORCH_CHECK(values_left.size(1) == values_right.size(1));

  auto B = lengths_left.size(0);
  auto D = values_left.size(1);
  auto L_out = lengths_left.sum().item<int64_t>();
  TORCH_CHECK(L_out < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(values_left.get_device() == lengths_left.get_device());
  TORCH_CHECK(values_left.get_device() == lengths_right.get_device());
  TORCH_CHECK(values_left.get_device() == values_right.get_device());

  auto output = at::empty({L_out, D}, values_left.options());

  if (L_out == 0) {
    return output;
  }

  const auto offsets_left =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(lengths_left.view({-1}));
  const auto offsets_right =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(lengths_right.view({-1}));

  // Optimized thread block configuration based on benchmark results
  uint32_t B_blocks, threads_x;
  B_blocks = 4;
  threads_x = 256;

  dim3 threads(threads_x, B_blocks);
  auto blocks = div_round_up(B, B_blocks);

  AT_DISPATCH_INTEGRAL_TYPES(
      lengths_left.scalar_type(),
      "replace_last_n_with_jagged_cuda_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values_left.scalar_type(),
            "replace_last_n_with_jagged_cuda_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _replace_last_n_with_jagged_cuda_kernel<index_t, val_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  B,
                  D,
                  lengths_left
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  offsets_left
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  values_left
                      .packed_accessor32<val_t, 2, at::RestrictPtrTraits>(),
                  lengths_right
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  offsets_right
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  values_right
                      .packed_accessor32<val_t, 2, at::RestrictPtrTraits>(),
                  output.packed_accessor32<val_t, 2, at::RestrictPtrTraits>());
            });
      });

  return output;
}
} // namespace hstu
