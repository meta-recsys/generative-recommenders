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
__global__ __launch_bounds__(kMaxThreads) void _jagged_transpose_1d_cuda_kernel(
    int32_t size1,
    int32_t size2,
    int32_t max_len,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> offsets,
    const at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits> values,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits> lengths,
    const at::PackedTensorAccessor32<index_t, 1, at::RestrictPtrTraits>
        trans_offsets,
    at::PackedTensorAccessor32<val_t, 1, at::RestrictPtrTraits> trans_values) {
  for (auto idx = blockIdx.x * blockDim.y + threadIdx.y;
       idx < static_cast<uint32_t>(size1 * size2);
       idx += gridDim.x * blockDim.y) {
    auto i = idx / size2;
    auto j = idx % size2;
    auto src_idx = i * size2 + j;
    auto dst_idx = j * size1 + i;
    auto src_offset = offsets[src_idx];
    auto src_length = lengths[src_idx];
    auto dst_offset = trans_offsets[dst_idx];

    for (auto k = threadIdx.x; k < static_cast<uint32_t>(src_length);
         k += blockDim.x) {
      trans_values[dst_offset + k] = values[src_offset + k];
    }
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> jagged_transpose_1d_cuda(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const at::Tensor& lengths,
    const int64_t max_len,
    const int64_t size1,
    const int64_t size2) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(offsets.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(lengths.device().type() == at::DeviceType::CUDA);
  TORCH_CHECK(offsets.size(0) == size1 * size2 + 1);
  TORCH_CHECK(lengths.size(0) == size1 * size2);
  TORCH_CHECK(values.get_device() == offsets.get_device());
  TORCH_CHECK(values.get_device() == lengths.get_device());

  auto trans_lengths =
      lengths.view({size1, size2}).transpose(0, 1).contiguous().view({-1});
  auto trans_offsets =
      fbgemm_gpu::asynchronous_complete_cumsum_gpu(trans_lengths);
  auto L_out = trans_offsets[-1].item<int64_t>();
  TORCH_CHECK(L_out < std::numeric_limits<int32_t>::max());
  auto trans_values = at::empty({L_out}, values.options());

  if (L_out == 0) {
    return std::make_tuple(trans_values, trans_offsets, trans_lengths);
  }

  // Optimized thread block configuration based on benchmark results
  uint32_t B_blocks = 4;
  dim3 threads(256, B_blocks);
  auto blocks = div_round_up(size1 * size2, B_blocks);

  AT_DISPATCH_INTEGRAL_TYPES(
      lengths.scalar_type(), "jagged_transpose_1d_cuda_kernel_input1", [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values.scalar_type(),
            "jagged_transpose_1d_cuda_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _jagged_transpose_1d_cuda_kernel<index_t, val_t><<<
                  blocks,
                  threads,
                  0,
                  at::cuda::getCurrentCUDAStream()>>>(
                  size1,
                  size2,
                  max_len,
                  offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  values.packed_accessor32<val_t, 1, at::RestrictPtrTraits>(),
                  lengths
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  trans_offsets
                      .packed_accessor32<index_t, 1, at::RestrictPtrTraits>(),
                  trans_values
                      .packed_accessor32<val_t, 1, at::RestrictPtrTraits>());
            });
      });

  return std::make_tuple(trans_values, trans_offsets, trans_lengths);
}
} // namespace hstu
