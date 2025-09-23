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
void _replace_last_n_with_jagged_cpu_kernel(
    int32_t B,
    const at::TensorAccessor<index_t, 1>& lengths_left,
    const at::TensorAccessor<index_t, 1>& offsets_left,
    const at::TensorAccessor<val_t, 2>& values_left,
    const at::TensorAccessor<index_t, 1>& lengths_right,
    const at::TensorAccessor<index_t, 1>& offsets_right,
    const at::TensorAccessor<val_t, 2>& values_right,
    const at::TensorAccessor<index_t, 1>& output_offsets,
    at::TensorAccessor<val_t, 2> output) {
  for (auto b : c10::irange(B)) {
    auto left_start = offsets_left[b];
    auto left_len = lengths_left[b];
    auto right_start = offsets_right[b];
    auto right_len = lengths_right[b];
    auto output_start = output_offsets[b];

    auto keep_len = left_len - right_len;

    for (auto i = 0; i < left_len; ++i) {
      for (auto d = 0; d < values_left.size(1); ++d) {
        if (i < keep_len) {
          output[output_start + i][d] = values_left[left_start + i][d];
        } else {
          auto right_idx = i - keep_len;
          if (right_idx < right_len) {
            output[output_start + i][d] =
                values_right[right_start + right_idx][d];
          }
        }
      }
    }
  }
}

at::Tensor replace_last_n_with_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right) {
  TORCH_INTERNAL_ASSERT(lengths_left.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(lengths_right.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(values_left.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(values_right.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(lengths_left.size(0) == lengths_right.size(0));
  TORCH_CHECK(values_left.size(1) == values_right.size(1));

  auto B = lengths_left.size(0);
  auto D = values_left.size(1);

  auto L_out = lengths_left.sum().item<int64_t>();

  auto output = at::empty({L_out, D}, values_left.options());

  if (L_out == 0) {
    return output;
  }

  const auto offsets_left =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(lengths_left.view({-1}));
  const auto offsets_right =
      fbgemm_gpu::asynchronous_complete_cumsum_cpu(lengths_right.view({-1}));
  const auto output_offsets = offsets_left;

  AT_DISPATCH_INTEGRAL_TYPES(
      lengths_left.scalar_type(),
      "replace_last_n_with_jagged_cpu_kernel_input1",
      [&] {
        using index_t = scalar_t;
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::BFloat16,
            at::ScalarType::Half,
            values_left.scalar_type(),
            "replace_last_n_with_jagged_cpu_kernel_input2",
            [&] {
              using val_t = scalar_t;
              _replace_last_n_with_jagged_cpu_kernel<index_t, val_t>(
                  B,
                  lengths_left.accessor<index_t, 1>(),
                  offsets_left.accessor<index_t, 1>(),
                  values_left.accessor<val_t, 2>(),
                  lengths_right.accessor<index_t, 1>(),
                  offsets_right.accessor<index_t, 1>(),
                  values_right.accessor<val_t, 2>(),
                  output_offsets.accessor<index_t, 1>(),
                  output.accessor<val_t, 2>());
            });
      });

  return output;
}

at::Tensor replace_last_n_with_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right) {
  auto L_out = lengths_left.sum().item<int64_t>();
  auto D = values_left.size(1);

  auto output = at::native::empty_meta_symint(
      {L_out, D},
      /*dtype=*/::std::make_optional(values_left.scalar_type()),
      /*layout=*/::std::make_optional(values_left.layout()),
      /*device=*/::std::make_optional(c10::Device(c10::kMeta)),
      /*pin_memory=*/::std::nullopt);

  return output;
}
} // namespace hstu
