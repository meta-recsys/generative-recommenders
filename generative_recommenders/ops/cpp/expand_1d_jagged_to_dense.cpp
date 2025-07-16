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

namespace hstu {

template <typename index_t, typename val_t>
void expand_1d_jagged_to_dense_cpu_kernel_(
    int64_t B,
    int64_t max_len,
    const at::TensorAccessor<val_t, 1>& values,
    const at::TensorAccessor<index_t, 1>& offsets,
    at::TensorAccessor<val_t, 2> output) {
  for (auto i : c10::irange(B)) {
    int64_t begin = offsets[i];
    int64_t end = offsets[i + 1];
    if (end - begin == 0) {
      for (int64_t j : c10::irange(max_len)) {
        output[i][j] = 0;
        continue;
      }
    } else {
      int64_t j = 0;
      for (; j < std::min(end - begin, max_len); ++j) {
        output[i][j] = values[begin + j];
      }
      for (; j < max_len; ++j) {
        output[i][j] = values[end - 1];
      }
    }
  } // for each i
}

at::Tensor expand_1d_jagged_to_dense_cpu(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_len) {
  TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(offsets.device().type() == at::DeviceType::CPU);
  TORCH_CHECK(values.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(max_len >= 0);
  auto B = offsets.size(0) - 1;
  auto output = at::empty({B, max_len}, values.options());
  if (values.numel() == 0 || max_len == 0) {
    return output;
  }
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      values.scalar_type(),
      "expand_1d_jagged_to_dense_cpu_input1",
      [&] {
        using val_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(
            offsets.scalar_type(), "expand_1d_jagged_to_dense_cpu_input2", [&] {
              using index_t = scalar_t;
              expand_1d_jagged_to_dense_cpu_kernel_<index_t, val_t>(
                  B,
                  max_len,
                  values.accessor<val_t, 1>(),
                  offsets.accessor<index_t, 1>(),
                  output.accessor<val_t, 2>());
            });
      });
  return output;
}

at::Tensor expand_1d_jagged_to_dense_meta(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const c10::SymInt max_len) {
  auto B = offsets.sym_size(0) - 1;
  auto output = at::empty_symint({B, max_len}, values.options());
  return output;
}

} // namespace hstu
