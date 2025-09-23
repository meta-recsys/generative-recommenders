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

/*
 * Because different .SO may include the same CUDA CUB kernels, this results in
 * confusion, where libA may end up calling libB's cub kernel and causing
 * failures when we static link libcudart_static.a. To avoid this, we annotate
 * only the public functions and hide the rest.
 */
#define DLL_PUBLIC __attribute__((visibility("default")))

namespace hstu {
at::Tensor expand_1d_jagged_to_dense_cpu(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_len);

at::Tensor expand_1d_jagged_to_dense_meta(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const c10::SymInt max_len);

at::Tensor expand_1d_jagged_to_dense_cuda(
    const at::Tensor& values,
    const at::Tensor& offsets,
    const int64_t max_len);

at::Tensor complete_cumsum_cpu(const at::Tensor& values);

at::Tensor complete_cumsum_cuda(const at::Tensor& values);

at::Tensor complete_cumsum_meta(const at::Tensor& values);

at::Tensor concat_1d_jagged_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

at::Tensor concat_1d_jagged_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

at::Tensor concat_1d_jagged_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

std::tuple<at::Tensor, at::Tensor> split_1d_jagged_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& lengths_right,
    const at::Tensor& combined_values);

std::tuple<at::Tensor, at::Tensor> split_1d_jagged_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& lengths_right,
    const at::Tensor& combined_values);

std::tuple<at::Tensor, at::Tensor> split_1d_jagged_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& lengths_right,
    const at::Tensor& combined_values);

at::Tensor replace_last_n_with_jagged_cpu(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

at::Tensor replace_last_n_with_jagged_cuda(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

at::Tensor replace_last_n_with_jagged_meta(
    const at::Tensor& lengths_left,
    const at::Tensor& values_left,
    const at::Tensor& lengths_right,
    const at::Tensor& values_right);

DLL_PUBLIC std::tuple<at::Tensor, at::Tensor> sort_kv_pairs_meta(
    const at::Tensor& keys,
    const at::Tensor& values,
    const std::optional<int64_t>& end_bit,
    const bool descending = false) {
  TORCH_CHECK(
      keys.dtype() == at::kInt || keys.dtype() == at::kLong ||
      keys.dtype() == at::kByte || keys.dtype() == at::kShort);
  TORCH_CHECK(keys.dim() == 1);
  TORCH_CHECK(values.dim() == 1);
  return {at::empty_like(keys), at::empty_like(values)};
}

std::tuple<at::Tensor, at::Tensor> sort_kv_pairs_cuda(
    const at::Tensor& keys,
    const at::Tensor& values,
    const std::optional<int64_t>& end_bit,
    const bool descending = false);

} // namespace hstu

TORCH_LIBRARY_FRAGMENT(hstu, m) {
  m.def(
      "expand_1d_jagged_to_dense(Tensor values, Tensor offsets, SymInt max_len) -> Tensor");
  m.def(
      "concat_1d_jagged_jagged(Tensor lengths_left, Tensor values_left, Tensor lengths_right, Tensor values_right) -> Tensor");
  m.def(
      "split_1d_jagged_jagged(Tensor lengths_left, Tensor lengths_right, Tensor combined_values) -> (Tensor, Tensor)");
  m.def(
      "replace_last_n_with_jagged(Tensor lengths_left, Tensor values_left, Tensor lengths_right, Tensor values_right) -> Tensor");
  m.def("complete_cumsum(Tensor values) -> Tensor");
  m.def(
      "sort_kv_pairs(Tensor keys, Tensor values, int? end_bit=None, bool descending=False) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(hstu, CPU, m) {
  m.impl("expand_1d_jagged_to_dense", hstu::expand_1d_jagged_to_dense_cpu);
  m.impl("concat_1d_jagged_jagged", hstu::concat_1d_jagged_jagged_cpu);
  m.impl("split_1d_jagged_jagged", hstu::split_1d_jagged_jagged_cpu);
  m.impl("replace_last_n_with_jagged", hstu::replace_last_n_with_jagged_cpu);
  m.impl("complete_cumsum", hstu::complete_cumsum_cpu);
}

TORCH_LIBRARY_IMPL(hstu, CUDA, m) {
  m.impl("expand_1d_jagged_to_dense", hstu::expand_1d_jagged_to_dense_cuda);
  m.impl("concat_1d_jagged_jagged", hstu::concat_1d_jagged_jagged_cuda);
  m.impl("split_1d_jagged_jagged", hstu::split_1d_jagged_jagged_cuda);
  m.impl("replace_last_n_with_jagged", hstu::replace_last_n_with_jagged_cuda);
  m.impl("complete_cumsum", hstu::complete_cumsum_cuda);
  m.impl(
      "sort_kv_pairs",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(hstu::sort_kv_pairs_cuda)));
}

TORCH_LIBRARY_IMPL(hstu, Meta, m) {
  m.impl("expand_1d_jagged_to_dense", hstu::expand_1d_jagged_to_dense_meta);
  m.impl("concat_1d_jagged_jagged", hstu::concat_1d_jagged_jagged_meta);
  m.impl("split_1d_jagged_jagged", hstu::split_1d_jagged_jagged_meta);
  m.impl("replace_last_n_with_jagged", hstu::replace_last_n_with_jagged_meta);
  m.impl("complete_cumsum", hstu::complete_cumsum_meta);
  m.impl(
      "sort_kv_pairs",
      torch::dispatch(
          c10::DispatchKey::Meta, TORCH_FN(hstu::sort_kv_pairs_meta)));
}

TORCH_LIBRARY_IMPL(hstu, Autograd, m) {
  m.impl(
      "expand_1d_jagged_to_dense",
      torch::autograd::autogradNotImplementedFallback());
  m.impl("complete_cumsum", torch::autograd::autogradNotImplementedFallback());
}
