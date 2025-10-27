#include <generative_recommenders/ops/cpp/common.h>
#include <generative_recommenders/ops/cpp/sort_kv_pairs_cuda_kernels_template.h>

#include <cub/device/device_radix_sort.cuh>

namespace hstu {

template <>
DLL_PUBLIC std::tuple<at::Tensor, at::Tensor>
sort_kv_pairs_cuda_dispatched<SUB_KEY_T, SUB_VALUE_T>(
    const at::Tensor& keys,
    const at::Tensor& values,
    const std::optional<int64_t>& end_bit,
    const bool descending) {
  size_t temp_storage_bytes = 0;
  auto keys_contig = keys.contiguous();
  auto values_contig = values.contiguous();
  auto sorted_keys = at::empty_like(keys_contig);
  auto sorted_values = at::empty_like(values_contig);

  if (descending) {
    AT_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr,
            temp_storage_bytes,
            keys_contig.data_ptr<SUB_KEY_T>(),
            sorted_keys.data_ptr<SUB_KEY_T>(),
            values_contig.data_ptr<SUB_VALUE_T>(),
            sorted_values.data_ptr<SUB_VALUE_T>(),
            keys_contig.numel(),
            0,
            end_bit.has_value() ? end_bit.value() : sizeof(SUB_KEY_T) * 8,
            at::cuda::getCurrentCUDAStream(),
            false));
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        keys_contig.options().dtype(at::kByte));
    AT_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairsDescending(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            keys_contig.data_ptr<SUB_KEY_T>(),
            sorted_keys.data_ptr<SUB_KEY_T>(),
            values_contig.data_ptr<SUB_VALUE_T>(),
            sorted_values.data_ptr<SUB_VALUE_T>(),
            keys_contig.numel(),
            0,
            end_bit.has_value() ? end_bit.value() : sizeof(SUB_KEY_T) * 8,
            at::cuda::getCurrentCUDAStream(),
            false));
  } else {
    AT_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            keys_contig.data_ptr<SUB_KEY_T>(),
            sorted_keys.data_ptr<SUB_KEY_T>(),
            values_contig.data_ptr<SUB_VALUE_T>(),
            sorted_values.data_ptr<SUB_VALUE_T>(),
            keys_contig.numel(),
            0,
            end_bit.has_value() ? end_bit.value() : sizeof(SUB_KEY_T) * 8,
            at::cuda::getCurrentCUDAStream(),
            false));
    auto temp_storage = at::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        keys_contig.options().dtype(at::kByte));
    AT_CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            keys_contig.data_ptr<SUB_KEY_T>(),
            sorted_keys.data_ptr<SUB_KEY_T>(),
            values_contig.data_ptr<SUB_VALUE_T>(),
            sorted_values.data_ptr<SUB_VALUE_T>(),
            keys_contig.numel(),
            0,
            end_bit.has_value() ? end_bit.value() : sizeof(SUB_KEY_T) * 8,
            at::cuda::getCurrentCUDAStream(),
            false));
  }

  return {std::move(sorted_keys), std::move(sorted_values)};
}

} // namespace hstu
