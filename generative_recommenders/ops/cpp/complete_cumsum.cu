#include "common.h"

#include <cub/device/device_scan.cuh>

namespace hstu {

DLL_PUBLIC at::Tensor complete_cumsum_cuda(const at::Tensor& values) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(values.get_device());

  TORCH_CHECK(values.numel() < std::numeric_limits<int32_t>::max());
  TORCH_CHECK(values.dim() == 1);
  const auto values_contig = values.contiguous();

  auto cumsum = at::empty({values_contig.numel() + 1}, values_contig.options());
  cumsum[0].zero_();

  AT_DISPATCH_FLOATING_TYPES_AND4(
      at::ScalarType::Int,
      at::ScalarType::Long,
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      values_contig.scalar_type(),
      "complete_cumsum_cuda",
      [&] {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(
            cub::DeviceScan::InclusiveSum(
                nullptr,
                temp_storage_bytes,
                values_contig.data_ptr<scalar_t>(),
                cumsum.data_ptr<scalar_t>() + 1,
                values_contig.numel(),
                at::cuda::getCurrentCUDAStream()));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            values_contig.options().dtype(at::kByte));
        AT_CUDA_CHECK(
            cub::DeviceScan::InclusiveSum(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                values_contig.data_ptr<scalar_t>(),
                cumsum.data_ptr<scalar_t>() + 1,
                values_contig.numel(),
                at::cuda::getCurrentCUDAStream()));
      });

  return cumsum;
}

} // namespace hstu
