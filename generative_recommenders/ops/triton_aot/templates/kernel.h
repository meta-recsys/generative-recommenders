#pragma once

#include <torch/csrc/stable/tensor.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace triton {
namespace aot {

#ifndef GRID_DIM_DEFINED_MACRO
struct gridDims {
  int x = 1;
  int y = 1;
  int z = 1;
  cudaStream_t stream = nullptr;
  gridDims(int _x = 1, int _y = 1, int _z = 1, cudaStream_t _stream = nullptr)
      : x(_x), y(_y), z(_z), stream(_stream) {}
};
#define GRID_DIM_DEFINED_MACRO
#endif

#ifndef FITS_I32_DEFINED_MACRO
constexpr bool fits_i32(int64_t v) {
  return v >= INT32_MIN && v <= INT32_MAX;
}
#define FITS_I32_DEFINED_MACRO
#endif

// __TRITON_AOT_GENERATE_BEGIN__ TUNER_META_CPP
// __TRITON_AOT_GENERATE_END__ TUNER_META_CPP
// __TRITON_AOT_GENERATE_BEGIN__ SELECTOR_PROTO
// __TRITON_AOT_GENERATE_END__ SELECTOR_PROTO

} // namespace aot
} // namespace triton
