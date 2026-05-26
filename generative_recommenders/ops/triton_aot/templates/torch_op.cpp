// __TRITON_AOT_GENERATE_BEGIN__ HEADER_INCLUDE
#include "kernel.h"
// __TRITON_AOT_GENERATE_END__ HEADER_INCLUDE
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h> // NOLINT(facebook-unused-include-check)

// __TRITON_AOT_GENERATE_BEGIN__ TORCH_OP
namespace {
// no-op, force link StableLibrary
torch::stable::Tensor _triton_aot_placeholder_noop(
    torch::stable::Tensor input) {
  return input;
}
} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(triton_aot, m) {
  m.def("_placeholder_noop(Tensor input) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(triton_aot, CPU, m) {
  m.impl("_placeholder_noop", TORCH_BOX(&_triton_aot_placeholder_noop));
}
// __TRITON_AOT_GENERATE_END__ TORCH_OP
