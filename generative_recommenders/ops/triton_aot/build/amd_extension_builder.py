# pyre-strict
"""
AMD HIP (ROCm) extension builder for Triton AOT kernels.

This module contains the AmdExtensionBuilder class that inherits from
(Nvidia)ExtensionBuilder and overrides GPU-specific methods for AMD/HIP compilation.
"""

import os

# @manual=//generative_recommenders/ops/triton_aot/build:torch_cpp_headers
from importlib.resources import files

from generative_recommenders.ops.triton_aot.build import cubin_embedder
from generative_recommenders.ops.triton_aot.build.nvidia_extension_builder import (
    BUNDLED_HEADERS_PACKAGE,
    BUNDLED_HEADERS_RESOURCE,
    ExtensionBuilder,
)
from torch.utils.cpp_extension import COMMON_HIP_FLAGS, ROCM_HOME


# HIP compute capability to architecture mapping (from triton/cc/compiler.py)
HIP_CC_TO_ARCH_INFO: dict[int, str] = {
    90: "gfx90a",
    94: "gfx942",
    95: "gfx950",
}

# Fallback ROCm include path for missing headers in certain ROCm versions (e.g., 6.2.x)
DEFAULT_ROCM_INCLUDE: str = "/opt/rocm/include"


class AmdExtensionBuilder(ExtensionBuilder):
    """
    Extension builder for AMD HIP (ROCm) backend.

    Inherits from ExtensionBuilder and overrides GPU-specific methods
    for AMD/HIP compilation.

    Templates are pre-hipified at Buck build time, and compiler.py generates
    HIP code directly (hipFunction_t, hipModuleLaunchKernel, etc.).
    Thus we need ROCm, but no runtime hipification is needed here.
    """

    def __init__(
        self,
        source_dir: str,
        kernel_name: str,
        output_dir: str = "/tmp",
        gpu_toolkit_path: str | None = None,
    ) -> None:
        """
        Initialize the AMD extension builder.

        Args:
            source_dir: Directory containing the generated C++ sources and hsaco files.
            kernel_name: Name of the kernel (e.g., "_addmm_fwd").
            output_dir: Directory to place the built .so file.
            gpu_toolkit_path: Path to GPU toolkit. Defaults to ROCM_HOME if not provided.

        Raises:
            RuntimeError: If gpu_toolkit_path is not provided and ROCM_HOME is not set.
        """
        if gpu_toolkit_path is None:
            if ROCM_HOME is None:
                raise RuntimeError(
                    "ROCM_HOME/HIP_HOME is not set. Install ROCm toolkit or set ROCM_HOME."
                )
            gpu_toolkit_path = ROCM_HOME
        super().__init__(source_dir, kernel_name, output_dir, gpu_toolkit_path)

    def get_gpu_include_dirs(self) -> list[str]:
        """
        Return HIP include directory paths, validated.

        Uses ROCM_HOME/include as the primary include directory.
        Attempts to find additional headers from /opt/rocm.
        """
        include_dir = os.path.join(self.gpu_toolkit_path, "include")
        hip_header = os.path.join(include_dir, "hip", "hip_runtime.h")
        if not os.path.exists(hip_header):
            raise RuntimeError(
                f"HIP header not found at {hip_header}. ROCm Toolkit must be installed."
            )

        include_dirs = [include_dir]

        # ROCm 6.2.x is missing hipblas-common headers. If not found, try /opt/rocm.
        hipblas_common = os.path.join(include_dir, "hipblas-common", "hipblas-common.h")
        if not os.path.exists(hipblas_common):
            if os.path.exists(
                os.path.join(DEFAULT_ROCM_INCLUDE, "hipblas-common", "hipblas-common.h")
            ):
                include_dirs.insert(0, DEFAULT_ROCM_INCLUDE)

        return include_dirs

    def get_gpu_library_dirs(self) -> list[str]:
        """
        Return directories containing HIP libraries (libamdhip64.so).
        """
        candidates = [
            os.path.join(self.gpu_toolkit_path, "lib"),
            os.path.join(self.gpu_toolkit_path, "lib64"),
            os.path.join(self.gpu_toolkit_path, "hip", "lib"),
        ]

        result = [d for d in candidates if os.path.isdir(d)]
        if not result:
            raise RuntimeError(
                f"No HIP library directories found in {self.gpu_toolkit_path}. Searched: {candidates}"
            )
        return result

    def get_libraries(self) -> list[str]:
        """Return HIP libraries to link against."""
        return ["amdhip64"]

    def get_extra_compile_args(self) -> list[str]:
        """Return HIP-specific compiler arguments."""
        args = super().get_extra_compile_args()
        args.extend(COMMON_HIP_FLAGS)

        # The c10/cuda/impl/cuda_cmake_macros.h is not generated for the
        # hip build yet.
        args.append("-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE")
        return args

    def generate_embedded_kernels(
        self, output_filename: str, kernel_variants: list[str]
    ) -> None:
        """Generate embedded hsaco cpp file."""
        cubin_embedder.generate_cpp_for_kernel_binaries(
            output_filename=output_filename,
            kernel_variants=kernel_variants,
            binary_dir=self.source_dir,
            is_amd=True,
        )

    def get_torch_device_type(self) -> str:
        """Return the torch device type for include path lookup."""
        return "hip"

    def get_torch_include_dirs(self) -> list[str]:
        """
        Return torch include directories for C++ extension compilation.

        For AMD/HIP builds, adds aten/src path for ATen-hip-headers.
        """
        include_dirs = super().get_torch_include_dirs()

        # ATen-hip-headers exports native HIP impl headers (like Masquerading headers)
        # under aten/src/ATen/hip/impl/, but the hipified HIPContext.h includes them
        # as <ATen/hip/impl/...>. Adding aten/src resolves this path mismatch.
        bundled_headers_path = str(
            files(BUNDLED_HEADERS_PACKAGE).joinpath(BUNDLED_HEADERS_RESOURCE)
        )
        aten_src_path = os.path.join(bundled_headers_path, "aten", "src")
        if os.path.isdir(aten_src_path) and aten_src_path not in include_dirs:
            include_dirs.append(aten_src_path)

        return include_dirs
