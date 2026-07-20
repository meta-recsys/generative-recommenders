# pyre-strict
"""
Router module for Triton AOT extension builders.

This module provides the main entry point for building Triton AOT kernels.
It automatically selects the appropriate builder (NVIDIA or AMD) based on
the PyTorch build configuration.
"""

import torch
from generative_recommenders.ops.triton_aot.build.amd_extension_builder import (
    AmdExtensionBuilder,
)
from generative_recommenders.ops.triton_aot.build.nvidia_extension_builder import (
    ExtensionBuilder,
)

IS_AMD: bool = torch.version.hip is not None


def build_triton_aot_extension(
    source_dir: str, kernel_name: str, output_dir: str
) -> str:
    """
    Build a Triton AOT kernel as a PyTorch C++ extension.

    This function compiles Triton AOT generated C++ sources into a shared library
    that can be loaded by PyTorch. Designed for fbcode's statically-linked PyTorch
    where torch symbols are resolved at runtime from the interpreter.

    Supports both NVIDIA CUDA and AMD HIP (ROCm) backends. The backend is
    automatically detected based on the PyTorch build configuration.

    Args:
        source_dir: Directory containing the generated C++ sources and cubin/hsaco files.
        kernel_name: Name of the kernel (e.g., "_addmm_fwd").
        output_dir: Directory to place the built .so file.

    Returns:
        Path to the built .so file.

    Raises:
        RuntimeError: If CUDA/HIP is not properly configured or build fails.
        AssertionError: If required source files are missing.
    """
    if IS_AMD:
        builder = AmdExtensionBuilder(source_dir, kernel_name, output_dir)
    else:
        builder = ExtensionBuilder(source_dir, kernel_name, output_dir)

    return builder.build()
