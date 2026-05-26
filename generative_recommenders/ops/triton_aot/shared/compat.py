# pyre-strict
"""
This module provides shared utilities that handle differences between
Triton versions.
"""

from typing import Any

# @manual=//triton:triton
import triton
from packaging.version import Version
from triton.runtime.jit import JITFunction

TRITON_VERSION: str = triton.__version__


def version_gte(version: str, target: str) -> bool:
    """
    Check if version >= target using semantic version comparison.
    Simple string comparison fails for versions like "3.10" vs "3.5"
    """
    return Version(version) >= Version(target)


def get_kernel_name(jit_fn: JITFunction[Any]) -> str:
    """
    Get the simple kernel name from a JITFunction.

    In Triton 3.5+, JITFunction._fn_name returns the full qualified name
    (e.g., "generative_recommenders.ops.triton_aot.triton_addmm._addmm_fwd").
    In older versions, it returns just the simple name (e.g., "_addmm_fwd").

    This function normalizes the behavior to always return the simple name.

    Args:
        jit_fn: A Triton JITFunction

    Returns:
        The simple kernel name (e.g., "_addmm_fwd")
    """
    fn_name = jit_fn._fn_name
    if version_gte(TRITON_VERSION, "3.5"):
        # Triton 3.5+ uses get_full_name(fn) which returns qualified name
        return fn_name.rsplit(".", 1)[-1]
    else:
        # Older versions use fn.__name__ which is already simple
        return fn_name


def get_scratch_parameters(kernel: Any) -> tuple[str, list[str]]:
    """
    Get scratch parameter declarations and argument pointers for the kernel launcher.

    Scratch parameters are backend and version-specific features for profiling
    and global memory management.

    Detection Strategy:
        1. Check metadata first for each parameter
        2. Fall back to version-based detection if metadata unavailable

    Version Requirements (fallback):
        - v3.4+: both global_scratch and profile_scratch
        - v3.3: only global_scratch
        - v3.2 and earlier: no scratch parameters

    Args:
        kernel: Compiled Triton kernel with metadata attribute

    Returns:
        Tuple of (declarations, arg_pointers):
        - declarations: C++ variable declarations for scratch parameters
        - arg_pointers: List of argument pointers to append to kernel args
    """
    declarations = []
    arg_pointers = []

    if hasattr(kernel.metadata, "global_scratch_size"):
        declarations.append("CUdeviceptr global_scratch = 0;")
        arg_pointers.append("&global_scratch")
    elif version_gte(TRITON_VERSION, "3.3"):
        declarations.append("CUdeviceptr global_scratch = 0;")
        arg_pointers.append("&global_scratch")

    if hasattr(kernel.metadata, "profile_scratch_size"):
        declarations.append("CUdeviceptr profile_scratch = 0;")
        arg_pointers.append("&profile_scratch")
    elif version_gte(TRITON_VERSION, "3.4"):
        declarations.append("CUdeviceptr profile_scratch = 0;")
        arg_pointers.append("&profile_scratch")

    return ("\n            ".join(declarations), arg_pointers)
