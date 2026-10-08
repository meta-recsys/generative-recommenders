# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pyre-strict

"""
Generate C++ source files that embed kernel binaries as byte arrays.

This module converts kernel binary files (.cubin for NVIDIA, .hsaco for AMD)
into C++ source files with the binary data embedded as hex arrays,
enabling static linking of GPU kernels.
"""

import binascii
import logging
import os

from generative_recommenders.ops.triton_aot.compile.utils import hash_kernel_name
from generative_recommenders.ops.triton_aot.templates.template_utils import (
    load_template,
    render_template,
)

logger: logging.Logger = logging.getLogger(__name__)

# Template for a single kernel binary array definition
#
# IMPORTANT: The pointer must NOT be in .triton section!
#
# Why this matters for large binaries (>4GB):
# - The cubin data is in .triton, which is placed beyond 4GB via linker script
# - Code in .text accesses the pointer via R_X86_64_PC32 (±2GB range)
# - If pointer is in .triton (>4GB away), R_X86_64_PC32 overflows
# - Solution: pointer in .data.rel.ro (near .text), data in .triton
# - Pointer initialization uses R_X86_64_64 to reference .triton data (no limit)
#
# The pointer is marked volatile to prevent the optimizer from constant-propagating
# through the pointer with -O2. Without volatile, the compiler sees that _cubin_ptr
# is const and initialized with (const void*)_cubin, then constant-propagates:
# image = _cubin_ptr -> image = (const void*)_cubin -> emits R_X86_64_32 to _cubin
KERNEL_BINARY_ARRAY_TEMPLATE = """\
    // Source: {binary_path}
    __attribute__((section(".triton"), visibility("default"), aligned(8)))
    unsigned char {symbol_name}[] = {{ {hex_array} }};
    // Pointer to cubin data - placed in .data.rel.ro (near .text) so code can
    // access it via R_X86_64_PC32. The pointer initialization uses R_X86_64_64
    // to reference the cubin data in .triton, which has no distance limit.
    // volatile prevents -O2 from constant-propagating through the pointer.
    __attribute__((section(".data.rel.ro"), visibility("default")))
    const void* volatile {symbol_name}_ptr = (const void*){symbol_name};
"""


def _kernel_binary_to_hex_array(binary_path: str) -> str:
    """Convert a kernel binary file to a C++ hex array string.

    Args:
        binary_path: Path to the kernel binary file (.cubin or .hsaco).

    Returns:
        Formatted hex string like "0xAB, 0xCD, ...".
    """
    with open(binary_path, "rb") as f:
        data = f.read()
    hex_str = binascii.hexlify(data).decode("ascii")
    return ", ".join(f"0x{hex_str[i : i + 2]}" for i in range(0, len(hex_str), 2))


def _embed_kernel_binaries(
    kernel_variants: list[str], binary_dir: str, is_amd: bool = False
) -> str:
    """Generate kernel binary array definitions for all kernel variants.

    Args:
        kernel_variants: List of kernel variant names to embed
            (e.g., '_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_').
        binary_dir: Directory containing kernel binary files.
        is_amd: If True, look for .hsaco files instead of .cubin files.

    Returns:
        Combined kernel binary array definitions as a string.
    """
    binary_type = "hsaco" if is_amd else "cubin"
    logger.info(
        f"Embedding {len(kernel_variants)} kernel binaries ({binary_type}) from {binary_dir}"
    )

    kernel_binary_arrays = []
    binary_suffix = "." + binary_type

    for kernel in kernel_variants:
        symbol_name = f"{kernel}_cubin"
        binary_path = os.path.join(
            binary_dir, f"{hash_kernel_name(kernel)}{binary_suffix}"
        )

        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Kernel binary file not found: {binary_path}")

        logger.info(f"  Embedding {os.path.basename(binary_path)} as {symbol_name}")

        hex_array = _kernel_binary_to_hex_array(binary_path)
        kernel_binary_arrays.append(
            KERNEL_BINARY_ARRAY_TEMPLATE.format(
                binary_path=binary_path,
                symbol_name=symbol_name,
                hex_array=hex_array,
            )
        )
    return "\n".join(kernel_binary_arrays)


def generate_cpp_for_kernel_binaries(
    output_filename: str,
    kernel_variants: list[str],
    binary_dir: str | None = None,
    is_amd: bool = False,
) -> None:
    """Generate a C++ file embedding multiple kernel binaries.

    Args:
        output_filename: Output .cpp file path (e.g., 'embedded_kernels.cpp').
        kernel_variants: List of kernel variant names to embed
            (e.g., '_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_').
        binary_dir: Directory containing kernel binary files. Defaults to script directory.
        is_amd: If True, look for .hsaco files instead of .cubin files.
    """
    if binary_dir is None:
        binary_dir = os.path.dirname(os.path.realpath(__file__))

    template = load_template("embedded_cubins.cpp")
    kernel_binary_arrays = _embed_kernel_binaries(
        kernel_variants, binary_dir, is_amd=is_amd
    )
    cpp_content = render_template(template, {"CUBIN_ARRAYS": kernel_binary_arrays})

    with open(output_filename, "w") as f:
        f.write(cpp_content)

    logger.info(f"Generated {output_filename}")
