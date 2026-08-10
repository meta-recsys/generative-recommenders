# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import hashlib
from typing import Any, Type, TypeVar

T = TypeVar("T")


def unwrap_heuristic(func: Any, return_type: Type[T]) -> T:
    while not isinstance(func, return_type):
        func = func.fn
        if not hasattr(func, "fn"):
            # pyre-fixme[7]: Incompatible return type [7]: Expected `Variable[T]` but got `None`.
            return None
    return func


def is_autotuner(obj: Any) -> bool:
    """Check whether *obj* is a Triton Autotuner using duck typing.

    In Buck builds the ``Autotuner`` class can be loaded from multiple module
    paths (e.g. via ``torch.package`` re-imports), causing ``isinstance`` to
    return ``False`` for genuine Autotuner instances.  We combine a class-name
    check with duck-typing on the attributes that callers actually need
    (``cache``, ``configs``, ``arg_names``), making detection robust against
    module-path aliasing.
    """
    return "Autotuner" in type(obj).__name__ and all(
        hasattr(obj, attr) for attr in ("cache", "configs", "arg_names")
    )


def hash_kernel_name(kernel_name: str) -> str:
    """Hash kernel name to create shorter, filesystem-safe names.

    Args:
        kernel_name: Full kernel name (can be very long with specialization suffixes).
            e.g., "_addmm_fwd_sm80_pfp32_pfp32_pfp32_pfp32_i32_..."

    Returns:
        Hashed name in format "kernel_<sha256_hex>".
            e.g., "kernel_a1b2c3d4e5f6..."

    """
    return "kernel_" + hashlib.sha256(kernel_name.encode("utf-8")).hexdigest()
