# pyre-strict

"""AOTT-local type mappings for stable ABI codegen.

These replace ``shared.types.ATYPES`` and ``shared.types.PY_TYPES_TO_CPP_TYPES``
with versions that have zero link dependency on ATen.  The shared dicts are kept
unchanged so TritonCC is not affected.
"""

from typing import Any

# Stable ABI scalar type mapping: Triton pointer dtype → c10::ScalarType enum.
# Uses c10::ScalarType:: (from torch/headeronly/core/ScalarType.h) instead of
# at::kFloat aliases (which require ATen headers).
SCALAR_TYPES: dict[str, str] = {
    "*i1": "c10::ScalarType::Bool",
    "*u8": "c10::ScalarType::Byte",
    "*i8": "c10::ScalarType::Char",
    "*i16": "c10::ScalarType::Short",
    "*i32": "c10::ScalarType::Int",
    "*i64": "c10::ScalarType::Long",
    "*fp16": "c10::ScalarType::Half",
    "*fp32": "c10::ScalarType::Float",
    "*fp64": "c10::ScalarType::Double",
    "*bf16": "c10::ScalarType::BFloat16",
    "*fp8e4nv": "c10::ScalarType::Float8_e4m3fn",
    "*fp8e4b8": "c10::ScalarType::Float8_e4m3fnuz",
}

# Stable ABI override: str → "std::string" instead of "at::string".
PY_TYPES_TO_CPP_TYPES: dict[type[Any], str] = {
    int: "int64_t",
    str: "std::string",
    float: "double",
}
