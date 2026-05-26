# pyre-strict

"""Functions for converting kernel specs to architecture-specific formats.

A "spec" (specification) describes how to compile a Triton kernel for a specific
set of input shapes and types. Users provide "base specs" in a human-friendly
format that describes kernel arguments:

    {"signature": [("*fp32", 16), ("*bf16", 16), ("i32", None), 128]}

This format encodes dtypes, alignment hints, and constant values together.
Before compilation, base specs must be converted to "compiled specs" that
separate this information into distinct fields the compiler understands:

    {"signature": {0: "*fp32", 1: "*bf16"},
     "constants": {2: None, 3: 128},
     "configs": (instance_descriptor(...),),
     "cc": 80}

This module provides the functionality to perform this transformation,
extracting constraints, identifying constants, and preparing specs for each
target GPU architecture.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, TypeAlias

from generative_recommenders.ops.triton_aot.shared.types import CTYPES

# Compile-time constant values that can appear in signatures or be returned
# by constexpr(). These are values the compiler can fold into generated code.
ConstantValue: TypeAlias = str | int | float | bool | None

# A single element in a kernel signature list.
# Can be: dtype string, (dtype, alignment) tuple, (dtype, alignment, has_value)
# triple for optional args, or a bare literal constant.
SignatureElement: TypeAlias = (
    ConstantValue | tuple[str, int | None] | tuple[str, int | None, bool]
)


instance_descriptor = namedtuple(
    "instance_descriptor",
    [
        "divisible_by_16",
        "equal_to_1",
        "ids_of_folded_args",
        "divisible_by_8",
    ],
)


def constexpr(s: SignatureElement) -> ConstantValue:
    """Identify compile-time constant expressions in signature elements.

    Args:
        s: A signature element.

    Returns:
        The constant value if s is a compile-time constant, None otherwise.
        Constants are: int, float, bool, or strings that aren't dtype names.
    """
    expr = s[0] if isinstance(s, tuple) and len(s) > 1 else s

    if expr is None:
        return expr

    try:
        ret = int(expr)
        return ret
    except (ValueError, TypeError):
        pass
    try:
        ret = float(expr)
        return ret
    except (ValueError, TypeError):
        pass

    if isinstance(expr, bool):
        return expr
    if isinstance(expr, str) and expr not in CTYPES and not expr.startswith("*"):
        return expr
    return None


@dataclass
class SignatureConstraints:
    """Constraints extracted from parsing a kernel signature.

    When compiling a Triton kernel, the compiler can generate more efficient
    code if it knows certain properties about the arguments:

    - Pointer alignment: If a pointer is always 16-byte aligned, the compiler
      can use faster aligned memory operations.
    - Constant values: Arguments known at compile time can be folded into the
      generated code, eliminating runtime checks.
    - FP8 dtypes: Some GPU architectures require dtype substitutions for FP8
      types (e.g., gfx942 needs fp8e4b8 instead of fp8e4nv).

    This dataclass collects all these constraints from a single pass over the
    signature, so downstream code can use them without re-parsing.

    Attributes:
        divisible_by_16: Indices of args with values divisible by 16.
        divisible_by_8: Indices of args with values divisible by 8.
        equal_to_1: Indices of args with value equal to 1.
        none_args: Indices of args that are None (not provided).
        optional_args: Indices of optional arguments.
        has_fp8: Whether any argument has an FP8 dtype.
    """

    divisible_by_16: set[int]
    divisible_by_8: set[int]
    equal_to_1: set[int]
    none_args: set[int]
    optional_args: set[int]
    has_fp8: bool


def collect_constraints(signature: list[SignatureElement]) -> SignatureConstraints:
    """Collect divisibility and type constraints from a signature list.

    Iterates through signature elements and identifies:
    - Arguments divisible by 16 or 8 (for memory alignment)
    - Arguments equal to 1 (for optimization)
    - Optional arguments and those not provided (None)
    - Whether any FP8 dtypes are present

    Args:
        signature: List of signature elements. The input format is unfortunately
            variable; each element can be one of several types:

            1. Plain string (dtype only, no alignment info):
               "*fp32"           - A float32 pointer
               "i32"             - A 32-bit integer scalar

            2. Tuple of (dtype, value) where value indicates alignment or constness:
               ("*fp32", 16)     - Float32 pointer, 16-byte aligned
               ("i32", None)     - Integer arg not provided (becomes constant None)
               ("*bf16", 1)      - Pointer with value=1 (folded as constant)

            3. Triple of (dtype, value, has_value) for optional arguments:
               ("*fp32", 16, True)  - Optional arg that IS provided, 16-byte aligned
               ("*fp32", 16, False) - Optional arg NOT provided (becomes None)

            4. Bare literals (become compile-time constants):
               128               - Integer constant
               "leaky_relu"      - String constant (e.g., activation name)

    Returns:
        SignatureConstraints with all constraint sets populated.

    Example:
        >>> sig = [("*fp32", 16), ("i32", None), ("*fp8e4nv", 8)]
        >>> c = collect_constraints(sig)
        >>> 0 in c.divisible_by_16
        True
        >>> c.has_fp8
        True
    """
    divisible_by_16: set[int] = set()
    divisible_by_8: set[int] = set()
    equal_to_1: set[int] = set()
    none_args: set[int] = set()
    optional_args: set[int] = set()
    has_fp8: bool = False

    for i, s in enumerate(signature):
        # Handle optional tensor case: tuple with 3 elements where s[2] indicates
        # whether the optional arg has a value
        if isinstance(s, tuple) and len(s) > 2:
            optional_args.add(i)
            # pyrefly: ignore [bad-index]
            if not s[2]:  # has_value is False
                none_args.add(i)
                continue

        # Extract dtype
        dtype = s[0] if isinstance(s, tuple) else s

        # Check for FP8 types
        if isinstance(dtype, str) and ("fp8e4nv" in dtype or "fp8e4b8" in dtype):
            has_fp8 = True

        # Extract value (alignment or constant)
        value = s[1] if isinstance(s, tuple) else s

        # Check divisibility and equality constraints
        if isinstance(value, int):
            if value % 16 == 0:
                divisible_by_16.add(i)
            if value % 8 == 0:
                divisible_by_8.add(i)
            if value == 1:
                equal_to_1.add(i)

        if value is None:
            none_args.add(i)

    return SignatureConstraints(
        divisible_by_16=divisible_by_16,
        divisible_by_8=divisible_by_8,
        equal_to_1=equal_to_1,
        none_args=none_args,
        optional_args=optional_args,
        has_fp8=has_fp8,
    )


def make_instance_descriptor(
    constraints: SignatureConstraints,
) -> tuple[instance_descriptor]:
    """Create an instance_descriptor tuple from constraints.

    Args:
        constraints: The collected signature constraints.

    Returns:
        A tuple containing a single instance_descriptor namedtuple with
        divisible_by_16, equal_to_1, ids_of_folded_args, and divisible_by_8.
    """
    ids_of_folded_args = constraints.equal_to_1 | constraints.none_args
    return (
        instance_descriptor(
            divisible_by_16=constraints.divisible_by_16,
            equal_to_1=constraints.equal_to_1,
            ids_of_folded_args=ids_of_folded_args,
            divisible_by_8=constraints.divisible_by_8,
        ),
    )


def extract_constants(
    signature: list[SignatureElement],
    constraints: SignatureConstraints,
) -> dict[int, ConstantValue]:
    """Extract compile-time constant values from signature elements.

    Identifies arguments that can be folded into generated code at compile time.
    Constants come from three sources:

    1. Bare literals in the signature (e.g., 128 for block size, "leaky_relu"
       for activation type, True for a boolean flag)
    2. Arguments with value=1 (tracked in constraints.equal_to_1)
    3. Arguments not provided (tracked in constraints.none_args)

    Args:
        signature: List of signature elements in input format.
        constraints: The collected signature constraints.

    Returns:
        Dict mapping argument indices to their constant values.
    """
    # Use constexpr to identify constant expressions
    constexprs = {i: constexpr(s) for i, s in enumerate(signature)}
    constants: dict[int, ConstantValue] = {
        k: v for k, v in constexprs.items() if v is not None
    }

    # Add equal_to_1 args with value 1
    for k in constraints.equal_to_1:
        constants[k] = 1

    # Add none_args with value None
    for k in constraints.none_args:
        constants[k] = None

    return constants


def signature_list_to_dict(
    signature: list[SignatureElement],
    constants: dict[int, ConstantValue],
) -> dict[int, str]:
    """Convert signature from list format to dict format.

    Transforms the input signature list into a dict mapping argument
    indices to dtype strings. Arguments that are constants are excluded
    since they don't need runtime type information.

    Args:
        signature: List of signature elements in input format.
        constants: Dict of constant argument indices to exclude.

    Returns:
        Dict mapping non-constant argument indices to their dtype strings.
    """
    result: dict[int, str] = {}
    for i, s in enumerate(signature):
        if i in constants:
            continue
        # After filtering out constants, remaining elements are dtype declarations.
        # For tuples like ("*fp32", 16), s[0] is the dtype string.
        # For plain strings like "*fp32", the element itself is the dtype.
        if isinstance(s, tuple) and len(s) > 1:
            dtype = s[0]
        else:
            dtype = s
        assert isinstance(dtype, str)
        result[i] = dtype
    return result


# CC (compute capability) to AMD GPU architecture mapping
# CC is a 2-digit shorthand: 94 -> gfx942, 95 -> gfx950
HIP_CC_TO_ARCH_INFO: dict[int, str] = {
    90: "gfx90a",
    94: "gfx942",
    95: "gfx950",
}

# Reverse mapping: architecture string -> CC string
HIP_ARCH_TO_CC: dict[str, str] = {v: str(k) for k, v in HIP_CC_TO_ARCH_INFO.items()}

HIP_CC_MI350X: str = "95"  # CC string for gfx950 (MI350X/MI355X)


def _normalize_cc(cc: set[str]) -> set[str]:
    """Normalize CC values to 2-digit format for internal comparison.

    Accepts both tritoncc format ("94", "95") and Triton driver format
    ("gfx942", "gfx950"). Returns 2-digit CC strings.
    """
    return {HIP_ARCH_TO_CC.get(c, c) for c in cc}


def get_fp8_replacement_signature_for_amd(
    spec: dict[str, Any], cc: set[str]
) -> dict[int, str]:
    """Replace FP8 dtypes in signature for AMD architectures.

    Args:
        spec: Compiled spec dict with 'signature' in dict format.
        cc: Set of CC strings in either format:
            - 2-digit tritoncc format: {"94"} for gfx942
            - Triton driver format: {"gfx942"}
            See HIP_CC_TO_ARCH_INFO.

    Returns:
        Dict mapping argument indices to dtype strings with FP8 types replaced.
    """
    normalized_cc: set[str] = _normalize_cc(cc)

    def replace_fp8_type(dtype_str: str) -> str:
        if "fp8e4nv" in dtype_str:
            if HIP_CC_MI350X not in normalized_cc:
                return dtype_str.replace("fp8e4nv", "fp8e4b8")
        elif "fp8e4b8" in dtype_str and HIP_CC_MI350X in normalized_cc:
            return dtype_str.replace("fp8e4b8", "fp8e4nv")
        return dtype_str

    replace_fp8_signatures: dict[int, str] = {}
    for key, value in spec["signature"].items():
        if isinstance(value, str):
            replace_fp8_signatures[key] = replace_fp8_type(value)
        else:
            replace_fp8_signatures[key] = value

    return replace_fp8_signatures


def get_fp8_replacement_signature_for_sm80(
    spec: dict[str, Any],
) -> dict[int, Any]:
    """Replace FP8 dtypes with bf16 for SM80 (A100) which lacks native FP8 support.

    Args:
        spec: Compiled spec dict with 'signature' in dict format.

    Returns:
        Dict mapping argument indices to dtype strings with FP8 types replaced by bf16.
    """

    def replace_fp8_type(dtype_str: str) -> str:
        if "fp8e4nv" in dtype_str:
            return dtype_str.replace("fp8e4nv", "bf16")
        return dtype_str

    replace_fp8_signatures: dict[int, Any] = {}
    for key, value in spec["signature"].items():
        if isinstance(value, tuple) and isinstance(value[0], str):
            replace_fp8_signatures[key] = (replace_fp8_type(value[0]), value[1])
        elif isinstance(value, str):
            replace_fp8_signatures[key] = replace_fp8_type(value)
        else:
            replace_fp8_signatures[key] = value

    return replace_fp8_signatures
