# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""Kernel spec processing for AOT-T compilation.

Transforms raw kernel specs (from infer_spec) into compiled specs ready
for Triton native compile and C++ codegen.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, cast

# @manual=//triton:triton
import triton
from generative_recommenders.ops.triton_aot.compile.compile_state import hash_spec
from generative_recommenders.ops.triton_aot.shared.spec_conversion import (
    collect_constraints,
    extract_constants,
    get_fp8_replacement_signature_for_amd,
    get_fp8_replacement_signature_for_sm80,
    signature_list_to_dict,
    SignatureElement,
)
from generative_recommenders.ops.triton_aot.shared.types import AUTOTUNE_ATTRs
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.compiler.compiler import ASTSource
from triton.runtime.jit import JITFunction

logger: logging.Logger = logging.getLogger(__name__)

TRITON_VERSION: str = triton.__version__

# A raw kernel spec produced by infer_spec.  The only key is "signature".
RawKernelSpec = dict[str, list[SignatureElement]]


@dataclass
class KernelSpec:
    """A single compilation variant for a kernel.

    Each variant represents one combination of dtypes, constant values,
    alignment constraints, and autotune configuration.  Multiple variants
    are grouped together in an ``OpsUnit``.

    Attributes:
        signature: Non-constant arg index → dtype string (e.g., ``{0: "*fp32", 4: "i32"}``).
        constants: Arg index → compile-time constant value.  Includes bare
            literals (128, ``"leaky_relu"``), absent optional tensors (None),
            and equal-to-1 specializations (stride=1 → constexpr folding).
        divisible_by_16: Indices of args whose values are divisible by 16.
            For pointers this means the address is 16-byte aligned;
            for scalars it means the value itself is a multiple of 16.
        divisible_by_8: Indices of args whose values are divisible by 8.
            Only meaningful for scalars (pointer alignment is always ≥16).
        num_warps: Number of warps per block.
        num_stages: Number of pipeline stages.
        matrix_instr_nonkdim: AMD matrix instruction non-K dimension.
        waves_per_eu: AMD waves per execution unit.
        kpack: AMD kpack factor.
    """

    signature: dict[int, str]
    constants: dict[int, Any]
    divisible_by_16: set[int]
    divisible_by_8: set[int]
    num_warps: int = 4
    num_stages: int = 3
    matrix_instr_nonkdim: int = 0
    waves_per_eu: int = 1
    kpack: int = 1


@dataclass
class OpsUnit:
    """All compilation variants for a single kernel op.

    Groups per-kernel invariants with the list of ``KernelSpec`` variants.
    Use ``OpsUnit.from_raw_specs()`` to build — it performs the complete
    spec processing pipeline (convert → detect optional → validate →
    autotune → dedup → compute invariants).

    Attributes:
        cc: Compute capability (int for NVIDIA, str for AMD).
        optional: Indices of optional tensor args (unified across all call sites).
        pointer_args: Indices of all tensor pointer args (required + optional).
            Invariant across specs — a pointer arg never becomes a non-pointer.
        scalar_dtypes: Non-pointer signature arg index → widest dtype string
            across all specs (e.g., ``"i32"``, ``"i64"``, ``"fp32"``).
            Computed by ``_wider_type`` — individual specs may use narrower types.
        constant_types: Python type per constant arg position (e.g., ``{15: int, 19: bool}``).
            Excludes optional tensor args (None constants).
            Invariant across specs — same Python type for each position.
        specs: Per-variant compilation specs.
    """

    cc: int | str
    optional: set[int]
    pointer_args: set[int]
    scalar_dtypes: dict[int, str]
    constant_types: dict[int, type[Any]]
    specs: list[KernelSpec]

    @classmethod
    def from_raw_specs(
        cls,
        base_specs: list[RawKernelSpec],
        gpu_target: GPUTarget,
        tuned_func: triton.runtime.autotuner.Autotuner | None = None,
    ) -> OpsUnit:
        """Build an OpsUnit from raw kernel specs.

        Performs the complete spec processing pipeline:
        1. Convert raw specs to KernelSpecs
        2. Detect optional tensor args (cross-spec + 3-tuple)
        3. Validate consistency across converted specs
        4. Apply autotuning (if tuned_func provided)
        5. Deduplicate specs
        6. Compute shared invariants (pointer_args, scalar_dtypes, constant_types)
        """
        # Validate raw specs upfront, before any rewriting.
        num_params = _check_uniform_signature_length(base_specs)
        specs, three_tuple_optional = _convert_raw_specs(base_specs, gpu_target)
        optional = _detect_optional_args(specs) | three_tuple_optional

        _validate_converted_specs(specs, optional, num_params)

        # Plain @triton.jit kernels (no @triton.autotune) skip config expansion.
        if tuned_func is not None:
            specs = _autotune_specs(tuned_func, gpu_target, specs)

        specs = _dedup_specs(specs)

        pointer_args, scalar_dtypes, constant_types = _compute_invariants(
            specs, optional
        )

        return cls(
            cc=gpu_target.arch,
            optional=optional,
            pointer_args=pointer_args,
            scalar_dtypes=scalar_dtypes,
            constant_types=constant_types,
            specs=specs,
        )


# ---------------------------------------------------------------------------
# Public helpers (used outside spec processing)
# ---------------------------------------------------------------------------


def gen_compile_arg(
    spec: KernelSpec,
    func: JITFunction[list[Any]],
) -> tuple[ASTSource]:
    # ASTSource expects tuple-keyed dicts: {(idx,): value} for constants,
    # {(idx,): [[attr_name, attr_val], ...]} for attrs.  Tuple keys support
    # nested paths into structured types (asserted by ASTSource.__init__).
    new_signature = {}
    new_constants = {}
    param_names = list(func.signature.parameters.keys())
    for idx, param in enumerate(param_names):
        if idx in spec.signature:
            new_signature[param] = spec.signature[idx]
        if idx in spec.constants:
            new_constants[(idx,)] = spec.constants[idx]
            new_signature[param] = "constexpr"

    # parse_attr("D") returns a fresh [["tt.divisibility", 16]] each call.
    new_attrs = {(idx,): BaseBackend.parse_attr("D") for idx in spec.divisible_by_16}

    return (
        ASTSource(
            func,
            new_signature,
            constexprs=new_constants,
            attrs=new_attrs,
        ),
    )


# ---------------------------------------------------------------------------
# Int width helpers
# ---------------------------------------------------------------------------

_INT_WIDTH_RANK: dict[str, int] = {"i32": 0, "i64": 1}


def _wider_type(t1: str, t2: str) -> str:
    """Return the wider of two scalar dtypes.

    Only i32/i64 widening is supported.  All other types must match exactly.
    """
    if t1 == t2:
        return t1
    r1 = _INT_WIDTH_RANK.get(t1)
    r2 = _INT_WIDTH_RANK.get(t2)
    if r1 is not None and r2 is not None:
        return t1 if r1 >= r2 else t2
    raise ValueError(f"Cannot widen incompatible types: {t1!r} vs {t2!r}")


# ---------------------------------------------------------------------------
# Private helpers — called by OpsUnit.from_raw_specs
# ---------------------------------------------------------------------------


def _detect_optional_args(specs: list[KernelSpec]) -> set[int]:
    """Detect optional tensor args by cross-spec comparison.

    An arg at index ``i`` is optional if:
    - Some specs have ``i`` in ``signature`` as a pointer type (``*...``)
    - Other specs have ``constants[i] = None``

    Single-spec None args (always-absent tensors) are NOT detected here
    but are handled by ``_compute_invariants`` which adds any
    ``constants[i] = None`` to ``pointer_args``.
    """
    if len(specs) <= 1:
        return set()
    optional: set[int] = set()
    all_indices: set[int] = set()
    for spec in specs:
        all_indices |= spec.signature.keys()
        all_indices |= spec.constants.keys()
    for i in all_indices:
        has_pointer = any(
            i in s.signature and s.signature[i].startswith("*") for s in specs
        )
        has_none_const = any(i in s.constants and s.constants[i] is None for s in specs)
        if has_pointer and has_none_const:
            optional.add(i)
    return optional


def _check_uniform_signature_length(base_specs: list[RawKernelSpec]) -> int:
    """All raw specs must declare the same param count; return that count.

    Each raw spec is one ``infer_spec`` call site for the same kernel,
    so all should have ``len(fn.signature.parameters)`` entries.  Differing
    lengths means upstream bug (mixed kernels, truncated spec, etc.) and
    would surface later as silent IndexError or wrong bound checks.
    """
    if not base_specs:
        return 0
    sig_lens = {len(spec["signature"]) for spec in base_specs}
    if len(sig_lens) != 1:
        raise ValueError(
            f"Raw specs declare inconsistent signature lengths: "
            f"{sorted(sig_lens)}.  All specs for the same kernel must have "
            f"one entry per declared param."
        )
    return sig_lens.pop()


def _check_arg_indices_in_range(
    specs: list[KernelSpec],
    num_params: int,
) -> None:
    """Every spec arg index must be in ``[0, num_params)``.

    Out-of-range indices would silently drop in ``gen_compile_arg``'s
    ``enumerate(param_names)`` loop.  ``num_params <= 0`` disables the check.
    """
    if num_params <= 0:
        return
    for idx, spec in enumerate(specs):
        all_indices = (
            spec.signature.keys()
            | spec.constants.keys()
            | spec.divisible_by_16
            | spec.divisible_by_8
        )
        for i in all_indices:
            if not 0 <= i < num_params:
                raise ValueError(
                    f"Spec {idx}: arg index {i} out of range "
                    f"[0, {num_params}) — kernel has {num_params} declared params"
                )


def _collect_pointer_args(
    specs: list[KernelSpec],
    optional: set[int],
) -> set[int]:
    """Collect all tensor pointer indices across all specs.

    Includes optional args (from _detect_optional_args) AND any arg
    whose constant value is None (single-spec optional tensor case
    where _detect_optional_args didn't fire).
    """
    pointer_args: set[int] = set(optional)
    for spec in specs:
        for i, dtype in spec.signature.items():
            if dtype.startswith("*"):
                pointer_args.add(i)
        for i, val in spec.constants.items():
            if val is None:
                pointer_args.add(i)
    return pointer_args


def _collect_scalar_dtypes(
    specs: list[KernelSpec],
    pointer_args: set[int],
) -> dict[int, str]:
    """Collect non-pointer signature arg dtypes, widening compatible int types.

    Invariant across specs (validated by _validate_converted_specs).
    """
    scalar_dtypes: dict[int, str] = {}
    for spec in specs:
        for i, dtype in spec.signature.items():
            if i not in pointer_args:
                if i in scalar_dtypes:
                    scalar_dtypes[i] = _wider_type(scalar_dtypes[i], dtype)
                else:
                    scalar_dtypes[i] = dtype
    return scalar_dtypes


def _collect_constant_types(
    specs: list[KernelSpec],
) -> dict[int, type[Any]]:
    """Collect Python type per constant position.

    Excludes None constants (optional tensor args — already in pointer_args).
    """
    constant_types: dict[int, type[Any]] = {}
    for spec in specs:
        for i, val in spec.constants.items():
            if val is not None and i not in constant_types:
                constant_types[i] = type(val)
    return constant_types


def _compute_invariants(
    specs: list[KernelSpec],
    optional: set[int],
) -> tuple[set[int], dict[int, str], dict[int, type[Any]]]:
    """Compute shared invariants from processed specs.

    Returns (pointer_args, scalar_dtypes, constant_types).

    When annotation-as-variant produces mixed partitions (arg in
    ``signature`` in some specs, ``constants`` in others), the arg
    appears in both ``scalar_dtypes`` and ``constant_types``.  The
    selector must receive it as a runtime parameter for dispatch,
    so ``scalar_dtypes`` wins and the arg is removed from
    ``constant_types``.
    """
    pointer_args = _collect_pointer_args(specs, optional)
    scalar_dtypes = _collect_scalar_dtypes(specs, pointer_args)
    constant_types = _collect_constant_types(specs)

    # Resolve overlap: if any spec has the arg in signature (scalar),
    # the selector needs it as a runtime parameter → not a constant.
    for i in scalar_dtypes:
        constant_types.pop(i, None)

    return pointer_args, scalar_dtypes, constant_types


def _validate_converted_specs(
    specs: list[KernelSpec],
    optional: set[int],
    num_params: int = 0,
) -> None:
    """Validate that converted specs are consistent before further processing.

    Checks that all specs produce identical C++ function signatures:
    - All arg indices are in ``[0, num_params)`` (when ``num_params > 0``)
    - Optional args: each spec has either a pointer in signature or None in constants
    - Non-optional scalar args: same dtype (or compatible int widths)
    - Non-optional constant args: same Python type

    Called after _convert_raw_specs + _detect_optional_args, before autotuning.
    """
    _check_arg_indices_in_range(specs, num_params)
    if len(specs) <= 1:
        return
    ref = specs[0]
    for idx, spec in enumerate(specs[1:], 1):
        _check_optional_consistency(ref, spec, idx, optional)
        _check_signature_consistency(ref, spec, idx, optional)
        _check_constants_consistency(ref, spec, idx, optional)


def _check_optional_consistency(
    ref: KernelSpec,
    spec: KernelSpec,
    idx: int,
    optional: set[int],
) -> None:
    """Optional positions must be pointer-in-signature or None-in-constants.

    Validates that optional tensor args are not misclassified as scalars
    or non-None constants, which would produce incompatible C++ types.
    """
    for i in optional:
        for label, s in [("spec 0", ref), (f"spec {idx}", spec)]:
            if i in s.signature:
                if not s.signature[i].startswith("*"):
                    raise ValueError(
                        f"Arg {i}: optional position has non-pointer type "
                        f"'{s.signature[i]}' in {label}"
                    )
            elif i in s.constants:
                if s.constants[i] is not None:
                    raise ValueError(
                        f"Arg {i}: optional position has non-None constant "
                        f"{s.constants[i]!r} in {label}"
                    )


def _check_signature_consistency(
    ref: KernelSpec,
    spec: KernelSpec,
    idx: int,
    optional: set[int],
) -> None:
    """Non-optional, non-pointer scalar args must have compatible dtypes.

    Pointer args are skipped (different tensor dtypes are dispatched by
    the dtype guard in ``gen_guarded_calls``).  Compatible int widths
    (i32/i64) are allowed — handled by ``_wider_type`` and int range guards.
    Optional positions are validated by ``_check_optional_consistency``.

    Partition differences are allowed: an arg may be in ``signature`` in
    one spec and in ``constants`` in another (e.g., annotation-as-variant
    where stride=1 is constexpr in one spec but a runtime parameter in
    another).  The per-spec codegen handles this correctly.
    """
    for i in ref.signature.keys() | spec.signature.keys():
        if i in optional:
            continue
        if (i in ref.signature and ref.signature[i].startswith("*")) or (
            i in spec.signature and spec.signature[i].startswith("*")
        ):
            continue
        # Allow partition differences: arg in signature in one spec,
        # in constants in another (annotation-as-variant pattern).
        if i not in ref.signature or i not in spec.signature:
            continue
        if ref.signature[i] != spec.signature[i]:
            r1 = _INT_WIDTH_RANK.get(ref.signature[i])
            r2 = _INT_WIDTH_RANK.get(spec.signature[i])
            if r1 is not None and r2 is not None:
                continue
            raise ValueError(
                f"Arg {i}: dtype mismatch '{ref.signature[i]}' vs "
                f"'{spec.signature[i]}' (spec 0 vs spec {idx})"
            )


def _check_constants_consistency(
    ref: KernelSpec,
    spec: KernelSpec,
    idx: int,
    optional: set[int],
) -> None:
    """Non-optional constant args must have the same Python type across specs.

    C++ codegen uses one type per constant arg position (``PY_TYPES_TO_CPP_TYPES``),
    so ``BLOCK_M=64`` (int) and ``BLOCK_M=64.0`` (float) would produce
    incompatible launchers.  Optional positions are validated separately
    by ``_check_optional_consistency``.
    """
    for i in ref.constants.keys() | spec.constants.keys():
        if i in optional:
            continue
        if ref.constants.get(i) is None or spec.constants.get(i) is None:
            continue
        if type(ref.constants[i]) is not type(spec.constants[i]):
            raise ValueError(
                f"Arg {i}: constant type mismatch "
                f"{type(ref.constants[i]).__name__} vs "
                f"{type(spec.constants[i]).__name__} (spec 0 vs spec {idx})"
            )


def _convert_raw_specs(
    base_specs: list[RawKernelSpec],
    gpu_target: GPUTarget,
) -> tuple[list[KernelSpec], set[int]]:
    """Convert raw specs to KernelSpecs.

    Returns (specs, three_tuple_optional) where three_tuple_optional is the
    union of optional_args detected from 3-tuple signature elements across
    all specs (backward compat with ``collect_constraints``).
    """
    raw_specs = cast(list[dict[str, Any]], copy.deepcopy(base_specs))
    is_amd = gpu_target.backend == "hip"

    result: list[KernelSpec] = []
    three_tuple_optional: set[int] = set()
    for raw_spec in raw_specs:
        constraints = collect_constraints(raw_spec["signature"])
        constants = extract_constants(raw_spec["signature"], constraints)
        signature: dict[int, str] = signature_list_to_dict(
            raw_spec["signature"], constants
        )
        three_tuple_optional |= constraints.optional_args

        spec = KernelSpec(
            signature=signature,
            constants=constants,
            divisible_by_16=constraints.divisible_by_16,
            divisible_by_8=constraints.divisible_by_8,
        )

        if constraints.has_fp8:
            if is_amd:
                spec.signature = get_fp8_replacement_signature_for_amd(
                    {"signature": spec.signature}, {str(gpu_target.arch)}
                )
            elif gpu_target.arch == 80:
                spec.signature = get_fp8_replacement_signature_for_sm80(
                    {"signature": spec.signature}
                )

        result.append(spec)

    return result, three_tuple_optional


def _autotune_specs(
    func: triton.runtime.autotuner.Autotuner,
    target: GPUTarget,
    specs: list[KernelSpec],
) -> list[KernelSpec]:
    tuned_specs: list[KernelSpec] = []
    for spec in specs:
        for cfg in func.cache.values():
            constants = spec.constants.copy()
            for arg_name, arg_val in cfg.kwargs.items():
                if arg_name in AUTOTUNE_ATTRs:
                    continue
                arg_idx = func.arg_names.index(arg_name)
                if constants.get(arg_idx, -1) == -1:
                    constants[arg_idx] = arg_val

            autotune_values: dict[str, int] = {}
            for name, default in AUTOTUNE_ATTRs.items():
                if name in cfg.kwargs:
                    autotune_values[name] = cfg.kwargs[name]
                else:
                    autotune_values[name] = getattr(cfg, name, default)
                # AMD has changed their software pipeliner in Triton
                # It now expects num_stages == 2 instead of 0
                # see: https://github.com/pytorch/pytorch/pull/139881
                # if we see someone try to set num_stages == 0, set it to the default (2) instead
                # We can't use the Triton hook to get the default value because it requires the AMD runtime to be loaded
                if (
                    target.backend == "hip"
                    and name == "num_stages"
                    and autotune_values[name] == 0
                    and TRITON_VERSION >= "3.2.0"
                ):
                    autotune_values[name] = 2

            tuned_spec = dataclasses.replace(
                spec,
                constants=constants,
                # pyrefly: ignore [bad-argument-type]
                **autotune_values,
            )
            tuned_specs.append(tuned_spec)
    return tuned_specs


def _dedup_specs(specs: list[KernelSpec]) -> list[KernelSpec]:
    deduped_specs: list[KernelSpec] = []
    duplicated_specs: list[KernelSpec] = []
    hash_spec_ids: set[str] = set()
    for spec in specs:
        id = hash_spec(dataclasses.asdict(spec))
        if id in hash_spec_ids:
            duplicated_specs.append(spec)
        else:
            hash_spec_ids.add(id)
            deduped_specs.append(spec)

    logger.debug(
        f"[TritonAOT Dedup] {len(specs)=} {len(deduped_specs)=} {len(duplicated_specs)=}"
    )
    return deduped_specs
