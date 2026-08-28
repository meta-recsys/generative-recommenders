# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from inspect import getcallargs, Parameter, signature
from typing import Any, Callable, Dict, List, Optional, Set

import torch

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.ops.triton_aot.compile.stable_types import SCALAR_TYPES
from generative_recommenders.ops.triton_aot.compile.utils import is_autotuner
from generative_recommenders.ops.triton_aot.types import (
    Annotation,
    AnnotationHint,
    TritonAOT,
)

# @manual=//triton:triton
from triton.runtime.jit import KernelInterface, mangle_type


class CustomEncoder(json.JSONEncoder):
    # pyre-ignore[14]: Inconsistent override
    def default(self, obj: object) -> Any:
        if isinstance(obj, set):
            return {"__set__": True, "items": sorted(obj)}
        # Handle other non-serializable types
        return super().default(obj)


def hash_spec(spec: Dict[str, Any]) -> str:
    serialized_dict = json.dumps(spec, cls=CustomEncoder, sort_keys=True)
    return hashlib.sha256(serialized_dict.encode("utf-8")).hexdigest()


class AOTTCompileState:
    """
    Singleton state container for Triton AOT compilation.

    Description:
    This singleton pattern enables state sharing between code loaded via
    torch.package (which creates isolated module namespaces) and the regular
    Python import system. Without this pattern, the packaged module would have
    its own copy of global state, leading to inconsistencies.

    Usage:
        # Normal usage - get the singleton instance
        state = AOTTCompileState.get_instance()

        # For torch.package integration - inject shared instance into packaged module
        packaged_module = package_importer.import_module("triton_aot.compile.compile_state")
        packaged_module.AOTTCompileState.set_instance(AOTTCompileState.get_instance())
    """

    _instance: Optional["AOTTCompileState"] = None

    kernel_specs: Dict[KernelInterface[List[Any]], List[Dict[str, List[Any]]]] = {}
    specs_hashset: Dict[KernelInterface[List[Any]], Set[str]] = {}
    enable_aott_compile: bool = False
    compile_base_dir: str = ""
    compile_path: str = ""

    def __new__(cls) -> "AOTTCompileState":
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialize()
            cls._instance = instance
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the singleton state. Called only once."""
        self.kernel_specs: Dict[
            KernelInterface[List[Any]], List[Dict[str, List[Any]]]
        ] = {}
        self.specs_hashset: Dict[KernelInterface[List[Any]], Set[str]] = {}
        self.enable_aott_compile: bool = False
        self.compile_base_dir: str = os.getenv("TRITON_AOT_PATH_PREFIX", "/var/tmp")
        self.compile_path: str = tempfile.mkdtemp(
            dir=self.compile_base_dir, prefix="triton_aot_compile_"
        )

    @classmethod
    def get_instance(cls) -> "AOTTCompileState":
        """Get the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set_instance(cls, instance: "AOTTCompileState") -> None:
        """
        Set the singleton instance. Used for torch.package integration.

        When code is loaded via torch.package, it creates a separate module
        namespace with its own class objects. This method allows injecting
        a shared instance from the main module into the packaged module.
        """
        cls._instance = instance

    def reset(self) -> None:
        """Reset all state to initial values."""
        self.kernel_specs = {}
        self.specs_hashset = {}
        self.disable()
        self.compile_base_dir = os.getenv("TRITON_AOT_PATH_PREFIX", "/var/tmp")
        self.compile_path = tempfile.mkdtemp(
            dir=self.compile_base_dir, prefix="triton_aot_compile_"
        )

    def add_kernel_spec(
        self,
        fn: KernelInterface[List[Any]],
        spec: Dict[str, List[Any]],
        hashed_spec: str,
    ) -> None:
        """Add a kernel spec if not already present (based on hash).
        If the same Triton kernel is used at multiple locations in a model:
            - All calls share one spec list under the same kernel function key
            - Specs with identical signatures (same dtypes, shapes) are deduplicated via hash
            - Specs with different signatures (e.g., fp32 vs bf16) are recorded separately

        Example:
            # Two call sites using the same kernel:
            my_kernel[grid](tensor_fp32, ...)  # Records spec with "*fp32"
            my_kernel[grid](tensor_bf16, ...)  # Records spec with "*bf16"
            my_kernel[grid](tensor_fp32, ...)  # Deduplicated, same hash as first call

            # Result: kernel_specs[my_kernel] = [fp32_spec, bf16_spec]
        """
        if fn not in self.kernel_specs:
            self.kernel_specs[fn] = []
            self.specs_hashset[fn] = set()
        if hashed_spec not in self.specs_hashset[fn]:
            self.kernel_specs[fn].append(spec)
            self.specs_hashset[fn].add(hashed_spec)

    def _collect_spec(
        self,
        fn: KernelInterface[List[Any]],
        annotations: Dict[str, Annotation],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Spec collection callback registered on TritonAOT during compile.

        Always collects the annotated spec (which equals the inferred spec
        when no annotations are present).  Also collects the inferred spec
        when it differs and either:
        - annotations conflict with sample (fallback for safety), or
        - inferred has perf hints the annotation lacks (perf variant).
        """
        spec = infer_spec(fn, annotations, *args, **kwargs)
        annotated_hash = hash_spec(spec)
        self.add_kernel_spec(fn, spec, annotated_hash)

        if annotations:
            inferred = infer_spec(fn, {}, *args, **kwargs)
            inferred_hash = hash_spec(inferred)
            if inferred_hash == annotated_hash:
                return
            if _annotation_conflicts_with_sample(
                fn, annotations, *args, **kwargs
            ) or _inferred_has_perf_advantage(spec, inferred):
                self.add_kernel_spec(fn, inferred, inferred_hash)

    def enable(self) -> None:
        """Enable AOT compile and register the spec collection hook."""
        self.enable_aott_compile = True
        TritonAOT.set_spec_collector(self._collect_spec)

    def disable(self) -> None:
        """Disable AOT compile and unregister the spec collection hook."""
        self.enable_aott_compile = False
        TritonAOT.set_spec_collector(None)


def get_aott_compile_state() -> AOTTCompileState:
    """Get the current AOTTCompileState singleton.

    Uses get_instance() so injected instances (via set_instance() for
    torch.package integration) are respected.
    """
    return AOTTCompileState.get_instance()


########
# Module-level global accessors that delegate to singleton
########


def get_triton_aot_kernel_specs() -> Dict[
    KernelInterface[List[Any]], List[Dict[str, List[Any]]]
]:
    return get_aott_compile_state().kernel_specs


def get_triton_aot_specs_hashset() -> Dict[KernelInterface[List[Any]], Set[str]]:
    return get_aott_compile_state().specs_hashset


def get_aott_compile_path() -> str:
    return get_aott_compile_state().compile_path


def add_kernel_spec(
    fn: KernelInterface[List[Any]], spec: Dict[str, List[Any]], hashed_spec: str
) -> None:
    get_aott_compile_state().add_kernel_spec(fn, spec, hashed_spec)


def _unwrap_triton_fn(
    fn: KernelInterface[List[Any]],
) -> Callable[..., Any]:
    while isinstance(fn, KernelInterface):
        # pyre-ignore[16]: KernelInterface has `fn` attribute at runtime
        fn = fn.fn
    return fn


def _inferred_has_perf_advantage(
    annotated_spec: Dict[str, List[Any]],
    inferred_spec: Dict[str, List[Any]],
) -> bool:
    """True if inferred spec has alignment/divisibility hints the annotated lacks.

    A tuple element ``(type, N)`` carries alignment or divisibility info
    that a bare string does not.  When inference adds such hints (e.g.,
    tensor alignment from ``data_ptr() % 16 == 0``), the inferred spec
    produces a more optimized cubin worth keeping as a perf variant.
    """
    for ann_elem, inf_elem in zip(
        annotated_spec["signature"], inferred_spec["signature"]
    ):
        if isinstance(inf_elem, tuple) and not isinstance(ann_elem, tuple):
            return True
    return False


# Triton-internal kwargs injected by KernelInterface.__getitem__
# (triton/runtime/jit.py).  These are not kernel parameters and must
# be stripped before getcallargs.
_TRITON_INTERNAL_KWARGS: frozenset[str] = frozenset({"warmup", "grid"})


def _resolve_call_args(
    fn: KernelInterface[List[Any]],
    *args: Any,
    **kwargs: Any,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Unwrap kernel and resolve call args with autotune placeholder fill."""
    triton_fn = _unwrap_triton_fn(fn)
    # Filter Triton-internal kwargs injected by KernelInterface.__getitem__
    # (triton/runtime/jit.py) — not part of the kernel signature.
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in _TRITON_INTERNAL_KWARGS}
    if is_autotuner(fn):
        # pyre-ignore[16]: Attributes checked by is_autotuner
        for arg_name in fn.configs[0].kwargs.keys():
            if arg_name not in clean_kwargs:
                clean_kwargs[arg_name] = -1
    return triton_fn, getcallargs(triton_fn, *args, **clean_kwargs)


_I32_MIN: int = -(2**31)
_I32_MAX: int = 2**31 - 1


def _sample_satisfies_int_type(sample: int, ann_type: str) -> bool:
    """True if sample int fits the annotated type range."""
    if ann_type == "i32":
        return _I32_MIN <= sample <= _I32_MAX
    return True


def _sample_satisfies_annotation(sample: Any, ann: Annotation) -> bool:
    """True if a single sample value satisfies its annotation constraint."""
    if isinstance(ann, AnnotationHint):
        if isinstance(sample, torch.Tensor):
            return sample.data_ptr() % ann.hint == 0
        if isinstance(sample, int):
            if ann.hint == 1:
                return sample == 1
            if not _sample_satisfies_int_type(sample, ann.dtype):
                return False
            if ann.hint > 1:
                return sample % ann.hint == 0
        return True
    if isinstance(ann, str) and not ann.startswith("*") and isinstance(sample, int):
        return _sample_satisfies_int_type(sample, ann)
    return True


def _annotation_conflicts_with_sample(
    fn: KernelInterface[List[Any]],
    annotations: Dict[str, Annotation],
    *args: Any,
    **kwargs: Any,
) -> bool:
    """True if any annotated param's sample value doesn't satisfy the annotation.

    Used by ``_collect_spec`` to decide whether to generate an inferred
    fallback spec.  When the sample satisfies all annotations, only the
    annotated spec is needed (the user's constraints hold for this input).
    """
    _, sample_args = _resolve_call_args(fn, *args, **kwargs)

    for param_name, ann in annotations.items():
        sample = sample_args.get(param_name)
        if sample is None:
            continue
        if not _sample_satisfies_annotation(sample, ann):
            return True

    return False


def _infer_spec_entry(
    arg_name: str,
    arg: Any,
    arg_annotation: Any,
    annotations: Dict[str, Annotation],
) -> Any:
    if arg_annotation != Parameter.empty:
        if arg_annotation == tl.constexpr:
            return arg
        raise RuntimeError(
            f"TritonAOT: unsupported scalar annotation {arg_annotation}."
        )

    if arg_name in annotations:
        ann = annotations[arg_name]
        # Convert to tuple for raw spec format (shared/spec_conversion
        # processes plain tuples).
        return ann.to_tuple() if isinstance(ann, AnnotationHint) else ann

    if arg is None:
        return None

    if isinstance(arg, torch.Tensor):
        # Reject dtypes SCALAR_TYPES can't render (e.g. *u1, *u16, *fp8e5)
        # so codegen doesn't KeyError downstream.
        type_str = mangle_type(arg)
        if type_str not in SCALAR_TYPES:
            raise RuntimeError(
                f"TritonAOT: unsupported tensor type for {arg_name}: "
                f"{arg.dtype} (Triton mangled to {type_str!r}). "
                f"Supported tensor dtypes: {sorted(SCALAR_TYPES.keys())}."
            )
        return (type_str, 16) if arg.data_ptr() % 16 == 0 else type_str

    if isinstance(arg, bool):
        # bool is subclass of int; must check before int.
        # Non-constexpr bools have no CTYPES entry for codegen.
        raise RuntimeError(
            f"TritonAOT: parameter {arg_name} is a bool without "
            f"tl.constexpr annotation.  Add `{arg_name}: tl.constexpr` "
            f"to the kernel signature."
        )

    if isinstance(arg, int):
        # Always i64 for safety; users annotate "i32" for narrower variant via
        # annotation-as-variant.
        if not -(2**63) <= arg <= 2**63 - 1:
            raise RuntimeError(
                f"TritonAOT: unsupported int value for {arg_name}: "
                f"value exceeds i64 range. Use a smaller value or tl.constexpr."
            )
        return "i64"

    if isinstance(arg, float):
        return "fp32"

    raise RuntimeError(f"TritonAOT: parameter {arg_name} needs annotation.")


def infer_spec(
    fn: KernelInterface[List[Any]],
    annotations: Dict[str, Annotation],
    *args: Any,
    **kwargs: Any,
) -> Dict[str, List[Any]]:
    """Infer kernel spec from sample args.

    Tensor dtype: ``mangle_type``, alignment: ``data_ptr() % 16``.
    Scalar int: always ``"i64"`` (safe default; user can annotate ``"i32"``
    to get a narrower variant via annotation-as-variant).
    Float: ``mangle_type`` → fp32.
    """
    triton_fn, call_args = _resolve_call_args(fn, *args, **kwargs)
    fn_sig = signature(triton_fn)
    arg_annotations = {
        name: param.annotation for name, param in fn_sig.parameters.items()
    }
    spec = []

    for arg_name in fn_sig.parameters.keys():
        arg = call_args[arg_name]
        spec.append(
            _infer_spec_entry(arg_name, arg, arg_annotations[arg_name], annotations)
        )
    return {"signature": spec}
