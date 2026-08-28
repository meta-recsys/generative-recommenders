# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional, Protocol, Union

from generative_recommenders.ops.triton_aot.compile.utils import is_autotuner

# @manual=//triton:triton
from triton.runtime.jit import KernelInterface
# triton.fb.triton_util depends on torch
# @dep=//caffe2:_torch


_VALID_HINTS: frozenset[int] = frozenset({1, 8, 16})
_VALID_POINTER_HINTS: frozenset[int] = frozenset({16})


@dataclass(frozen=True)
class AnnotationHint:
    """Annotation with a value hint (dtype + divisibility/alignment).

    Valid hints: 16 (divisible_by_16), 8 (divisible_by_8), 1 (equal_to_1).
    For pointers (dtype starts with ``*``), only 16 is valid — other values
    would cause incorrect codegen (e.g. alignment=1 folds the pointer as a
    constexpr constant, causing a segfault at launch).
    """

    dtype: str
    hint: int

    def __post_init__(self) -> None:
        if self.hint not in _VALID_HINTS:
            raise RuntimeError(
                f"TritonAOT: invalid annotation hint {self.hint!r} for "
                f"dtype {self.dtype!r}. Valid hints: {sorted(_VALID_HINTS)}."
            )
        if self.dtype.startswith("*") and self.hint not in _VALID_POINTER_HINTS:
            raise RuntimeError(
                f"TritonAOT: invalid pointer alignment {self.hint!r} for "
                f"dtype {self.dtype!r}. Pointer annotations only support "
                f"alignment={sorted(_VALID_POINTER_HINTS)}."
            )

    def to_tuple(self) -> tuple[str, int]:
        """Convert to plain tuple for raw spec format."""
        return (self.dtype, self.hint)


# Internal annotation type (after normalization).
Annotation = Union[str, AnnotationHint]

# User-facing input type (also accepts raw tuples).
AnnotationInput = Union[str, tuple[str, int], AnnotationHint]


def _normalize_annotation(ann: AnnotationInput) -> Annotation:
    """Convert a raw tuple to AnnotationHint (triggers validation)."""
    if isinstance(ann, AnnotationHint):
        return ann
    if isinstance(ann, tuple):
        return AnnotationHint(ann[0], ann[1])
    return ann


class SpecCollector(Protocol):
    """Callback invoked by TritonAOT.run() to collect kernel specs during AOT compile."""

    def __call__(
        self,
        fn: KernelInterface[List[Any]],
        annotations: Dict[str, Annotation],
        *args: Any,
        **kwargs: Any,
    ) -> None: ...


logger: logging.Logger = logging.getLogger(__name__)


class TritonAOTMeta(type):
    # TODO consider merge with AOTTCompileState
    def __init__(cls, name, bases, attrs):  # pyre-ignore [2,3]
        super().__init__(name, bases, attrs)
        # Initialize an empty list for each new class created
        cls._instances: List["TritonAOT"] = []

    def __call__(cls, *args, **kwargs):  # pyre-ignore [2,3]
        # Create the instance using the default behavior
        instance = super().__call__(*args, **kwargs)
        # Store the instance in the class-specific list
        cls._instances.append(instance)
        return instance

    def get_instances(cls) -> List["TritonAOT"]:
        return cls._instances


class TritonAOT(KernelInterface[List[Any]], metaclass=TritonAOTMeta):
    """Wraps a Triton kernel for ahead-of-time compilation.

    Annotations specify dtype and optional value hints for kernel parameters:

    - Scalar:  ``"i32"``, ``"fp32"``, or ``AnnotationHint("i32", 16)``
      where 16 means the runtime value is divisible by 16.
    - Pointer: ``AnnotationHint("*fp32", 16)`` for 16-byte aligned tensors.
      Only alignment=16 is valid for pointers.
    - Tensor:  typically inferred from runtime ``torch.Tensor.dtype``.
    - Optional tensor:  auto-detected when the same kernel is called
      with a tensor at one site and ``None`` at another.
    """

    _spec_collector: ClassVar[Optional[SpecCollector]] = None

    def __init__(
        self,
        fn: KernelInterface[List[Any]],
        annotations: Dict[str, AnnotationInput],
    ) -> None:
        self.fn: KernelInterface[List[Any]] = fn
        self.annotations: Dict[str, Annotation] = {
            k: _normalize_annotation(v) for k, v in annotations.items()
        }

    @classmethod
    def set_spec_collector(cls, collector: Optional[SpecCollector]) -> None:
        """Register or unregister the spec collection callback.

        When a collector is registered (not None), TritonAOT.run() will call
        it to collect kernel specs for AOT compilation.  When None, run()
        simply delegates to the underlying Triton kernel (normal JIT path).
        """
        cls._spec_collector = collector

    # pyrefly: ignore [bad-override]
    def run(self, *args: Any, **kwargs: Any) -> Any:
        if self._spec_collector is not None:
            self._spec_collector(self.fn, self.annotations, *args, **kwargs)
        # pyre-ignore[29]: KernelInterface.run is callable at runtime
        return self.fn.run(*args, **kwargs)


def triton_aot(
    annotations: Dict[str, AnnotationInput],
) -> Callable[[KernelInterface[List[Any]]], TritonAOT]:
    def decorator(fn: KernelInterface[List[Any]]) -> TritonAOT:
        return TritonAOT(fn, annotations)

    return decorator


def get_all_triton_aot_instances() -> List[TritonAOT]:
    """Return all triton aot function instances (e.g. decorated with @triton_aot)."""
    return TritonAOT.get_instances()


def reset_all_triton_aot_autotune_cache() -> bool:
    """Reset triton autotune cache for all triton aot kernels.

    If triton aot compile is not enabled, this function is no op. Return True if any
    kernel's autotune cache is reset. Else return False.

    """
    if TritonAOT._spec_collector is None:
        return False

    reset = False
    for triton_aot_kernel in get_all_triton_aot_instances():
        if is_autotuner(triton_aot_kernel.fn):
            autotune_fn = triton_aot_kernel.fn
            autotune_fn.cache.clear()  # pyre-ignore [16]
            logger.info(
                f"Reset autotune cache for triton kernel {autotune_fn.fn.__name__}"  # pyre-ignore [16]
            )
            reset = True

    return reset
