# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

import importlib
import logging
import os
import pickle
from types import ModuleType, TracebackType
from typing import Any, Callable, Optional, Type

from generative_recommenders.ops.triton_aot.build.extension_builder import (
    build_triton_aot_extension,
)
from generative_recommenders.ops.triton_aot.compile.codegen import (
    is_non_empty_mapping_of_type,
)
from generative_recommenders.ops.triton_aot.compile.compile_state import (
    get_aott_compile_path,
    get_aott_compile_state,
    get_triton_aot_kernel_specs,
)
from generative_recommenders.ops.triton_aot.compile.pipeline import compile_to_cpp
from generative_recommenders.ops.triton_aot.compile.utils import unwrap_heuristic
from torch import package
from triton.backends.compiler import GPUTarget
from triton.runtime import driver, JITFunction

# @manual=//triton:triton
from triton.runtime.autotuner import Config

logger: logging.Logger = logging.getLogger(__name__)


class TritonAOTCompile:
    """
    Context manager to compile Triton kernels to C++ and build a shared library.
    The compiled kernels are cached in a temporary directory.

    - package_importer:
        torch.package importer for loading kernels source code (aott/ops).
        If not provided, the default importlib is used (for local use cases)
    - gpu_target:
        GPU target to compile for (default: active GPU target, determined by Triton driver)
    This local copy intentionally omits Manifold autotune-cache overrides. The
    HSTU e2e path only needs representative-input autotuning captured during
    the compile context.
    """

    def __init__(
        self,
        package_importer: Optional[package.PackageImporter] = None,
        gpu_target: Optional[GPUTarget] = None,
        auto_tune_cache_override_path: Optional[str] = None,
    ) -> None:
        self._import_module: Callable[[str], ModuleType] = (
            package_importer.import_module
            if package_importer is not None
            else importlib.import_module
        )
        self.gpu_target: GPUTarget = gpu_target or driver.active.get_current_target()
        self.auto_tune_cache_override_path: Optional[str] = (
            auto_tune_cache_override_path
        )

    def _load_autotune_cache_overrides(
        self,
    ) -> dict[str, Any]:
        if self.auto_tune_cache_override_path is None:
            return {}
        raise NotImplementedError(
            "Local generative_recommenders AOT-T compile does not support "
            "auto_tune_cache_override_path."
        )

    def __enter__(self) -> None:
        state = get_aott_compile_state()
        state.reset()
        state.enable()
        logger.info(
            f"Start AOTT compile, output dir: {get_aott_compile_path()}, gpu_target: {self.gpu_target}"
        )

    def _resolve_autotune_cache(
        self,
        fn: Any,
        fn_name: str,
        fn_dir: str,
        overrides: dict[str, Any],
    ) -> None:
        """Apply override (if matched) and dump the autotune cache to fn_dir."""
        override = overrides.get(fn_name)
        if override is not None:
            logger.info(
                f"[AOTT]: Overriding autotune cache for {fn_name} "
                f"from {self.auto_tune_cache_override_path}"
            )
            fn.cache = override

        # cache are dumped just for testing
        if hasattr(fn, "cache") and is_non_empty_mapping_of_type(fn.cache, Config):
            with open(f"{fn_dir}/{fn_name}_autotune_cache", "wb") as data:
                # @lint-ignore PYTHONPICKLEISBAD
                pickle.dump(fn.cache, data)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        compile_path = get_aott_compile_path()
        if not os.path.exists(compile_path):
            os.makedirs(compile_path)

        kernel_specs = get_triton_aot_kernel_specs()
        auto_tune_overrides = self._load_autotune_cache_overrides()

        logger.info(f"[AOTT]: compiling {len(kernel_specs)} kernels")

        for fn, specs in kernel_specs.items():
            jit_fn = unwrap_heuristic(fn, JITFunction)
            fn_name = jit_fn.__name__

            logger.info(f"[AOTT]: compiling {fn_name} with specs: {specs}")

            module_suffix = jit_fn.__module__.rsplit(".", 1)[-1]
            fn_dir = f"{compile_path}/{module_suffix}_{fn_name}"
            if not os.path.exists(fn_dir):
                os.makedirs(fn_dir)

            self._resolve_autotune_cache(fn, fn_name, fn_dir, auto_tune_overrides)

            compile_to_cpp(
                func=fn,
                base_specs=specs,
                install_dir=f"{fn_dir}",
                prefix=f"{fn_name}",
                gpu_target=self.gpu_target,
                tuner_fallback=True,
                import_module=self._import_module,
            )

            build_triton_aot_extension(
                source_dir=fn_dir,
                kernel_name=fn_name,
                output_dir=fn_dir,
            )

        get_aott_compile_state().disable()
