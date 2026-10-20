# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python3

# pyre-strict

import importlib.util
import logging
import os
import sys
from typing import Any, Dict, Optional

from generative_recommenders.ops.triton_aot.compile.compile_state import (
    get_aott_compile_path,
)
from torch import package
from torch.fx import GraphModule

logger: logging.Logger = logging.getLogger(__name__)


def _find_wrapper_files(
    compile_path: str,
) -> list[tuple[str, str, str]]:
    """Find all ``*_wrapper.py`` files under *compile_path*.

    Walks one level deep into kernel subdirectories and returns a list of
    ``(wrapper_name, fn_name, wrapper_path)`` tuples.
    """
    results: list[tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(compile_path):
        if dirpath != compile_path:
            dirnames.clear()  # only recurse one level into kernel subdirs
        for item in filenames:
            if item.endswith("_wrapper.py"):
                wrapper_name = item.removesuffix(".py")
                fn_name = wrapper_name.removesuffix("_wrapper")
                wrapper_path = os.path.join(dirpath, item)
                results.append((wrapper_name, fn_name, wrapper_path))
    return results


def _load_wrapper_module(
    wrapper_name: str,
    fn_name: str,
    wrapper_path: str,
    package_importer: Optional[package.PackageImporter],
) -> Optional[Any]:
    """Dynamically import a single ``*_wrapper.py`` and return its wrapper callable.

    Returns ``None`` if the module does not expose a function named *fn_name*.
    """
    spec = importlib.util.spec_from_file_location(wrapper_name, wrapper_path)
    assert spec is not None, f"Failed to create spec for {wrapper_path}"
    assert spec.loader is not None, f"Spec has no loader for {wrapper_path}"

    loader = spec.loader
    wrapper_module = importlib.util.module_from_spec(spec)

    sys.modules[wrapper_name] = wrapper_module

    if package_importer is not None:
        wrapper_module._package_importer = package_importer  # type: ignore[attr-defined]

    loader.exec_module(wrapper_module)

    if hasattr(wrapper_module, fn_name):
        return getattr(wrapper_module, fn_name)
    return None


def replace_kernels(
    fx_m: GraphModule,
    eager: bool = False,
    package_importer: Optional[package.PackageImporter] = None,
) -> GraphModule:
    if eager:
        raise NotImplementedError(
            "Local generative_recommenders AOT-T transform does not support "
            "eager replacement."
        )

    compile_path = get_aott_compile_path()
    assert os.path.exists(compile_path), "triton_aot_compile dir does not exist"

    wrapper_dict: Dict[str, Any] = {}
    for wrapper_name, fn_name, wrapper_path in _find_wrapper_files(compile_path):
        wrapper_fn = _load_wrapper_module(
            wrapper_name, fn_name, wrapper_path, package_importer
        )
        if wrapper_fn is not None:
            wrapper_dict[fn_name] = wrapper_fn

    logger.info(f"replace_kernels: {wrapper_dict=}")

    # Phase 2: Replace FX graph nodes
    #   Walk the FX graph, find call_function nodes whose target name
    #   matches a loaded wrapper, and swap the target so that
    #   kernel[grid](...) calls become torch.ops.triton_aot.* calls.
    replaced_count = 0
    for nodes in fx_m.graph.nodes:
        if nodes.op == "call_function" and nodes.target.__name__ in wrapper_dict.keys():
            logger.info(
                f"Replaced node: {nodes.op} {nodes.target} -> {wrapper_dict[nodes.target.__name__]} {nodes.meta}"
            )
            nodes.target = wrapper_dict[nodes.target.__name__]
            replaced_count += 1

    assert replaced_count > 0, (
        f"No ops were replaced with triton_aot wrappers. "
        f"wrapper_dict={wrapper_dict}, compile_path={compile_path}"
    )
    logger.info(
        f"Successfully replaced {replaced_count} op(s) with triton_aot wrappers."
    )

    fx_m.recompile()
    return fx_m
