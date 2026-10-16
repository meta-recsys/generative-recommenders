# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import Optional

from generative_recommenders.ops.triton_aot.transform.kernel_wrapper_codegen import (
    kernel_wrapper_codegen,
)
from generative_recommenders.ops.triton_aot.transform.replace_kernels import (
    replace_kernels,
)
from torch import package
from torch.fx import GraphModule


def transform_kernels(
    fx_m: GraphModule,
    eager: bool = False,
    package_importer: Optional[package.PackageImporter] = None,
) -> GraphModule:
    """Generate AOT wrappers and replace FX graph nodes in one step.

    1. kernel_wrapper_codegen: AST-transforms wrapper functions,
       rewrites kernel[grid](...) -> torch.ops.triton_aot.kernel(...),
       writes {fn}_wrapper.py
    2. replace_kernels: loads wrappers and replaces graph node targets
    """
    kernel_wrapper_codegen(fx_m, package_importer)
    return replace_kernels(fx_m, eager=eager, package_importer=package_importer)
