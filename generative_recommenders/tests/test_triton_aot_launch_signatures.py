# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

"""Checks every AOT-T kernel launch against the kernel it launches, on CPU.

Each AOT-T wrapper module hand-writes a ``kernel[grid](...)`` launch for a
Triton kernel that usually lives in ``generative_recommenders/ops/triton/`` and
keeps evolving there. When the kernel gains a required argument, or loses one,
the wrapper's launch goes stale and nothing fails until someone runs the AOT-T
path on a GPU -- which is how ``_hstu_attn_fwd``'s ``min_full_attn_seq_len`` /
``HAS_MIN_FULL_ATTN_SEQ_LEN`` broke the AOT-T end-to-end test.

This test reads each wrapper's source, finds its launches, and compares the
arguments passed with the kernel's real signature: every parameter without a
default must be passed, set by an autotune config or set by a heuristic, and
every keyword passed must be a kernel parameter. It needs no GPU, so it runs
in the diff that changes a kernel.
"""

import ast
import importlib
import inspect
import unittest
from typing import Any, Dict, List, Set, Tuple

from generative_recommenders.ops.triton_aot.types import TritonAOT
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction

_TRITON_AOT = "generative_recommenders.ops.triton_aot"

# Every AOT-T wrapper module that launches a kernel.
WRAPPER_MODULES: List[str] = [
    f"{_TRITON_AOT}.triton_addmm",
    f"{_TRITON_AOT}.triton_concat_2d_jagged",
    f"{_TRITON_AOT}.triton_group_norm_mul_dropout",
    f"{_TRITON_AOT}.triton_layer_norm",
    f"{_TRITON_AOT}.triton_layer_norm_mul_dropout",
    f"{_TRITON_AOT}.triton_position",
    f"{_TRITON_AOT}.triton_ragged_hstu_attention",
    f"{_TRITON_AOT}.triton_rms_norm",
    f"{_TRITON_AOT}.triton_split_2d_jagged",
]

# Launch options Triton accepts on any kernel, not kernel parameters.
_LAUNCH_OPTIONS: Set[str] = {"num_warps", "num_stages", "num_ctas", "maxnreg"}


def _kernel_contract(kernel: TritonAOT) -> Tuple[List[str], Set[str], Set[str]]:
    """Return (parameter names, parameters with defaults, parameters set for you).

    "Set for you" covers what autotune configs and heuristics fill in, so the
    launch does not pass them.
    """
    provided: Set[str] = set()
    fn: Any = kernel.fn
    while not isinstance(fn, JITFunction):
        if isinstance(fn, Autotuner):
            for config in fn.configs:
                provided |= set(config.kwargs)
        elif isinstance(fn, Heuristics):
            provided |= set(fn.values)
        fn = fn.fn
    signature = inspect.signature(fn.fn)
    defaulted = {
        name
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    return list(fn.arg_names), defaulted, provided


def _launches(module: Any, kernel_names: Set[str]) -> List[Tuple[str, int, ast.Call]]:
    """Find every ``kernel[grid](...)`` call to one of ``kernel_names``."""
    tree = ast.parse(inspect.getsource(module))
    launches = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in kernel_names
        ):
            launches.append((node.func.value.id, node.lineno, node))
    return launches


class TritonAotLaunchSignatureTest(unittest.TestCase):
    def test_launches_match_kernel_signatures(self) -> None:
        for module_name in WRAPPER_MODULES:
            module = importlib.import_module(module_name)
            kernels: Dict[str, TritonAOT] = {
                name: value
                for name, value in vars(module).items()
                if isinstance(value, TritonAOT)
            }
            launches = _launches(module, set(kernels))
            with self.subTest(module=module_name):
                # Every wrapper module launches at least one kernel; an empty
                # result means the scan itself stopped finding them.
                self.assertGreater(len(launches), 0)
            for kernel_name, lineno, call in launches:
                with self.subTest(module=module_name, launch=f"{kernel_name}:{lineno}"):
                    params, defaulted, provided = _kernel_contract(kernels[kernel_name])
                    passed = set(params[: len(call.args)]) | {
                        kw.arg for kw in call.keywords if kw.arg is not None
                    }
                    missing = [
                        p
                        for p in params
                        if p not in passed and p not in defaulted and p not in provided
                    ]
                    unknown = sorted(passed - set(params) - _LAUNCH_OPTIONS)
                    self.assertEqual(
                        missing,
                        [],
                        f"{module_name}:{lineno} does not pass required "
                        f"{kernel_name} arguments {missing}",
                    )
                    self.assertEqual(
                        unknown,
                        [],
                        f"{module_name}:{lineno} passes {unknown}, which "
                        f"{kernel_name} does not take",
                    )
