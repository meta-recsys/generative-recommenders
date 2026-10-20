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

# pyre-strict

"""Guards the AOT-T wrappers against being un-deserializable from a model package.

The shared ``generative_recommenders/ops/*.py`` modules import the AOT-T
wrappers at module scope, inside ``try: ... except ImportError``.  torch.package's
dependency scanner is a plain AST walk, so it cannot see that guard: every
module the wrappers can reach is pulled into any model package that interns
``generative_recommenders.**``.  An interned module then runs under a mangled
``<torch_package_N>.`` name that is absent from ``sys.modules``.

This has broken model deserialization before: ``triton_aot/types.py`` combined
``from __future__ import annotations`` with an ``@dataclass``, so CPython's
``dataclasses._is_type`` evaluated ``sys.modules.get(cls.__module__).__dict__``,
got ``None``, and raised ``AttributeError: 'NoneType' object has no attribute
'__dict__'`` while deserializing.  An ``AttributeError`` is not an
``ImportError``, so the fallback stubs did not catch it and loading the whole
package failed.

These tests intern each wrapper exactly the way a model package would and
import it back, so the next module that reintroduces the hazard fails here
instead of when a packaged model is loaded.
"""

import importlib
import io
import unittest

from torch import package

_TRITON_AOT = "generative_recommenders.ops.triton_aot"

# Every module reachable from the shared ops' module-scope imports.
WRAPPER_MODULES: list[str] = [
    f"{_TRITON_AOT}.types",
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


class TritonAotTorchPackageTest(unittest.TestCase):
    def _intern_and_reimport(self, module_name: str) -> None:
        """Package these modules the way a model exporter does, then load them back."""
        importlib.import_module(module_name)

        buf = io.BytesIO()
        with package.PackageExporter(buf) as exporter:
            # Order matters: the first matching pattern wins, so the AOT-T tree
            # is interned and everything else (torch, triton, the rest of GR)
            # is externed, so the AOT-T modules run under the mangled package
            # name exactly as they would inside an exported model.
            exporter.intern(f"{_TRITON_AOT}.**")
            exporter.extern("**")
            exporter.save_module(module_name)
        buf.seek(0)

        imported = package.PackageImporter(buf).import_module(module_name)
        self.assertIsNotNone(imported)

    def test_wrapper_modules_survive_torch_package(self) -> None:
        for module_name in WRAPPER_MODULES:
            with self.subTest(module=module_name):
                self._intern_and_reimport(module_name)


if __name__ == "__main__":
    unittest.main()
