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

import unittest

import torch
from generative_recommenders.common import switch_to_contiguous_if_needed


class SwitchToContiguousIfNeededTest(unittest.TestCase):
    def test_torchscript_does_not_compile_fx_tracing_helper(self) -> None:
        class ContiguousModule(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return switch_to_contiguous_if_needed(x)

        scripted = torch.jit.script(ContiguousModule())
        x = torch.arange(12).reshape(3, 4).transpose(0, 1)

        out = scripted(x)

        self.assertTrue(torch.equal(out, x))
        self.assertTrue(out.is_contiguous())


if __name__ == "__main__":
    unittest.main()
