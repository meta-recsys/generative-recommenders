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
from torch._dynamo.testing import CompileCounter


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

    def test_compile_accepts_statically_empty_tensor(self) -> None:
        torch._dynamo.reset()
        compiled = torch.compile(
            switch_to_contiguous_if_needed, backend="eager", fullgraph=True
        )
        x = torch.empty(0, 4)

        out = compiled(x)

        self.assertEqual(out.shape, (0, 4))

    def test_compile_accepts_dynamic_leading_size(self) -> None:
        # A symbolic size takes the range-hint branch.
        torch._dynamo.reset()
        compiled = torch.compile(
            switch_to_contiguous_if_needed, backend="eager", fullgraph=True
        )
        x = torch.arange(12.0).reshape(3, 4).transpose(0, 1)
        torch._dynamo.mark_dynamic(x, 0)

        out = compiled(x)

        self.assertTrue(torch.equal(out, x))
        self.assertTrue(out.is_contiguous())

    def test_compile_recompiles_dynamic_size_for_empty_tensor(self) -> None:
        # A dynamic size is never compiled as 0: an empty tensor fails the guard
        # of the code compiled for size 3 and is recompiled with the int size 0.
        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(
            switch_to_contiguous_if_needed,
            backend=counter,
            fullgraph=True,
            dynamic=True,
        )

        compiled(torch.randn(3, 4))
        out = compiled(torch.empty(0, 4))

        self.assertEqual(out.shape, (0, 4))
        self.assertEqual(counter.frame_count, 2)

    def test_compile_assumes_data_dependent_size_is_non_empty(self) -> None:
        # The number of nonzeros is known only at run time. The checks let the
        # compiler decide the branch below; a run with no nonzeros fails them.
        torch._dynamo.reset()

        @torch.compile(backend="aot_eager", fullgraph=True)
        def ones_per_nonzero(t: torch.Tensor) -> torch.Tensor:
            n = torch.nonzero(t).size(0)
            x = switch_to_contiguous_if_needed(torch.ones(n, 4))
            if x.size(0) == 0:
                return x
            return x + 1

        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            self.assertEqual(ones_per_nonzero(torch.tensor([0, 1, 1])).shape, (2, 4))
            with self.assertRaises(RuntimeError):
                ones_per_nonzero(torch.zeros(3))

    def test_compile_makes_non_contiguous_tensor_contiguous(self) -> None:
        compiled = torch.compile(
            switch_to_contiguous_if_needed, backend="eager", fullgraph=True
        )
        x = torch.arange(12.0).reshape(3, 4).transpose(0, 1)

        out = compiled(x)

        self.assertTrue(torch.equal(out, x))
        self.assertTrue(out.is_contiguous())


if __name__ == "__main__":
    unittest.main()
