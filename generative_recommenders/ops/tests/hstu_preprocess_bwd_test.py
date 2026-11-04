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

import unittest

import torch
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.ops.triton.triton_hstu_preprocess_bwd import (
    copy_u_grad_and_column_sum,
)


class CopyUGradAndColumnSumTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_matches_copy_then_sum(self) -> None:
        torch.manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16):
            for T, U, N in (
                (0, 64, 192),
                (77, 64, 192),
                (5000, 512, 2048),
                (300, 100, 330),
            ):
                duvqk = torch.randn(T, N, device="cuda", dtype=dtype)
                source = torch.randn(T, 2 * U, device="cuda", dtype=dtype)[:, :U]
                ref = duvqk.clone()
                ref[:, :U].copy_(source)
                ref_sum = ref.float().sum(dim=0)

                out = duvqk.clone()
                col_sum = copy_u_grad_and_column_sum(source, out)

                self.assertTrue(torch.equal(out, ref))
                self.assertEqual(col_sum.dtype, dtype)
                self.assertEqual(col_sum.shape, (N,))
                tol = 1e-2 if dtype == torch.bfloat16 else 1e-4
                torch.testing.assert_close(
                    col_sum.float(), ref_sum, atol=tol * max(1, T) ** 0.5, rtol=tol
                )

    @unittest.skipIf(*gpu_unavailable)
    def test_deterministic(self) -> None:
        torch.manual_seed(0)
        duvqk = torch.randn(9000, 1024, device="cuda", dtype=torch.bfloat16)
        source = torch.randn(9000, 256, device="cuda", dtype=torch.bfloat16)
        first = copy_u_grad_and_column_sum(source, duvqk.clone())
        for _ in range(3):
            self.assertTrue(
                torch.equal(copy_u_grad_and_column_sum(source, duvqk.clone()), first)
            )


if __name__ == "__main__":
    unittest.main()
