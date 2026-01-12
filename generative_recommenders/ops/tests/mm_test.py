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
from typing import Optional

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.ops.mm import addmm
from hypothesis import given, settings, strategies as st, Verbosity


class MMlTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        M=st.integers(min_value=100, max_value=300),
        N=st.integers(min_value=100, max_value=300),
        K=st.sampled_from([128, 256]),
        broadcast=st.booleans(),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16, torch.float16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_addmm(
        self,
        M: int,
        N: int,
        K: int,
        broadcast: bool,
        dtype: torch.dtype,
    ) -> None:
        self._test_addmm(
            M=M,
            N=N,
            K=K,
            broadcast=broadcast,
            dtype=dtype,
            kernel_type=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        M=st.integers(min_value=100, max_value=300),
        N=st.sampled_from([16, 48, 128, 144, 256]),
        K=st.sampled_from([16, 48, 128, 144, 256]),
        broadcast=st.booleans(),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16, torch.float16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_addmm_tma(
        self,
        M: int,
        N: int,
        K: int,
        broadcast: bool,
        dtype: torch.dtype,
    ) -> None:
        self._test_addmm(
            M=M,
            N=N,
            K=K,
            broadcast=broadcast,
            dtype=dtype,
            kernel_type=HammerKernel.TRITON,
        )

    def _test_addmm(
        self,
        M: int,
        N: int,
        K: int,
        broadcast: bool,
        dtype: torch.dtype,
        kernel_type: HammerKernel,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        # to enable more deterministic results.
        torch.manual_seed(0)

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        x: torch.Tensor = torch.rand((M, K), dtype=dtype, device="cuda").requires_grad_(
            True
        )
        w: torch.Tensor = torch.rand((K, N), dtype=dtype, device="cuda").requires_grad_(
            True
        )

        if broadcast:
            y: torch.Tensor = torch.rand(
                (N), dtype=dtype, device="cuda"
            ).requires_grad_(True)
        else:
            y: torch.Tensor = torch.rand(
                (M, N), dtype=dtype, device="cuda"
            ).requires_grad_(True)

        ref_z = addmm(y, x, w, kernel=HammerKernel.PYTORCH)
        dz = torch.randn_like(ref_z) * 0.1
        ref_z.backward(dz)
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_dw, w.grad = w.grad.detach().clone(), None
        ref_dy, y.grad = y.grad.detach().clone(), None

        x = x.detach().clone().requires_grad_(True)
        w = w.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        real_z = addmm(y, x, w, kernel=kernel_type)

        torch.testing.assert_close(ref_z, real_z, atol=atol, rtol=rtol)

        # triton cc doesn't support backward
        if kernel_type != HammerKernel.TRITON_CC:
            real_z.backward(dz)
            real_dx, x.grad = x.grad.detach().clone(), None
            real_dw, w.grad = w.grad.detach().clone(), None
            real_dy, y.grad = y.grad.detach().clone(), None

            torch.testing.assert_close(ref_dx, real_dx, atol=atol, rtol=rtol)
            torch.testing.assert_close(ref_dw, real_dw, atol=atol, rtol=rtol)
            torch.testing.assert_close(ref_dy, real_dy, atol=atol, rtol=rtol)
