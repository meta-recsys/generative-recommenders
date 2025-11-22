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
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.ops.layer_norm import rms_norm


class RMSNormSILUTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_rms_norm_silu_forward(self) -> None:
        """Test RMS norm with SILU activation"""
        torch.manual_seed(42)
        N, D = 128, 512
        device = torch.device("cuda")
        dtype = torch.bfloat16

        x = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

        # Test with SILU=False (should match original behavior)
        y_no_silu_triton = rms_norm(
            x, weight, eps=1e-5, kernel=HammerKernel.TRITON, silu=False
        )
        y_no_silu_pytorch = rms_norm(
            x, weight, eps=1e-5, kernel=HammerKernel.PYTORCH, silu=False
        )

        torch.testing.assert_close(
            y_no_silu_triton,
            y_no_silu_pytorch,
            rtol=1e-2,
            atol=1e-2,
            msg="RMS norm without SILU should match between Triton and PyTorch",
        )

        # Test with SILU=True
        y_silu_triton = rms_norm(
            x, weight, eps=1e-5, kernel=HammerKernel.TRITON, silu=True
        )
        y_silu_pytorch = rms_norm(
            x, weight, eps=1e-5, kernel=HammerKernel.PYTORCH, silu=True
        )

        torch.testing.assert_close(
            y_silu_triton,
            y_silu_pytorch,
            rtol=1e-2,
            atol=1e-2,
            msg="RMS norm with SILU should match between Triton and PyTorch",
        )

        # SILU should produce different output than no SILU
        self.assertFalse(
            torch.allclose(y_no_silu_triton, y_silu_triton, rtol=1e-2, atol=1e-2),
            "SILU should change the output",
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_rms_norm_silu_backward(self) -> None:
        """Test RMS norm with SILU backward pass"""
        torch.manual_seed(42)
        N, D = 128, 512
        device = torch.device("cuda")
        dtype = torch.bfloat16

        x_triton = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        weight_triton = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

        x_pytorch = x_triton.clone().detach().requires_grad_(True)
        weight_pytorch = weight_triton.clone().detach().requires_grad_(True)

        # Forward pass
        y_triton = rms_norm(
            x_triton, weight_triton, eps=1e-5, kernel=HammerKernel.TRITON, silu=True
        )
        y_pytorch = rms_norm(
            x_pytorch,
            weight_pytorch,
            eps=1e-5,
            kernel=HammerKernel.PYTORCH,
            silu=True,
        )

        # Backward pass
        grad_output = torch.randn_like(y_triton)
        y_triton.backward(grad_output)
        y_pytorch.backward(grad_output)

        # Compare gradients
        assert x_triton.grad is not None
        assert x_pytorch.grad is not None
        assert weight_triton.grad is not None
        assert weight_pytorch.grad is not None

        torch.testing.assert_close(
            x_triton.grad,
            x_pytorch.grad,
            rtol=1e-1,
            atol=1e-1,
            msg="Input gradients should match between Triton and PyTorch",
        )

        torch.testing.assert_close(
            weight_triton.grad,
            weight_pytorch.grad,
            rtol=1e-1,
            atol=1e-1,
            msg="Weight gradients should match between Triton and PyTorch",
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_rms_norm_silu_triton_cc_raises(self) -> None:
        """Test that TRITON_CC raises error with SILU=True"""
        torch.manual_seed(42)
        N, D = 128, 512
        device = torch.device("cuda")
        dtype = torch.bfloat16

        x = torch.randn(N, D, device=device, dtype=dtype)
        weight = torch.randn(D, device=device, dtype=dtype)

        # TRITON_CC with SILU=True should raise ValueError
        with self.assertRaises(ValueError):
            rms_norm(x, weight, eps=1e-5, kernel=HammerKernel.TRITON_CC, silu=True)

        # TRITON_CC with SILU=False should work
        try:
            rms_norm(x, weight, eps=1e-5, kernel=HammerKernel.TRITON_CC, silu=False)
        except ValueError:
            self.fail("TRITON_CC with SILU=False should not raise ValueError")


if __name__ == "__main__":
    unittest.main()
