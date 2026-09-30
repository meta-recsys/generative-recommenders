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

from __future__ import annotations

import unittest

import torch
from generative_recommenders.ops.fp8 import fp8_rowwise_addmm, quantize_fp8_per_row
from generative_recommenders.ops.hstu_compute import (
    hstu_compute_output,
    hstu_compute_uqvk,
)


class FP8Test(unittest.TestCase):
    def test_quantize_fp8_per_row(self) -> None:
        x = torch.tensor([[0.0, 0.0], [-2.0, 1.0], [0.25, -0.5]])

        x_fp8, scale = quantize_fp8_per_row(x)

        self.assertEqual(x_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (3,))
        torch.testing.assert_close(
            x_fp8.float() * scale.unsqueeze(1), x, atol=0.02, rtol=0.02
        )

    def test_cpu_inference_fallback(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(8, 16)
        w = torch.randn(16, 12)
        input = torch.randn(8, 12)

        actual = fp8_rowwise_addmm(
            input=input,
            mat1=x,
            mat2=w,
            out_dtype=torch.float32,
            is_inference=True,
        )
        expected = torch.addmm(input, x, w)

        torch.testing.assert_close(actual, expected, atol=0.35, rtol=0.12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_inference_and_prequantized_inputs(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16) * 0.1
        w = torch.randn(64, 48, device=device, dtype=torch.bfloat16) * 0.1
        bias = torch.randn(48, device=device, dtype=torch.bfloat16) * 0.1
        x_fp8, x_scale = quantize_fp8_per_row(x)
        w_fp8, w_scale = quantize_fp8_per_row(w.t().contiguous())

        actual = fp8_rowwise_addmm(
            input=bias,
            mat1_fp8=x_fp8,
            mat1_scale=x_scale,
            mat2_fp8=w_fp8,
            mat2_scale=w_scale,
            is_inference=True,
        )
        end_to_end = fp8_rowwise_addmm(
            input=bias,
            mat1=x,
            mat2=w,
            is_inference=True,
        )
        expected = torch.addmm(bias, x, w)

        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.12)
        torch.testing.assert_close(end_to_end, actual)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_training_backward(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(64, 48, device=device, dtype=torch.bfloat16, requires_grad=True)
        bias = torch.randn(48, device=device, dtype=torch.bfloat16, requires_grad=True)
        grad_output = torch.randn(32, 48, device=device, dtype=torch.bfloat16)

        output = fp8_rowwise_addmm(input=bias, mat1=x, mat2=w)
        output.backward(grad_output)

        torch.testing.assert_close(x.grad, torch.mm(grad_output, w.detach().t()))
        torch.testing.assert_close(w.grad, torch.mm(x.detach().t(), grad_output))
        torch.testing.assert_close(bias.grad, grad_output.sum(dim=0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_hstu_input_projection_can_enable_fp8(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        num_tokens = 32
        model_dim = 64
        num_heads = 2
        head_dim = 16
        projection_dim = 4 * num_heads * head_dim
        x = torch.randn(num_tokens, model_dim, device=device, dtype=torch.bfloat16)
        norm_weight = torch.ones(model_dim, device=device, dtype=torch.bfloat16)
        norm_bias = torch.zeros(model_dim, device=device, dtype=torch.bfloat16)
        uvqk_weight = (
            torch.randn(model_dim, projection_dim, device=device, dtype=torch.bfloat16)
            * 0.1
        )
        uvqk_bias = (
            torch.randn(projection_dim, device=device, dtype=torch.bfloat16) * 0.1
        )

        reference_uqvk = hstu_compute_uqvk(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=1e-6,
            num_heads=num_heads,
            attn_dim=head_dim,
            hidden_dim=head_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
        )
        fp8_uqvk = hstu_compute_uqvk(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=1e-6,
            num_heads=num_heads,
            attn_dim=head_dim,
            hidden_dim=head_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            fp8_in_addmm_fwd=True,
        )
        for actual, expected in zip(fp8_uqvk, reference_uqvk):
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.15)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_hstu_output_projection_can_enable_fp8(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        num_tokens = 32
        model_dim = 64
        num_heads = 2
        attn = torch.randn(num_tokens, model_dim, device=device, dtype=torch.bfloat16)
        u = torch.randn_like(attn)
        x = torch.randn_like(attn)
        norm_weight = torch.ones(model_dim, device=device, dtype=torch.bfloat16)
        norm_bias = torch.zeros(model_dim, device=device, dtype=torch.bfloat16)
        output_weight = (
            torch.randn(3 * model_dim, model_dim, device=device, dtype=torch.bfloat16)
            * 0.1
        )

        def compute(fp8: bool) -> torch.Tensor:
            return hstu_compute_output(
                attn=attn,
                u=u,
                x=x,
                norm_weight=norm_weight,
                norm_bias=norm_bias,
                norm_eps=1e-6,
                output_weight=output_weight,
                num_heads=num_heads,
                linear_dim=model_dim // num_heads,
                dropout_ratio=0.0,
                training=False,
                concat_u=True,
                concat_x=True,
                mul_u_activation_type="none",
                group_norm=False,
                recompute_y_in_backward=False,
                fp8_in_addmm_fwd=fp8,
            )

        reference_output = compute(False)
        fp8_output = compute(True)
        torch.testing.assert_close(fp8_output, reference_output, atol=0.2, rtol=0.05)
