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

"""Portable PyTorch reference operations for row-wise FP8 linear layers."""

from __future__ import annotations

import torch
from torch.autograd.function import FunctionCtx


class _FP8AddmmContext(FunctionCtx):
    input_shape: torch.Size
    saved_tensors: tuple[torch.Tensor, torch.Tensor]


@torch.fx.wrap
def pytorch_quantize_fp8_per_row(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a two-dimensional tensor to E4M3 with one scale per row."""
    if x.ndim != 2:
        raise ValueError("row-wise FP8 quantization requires a 2D tensor")
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    row_max = torch.amax(torch.abs(x.float()), dim=1)
    scale = torch.where(row_max > 0, row_max / fp8_max, torch.ones_like(row_max))
    x_fp8 = torch.clamp(x.float() / scale.unsqueeze(1), min=-fp8_max, max=fp8_max).to(
        fp8_dtype
    )
    return x_fp8, scale


@torch.fx.wrap
def pytorch_fp8_scaled_mm(
    input: torch.Tensor,
    mat1_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat2_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """Multiply row-wise quantized inputs and weights, then add a bias."""
    if mat1_fp8.ndim != 2 or mat2_fp8.ndim != 2:
        raise ValueError("FP8 inputs must be two-dimensional")
    if mat1_fp8.shape[1] != mat2_fp8.shape[1]:
        raise ValueError("FP8 inputs must have the same reduction dimension")
    if mat1_scale.shape != (mat1_fp8.shape[0],):
        raise ValueError("mat1_scale must contain one value per mat1 row")
    if mat2_scale.shape != (mat2_fp8.shape[0],):
        raise ValueError("mat2_scale must contain one transposed mat2 row")
    output_shape = (mat1_fp8.shape[0], mat2_fp8.shape[0])
    if input.shape not in ((output_shape[1],), output_shape):
        raise ValueError("input must be a vector or matrix broadcastable to the output")

    if mat1_fp8.device.type == "cuda":
        fused_bias = input.to(torch.bfloat16) if input.ndim == 1 else None
        output = torch._scaled_mm(
            mat1_fp8,
            mat2_fp8.t(),
            scale_a=mat1_scale.unsqueeze(1),
            scale_b=mat2_scale.unsqueeze(0),
            bias=fused_bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=use_fast_accum,
        ).to(out_dtype)
        if fused_bias is not None:
            return output
    else:
        output = torch.mm(mat1_fp8.float(), mat2_fp8.float().t())
        output = output * mat1_scale.unsqueeze(1) * mat2_scale.unsqueeze(0)
        output = output.to(out_dtype)
    return output + input.to(out_dtype)


class _PytorchFP8Addmm(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: _FP8AddmmContext,
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        mat1_fp8: torch.Tensor,
        mat2_fp8: torch.Tensor,
        mat1_scale: torch.Tensor,
        mat2_scale: torch.Tensor,
        out_dtype: torch.dtype,
        use_fast_accum: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(mat1, mat2)
        ctx.input_shape = input.shape
        return pytorch_fp8_scaled_mm(
            input=input,
            mat1_fp8=mat1_fp8,
            mat2_fp8=mat2_fp8,
            mat1_scale=mat1_scale,
            mat2_scale=mat2_scale,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: _FP8AddmmContext,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        mat1, mat2 = ctx.saved_tensors
        grad_input = grad_output.sum_to_size(ctx.input_shape)
        grad_mat1 = torch.mm(grad_output, mat2.t())
        grad_mat2 = torch.mm(mat1.t(), grad_output)
        return grad_input, grad_mat1, grad_mat2, None, None, None, None, None, None


def pytorch_fp8_rowwise_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor | None,
    mat2: torch.Tensor | None,
    mat1_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat2_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = True,
    is_inference: bool = False,
) -> torch.Tensor:
    """Run row-wise FP8 addmm with a full-precision training backward."""
    if is_inference:
        return pytorch_fp8_scaled_mm(
            input=input,
            mat1_fp8=mat1_fp8,
            mat2_fp8=mat2_fp8,
            mat1_scale=mat1_scale,
            mat2_scale=mat2_scale,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    if mat1 is None or mat2 is None:
        raise ValueError("training requires the unquantized mat1 and mat2 tensors")
    return _PytorchFP8Addmm.apply(
        input,
        mat1,
        mat2,
        mat1_fp8,
        mat2_fp8,
        mat1_scale,
        mat2_scale,
        out_dtype,
        use_fast_accum,
    )
