# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# pyre-strict

#!/usr/bin/env python3


import torch
from generative_recommenders.common import next_power_of_2, should_trigger_eager_impl
from generative_recommenders.ops.pytorch.pt_hstu_linear import pytorch_norm_mul_dropout
from generative_recommenders.ops.triton.triton_hstu_linear import _ln_mul_dropout_fwd
from generative_recommenders.ops.triton_aot.types import triton_aot

_ln_mul_dropout_fwd = triton_aot(
    annotations={
        "D": ("i32", 16),
        "stride_x": ("i32", 16),
        "stride_u": ("i32", 16),
        "stride_y": ("i32", 16),
    },
    # pyrefly: ignore [bad-argument-type]
)(_ln_mul_dropout_fwd)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_layer_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_ux: bool,
    mul_u_activation_type: str,
) -> torch.Tensor:
    assert x.dim() == 2
    if x.stride(1) != 1:
        x = x.contiguous()
    N, D = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D

    if concat_ux:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty_like(x)
    if N == 0:
        return y
    mean = x.new_empty((N,))
    rstd = x.new_empty((N,))

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D = min(MAX_FUSED_SIZE, next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    seed = 0
    # num_warps = min(max(BLOCK_D // 256, 1), 8)
    grid = (N,)
    # pyrefly: ignore [not-callable]
    _ln_mul_dropout_fwd[grid](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        D,
        eps,
        seed,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        # pyrefly: ignore [bad-argument-type]
        SILU_U=silu_u,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_D=BLOCK_D,
        # pyrefly: ignore [bad-argument-type]
        TRAINING=training,
        # pyrefly: ignore [bad-argument-type]
        CONCAT_U=concat_ux,
        # pyrefly: ignore [bad-argument-type]
        CONCAT_X=concat_ux,
        # pyrefly: ignore [bad-argument-type]
        MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
        # pyrefly: ignore [bad-argument-type]
        FAST_DROPOUT=False,
    )
    return y


@torch.fx.wrap
# "aot_triton_kernel_wrapper_" is a pre-defined prefix for
# AOT-T triton kernel wrapper functions. This is required for
# AOT-T backend to recognize and trace correctly for ops transformation.
def aot_triton_kernel_wrapper_layer_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_ux: bool,
    mul_u_activation_type: str,
) -> torch.Tensor:
    if should_trigger_eager_impl():
        return pytorch_norm_mul_dropout(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=dropout_ratio,
            training=training,
            silu_u=silu_u,
            concat_u=concat_ux,
            concat_x=concat_ux,
            mul_u_activation_type=mul_u_activation_type,
            group_norm=False,
        )
    else:
        return _triton_aot_layer_norm_mul_dropout(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=dropout_ratio,
            training=training,
            silu_u=silu_u,
            concat_ux=concat_ux,
            mul_u_activation_type=mul_u_activation_type,
        )
