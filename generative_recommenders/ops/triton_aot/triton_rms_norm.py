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
from generative_recommenders.common import (
    cdiv,
    next_power_of_2,
    should_trigger_eager_impl,
    switch_to_contiguous_if_needed,
)
from generative_recommenders.ops.pytorch.pt_layer_norm import pytorch_rms_norm
from generative_recommenders.ops.triton.triton_layer_norm import _weighted_rms_norm_fwd
from generative_recommenders.ops.triton_aot.types import triton_aot

_weighted_rms_norm_fwd = triton_aot(
    annotations={
        "N": "i32",
        "D": ("i32", 16),
        "stride_x": ("i32", 16),
        "stride_y": ("i32", 16),
    },
)(_weighted_rms_norm_fwd)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    silu: bool,
) -> torch.Tensor:
    """Internal AOTT kernel function for RMS norm."""
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape

    assert weight.dim() == 1
    assert weight.numel() == D

    y = torch.empty_like(x)
    rstd = torch.empty(N, dtype=torch.float32, device=x.device)

    BLOCK_D = next_power_of_2(D)

    grid = lambda meta: (  # noqa E731
        cdiv(N, meta["BLOCK_N"]),
    )
    # pyrefly: ignore [not-callable]
    _weighted_rms_norm_fwd[grid](
        x,
        y,
        weight,
        rstd,
        N,
        D,
        eps,
        stride_x=x.stride(0),
        stride_y=y.stride(0),
        SILU=silu,
        BLOCK_D=BLOCK_D,
    )

    return y


def _pytorch_rms_norm_fallback(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    silu: bool,
) -> torch.Tensor:
    """PyTorch fallback for RMS norm in eager mode."""

    return pytorch_rms_norm(x, [x.shape[-1]], weight, eps, silu)


@torch.fx.wrap
def aot_triton_kernel_wrapper_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    silu: bool,
) -> torch.Tensor:
    """AOT-T wrapper for RMS norm.

    Routes between PyTorch fallback (for tracing/serialization) and AOTT kernel path.
    """
    if should_trigger_eager_impl():
        return _pytorch_rms_norm_fallback(x, weight, eps, silu)
    else:
        return _triton_aot_rms_norm(x, weight, eps, silu)
