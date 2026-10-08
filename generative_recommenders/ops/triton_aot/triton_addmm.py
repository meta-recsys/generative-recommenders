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
# pyre-ignore-all-errors[2]: Triton has its own type system on func's input

#!/usr/bin/env python3


from typing import Any, List, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import (
    BACKEND_ALLOW_TF32,
    cdiv,
    should_trigger_eager_impl,
)
from generative_recommenders.ops.triton_aot.types import triton_aot


def get_mm_configs() -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 256,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 256,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


@triton_aot(
    annotations={
        "M": "i32",
        "N": ("i32", 16),
        "K": ("i32", 16),
        "stride_xm": ("i32", 16),
        "stride_xk": ("i32", 1),
        "stride_wk": ("i32", 16),
        "stride_wn": ("i32", 1),
        "stride_ym": ("i32", 16),
        "stride_yn": ("i32", 1),
        "stride_zm": ("i32", 16),
        "stride_zn": ("i32", 1),
    },
)
# pyre-ignore[56]: Pyre cannot infer triton.autotune decorator type
@triton.autotune(
    configs=get_mm_configs(),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
) -> None:
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    allow_tf32: bool = BACKEND_ALLOW_TF32,
) -> torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    grid = lambda meta: (  # noqa E731
        cdiv(M, meta["BLOCK_M"]),
        cdiv(N, meta["BLOCK_N"]),
    )

    _addmm_fwd[grid](
        x,
        w,
        y,
        z,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0) if not is_y_1d else 0,
        y.stride(1) if not is_y_1d else y.stride(0),
        z.stride(0),
        z.stride(1),
        ALLOW_TF32=allow_tf32,
        BROADCAST_Y=is_y_1d,
    )
    return z


def _triton_aot_addmm_fwd_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return torch.addmm(y, x, w)


@torch.fx.wrap
def _triton_aot_addmm_fwd_maybe_eager(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    if torch.jit.is_scripting():
        # call eager
        return torch.addmm(y, x, w)
    else:
        return _triton_aot_addmm_fwd(x, w, y)


def triton_addmm_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    dz: torch.Tensor,
    is_y_1d: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())

    return dx, dw, dy


class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]: autograd.Function signature override
    def forward(
        ctx: Any,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = y.dim() == 1
        return _triton_aot_addmm_fwd(x, w, y)

    @staticmethod
    # pyre-ignore[14]: autograd.Function signature override
    def backward(
        ctx: Any, dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x, w) = ctx.saved_tensors
        return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)


def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)


@torch.fx.wrap
def aot_triton_kernel_wrapper_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    allow_tf32: bool = BACKEND_ALLOW_TF32,
) -> torch.Tensor:
    if should_trigger_eager_impl():
        return torch.addmm(input, mat1, mat2)
    else:
        return _triton_aot_addmm_fwd(mat1, mat2, input, allow_tf32=allow_tf32)
