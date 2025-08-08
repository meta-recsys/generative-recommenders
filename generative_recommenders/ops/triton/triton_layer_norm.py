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


from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from generative_recommenders.common import (
    switch_to_contiguous_if_needed,
    triton_autotune,
)


@triton.jit
def _layer_norm_fwd(
    X,
    Y,
    Mean,
    Rstd,
    D,
    eps,
    stride_x,
    stride_y,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    if ENABLE_TMA:
        x_desc = tl.make_tensor_descriptor(
            X,
            shape=[D],
            # pyre-ignore[6]: Triton tensor definition doesn't allow constants
            strides=[1],
            block_shape=[BLOCK_D],
        )
        y_desc = tl.make_tensor_descriptor(
            Y,
            shape=[D],
            # pyre-ignore[6]: Triton tensor definition doesn't allow constants
            strides=[1],
            block_shape=[BLOCK_D],
        )
        # pyre-ignore[6]: Triton tensor definition doesn't allow constants
        x = x_desc.load([0])
    else:
        x = tl.load(X + cols, mask=cols < D, other=0.0)

    x = x.to(tl.float32)

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x, axis=0) / D
    else:
        mean = tl.load(Mean + row)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = tl.zeros([BLOCK_D], dtype=tl.float32)
        _var += x_mean * x_mean
        var = tl.sum(_var, axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
    else:
        rstd = tl.load(Rstd + row)

    # Normalize and apply linear transformation
    y = x_mean * rstd
    if ENABLE_TMA:
        # pyre-ignore[6]: Triton tensor definition doesn't allow constants
        y_desc.store([0], y)
    else:
        mask = cols < D
        # Write output
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _weighted_layer_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    eps,
    stride_x,
    stride_y,
    IS_SWISH: tl.constexpr,
    TRAINING: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPUTE_MEAN_AND_RSTD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    if ENABLE_TMA:
        x_desc = tl.make_tensor_descriptor(
            X,
            shape=[D],
            # pyre-ignore[6]: Triton tensor definition doesn't allow constants
            strides=[1],
            block_shape=[BLOCK_D],
        )
        y_desc = tl.make_tensor_descriptor(
            Y,
            shape=[D],
            # pyre-ignore[6]: Triton tensor definition doesn't allow constants
            strides=[1],
            block_shape=[BLOCK_D],
        )
        w_desc = tl.make_tensor_descriptor(
            W,
            shape=[D],
            # pyre-ignore[6]: Triton tensor definition doesn't allow constants
            strides=[1],
            block_shape=[BLOCK_D],
        )
        b_desc = tl.make_tensor_descriptor(
            B,
            shape=[D],
            # pyre-ignore[6]: Triton tensor definition doesn't allow constants
            strides=[1],
            block_shape=[BLOCK_D],
        )
        # pyre-ignore[6]: Triton tensor definition doesn't allow constants
        x = x_desc.load([0])
    else:
        x = tl.load(X + cols, mask=cols < D, other=0.0)

    x = x.to(tl.float32)

    if COMPUTE_MEAN_AND_RSTD:
        mean = tl.sum(x, axis=0) / D
    else:
        mean = tl.load(Mean + row)

    x_mean = tl.where(cols < D, x - mean, 0.0)
    if COMPUTE_MEAN_AND_RSTD:
        _var = tl.zeros([BLOCK_D], dtype=tl.float32)
        _var += x_mean * x_mean
        var = tl.sum(_var, axis=0) / D
        rstd = 1 / tl.sqrt(var + eps)
        if TRAINING:
            tl.store(Mean + row, mean)
            tl.store(Rstd + row, rstd)
    else:
        rstd = tl.load(Rstd + row)

    # Normalize and apply linear transformation
    y = x_mean * rstd
    if ENABLE_TMA:
        # pyre-ignore[6]: Triton tensor definition doesn't allow constants
        w = w_desc.load([0]).to(tl.float32)
        # pyre-ignore[6]: Triton tensor definition doesn't allow constants
        b = b_desc.load([0]).to(tl.float32)

        y = y * w + b
        if IS_SWISH:
            y = tl.sigmoid(y) * x
        # Write output
        # pyre-ignore[6]: Triton tensor definition doesn't allow constants
        y_desc.store([0], y)
    else:
        mask = cols < D
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        b = tl.load(B + cols, mask=mask).to(tl.float32)

        y = y * w + b
        if IS_SWISH:
            y = tl.sigmoid(y) * x
        # Write output
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _layer_norm_bwd_dx(
    DX,
    DY,
    X,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / D
    c2 = tl.sum(dy, axis=0) / D
    dx = (dy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)


@triton.jit
def _weighted_layer_norm_bwd_dx(
    DX,
    DY,
    DW,
    DB,
    X,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    IS_SWISH: tl.constexpr,
    N,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row = pid
    acc_dw = tl.zeros(
        [
            BLOCK_D,
        ],
        dtype=tl.float32,
    )
    acc_db = tl.zeros(
        [
            BLOCK_D,
        ],
        dtype=tl.float32,
    )

    for idx in range(rows_per_tile):
        x_ptrs = X + row.to(tl.int64) * stride_x
        dy_ptrs = DY + row.to(tl.int64) * stride_dy
        dx_ptrs = DX + row.to(tl.int64) * stride_dx

        # Load data to SRAM
        x = tl.load(x_ptrs + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(dy_ptrs + cols, mask=mask, other=0).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        sigmoid_layer_norm = None
        if IS_SWISH:
            b = tl.load(B + cols, mask=mask).to(tl.float32)
            sigmoid_layer_norm = tl.sigmoid(xhat * w + b)
            sigmoid_layer_norm = tl.where(mask, sigmoid_layer_norm, 0.0)
            x_ = wdy * x * sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            x_ = tl.where(mask, x_, 0.0)

            c1 = tl.sum(xhat * x_, axis=0) / D
            c2 = tl.sum(x_, axis=0) / D
            dx = (x_ - (xhat * c1 + c2)) * rstd
            dx = dy * sigmoid_layer_norm + dx
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / D
            c2 = tl.sum(wdy, axis=0) / D
            dx = (wdy - (xhat * c1 + c2)) * rstd

        # Write dx
        tl.store(dx_ptrs + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        if IS_SWISH:
            partial_dw = dy * x * xhat * sigmoid_layer_norm * (1 - sigmoid_layer_norm)
            partial_db = dy * x * sigmoid_layer_norm * (1 - sigmoid_layer_norm)
        else:
            partial_dw = dy * xhat
            partial_db = dy
        # Accumulate partial sums in shared memory
        acc_dw += partial_dw
        acc_db += partial_db
        row += tile_num

    # Store accumulated sums back to global memory
    dw_ptrs = DW + pid.to(tl.int64) * D + cols
    db_ptrs = DB + pid.to(tl.int64) * D + cols
    tl.store(dw_ptrs, acc_dw, mask=mask)
    tl.store(db_ptrs, acc_db, mask=mask)


def _get_bwd_dwdb_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [32, 64, 128, 256]:
        for num_warps in [8, 16] + ([] if torch.ops.hip else [32]):
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    db = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


def triton_weighted_layer_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    mean: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    enable_tma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    learnable = weight is not None
    if learnable:
        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

    y = torch.empty_like(x)
    compute_mean_and_rstd = mean is None or rstd is None
    if mean is None:
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    if rstd is None:
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    if N == 0:
        return y, mean, rstd, BLOCK_D, num_warps

    if enable_tma:
        # TMA descriptors require a global memory allocation
        def alloc_fn(size: int, alignment: int, stream: int | None):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    if learnable:
        # pyre-ignore[28]
        _weighted_layer_norm_fwd[(N,)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            IS_SWISH=False,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
            num_warps=num_warps,
            ENABLE_TMA=enable_tma,
        )
    else:
        # pyre-ignore[28]
        _layer_norm_fwd[(N,)](
            x,
            y,
            mean,
            rstd,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=compute_mean_and_rstd,
            num_warps=num_warps,
            ENABLE_TMA=enable_tma,
        )
    return y, mean, rstd, BLOCK_D, num_warps


def triton_weighted_layer_norm_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    mean: torch.Tensor,
    rstd: torch.Tensor,
    learnable: bool,
    eps: float,
    BLOCK_D: int,
    num_warps: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if learnable:
        assert weight is not None and bias is not None
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        tile_num = max(1, min(sms * 8, N // 4))
        _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias
        # pyre-ignore[28]
        _weighted_layer_norm_bwd_dx[(tile_num,)](
            dx,
            dy,
            _dweight,
            _dbias,
            x,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            eps,
            IS_SWISH=False,
            N=N,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](
            _dweight,
            _dbias,
            dweight,
            dbias,
            tile_num,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, dbias
    else:
        N, D = x.shape
        dx = torch.empty_like(x)
        if N == 0:
            return dx, None, None
        # pyre-ignore[28]
        _layer_norm_bwd_dx[(N,)](
            dx,
            dy,
            x,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            eps,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
        return dx, None, None


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
        enable_tma: bool = False,
    ) -> torch.Tensor:
        y, mean, rstd, BLOCK_D, num_warps = triton_weighted_layer_norm_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            enable_tma=enable_tma,
        )
        learnable = weight is not None
        if learnable:
            ctx.save_for_backward(x, weight, bias, mean, rstd)
        else:
            ctx.save_for_backward(x, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.learnable = learnable
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None]:
        if ctx.learnable:
            x, weight, bias, mean, rstd = ctx.saved_tensors
        else:
            x, mean, rstd = ctx.saved_tensors
            weight, bias = None, None
        dx, dweight, dbias = triton_weighted_layer_norm_bwd(
            dy=dy,
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            learnable=ctx.learnable,
            eps=ctx.eps,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
        )
        return dx, dweight, dbias, None


@triton.jit
def _weighted_rms_norm_fwd(
    X,
    Y,
    W,
    Rstd,
    D,
    eps,
    stride_x,
    stride_y,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)

    # Compute variance
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = y * w
    # Write output
    tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _weighted_rms_norm_bwd_dx(
    DX,
    DY,
    DW,
    X,
    W,
    Rstd,
    Lock,
    stride_dx,
    stride_dy,
    stride_x,
    D,
    eps,
    GROUP_N,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    X += row.to(tl.int64) * stride_x
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    rstd = tl.load(Rstd + row)

    # Compute dx
    xhat = x * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    wdy = w * dy

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / D
    dx = (wdy - (xhat * c1)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    Lock += lock_id
    Count = Lock + GROUP_N
    DW = DW + lock_id * D + cols
    # Accumulate partial sums for dw/db
    partial_dw = dy * xhat
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _rms_norm_bwd_dwdb(
    DW,
    FINAL_DW,
    N,
    D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    dw = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        # pyre-fixme[16]: `int` has no attribute `__getitem__`.
        mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        assert x.dim() == 2
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape
        assert weight.dim() == 1
        assert weight.numel() == D

        y = torch.empty_like(x)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_D = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
        if D > BLOCK_D:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        num_warps = min(max(BLOCK_D // 256, 1), 8)

        # pyre-ignore[28]
        _weighted_rms_norm_fwd[(N,)](
            x,
            y,
            weight,
            rstd,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, rstd)

        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x, weight, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        if D <= 1024:
            GROUP_N = 256 * 8
        elif D <= 4096:
            GROUP_N = 128 * 8
        elif D <= 8192:
            GROUP_N = 96 * 8
        else:
            GROUP_N = 64 * 8
        GROUP_N = N if GROUP_N > N else GROUP_N
        locks = torch.zeros(2 * GROUP_N, dtype=torch.int32, device=x.device)
        _dweight = torch.empty((GROUP_N, D), dtype=torch.float32, device=x.device)
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        # pyre-ignore[28]
        _weighted_rms_norm_bwd_dx[(N,)](
            dx,
            dy,
            _dweight,
            x,
            weight,
            rstd,
            locks,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            ctx.eps,
            GROUP_N=GROUP_N,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
        )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _rms_norm_bwd_dwdb[grid](
            _dweight,
            dweight,
            GROUP_N,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, None


class SwishLayerNormFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        enable_tma: bool = False,
    ) -> torch.Tensor:
        assert x.dim() == 2, f"x.dim() == {x.dim()}, expected 2"
        x = switch_to_contiguous_if_needed(x)
        N, D = x.shape

        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        assert weight.numel() == D
        assert bias.numel() == D

        y = torch.empty_like(x)
        mean = torch.empty((N,), dtype=torch.float32, device=x.device)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)

        BLOCK_D = triton.next_power_of_2(D)
        num_warps = min(max(BLOCK_D // 256, 1), 8)

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        if N == 0:
            return y

        if enable_tma:
            # TMA descriptors require a global memory allocation
            def alloc_fn(size: int, alignment: int, stream: int | None):
                return torch.empty(size, device="cuda", dtype=torch.int8)

            triton.set_allocator(alloc_fn)

        # pyre-ignore[28]
        _weighted_layer_norm_fwd[(N,)](
            x,
            y,
            weight,
            bias,
            mean,
            rstd,
            D,
            eps,
            x.stride(0),
            y.stride(0),
            IS_SWISH=True,
            TRAINING=True,
            BLOCK_D=BLOCK_D,
            COMPUTE_MEAN_AND_RSTD=True,
            num_warps=num_warps,
            ENABLE_TMA=enable_tma,
        )

        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None]:
        x, weight, bias, mean, rstd = ctx.saved_tensors
        N, D = x.shape
        dx = torch.empty_like(x)
        sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        tile_num = max(1, min(sms * 8, N // 4))
        _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
        dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
        dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)
        if N == 0:
            dweight.zero_()
            dbias.zero_()
            return dx, dweight, dbias, None
        # pyre-ignore[28]
        _weighted_layer_norm_bwd_dx[(tile_num,)](
            dx,
            dy,
            _dweight,
            _dbias,
            x,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            dy.stride(0),
            x.stride(0),
            D,
            ctx.eps,
            IS_SWISH=True,
            N=N,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
        )

        def grid(META):
            return (triton.cdiv(D, META["BLOCK_D"]),)

        blocks = triton.next_power_of_2(sms * 4)
        BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
        BLOCK_D = min(max(BLOCK_D, 4), 128)
        _layer_norm_bwd_dwdb[grid](
            _dweight,
            _dbias,
            dweight,
            dbias,
            tile_num,
            D,
            BLOCK_D=BLOCK_D,
        )

        return dx, dweight, dbias, None


@torch.fx.wrap
def triton_layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return LayerNormFunction.apply(x, weight, bias, eps)


@torch.fx.wrap
def triton_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return RMSNormFunction.apply(x, weight, eps)


@torch.fx.wrap
def triton_swish_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    return SwishLayerNormFunction.apply(x, weight, bias, eps)
