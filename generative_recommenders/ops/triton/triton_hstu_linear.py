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
from generative_recommenders.ops.triton.triton_addmm import maybe_triton_addmm_fwd


def _get_layer_norm_mul_dropout_fwd_multirow_configs() -> List[triton.Config]:
    """Generate autotune configs for multi-row LayerNorm multiplication with dropout kernels."""
    configs = []
    for BLOCK_N in [1, 2, 4, 8, 16]:
        for num_warps in [1, 2, 4]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


from generative_recommenders.ops.utils import is_sm100_plus

# @manual=//triton:triton
from triton.language.extra import libdevice

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef


COMPUTE_OUTPUT_LN_FAST_DROPOUT = False


def set_compute_output_ln_fast_dropout(value: bool) -> None:
    global COMPUTE_OUTPUT_LN_FAST_DROPOUT
    COMPUTE_OUTPUT_LN_FAST_DROPOUT = value


FUSE_OUTPUT_LN_RNG_BLACKWELL = False


# Only impact B200 training when CONCAT_UX is False
def set_fuse_output_ln_rng_blackwell(value: bool) -> None:
    global FUSE_OUTPUT_LN_RNG_BLACKWELL
    FUSE_OUTPUT_LN_RNG_BLACKWELL = value


@triton.jit
def rand3x(seed, offsets, n_rounds: tl.constexpr = 10):  # pyre-ignore [9]
    i1, i2, i3, _ = tl.randint4x(seed, offsets, n_rounds)
    u1 = tl.uint_to_uniform_float(i1)
    u2 = tl.uint_to_uniform_float(i2)
    u3 = tl.uint_to_uniform_float(i3)
    return u1, u2, u3


@triton.jit
def _generate_random_mask(
    MASK_BUFFER,
    N_MASK,
    dropout_ratio,
    seed,
    D: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # NOTE: This function appears to be incomplete/unused - kept for compatibility
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    random_offsets = pid * BLOCK_D + cols
    rand1, rand2, rand3, rand4 = tl.rand4x(seed, random_offsets)
    start_row = pid * 4
    MASK_BUFFER += start_row * STRIDE
    row_mask = start_row < N_MASK
    mask1 = rand1 > dropout_ratio
    tl.store(MASK_BUFFER + cols, mask1, mask=row_mask & col_mask)
    row_mask = (start_row + 1) < N_MASK
    mask2 = rand2 > dropout_ratio
    tl.store(MASK_BUFFER + STRIDE + cols, mask2, mask=row_mask & col_mask)
    row_mask = (start_row + 2) < N_MASK
    mask3 = rand3 > dropout_ratio
    tl.store(
        MASK_BUFFER + 2 * STRIDE + cols,
        mask3,
        mask=row_mask & col_mask,
    )
    row_mask = (start_row + 3) < N_MASK
    mask4 = rand4 > dropout_ratio
    tl.store(
        MASK_BUFFER + 3 * STRIDE + cols,
        mask4,
        mask=row_mask & col_mask,
    )


@triton_autotune(
    configs=_get_layer_norm_mul_dropout_fwd_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _ln_mul_dropout_fwd_rng(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    N,
    D,
    eps,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
):
    block_id = tl.program_id(0)
    start_row = block_id * BLOCK_N

    # Create block pointers for X, U, and Y
    X_block_ptr = tl.make_block_ptr(
        base=X,
        shape=(N, D),
        strides=(stride_x, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    U_block_ptr = tl.make_block_ptr(
        base=U,
        shape=(N, D),
        strides=(stride_u, 1),
        offsets=(start_row, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    # Load data blocks
    x_block = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    u_block = tl.load(U_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D
    rows = start_row + tl.arange(0, BLOCK_N)
    row_mask = rows < N

    mean = tl.sum(x_block, axis=1) / D
    tl.store(Mean + rows, mean, mask=row_mask)
    mean = tl.expand_dims(mean, 1)

    x_mean = x_block - mean
    x_mean = tl.where(row_mask[:, None] & col_mask[None, :], x_mean, 0.0)
    _var = x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd, mask=row_mask)
    rstd = tl.expand_dims(rstd, 1)

    y = x_mean * rstd
    w = tl.load(W + cols, mask=col_mask).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask).to(tl.float32)
    y = y * w[None, :] + b[None, :]

    if MUL_U_ACTIVATION_TYPE == "silu":
        # pyre-fixme[16]
        silu_u_activation_block = u_block * tl.sigmoid(u_block)
        y = y * silu_u_activation_block
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        # pyre-fixme[16]
        sigmoid_u_activation_block = tl.sigmoid(u_block)
        y = y * sigmoid_u_activation_block
    else:
        y = y * u_block

    if CONCAT_U and SILU_U:
        # pyre-fixme[16]
        u_block = u_block * tl.sigmoid(u_block)

    if TRAINING:
        row_offsets = start_row + tl.arange(0, BLOCK_N)
        col_offsets = tl.arange(0, BLOCK_D)
        if CONCAT_U and CONCAT_X:
            # Load precomputed random masks for u, x, y
            u_offsets = row_offsets[:, None] * stride_mask + col_offsets[None, :]
            x_offsets = (row_offsets[:, None] + N) * stride_mask + col_offsets[None, :]
            y_offsets = (row_offsets[:, None] + 2 * N) * stride_mask + col_offsets[
                None, :
            ]

            mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < D)

            u_keep = tl.load(RANDOM_MASK + u_offsets, mask=mask, other=True)
            x_keep = tl.load(RANDOM_MASK + x_offsets, mask=mask, other=True)
            y_keep = tl.load(RANDOM_MASK + y_offsets, mask=mask, other=True)

            u_block = tl.where(u_keep, u_block / (1.0 - dropout_ratio), 0.0)
            x_block = tl.where(x_keep, x_block / (1.0 - dropout_ratio), 0.0)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        elif CONCAT_U:
            # Load precomputed random mask for u, y
            u_offsets = row_offsets[:, None] * stride_mask + col_offsets[None, :]
            y_offsets = (row_offsets[:, None] + N) * stride_mask + col_offsets[None, :]
            mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < D)

            u_keep = tl.load(RANDOM_MASK + u_offsets, mask=mask, other=True)
            y_keep = tl.load(RANDOM_MASK + y_offsets, mask=mask, other=True)

            u_block = tl.where(u_keep, u_block / (1.0 - dropout_ratio), 0.0)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        elif CONCAT_X:
            # Load precomputed random mask for x, y
            x_offsets = row_offsets[:, None] * stride_mask + col_offsets[None, :]
            y_offsets = (row_offsets[:, None] + N) * stride_mask + col_offsets[None, :]
            mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < D)

            x_keep = tl.load(RANDOM_MASK + x_offsets, mask=mask, other=True)
            y_keep = tl.load(RANDOM_MASK + y_offsets, mask=mask, other=True)

            x_block = tl.where(x_keep, x_block / (1.0 - dropout_ratio), 0.0)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            # Load precomputed random mask for y
            y_offsets = row_offsets[:, None] * stride_mask + col_offsets[None, :]
            mask = (row_offsets[:, None] < N) & (col_offsets[None, :] < D)

            y_keep = tl.load(RANDOM_MASK + y_offsets, mask=mask, other=True)
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    if CONCAT_U and CONCAT_X:
        Y_block_ptr_u = tl.make_block_ptr(
            base=Y,
            shape=(N, 3 * D),
            strides=(stride_y, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        Y_block_ptr_x = tl.make_block_ptr(
            base=Y,
            shape=(N, 3 * D),
            strides=(stride_y, 1),
            offsets=(start_row, D),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        Y_block_ptr_y = tl.make_block_ptr(
            base=Y,
            shape=(N, 3 * D),
            strides=(stride_y, 1),
            offsets=(start_row, 2 * D),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        tl.store(Y_block_ptr_u, u_block.to(Y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(Y_block_ptr_x, x_block.to(Y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(Y_block_ptr_y, y.to(Y.dtype.element_ty), boundary_check=(0, 1))
    elif CONCAT_U:
        Y_block_ptr_u = tl.make_block_ptr(
            base=Y,
            shape=(N, 2 * D),
            strides=(stride_y, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        Y_block_ptr_y = tl.make_block_ptr(
            base=Y,
            shape=(N, 2 * D),
            strides=(stride_y, 1),
            offsets=(start_row, D),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        tl.store(Y_block_ptr_u, u_block.to(Y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(Y_block_ptr_y, y.to(Y.dtype.element_ty), boundary_check=(0, 1))
    elif CONCAT_X:
        Y_block_ptr_x = tl.make_block_ptr(
            base=Y,
            shape=(N, 2 * D),
            strides=(stride_y, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        Y_block_ptr_y = tl.make_block_ptr(
            base=Y,
            shape=(N, 2 * D),
            strides=(stride_y, 1),
            offsets=(start_row, D),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        tl.store(Y_block_ptr_x, x_block.to(Y.dtype.element_ty), boundary_check=(0, 1))
        tl.store(Y_block_ptr_y, y.to(Y.dtype.element_ty), boundary_check=(0, 1))
    else:
        Y_block_ptr = tl.make_block_ptr(
            base=Y,
            shape=(N, D),
            strides=(stride_y, 1),
            offsets=(start_row, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )

        tl.store(Y_block_ptr, y.to(Y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _ln_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    FAST_DROPOUT: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)

    # Compute mean
    mean = 0.0
    x = tl.load(X + cols, mask=cols < D, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D

    # Compute variance
    _var = tl.zeros([BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(cols < D, x - mean, 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=0) / D
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    mask = cols < D
    y = x_mean * rstd
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = y * w + b
    u = tl.load(U + cols, mask=cols < D, other=0.0).to(tl.float32)
    sigmoid_u = tl.sigmoid(u)
    silu_u = u * sigmoid_u

    if MUL_U_ACTIVATION_TYPE == "silu":
        y = y * silu_u
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        y = y * sigmoid_u
    else:
        y = y * u

    if CONCAT_U and SILU_U:
        u = silu_u

    if TRAINING:
        random_offsets = 3 * row * BLOCK_D + cols
        if CONCAT_U and CONCAT_X:
            # apply dropout on u
            if FAST_DROPOUT:
                random_u, random_x, random_y = rand3x(seed, random_offsets)
            else:
                random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            if not FAST_DROPOUT:
                random_x = tl.rand(seed, random_offsets + D)
            x_keep = random_x > dropout_ratio  # pyre-ignore [61]
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + 2 * D)
            y_keep = random_y > dropout_ratio  # pyre-ignore [61]
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        elif CONCAT_U:
            # apply dropout on u
            if FAST_DROPOUT:
                random_u, random_y, _ = rand3x(seed, random_offsets)
            else:
                random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + D)
            y_keep = random_y > dropout_ratio  # pyre-ignore [61]
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        elif CONCAT_X:
            # apply dropout on x
            if FAST_DROPOUT:
                random_x, random_y, _ = rand3x(seed, random_offsets)
            else:
                random_x = tl.rand(seed, random_offsets)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            if not FAST_DROPOUT:
                random_y = tl.rand(seed, random_offsets + D)
            y_keep = random_y > dropout_ratio  # pyre-ignore [61]
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_U and CONCAT_X:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
    elif CONCAT_U:
        tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
    elif CONCAT_X:
        tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du_rng(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    RANDOM_MASK,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    stride_mask,
    D,
    eps,
    dropout_ratio,
    N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    num_random = 1
    if CONCAT_U:
        num_random += 1
    if CONCAT_X:
        num_random += 1
    RANDOM_MASK += row.to(tl.int64) * stride_mask * num_random

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    for _ in range(0, rows_per_tile):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_U and CONCAT_X:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            if CONCAT_U and CONCAT_X:
                # Load dropout masks for u, x, y from pre-generated mask tensor
                du_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dx_keep = tl.load(
                    RANDOM_MASK + stride_mask + cols, mask=mask, other=True
                )
                dy_keep = tl.load(
                    RANDOM_MASK + 2 * stride_mask + cols, mask=mask, other=True
                )
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            elif CONCAT_U:
                du_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dy_keep = tl.load(
                    RANDOM_MASK + stride_mask + cols, mask=mask, other=True
                )
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            elif CONCAT_X:
                dx_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dy_keep = tl.load(
                    RANDOM_MASK + stride_mask + cols, mask=mask, other=True
                )
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                # Load dropout mask directly instead of generating random numbers
                dy_keep = tl.load(RANDOM_MASK + cols, mask=mask, other=True)
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du_y = dy * ln
        mul_u = u
        sig_u = tl.sigmoid(u)
        silu_u = u * sig_u

        if MUL_U_ACTIVATION_TYPE == "silu":
            mul_u = silu_u
            du_y = dy * ln * (sig_u + silu_u * (1.0 - sig_u))
            dy = dy * silu_u
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            mul_u = sig_u
            du_y = dy * ln * sig_u * (1.0 - sig_u)
            dy = dy * sig_u
        else:
            dy = dy * u

        du_u = du
        if CONCAT_U and SILU_U:
            du_u *= sig_u + silu_u * (1.0 - sig_u)
            u = silu_u

        du = du_y + du_u

        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)

        wdy = w * dy
        if COMPUTE_Y:
            y = ln * mul_u
            if TRAINING:
                if CONCAT_U:
                    u = tl.where(
                        du_keep,  # pyre-ignore [61]
                        u / (1.0 - dropout_ratio),
                        0.0,
                    )
                if CONCAT_X:
                    x = tl.where(
                        dx_keep,  # pyre-ignore [61]
                        x / (1.0 - dropout_ratio),
                        0.0,
                    )
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
            if CONCAT_U and CONCAT_X:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_U:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_X:
                tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw += dy * xhat
        partial_db += dy
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        RANDOM_MASK += tile_num.to(tl.int64) * stride_mask * num_random
        row += tile_num
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


@triton.jit
def _ln_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    eps,
    seed,
    dropout_ratio,
    N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
    FAST_DROPOUT: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    for _idx in range(0, rows_per_tile):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_U and CONCAT_X:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        elif CONCAT_U:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
        elif CONCAT_X:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if TRAINING:
            random_offsets = 3 * row * BLOCK_D + cols
            if CONCAT_U and CONCAT_X:
                # apply dropout on du
                if FAST_DROPOUT:
                    random_du, random_dx, random_dy = rand3x(seed, random_offsets)
                else:
                    random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dx
                if not FAST_DROPOUT:
                    random_dx = tl.rand(seed, random_offsets + D)
                dx_keep = random_dx > dropout_ratio  # pyre-ignore [61]
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + 2 * D)
                dy_keep = random_dy > dropout_ratio  # pyre-ignore [61]
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            elif CONCAT_U:
                # apply dropout on du
                if FAST_DROPOUT:
                    random_du, _, random_dy = rand3x(seed, random_offsets)
                else:
                    random_du = tl.rand(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + D)
                dy_keep = random_dy > dropout_ratio  # pyre-ignore [61]
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            elif CONCAT_X:
                # apply dropout on dx
                if FAST_DROPOUT:
                    _, random_dx, random_dy = rand3x(seed, random_offsets)
                else:
                    random_dx = tl.rand(seed, random_offsets)
                dx_keep = random_dx > dropout_ratio  # pyre-ignore [61]
                dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
                # apply dropout on dy
                if not FAST_DROPOUT:
                    random_dy = tl.rand(seed, random_offsets + D)
                dy_keep = random_dy > dropout_ratio  # pyre-ignore [61]
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
            else:
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                # write-back
                dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du_y = dy * ln
        mul_u = u
        sig_u = tl.sigmoid(u)

        if MUL_U_ACTIVATION_TYPE == "silu":
            mul_u = u * sig_u
            du_y = dy * ln * (sig_u + u * sig_u * (1.0 - sig_u))
            dy = dy * u * sig_u
        elif MUL_U_ACTIVATION_TYPE == "sigmoid":
            mul_u = sig_u
            du_y = dy * ln * (sig_u * (1.0 - sig_u))
            dy = dy * sig_u
        else:
            dy = dy * u

        du_u = du
        if CONCAT_U:
            if SILU_U:
                du_u *= sig_u + u * sig_u * (1.0 - sig_u)
                u = u * sig_u

        du = du_y + du_u

        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        wdy = w * dy
        if COMPUTE_Y:
            y = ln * mul_u
            if TRAINING:
                if CONCAT_U:
                    u = tl.where(
                        du_keep,  # pyre-ignore [61]
                        u / (1.0 - dropout_ratio),
                        0.0,
                    )
                if CONCAT_X:
                    x = tl.where(
                        dx_keep,  # pyre-ignore [61]
                        x / (1.0 - dropout_ratio),
                        0.0,
                    )
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
            if CONCAT_U and CONCAT_X:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_U:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            elif CONCAT_X:
                tl.store(Y + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        c1 = tl.sum(xhat * wdy, axis=0) / D
        c2 = tl.sum(wdy, axis=0) / D
        dx += (wdy - (xhat * c1 + c2)) * rstd
        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw += dy * xhat
        partial_db += dy
        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        row += tile_num
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


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
def _ln_mul_dropout_bwd_dwdb(
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


def triton_layer_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
]:  # y, mean, rstd, BLOCK_D, num_warps, seed
    assert x.dim() == 2
    x = switch_to_contiguous_if_needed(x)
    N, D = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == D
    assert bias.numel() == D

    if concat_u and concat_x:
        y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
    elif concat_u:
        y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
    elif concat_x:
        y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty_like(x)
    mean = torch.empty((N,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BLOCK_D:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D // 256, 1), 8)
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # Benchmark shows separating RNG from ln_mul_dropout kernel only benefits on
    # blackwell when CONCAT_UX is enabled. (fused RNG kernel can benefit from rand3x fast
    # dropout)
    if (
        not FUSE_OUTPUT_LN_RNG_BLACKWELL
        and is_sm100_plus()
        and (not concat_u and not concat_x)
        and training
    ):
        random_mask = torch.empty([N, D], dtype=torch.bool, device=x.device)

        _generate_random_mask[(triton.cdiv(N, 4),)](
            random_mask,
            N,
            dropout_ratio,
            seed,
            D,  # pyre-ignore [6]
            random_mask.stride(0),  # pyre-ignore [6]
            BLOCK_D,  # pyre-ignore [6]
        )

        def grid(META):
            return (triton.cdiv(N, META["BLOCK_N"]),)

        # pyre-ignore[28]
        _ln_mul_dropout_fwd_rng[grid](
            x,
            u,
            y,
            weight,
            bias,
            mean,
            rstd,
            random_mask,
            N,
            D,
            eps,
            dropout_ratio,
            x.stride(0),
            u.stride(0),
            y.stride(0),
            random_mask.stride(0),
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
        )

    else:
        # pyre-ignore[28]
        _ln_mul_dropout_fwd[(N,)](
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
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            FAST_DROPOUT=COMPUTE_OUTPUT_LN_FAST_DROPOUT,
            num_warps=num_warps,
        )
    return y, mean, rstd, BLOCK_D, num_warps, seed  # pyre-ignore [7]


def triton_layer_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, D = x.shape
    if compute_y:
        if concat_u and concat_x:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        elif concat_u:
            y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
        elif concat_x:
            y = torch.empty((N, 2 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)

    if (
        not FUSE_OUTPUT_LN_RNG_BLACKWELL
        and is_sm100_plus()
        and (not concat_u and not concat_x)
        and training
    ):
        random_mask = torch.empty([N, D], dtype=torch.bool, device=x.device)

        _generate_random_mask[(triton.cdiv(N, 4),)](
            random_mask,
            N,
            dropout_ratio,
            seed,
            D,  # pyre-ignore [6]
            random_mask.stride(0),  # pyre-ignore [6]
            BLOCK_D,  # pyre-ignore [6]
        )

        # pyre-ignore[28]
        _ln_mul_dropout_bwd_dx_du_rng[(tile_num,)](
            dx,
            du,
            dy,
            _dweight,
            _dbias,
            x,
            u,
            y,
            weight,
            bias,
            mean,
            rstd,
            random_mask,
            dx.stride(0),
            du.stride(0),
            dy.stride(0),
            x.stride(0),
            u.stride(0),
            y.stride(0) if compute_y else 0,  # pyre-ignore [16]
            random_mask.stride(0),
            D,
            eps,
            dropout_ratio,
            N=N,
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            COMPUTE_Y=compute_y,
            num_warps=num_warps,
        )

    else:
        # pyre-ignore[28]
        _ln_mul_dropout_bwd_dx_du[(tile_num,)](
            dx,
            du,
            dy,
            _dweight,
            _dbias,
            x,
            u,
            y,
            weight,
            bias,
            mean,
            rstd,
            dx.stride(0),
            du.stride(0),
            dy.stride(0),
            x.stride(0),
            u.stride(0),
            y.stride(0) if compute_y else 0,  # pyre-ignore [16]
            D,
            eps,
            seed,
            dropout_ratio,
            N=N,
            SILU_U=silu_u,
            BLOCK_D=BLOCK_D,
            TRAINING=training,
            CONCAT_U=concat_u,
            CONCAT_X=concat_x,
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,
            COMPUTE_Y=compute_y,
            FAST_DROPOUT=COMPUTE_OUTPUT_LN_FAST_DROPOUT,
            num_warps=num_warps,
        )

    def grid(META):
        return (triton.cdiv(D, META["BLOCK_D"]),)

    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D = min(max(BLOCK_D, 4), 128)
    _ln_mul_dropout_bwd_dwdb[grid](
        _dweight,
        _dbias,
        dweight,
        dbias,
        tile_num,
        D,
        BLOCK_D=BLOCK_D,
    )
    return dx, du, dweight, dbias, y


class LayerNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_ux: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if dropout_ratio == 0.0:
            # skip dropout computation if dropout ratio is 0
            training = False
        # skipping supporting concat_u and concat_x separately here because seems like this code path is only used in v1 of hstu_linear
        concat_u, concat_x = concat_ux, concat_ux
        y, mean, rstd, BLOCK_D, num_warps, seed = triton_layer_norm_mul_dropout_fwd(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=dropout_ratio,
            training=training,
            silu_u=silu_u,
            concat_u=concat_u,
            concat_x=concat_x,
            seed=seed,
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_ux = concat_ux
        ctx.silu_u = silu_u
        ctx.dropout_ratio = dropout_ratio
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
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
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = triton_layer_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            silu_u=ctx.silu_u,
            concat_u=ctx.concat_ux,
            concat_x=ctx.concat_ux,
            compute_y=False,
        )
        return dx, du, dweight, dbias, None, None, None, None, None, None


@triton.jit
def _group_norm_mul_dropout_fwd(
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    Y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    offsets = heads[:, None] * D + cols[None, :]
    mask_h = heads < Heads
    mask_c = cols < D
    mask = mask_c[None, :] & mask_h[:, None]

    # Compute mean
    mean = 0.0
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / D
    mean = tl.ravel(mean)

    # Compute variance
    _var = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(mask, x - mean[:, None], 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=1) / D
    var = tl.ravel(var)
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row * Heads + heads, mean, mask=mask_h)
    tl.store(Rstd + row * Heads + heads, rstd, mask=mask_h)

    # Normalize and apply linear transformation
    y = x_mean * rstd[:, None]  # pyre-ignore [16]
    w = tl.load(W + heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + heads, mask=mask_h).to(tl.float32)
    y = y * w[:, None] + b[:, None]
    u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
    if SILU_U:
        u = u * tl.sigmoid(u)
    y = y * u

    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on u
            random_u = tl.rand(seed, random_offsets)
            u_keep = random_u > dropout_ratio
            u = tl.where(u_keep, u / (1.0 - dropout_ratio), 0.0)
            # apply dropout on x
            random_x = tl.rand(seed, random_offsets + Heads * D)
            x_keep = random_x > dropout_ratio
            x = tl.where(x_keep, x / (1.0 - dropout_ratio), 0.0)
            # apply dropout on y
            random_y = tl.rand(seed, random_offsets + 2 * Heads * D)
            y_keep = random_y > dropout_ratio
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            y_keep = random > dropout_ratio
            # write-back
            y = tl.where(y_keep, y / (1.0 - dropout_ratio), 0.0)

    # Write output
    if CONCAT_UX:
        tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
        tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
    else:
        tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _group_norm_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D,
    Heads,
    eps,
    seed,
    dropout_ratio,
    SILU_U: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    off_heads = tl.arange(0, BLOCK_H)
    mask_c = cols < D
    mask_h = off_heads < Heads
    mask = mask_c[None, :] & mask_h[:, None]
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    offsets = off_heads[:, None] * D + cols[None, :]

    # Load data to SRAM
    x = tl.load(X + offsets, mask=mask, other=0).to(tl.float32)
    if CONCAT_UX:
        du = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
        dx = tl.load(DY + Heads * D + offsets, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + 2 * Heads * D + offsets, mask=mask, other=0).to(tl.float32)
    else:
        du = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dx = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
        dy = tl.load(DY + offsets, mask=mask, other=0).to(tl.float32)
    if TRAINING:
        if CONCAT_UX:
            random_offsets = row * 3 * D * Heads + offsets
            # apply dropout on du
            random_du = tl.rand(seed, random_offsets)
            du_keep = random_du > dropout_ratio
            du = tl.where(du_keep, du / (1.0 - dropout_ratio), 0.0)
            # apply dropout on dx
            random_dx = tl.rand(seed, random_offsets + Heads * D)
            dx_keep = random_dx > dropout_ratio
            dx = tl.where(dx_keep, dx / (1.0 - dropout_ratio), 0.0)
            # apply dropout on dy
            random_dy = tl.rand(seed, random_offsets + 2 * Heads * D)
            dy_keep = random_dy > dropout_ratio
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)
        else:
            random_offsets = row * D * Heads + offsets
            random = tl.rand(seed, random_offsets)
            dy_keep = random > dropout_ratio
            # write-back
            dy = tl.where(dy_keep, dy / (1.0 - dropout_ratio), 0.0)

    mean = tl.load(Mean + row * Heads + off_heads)
    rstd = tl.load(Rstd + row * Heads + off_heads)

    # Compute dx
    xhat = (x - mean[:, None]) * rstd[:, None]
    w = tl.load(W + off_heads, mask=mask_h).to(tl.float32)
    b = tl.load(B + off_heads, mask=mask_h).to(tl.float32)
    u = tl.load(U + offsets, mask=mask, other=0).to(tl.float32)
    ln = xhat * w[:, None] + b[:, None]
    du += dy * ln
    if SILU_U:
        sig_u = tl.sigmoid(u)
        silu_u = u * sig_u
        du = du * sig_u * (1 + u - silu_u)
        u = silu_u
    tl.store(DU + offsets, du.to(DU.dtype.element_ty), mask=mask)
    dy = dy * u
    wdy = w[:, None] * dy
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
        y = ln * u
        if TRAINING:
            if CONCAT_UX:
                u = tl.where(
                    du_keep,  # pyre-ignore [61]
                    u / (1.0 - dropout_ratio),
                    0.0,
                )
                x = tl.where(
                    dx_keep,  # pyre-ignore [61]
                    x / (1.0 - dropout_ratio),
                    0.0,
                )
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
            else:
                y = tl.where(
                    dy_keep,  # pyre-ignore [61]
                    y / (1.0 - dropout_ratio),
                    0.0,
                )
        if CONCAT_UX:
            tl.store(Y + offsets, u.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + Heads * D + offsets, x.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + 2 * Heads * D + offsets, y.to(Y.dtype.element_ty), mask=mask)
        else:
            tl.store(Y + offsets, y.to(Y.dtype.element_ty), mask=mask)

    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=1) / D
    c2 = tl.sum(wdy, axis=1) / D
    dx += (wdy - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
    # Write dx
    tl.store(DX + offsets, dx, mask=mask)

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_N
    DW = DW + lock_id * Heads + off_heads
    DB = DB + lock_id * Heads + off_heads
    # Accumulate partial sums for dw/db
    partial_dw = tl.sum(dy * xhat, axis=1)
    partial_dw = tl.ravel(partial_dw)
    partial_db = tl.sum(dy, axis=1)
    partial_db = tl.ravel(partial_db)
    tl.atomic_add(
        DW,
        partial_dw,
        mask=mask_h,
        sem="relaxed",
    )
    tl.atomic_add(
        DB,
        partial_db,
        mask=mask_h,
        sem="relaxed",
    )


def triton_group_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int
]:  # y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed
    assert x.dim() == 2
    assert x.shape == u.shape
    assert x.shape[1] == num_heads * linear_dim
    x = switch_to_contiguous_if_needed(x)
    u = switch_to_contiguous_if_needed(u)
    N, _ = x.shape
    assert weight.dim() == 1
    assert bias.dim() == 1
    assert weight.numel() == num_heads
    assert bias.numel() == num_heads

    if concat_ux:
        y = torch.empty((N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    mean = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N * num_heads,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y, mean, rstd, 0, 0, 0, 0
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D: int = triton.next_power_of_2(linear_dim)
    BLOCK_H: int = triton.next_power_of_2(num_heads)
    if BLOCK_D * BLOCK_H > MAX_FUSED_SIZE:
        raise RuntimeError(
            "This group norm doesn't support num_heads * linear_dim >= 64KB."
        )

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()
    num_warps: int = min(max(BLOCK_D * BLOCK_H // 256, 1), 8)
    # pyre-ignore[28]
    _group_norm_mul_dropout_fwd[(N,)](
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        linear_dim,
        num_heads,
        eps,
        seed,
        dropout_ratio,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        SILU_U=silu_u,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        num_warps=num_warps,
    )
    return y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed  # pyre-ignore [7]


def triton_group_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    BLOCK_H: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    silu_u: bool = False,
    concat_ux: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, dim = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty(
                (N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device
            )
        else:
            y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros_like(weight),
            torch.zeros_like(bias),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    if dim <= 1024:
        GROUP_N = 256 * 8
    elif dim <= 4096:
        GROUP_N = 128 * 8
    elif dim <= 8192:
        GROUP_N = 96 * 8
    else:
        GROUP_N = 64 * 8
    GROUP_N = N if GROUP_N > N else GROUP_N
    _dweight = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=x.device)
    _dbias = torch.zeros((GROUP_N, num_heads), dtype=torch.float32, device=x.device)
    dweight = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((num_heads,), dtype=weight.dtype, device=x.device)
    # pyre-ignore[28]
    _group_norm_mul_dropout_bwd_dx_du[(N,)](
        dx,
        du,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        y.stride(0) if compute_y else 0,  # pyre-ignore [16]
        linear_dim,
        num_heads,
        eps,
        seed,
        dropout_ratio,
        SILU_U=silu_u,
        GROUP_N=GROUP_N,
        BLOCK_D=BLOCK_D,
        BLOCK_H=BLOCK_H,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        COMPUTE_Y=compute_y,
        num_warps=num_warps,
    )
    _group_norm_bwd_dwdb[(num_heads,)](
        _dweight,
        _dbias,
        dweight,
        dbias,
        GROUP_N,
    )
    return dx, du, dweight, dbias, y


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
    key=[],
)
@triton.jit
def _group_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    N,
    BLOCK_N: tl.constexpr,
):
    col = tl.program_id(0)
    num_heads = tl.num_programs(0)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for i in range(0, N, BLOCK_N):
        rows = i + tl.arange(0, BLOCK_N)
        mask = rows < N
        offs = rows * num_heads + col
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + col, sum_dw.to(FINAL_DW.dtype.element_ty))
    tl.store(FINAL_DB + col, sum_db.to(FINAL_DB.dtype.element_ty))


class GroupNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_ux: bool = False,
        num_heads: int = 1,
        linear_dim: int = -1,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
            triton_group_norm_mul_dropout_fwd(
                x=x,
                u=u,
                weight=weight,
                bias=bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                silu_u=silu_u,
                concat_ux=concat_ux,
                num_heads=num_heads,
                linear_dim=linear_dim,
                seed=seed,
            )
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.BLOCK_H = BLOCK_H
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.silu_u = silu_u
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = triton_group_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            BLOCK_H=ctx.BLOCK_H,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            silu_u=ctx.silu_u,
            concat_ux=ctx.concat_ux,
            num_heads=ctx.num_heads,
            linear_dim=ctx.linear_dim,
            compute_y=False,
        )
        return (
            dx,
            du,
            dweight,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class HSTUComputeOutputFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        attn: torch.Tensor,
        u: torch.Tensor,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        output_weight: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_u: bool = False,
        concat_x: bool = False,
        mul_u_activation_type: str = "none",
        group_norm: bool = False,
        num_heads: int = 1,
        linear_dim: int = -1,
        seed: Optional[int] = None,
        recompute_y_in_backward: bool = False,
    ) -> torch.Tensor:
        if dropout_ratio == 0.0:
            training = False

        if group_norm:
            y, mean, rstd, BLOCK_D, BLOCK_H, num_warps, seed = (
                triton_group_norm_mul_dropout_fwd(
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    eps=eps,
                    dropout_ratio=dropout_ratio,
                    training=training,
                    silu_u=silu_u,
                    concat_ux=concat_u and concat_x,
                    num_heads=num_heads,
                    linear_dim=linear_dim,
                    seed=seed,
                )
            )
            ctx.BLOCK_H = BLOCK_H
        else:
            y, mean, rstd, BLOCK_D, num_warps, seed = triton_layer_norm_mul_dropout_fwd(
                x=attn,
                u=u,
                weight=norm_weight,
                bias=norm_bias,
                eps=eps,
                dropout_ratio=dropout_ratio,
                training=training,
                silu_u=silu_u,
                concat_u=concat_u,
                concat_x=concat_x,
                mul_u_activation_type=mul_u_activation_type,
                seed=seed,
            )

        out = maybe_triton_addmm_fwd(x=y, w=output_weight, y=x)

        saved_tensors = [attn, u, norm_weight, norm_bias, mean, rstd, output_weight]
        if not recompute_y_in_backward:
            saved_tensors.append(y)
        ctx.save_for_backward(*saved_tensors)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.concat_u = concat_u
        ctx.concat_x = concat_x
        ctx.dropout_ratio = dropout_ratio
        ctx.num_heads = num_heads
        ctx.linear_dim = linear_dim
        ctx.group_norm = group_norm
        ctx.recompute_y_in_backward = recompute_y_in_backward
        ctx.silu_u = silu_u
        ctx.mul_u_activation_type = mul_u_activation_type
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        torch.Tensor,  # dattn
        torch.Tensor,  # du
        torch.Tensor,  # dx
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        torch.Tensor,  # d_output_weight
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        attn, u, norm_weight, norm_bias, mean, rstd, output_weight = ctx.saved_tensors[
            :7
        ]
        dy = torch.mm(dout, output_weight.t())

        if ctx.group_norm:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_group_norm_mul_dropout_bwd(
                    dy=dy,
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    mean=mean,
                    rstd=rstd,
                    BLOCK_D=ctx.BLOCK_D,
                    BLOCK_H=ctx.BLOCK_H,
                    num_warps=ctx.num_warps,
                    eps=ctx.eps,
                    training=ctx.training,
                    dropout_ratio=ctx.dropout_ratio,
                    seed=ctx.seed,
                    silu_u=ctx.silu_u,
                    concat_ux=ctx.concat_u and ctx.concat_x,
                    num_heads=ctx.num_heads,
                    linear_dim=ctx.linear_dim,
                    compute_y=ctx.recompute_y_in_backward,
                )
            )
        else:
            dattn, du, d_norm_weight, d_norm_bias, y = (
                triton_layer_norm_mul_dropout_bwd(
                    dy=dy,
                    x=attn,
                    u=u,
                    weight=norm_weight,
                    bias=norm_bias,
                    mean=mean,
                    rstd=rstd,
                    BLOCK_D=ctx.BLOCK_D,
                    num_warps=ctx.num_warps,
                    eps=ctx.eps,
                    training=ctx.training,
                    dropout_ratio=ctx.dropout_ratio,
                    seed=ctx.seed,
                    silu_u=ctx.silu_u,
                    concat_u=ctx.concat_u,
                    concat_x=ctx.concat_x,
                    mul_u_activation_type=ctx.mul_u_activation_type,
                    compute_y=ctx.recompute_y_in_backward,
                )
            )
        if not ctx.recompute_y_in_backward:
            y = ctx.saved_tensors[7]
        d_output_weight = torch.mm(y.t(), dout)
        return (
            dattn,
            du,
            dout,
            d_norm_weight,
            d_norm_bias,
            d_output_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@triton.jit
def _helion_ln_mul_dropout_fwd(
    x,
    weight,
    bias,
    u,
    y,
    mean,
    rstd,
    eps,
    seed,
    dropout_ratio,
    D: tl.constexpr,
    stride_x: tl.constexpr,
    stride_u: tl.constexpr,
    stride_y: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    SILU_U: tl.constexpr,
    TRAINING: tl.constexpr,
):
    row = tl.program_id(0)
    x += row.to(tl.int64) * stride_x
    u += row.to(tl.int64) * stride_u
    y += row.to(tl.int64) * stride_y
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # Load input
    x_val = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)

    # Precompute inverse of D for faster computation
    inv_D = 1.0 / D

    # Compute mean
    mean_val = tl.sum(x_val, axis=0) * inv_D

    # Center the data
    x_mean = tl.where(mask, x_val - mean_val, 0.0)

    # Compute variance
    var = tl.sum(x_mean * x_mean, axis=0) * inv_D

    # Compute reciprocal standard deviation
    # pyre-fixme[16]
    rstd_val = libdevice.rsqrt(var + eps)

    # Normalize
    y_norm = x_mean * rstd_val

    # Apply weight and bias
    w = tl.load(weight + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias + cols, mask=mask, other=0.0).to(tl.float32)
    y_ln = y_norm * w + b

    # Load u and optionally apply SiLU activation
    u_val = tl.load(u + cols, mask=mask, other=0.0).to(tl.float32)
    if SILU_U:
        u_processed = u_val * tl.sigmoid(u_val)
    else:
        u_processed = u_val

    y_out = y_ln * u_processed

    if TRAINING:
        # Compute dropout scale
        # pyre-fixme[16]
        dropout_scale = fast_dividef(1.0, 1.0 - dropout_ratio)

        if CONCAT_UX:
            # Generate dropout masks
            random_offsets = 3 * row * BLOCK_D + cols
            random_u, random_x, random_y = rand3x(seed, random_offsets)

            u_keep = random_u > dropout_ratio
            x_keep = random_x > dropout_ratio
            y_keep = random_y > dropout_ratio

            # Apply dropout to u, x, y
            u_output = tl.where(u_keep, u_processed * dropout_scale, 0.0)
            x_output = tl.where(x_keep, x_val * dropout_scale, 0.0)
            y_output = tl.where(y_keep, y_out * dropout_scale, 0.0)
        else:
            # Generate dropout mask for y
            random_offsets = row * BLOCK_D + cols
            random_y = tl.rand(seed, random_offsets)
            y_keep = random_y > dropout_ratio

            # Apply dropout to y
            y_output = tl.where(y_keep, y_out * dropout_scale, 0.0)
    else:
        if CONCAT_UX:
            u_output = u_processed
            x_output = x_val
        y_output = y_out

    # Store outputs
    if CONCAT_UX:
        tl.store(y + cols, u_output.to(y.dtype.element_ty), mask=mask)
        tl.store(y + D + cols, x_output.to(y.dtype.element_ty), mask=mask)
        tl.store(y + 2 * D + cols, y_output.to(y.dtype.element_ty), mask=mask)
    else:
        tl.store(y + cols, y_output.to(y.dtype.element_ty), mask=mask)

    # Store mean and rstd
    tl.store(mean + row, mean_val)
    tl.store(rstd + row, rstd_val)


def helion_layer_norm_mul_dropout_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, int, int, int
]:  # y, mean, rstd, BLOCK_D, num_warps, seed
    N, D = x.shape

    if seed is None:
        seed = torch.randint(low=0, high=2**62, size=(1,), dtype=torch.int64).item()

    if concat_ux:
        y = torch.empty([N, 3 * D], dtype=x.dtype, device=x.device)
    else:
        y = torch.empty([N, D], dtype=x.dtype, device=x.device)
    mean = torch.empty([N], dtype=torch.float32, device=x.device)
    rstd = torch.empty([N], dtype=torch.float32, device=x.device)

    BLOCK_D = triton.next_power_of_2(D)
    # pyre-ignore[28]
    _helion_ln_mul_dropout_fwd[(N,)](
        x,
        weight,
        bias,
        u,
        y,
        mean,
        rstd,
        eps,
        seed,
        dropout_ratio,
        D,
        x.stride(0),
        u.stride(0),
        y.stride(0),
        BLOCK_D,
        CONCAT_UX=concat_ux,
        SILU_U=silu_u,
        TRAINING=training,
        num_warps=1,
    )

    return y, mean, rstd, BLOCK_D, 1, seed  # pyre-ignore [7]


@triton.jit
def _helion_ln_mul_dropout_bwd_dx_du(
    DX,
    DU,
    DY,
    DW,
    DB,
    X,
    U,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride_dx,
    stride_du,
    stride_dy,
    stride_x,
    stride_u,
    stride_y,
    D: tl.constexpr,
    eps,
    seed,
    dropout_ratio,
    N,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_UX: tl.constexpr,
    COMPUTE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    tile_num = tl.num_programs(0)
    rows_per_tile = N // tile_num
    if pid < N % tile_num:
        rows_per_tile += 1

    if rows_per_tile == 0:
        return

    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # precompute inverse of D
    inv_D: tl.constexpr = 1.0 / D

    row = pid
    X += row.to(tl.int64) * stride_x
    U += row.to(tl.int64) * stride_u
    if COMPUTE_Y:
        Y += row.to(tl.int64) * stride_y
    DY += row.to(tl.int64) * stride_dy
    DX += row.to(tl.int64) * stride_dx
    DU += row.to(tl.int64) * stride_du
    DW = DW + pid * D + cols
    DB = DB + pid * D + cols

    partial_dw = tl.zeros((BLOCK_D,), dtype=tl.float32)
    partial_db = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)

    for _idx in range(0, rows_per_tile):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        if CONCAT_UX:
            du = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
            dx = tl.load(DY + D + cols, mask=mask, other=0).to(tl.float32)
            dy = tl.load(DY + 2 * D + cols, mask=mask, other=0).to(tl.float32)
        else:
            du = tl.zeros([BLOCK_D], dtype=tl.float32)
            dx = tl.zeros([BLOCK_D], dtype=tl.float32)
            dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)

        if TRAINING:
            # pyre-fixme[16]
            dropout_scale = fast_dividef(1.0, 1.0 - dropout_ratio)
            if CONCAT_UX:
                random_offsets = 3 * row * BLOCK_D + cols
                # apply dropout on du
                random_du, random_dx, random_dy = rand3x(seed, random_offsets)
                du_keep = random_du > dropout_ratio
                du = tl.where(du_keep, du * dropout_scale, 0.0)
                # apply dropout on dx
                dx_keep = random_dx > dropout_ratio
                dx = tl.where(dx_keep, dx * dropout_scale, 0.0)
                # apply dropout on dy
                dy_keep = random_dy > dropout_ratio
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)
            else:
                random_offsets = row * BLOCK_D + cols
                random = tl.rand(seed, random_offsets)
                dy_keep = random > dropout_ratio
                dy = tl.where(dy_keep, dy * dropout_scale, 0.0)

        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)

        # Compute dx
        xhat = (x - mean) * rstd
        u = tl.load(U + cols, mask=mask, other=0).to(tl.float32)
        ln = xhat * w + b
        du += dy * ln

        if SILU_U:
            sig_u = tl.sigmoid(u)
            silu_u = u * sig_u
            du = du * sig_u * (1 + u - silu_u)
            u = silu_u

        tl.store(DU + cols, du.to(DU.dtype.element_ty), mask=mask)
        dy = dy * u
        wdy = w * dy

        if COMPUTE_Y:
            y = ln * u
            if TRAINING:
                # pyre-fixme[16]
                dropout_scale_y = fast_dividef(1.0, 1.0 - dropout_ratio)
                if CONCAT_UX:
                    u = tl.where(du_keep, u * dropout_scale_y, 0.0)  # pyre-ignore [61]
                    x = tl.where(dx_keep, x * dropout_scale_y, 0.0)  # pyre-ignore [61]
                    y = tl.where(dy_keep, y * dropout_scale_y, 0.0)  # pyre-ignore [61]
                else:
                    y = tl.where(dy_keep, y * dropout_scale_y, 0.0)  # pyre-ignore [61]
            if CONCAT_UX:
                tl.store(Y + cols, u.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + D + cols, x.to(Y.dtype.element_ty), mask=mask)
                tl.store(Y + 2 * D + cols, y.to(Y.dtype.element_ty), mask=mask)
            else:
                tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)
            Y += tile_num.to(tl.int64) * stride_y

        xhat = tl.where(mask, xhat, 0.0)
        wdy = tl.where(mask, wdy, 0.0)
        # multiply by inv_D
        c1 = tl.sum(xhat * wdy, axis=0) * inv_D
        c2 = tl.sum(wdy, axis=0) * inv_D
        dx += (wdy - (xhat * c1 + c2)) * rstd

        # Write dx
        tl.store(DX + cols, dx, mask=mask)

        # Accumulate partial sums for dw/db
        partial_dw += dy * xhat
        partial_db += dy

        X += tile_num.to(tl.int64) * stride_x
        U += tile_num.to(tl.int64) * stride_u
        DY += tile_num.to(tl.int64) * stride_dy
        DX += tile_num.to(tl.int64) * stride_dx
        DU += tile_num.to(tl.int64) * stride_du
        row += tile_num

    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)


@triton_autotune(
    configs=_get_bwd_dwdb_configs(),
    key=["D"],
)
@triton.jit
def _helion_ln_mul_dropout_bwd_dwdb(
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
        off_mask = (rows[:, None] < N) & (cols[None, :] < D)
        offs = rows[:, None] * D + cols[None, :]
        dw += tl.load(DW + offs, mask=off_mask, other=0.0)
        db += tl.load(DB + offs, mask=off_mask, other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw.to(FINAL_DW.dtype.element_ty), mask=cols < D)
    tl.store(FINAL_DB + cols, sum_db.to(FINAL_DB.dtype.element_ty), mask=cols < D)


def helion_layer_norm_mul_dropout_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    BLOCK_D: int,
    num_warps: int,
    eps: float,
    training: bool,
    dropout_ratio: float,
    seed: Optional[int] = None,
    silu_u: bool = False,
    concat_ux: bool = False,
    compute_y: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]
]:
    y = None
    N, D = x.shape
    if compute_y:
        if concat_ux:
            y = torch.empty((N, 3 * D), dtype=x.dtype, device=x.device)
        else:
            y = torch.empty_like(x)
    if N == 0:
        return (
            torch.zeros_like(x),
            torch.zeros_like(u),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            torch.zeros((D,), dtype=weight.dtype, device=x.device),
            y,
        )
    dx = torch.empty_like(x)
    du = torch.empty_like(u)
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    tile_num = max(1, min(sms * 64, N // 4))
    _dweight = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    _dbias = torch.empty((tile_num, D), dtype=torch.float32, device=x.device)
    dweight = torch.empty((D,), dtype=weight.dtype, device=x.device)
    dbias = torch.empty((D,), dtype=weight.dtype, device=x.device)

    # pyre-ignore[28]
    _helion_ln_mul_dropout_bwd_dx_du[(tile_num,)](
        dx,
        du,
        dy,
        _dweight,
        _dbias,
        x,
        u,
        y,
        weight,
        bias,
        mean,
        rstd,
        dx.stride(0),
        du.stride(0),
        dy.stride(0),
        x.stride(0),
        u.stride(0),
        y.stride(0) if compute_y else 0,  # pyre-ignore [16]
        D,
        eps,
        seed,
        dropout_ratio,
        N=N,
        SILU_U=silu_u,
        BLOCK_D=BLOCK_D,
        TRAINING=training,
        CONCAT_UX=concat_ux,
        COMPUTE_Y=compute_y,
        num_warps=num_warps,
    )

    blocks = triton.next_power_of_2(sms * 4)
    BLOCK_D_DWDB = triton.next_power_of_2(triton.cdiv(D, blocks))
    BLOCK_D_DWDB = min(max(BLOCK_D_DWDB, 4), 128)
    _helion_ln_mul_dropout_bwd_dwdb[(triton.cdiv(D, BLOCK_D_DWDB),)](
        _dweight,
        _dbias,
        dweight,
        dbias,
        tile_num,
        D,
        BLOCK_D=BLOCK_D_DWDB,
    )
    return dx, du, dweight, dbias, y


class HelionLayerNormMulDropoutFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        dropout_ratio: float,
        training: bool,
        silu_u: bool = False,
        concat_ux: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if dropout_ratio == 0.0:
            # skip dropout computation if dropout ratio is 0
            training = False
        y, mean, rstd, BLOCK_D, num_warps, seed = helion_layer_norm_mul_dropout_fwd(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=dropout_ratio,
            training=training,
            silu_u=silu_u,
            concat_ux=concat_ux,
            seed=seed,
        )
        ctx.save_for_backward(x, u, weight, bias, mean, rstd)
        ctx.BLOCK_D = BLOCK_D
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.seed = seed
        ctx.training = training
        ctx.silu_u = silu_u
        ctx.concat_ux = concat_ux
        ctx.dropout_ratio = dropout_ratio
        return y

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dy: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
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
        x, u, weight, bias, mean, rstd = ctx.saved_tensors
        dx, du, dweight, dbias, _ = helion_layer_norm_mul_dropout_bwd(
            dy=dy,
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            BLOCK_D=ctx.BLOCK_D,
            num_warps=ctx.num_warps,
            eps=ctx.eps,
            training=ctx.training,
            dropout_ratio=ctx.dropout_ratio,
            seed=ctx.seed,
            silu_u=ctx.silu_u,
            concat_ux=ctx.concat_ux,
            compute_y=False,
        )
        return dx, du, dweight, dbias, None, None, None, None, None, None


@torch.fx.wrap
def helion_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_ux: bool = False,
    seed: Optional[int] = None,
) -> torch.Tensor:
    return HelionLayerNormMulDropoutFunction.apply(
        x, u, weight, bias, eps, dropout_ratio, training, silu_u, concat_ux, seed
    )


@torch.fx.wrap
def triton_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if group_norm:
        return GroupNormMulDropoutFunction.apply(
            x,
            u,
            weight,
            bias,
            eps,
            dropout_ratio,
            training,
            silu_u,
            concat_u and concat_x,
            num_heads,
            linear_dim,
            seed,
        )
    else:
        return LayerNormMulDropoutFunction.apply(
            x,
            u,
            weight,
            bias,
            eps,
            dropout_ratio,
            training,
            silu_u,
            concat_u and concat_x,
            seed,
        )


@torch.fx.wrap
def triton_hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
    seed: Optional[int] = None,
    recompute_y_in_backward: bool = False,
) -> torch.Tensor:
    return HSTUComputeOutputFunction.apply(
        attn,
        u,
        x,
        norm_weight,
        norm_bias,
        output_weight,
        eps,
        dropout_ratio,
        training,
        silu_u,
        concat_u,
        concat_x,
        mul_u_activation_type,
        group_norm,
        num_heads,
        linear_dim,
        seed,
        recompute_y_in_backward,
    )
