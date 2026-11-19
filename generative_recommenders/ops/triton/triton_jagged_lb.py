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

"""Load-balanced Triton kernels for jagged+jagged concat and split.

Tile output rows and binary-search offsets to find each row's batch.
Split reuses concat's forward and backward kernels in reverse. With
skip_left, split returns an empty A and zeros A's rows in the input gradient.

Offsets must start at zero, be nondecreasing, and fit the logical row layout.
Trailing rows beyond the offsets are zero-filled. Outputs must be contiguous.
"""

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

# Fixed launch configuration: rows per program and warps per program.
_BLOCK_N = 16
_NUM_WARPS = 2


# Triton TR001: fixed launch configuration, no autotuning.
@triton.jit
def _concat_2D_jagged_jagged_lb_fwd(  # noqa: TR001
    ValuesA,
    ValuesB,
    OffsetsA,
    OffsetsB,
    Out,
    NOUT,
    NUM_BATCH,
    stride_oa,
    stride_ob,
    stride_ar,
    stride_ac,
    stride_br,
    stride_bc,
    D: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SKIP_LEFT: tl.constexpr = False,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = rows < NOUT
    # Largest z with offsets_a[z] + offsets_b[z] <= row.
    lo = tl.zeros([BLOCK_N], tl.int32)
    hi = tl.zeros([BLOCK_N], tl.int32) + NUM_BATCH
    for _ in tl.range(SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) >> 1
        boundary = tl.load(OffsetsA + mid * stride_oa) + tl.load(
            OffsetsB + mid * stride_ob
        )
        advance = active & (boundary <= rows)
        lo = tl.where(advance, mid + 1, lo)
        hi = tl.where(active & ~advance, mid, hi)
    z = lo - 1
    start_a = tl.load(OffsetsA + z * stride_oa)
    end_a = tl.load(OffsetsA + (z + 1) * stride_oa)
    start_b = tl.load(OffsetsB + z * stride_ob)
    end_b = tl.load(OffsetsB + (z + 1) * stride_ob)
    from_a = rows < end_a + start_b
    mapped = valid & (rows >= start_a + start_b) & (rows < end_a + end_b)
    if SKIP_LEFT:
        # A rows are stored as zeros without being loaded.
        mapped = mapped & ~from_a
    row_a = (rows - start_b).to(tl.int64) * stride_ar
    row_b = (rows - end_a).to(tl.int64) * stride_br
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols[None, :] < D
    load_mask = mapped[:, None] & col_mask
    store_mask = valid[:, None] & col_mask
    ptr_a = ValuesA + row_a[:, None] + cols[None, :] * stride_ac
    ptr_b = ValuesB + row_b[:, None] + cols[None, :] * stride_bc
    # One gather load per row from the side that owns it.
    values = tl.load(tl.where(from_a[:, None], ptr_a, ptr_b), load_mask, other=0)
    out_rows = rows.to(tl.int64)[:, None] * D + cols[None, :]
    tl.store(Out + out_rows, values, store_mask)


@triton.jit
def _concat_2D_jagged_jagged_lb_bwd_side(
    Grad,
    OffsetsOwn,
    stride_own,
    OffsetsOther,
    stride_other,
    Dst,
    N,
    pid,
    NUM_BATCH,
    stride_gr,
    stride_gc,
    D: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_A: tl.constexpr,
):
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = rows < N
    # Largest z with offsets_own[z] <= row.
    lo = tl.zeros([BLOCK_N], tl.int32)
    hi = tl.zeros([BLOCK_N], tl.int32) + NUM_BATCH
    for _ in tl.range(SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) >> 1
        advance = active & (tl.load(OffsetsOwn + mid * stride_own) <= rows)
        lo = tl.where(advance, mid + 1, lo)
        hi = tl.where(active & ~advance, mid, hi)
    z = lo - 1
    end_own = tl.load(OffsetsOwn + (z + 1) * stride_own)
    # A row a comes from output row a + offsets_b[z]; B row b comes from
    # output row b + offsets_a[z + 1].
    if IS_A:
        shift = tl.load(OffsetsOther + z * stride_other)
    else:
        shift = tl.load(OffsetsOther + (z + 1) * stride_other)
    mapped = valid & (rows < end_own)
    src_rows = (rows + shift).to(tl.int64) * stride_gr
    cols = tl.arange(0, BLOCK_D)
    col_mask = cols[None, :] < D
    load_mask = mapped[:, None] & col_mask
    store_mask = valid[:, None] & col_mask
    grad = tl.load(
        Grad + src_rows[:, None] + cols[None, :] * stride_gc, load_mask, other=0
    )
    dst_rows = rows.to(tl.int64)[:, None] * D + cols[None, :]
    tl.store(Dst + dst_rows, grad, store_mask)


# Triton TR001: fixed launch configuration, no autotuning.
@triton.jit
def _concat_2D_jagged_jagged_lb_bwd(  # noqa: TR001
    Grad,
    OffsetsA,
    OffsetsB,
    GradA,
    GradB,
    NA,
    NB,
    NUM_BATCH,
    stride_oa,
    stride_ob,
    stride_gr,
    stride_gc,
    D: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Programs [0, blocks_a) write GradA and the rest write GradB. Each branch
    # stores into its own destination so the stores stay contiguous.
    pid = tl.program_id(0)
    blocks_a = tl.cdiv(NA, BLOCK_N)
    if pid < blocks_a:
        _concat_2D_jagged_jagged_lb_bwd_side(
            Grad,
            OffsetsA,
            stride_oa,
            OffsetsB,
            stride_ob,
            GradA,
            NA,
            pid,
            NUM_BATCH,
            stride_gr,
            stride_gc,
            D,
            SEARCH_STEPS,
            BLOCK_N,
            BLOCK_D,
            True,
        )
    else:
        _concat_2D_jagged_jagged_lb_bwd_side(
            Grad,
            OffsetsB,
            stride_ob,
            OffsetsA,
            stride_oa,
            GradB,
            NB,
            pid - blocks_a,
            NUM_BATCH,
            stride_gr,
            stride_gc,
            D,
            SEARCH_STEPS,
            BLOCK_N,
            BLOCK_D,
            False,
        )


# The launch helpers use plain integer math because they run on every call.
def triton_concat_2D_jagged_jagged_lb_fwd(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    values_out: torch.Tensor,
    skip_left: bool = False,
) -> None:
    """Fill contiguous values_out [Na + Nb, D] with A[z] followed by B[z].

    With skip_left, values_a is not read and the A rows are stored as zeros.
    """
    n, d = values_out.shape
    if n == 0 or d == 0:
        return
    num_batch = offsets_a.size(0) - 1
    if num_batch == 0:
        values_out.zero_()
        return
    _concat_2D_jagged_jagged_lb_fwd[((n + _BLOCK_N - 1) // _BLOCK_N,)](
        values_a,
        values_b,
        offsets_a,
        offsets_b,
        values_out,
        n,
        num_batch,
        offsets_a.stride(0),
        offsets_b.stride(0),
        values_a.stride(0),
        values_a.stride(1),
        values_b.stride(0),
        values_b.stride(1),
        D=d,
        SEARCH_STEPS=num_batch.bit_length(),
        BLOCK_N=_BLOCK_N,
        BLOCK_D=1 << (d - 1).bit_length(),
        SKIP_LEFT=skip_left,
        num_warps=_NUM_WARPS,
    )


def triton_concat_2D_jagged_jagged_lb_bwd(
    d_out: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    d_values_a: torch.Tensor,
    d_values_b: torch.Tensor,
) -> None:
    """Split d_out [Na + Nb, D] into contiguous d_values_a [Na, D] and d_values_b [Nb, D].

    A 0-row d_values_a skips the A side, so only d_values_b is written.
    """
    na, d = d_values_a.shape
    nb = d_values_b.size(0)
    if (na == 0 and nb == 0) or d == 0:
        return
    num_batch = offsets_a.size(0) - 1
    if num_batch == 0:
        d_values_a.zero_()
        d_values_b.zero_()
        return
    grid = ((na + _BLOCK_N - 1) // _BLOCK_N + (nb + _BLOCK_N - 1) // _BLOCK_N,)
    _concat_2D_jagged_jagged_lb_bwd[grid](
        d_out,
        offsets_a,
        offsets_b,
        d_values_a,
        d_values_b,
        na,
        nb,
        num_batch,
        offsets_a.stride(0),
        offsets_b.stride(0),
        d_out.stride(0),
        d_out.stride(1),
        D=d,
        SEARCH_STEPS=num_batch.bit_length(),
        BLOCK_N=_BLOCK_N,
        BLOCK_D=1 << (d - 1).bit_length(),
        num_warps=_NUM_WARPS,
    )


def triton_split_2D_jagged_jagged_lb_fwd(
    values: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    values_a: torch.Tensor,
    values_b: torch.Tensor,
) -> None:
    """Split values [Na + Nb, D] into contiguous values_a and values_b.

    Pass a 0-row values_a for skip_left; only values_b is written then.
    """
    triton_concat_2D_jagged_jagged_lb_bwd(
        values, offsets_a, offsets_b, values_a, values_b
    )


def triton_split_2D_jagged_jagged_lb_bwd(
    d_values_a: torch.Tensor,
    d_values_b: torch.Tensor,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    d_values: torch.Tensor,
    skip_left: bool,
) -> None:
    """Fill contiguous d_values [Na + Nb, D] from d_values_a and d_values_b.

    With skip_left, d_values_a is not read and the A rows get zero gradient.
    """
    triton_concat_2D_jagged_jagged_lb_fwd(
        d_values_a, d_values_b, offsets_a, offsets_b, d_values, skip_left
    )
