# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import math
import os
from typing import Any, cast

import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.tlx as tlx
except ImportError:
    tlx = None

_TLX: Any = tlx


def can_use_tlx_jagged_packing(
    *values: torch.Tensor,
    is_forward: bool = False,
    use_multirow: bool = False,
    is_normal_jagged: bool = False,
) -> bool:
    """Resolve the opt-in at the runtime launcher, including during FX replay."""
    if os.environ.get("ONEFLOW_TLX_JAGGED_PACKING", "1") != "1":
        return False
    if tlx is None or not all(
        hasattr(tlx, name)
        for name in (
            "layout",
            "require_layout",
            "buffer_load",
            "buffer_store",
        )
    ):
        return False
    if not values or values[-1].ndim != 2 or torch.version.hip is None:
        return False
    first = values[0]
    if not first.is_cuda or first.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        return False
    if not all(
        value.device == first.device
        and value.dtype == first.dtype
        and value.ndim in (2, 3)
        and value.shape[-1] == first.shape[-1]
        and value.stride(-1) == 1
        and all(stride >= 0 for stride in value.stride())
        for value in values
    ):
        return False
    # Complete-call measurements only promote normal BF16 D100 forward over
    # the production multirow kernels. Backward keeps its separate TLX wins.
    if (
        is_forward
        and use_multirow
        and not (
            is_normal_jagged
            and first.dtype == torch.bfloat16
            and first.shape[-1] == 100
        )
    ):
        return False
    arch = torch.cuda.get_device_properties(first.device).gcnArchName
    return arch.split(":")[0] == "gfx950" and 0 < first.shape[-1] <= 512


def _packing_layout(
    block_rows: int, block_d: int, num_warps: int, vector: int
) -> object:
    if tlx is None:
        raise RuntimeError("TLX is required for jagged packing")
    thread_cols = min(64, block_d // vector)
    thread_rows = min(block_rows, num_warps * 64 // thread_cols)
    duplicate_threads = num_warps * 64 // (thread_cols * thread_rows)
    return _TLX.layout(
        shape=(
            (thread_cols, thread_rows, duplicate_threads),
            (vector, block_d // (thread_cols * vector), block_rows // thread_rows),
        ),
        stride=((vector, block_d, 0), (1, thread_cols * vector, block_d * thread_rows)),
    )


@triton.jit
def _sequence_bounds(
    OffsetsA,
    OffsetsB,
    user,
    DENSE_SIZE,
    DENSE_A: tl.constexpr,
    DENSE_B: tl.constexpr,
):
    user64 = user.to(tl.int64)
    if DENSE_A:
        start_a = user64 * DENSE_SIZE
        len_a = DENSE_SIZE
    else:
        start_a = tl.load(OffsetsA + user).to(tl.int64)
        len_a = tl.load(OffsetsA + user + 1).to(tl.int64) - start_a
    if DENSE_B:
        start_b = user64 * DENSE_SIZE
        len_b = DENSE_SIZE
    else:
        start_b = tl.load(OffsetsB + user).to(tl.int64)
        len_b = tl.load(OffsetsB + user + 1).to(tl.int64) - start_b
    return start_a, len_a, start_b, len_b


@triton.jit
def _find_packed_user(
    OffsetsA,
    OffsetsB,
    row,
    B,
    DENSE_SIZE,
    DENSE_A: tl.constexpr,
    DENSE_B: tl.constexpr,
    REPLACE: tl.constexpr,
):
    # Upper bound skips duplicate starts belonging to empty users.
    lo = 0
    hi = B + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if DENSE_A:
            start = mid.to(tl.int64) * DENSE_SIZE
        else:
            start = tl.load(OffsetsA + mid).to(tl.int64)
        if not REPLACE:
            if DENSE_B:
                start += mid.to(tl.int64) * DENSE_SIZE
            else:
                start += tl.load(OffsetsB + mid).to(tl.int64)
        before = start <= row
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    return lo - 1


@triton.jit
def _copy_packed_tile(
    ValuesA,
    ValuesB,
    ValuesC,
    user,
    rows,
    active,
    start_a,
    len_a,
    start_b,
    len_b,
    D: tl.constexpr,
    STRIDE_A: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_C: tl.constexpr,
    DENSE_STRIDE,
    PREFIX: tl.constexpr,
    DENSE_A: tl.constexpr,
    DENSE_B: tl.constexpr,
    REPLACE: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COPY_LAYOUT: tl.constexpr,
    VECTOR: tl.constexpr,
    USE_BUFFER: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_D)
    c_start = start_a if REPLACE else start_a + start_b
    b_boundary = len_a - len_b if REPLACE else len_a + PREFIX
    from_a = (rows >= PREFIX) & (rows < b_boundary)
    row_a = rows - PREFIX
    row_b = tl.where(rows < PREFIX, rows, rows - b_boundary + PREFIX)
    base_a = user.to(tl.int64) * DENSE_STRIDE if DENSE_A else start_a * STRIDE_A
    base_b = user.to(tl.int64) * DENSE_STRIDE if DENSE_B else start_b * STRIDE_B
    # Cast before multiplying: production dY row strides exceed the i32 address range.
    offs_a = row_a.to(tl.int64)[:, None] * STRIDE_A + cols[None, :]
    offs_b = row_b.to(tl.int64)[:, None] * STRIDE_B + cols[None, :]
    offs_c = rows.to(tl.int64)[:, None] * STRIDE_C + cols[None, :]
    mask = active[:, None] & (cols[None, :] < D)
    mask_a = mask & from_a[:, None]
    mask_b = mask & ~from_a[:, None]
    if USE_BUFFER:
        ptr_a = ValuesA + base_a
        ptr_b = ValuesB + base_b
        ptr_c = ValuesC + c_start * STRIDE_C
        offs_a = _TLX.require_layout(offs_a.to(tl.int32), COPY_LAYOUT)
        offs_b = _TLX.require_layout(offs_b.to(tl.int32), COPY_LAYOUT)
        offs_c = _TLX.require_layout(offs_c.to(tl.int32), COPY_LAYOUT)
        if SPLIT:
            value = _TLX.buffer_load(ptr_c, offs_c, mask, other=0, contiguity=VECTOR)
            # Replace backward writes every dA row: retained gradients or zero.
            # This lets autograd allocate dA without clearing the full tensor.
            if REPLACE and PREFIX == 0:
                value_a = tl.where(from_a[:, None], value, 0)
                _TLX.buffer_store(value_a, ptr_a, offs_a, mask)
            else:
                _TLX.buffer_store(value, ptr_a, offs_a, mask_a)
            _TLX.buffer_store(value, ptr_b, offs_b, mask_b)
        else:
            value_a = _TLX.buffer_load(
                ptr_a, offs_a, mask_a, other=0, contiguity=VECTOR
            )
            value_b = _TLX.buffer_load(
                ptr_b, offs_b, mask_b, other=0, contiguity=VECTOR
            )
            # Separate stores preserve copied bits and avoid an A/B pointer phi.
            _TLX.buffer_store(value_a, ptr_c, offs_c, mask_a)
            _TLX.buffer_store(value_b, ptr_c, offs_c, mask_b)
    else:
        ptr_a = _TLX.require_layout(ValuesA + base_a + offs_a, COPY_LAYOUT)
        ptr_b = _TLX.require_layout(ValuesB + base_b + offs_b, COPY_LAYOUT)
        ptr_c = _TLX.require_layout(ValuesC + c_start * STRIDE_C + offs_c, COPY_LAYOUT)
        if SPLIT:
            value = tl.load(ptr_c, mask, other=0)
            if REPLACE and PREFIX == 0:
                tl.store(ptr_a, tl.where(from_a[:, None], value, 0), mask)
            else:
                tl.store(ptr_a, value, mask_a)
            tl.store(ptr_b, value, mask_b)
        else:
            value_a = tl.load(ptr_a, mask_a, other=0)
            value_b = tl.load(ptr_b, mask_b, other=0)
            tl.store(ptr_c, value_a, mask_a)
            tl.store(ptr_c, value_b, mask_b)


@triton.jit(do_not_specialize=["DENSE_SIZE", "DENSE_STRIDE", "ROW_STEP"])
def _tlx_pack_rect(
    ValuesA,
    ValuesB,
    ValuesC,
    OffsetsA,
    OffsetsB,
    D: tl.constexpr,
    STRIDE_A: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_C: tl.constexpr,
    DENSE_SIZE,
    DENSE_STRIDE,
    PREFIX: tl.constexpr,
    DENSE_A: tl.constexpr,
    DENSE_B: tl.constexpr,
    REPLACE: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COPY_LAYOUT: tl.constexpr,
    VECTOR: tl.constexpr,
    USE_BUFFER: tl.constexpr,
    ROW_STEP,
):
    user = tl.program_id(1)
    first = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
    start_a, len_a, start_b, len_b = _sequence_bounds(
        OffsetsA, OffsetsB, user, DENSE_SIZE, DENSE_A, DENSE_B
    )
    length = len_a if REPLACE else len_a + len_b
    if first >= length:
        return
    for tile_first in tl.range(first, length, ROW_STEP):
        rows = tile_first + tl.arange(0, BLOCK_ROWS)
        _copy_packed_tile(
            ValuesA,
            ValuesB,
            ValuesC,
            user,
            rows,
            rows < length,
            start_a,
            len_a,
            start_b,
            len_b,
            D,
            STRIDE_A,
            STRIDE_B,
            STRIDE_C,
            DENSE_STRIDE,
            PREFIX,
            DENSE_A,
            DENSE_B,
            REPLACE,
            SPLIT,
            BLOCK_D,
            COPY_LAYOUT,
            VECTOR,
            USE_BUFFER,
        )


@triton.jit(do_not_specialize=["T", "B", "DENSE_SIZE", "DENSE_STRIDE"])
def _tlx_pack_compact(
    ValuesA,
    ValuesB,
    ValuesC,
    OffsetsA,
    OffsetsB,
    T,
    B,
    D: tl.constexpr,
    STRIDE_A: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_C: tl.constexpr,
    DENSE_SIZE,
    DENSE_STRIDE,
    PREFIX: tl.constexpr,
    DENSE_A: tl.constexpr,
    DENSE_B: tl.constexpr,
    REPLACE: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COPY_LAYOUT: tl.constexpr,
    VECTOR: tl.constexpr,
    USE_BUFFER: tl.constexpr,
):
    first = tl.program_id(0).to(tl.int64) * BLOCK_ROWS
    last = tl.minimum(first + BLOCK_ROWS, T)
    global_rows = first + tl.arange(0, BLOCK_ROWS)
    cursor = first
    while cursor < last:
        user = _find_packed_user(
            OffsetsA, OffsetsB, cursor, B, DENSE_SIZE, DENSE_A, DENSE_B, REPLACE
        )
        start_a, len_a, start_b, len_b = _sequence_bounds(
            OffsetsA, OffsetsB, user, DENSE_SIZE, DENSE_A, DENSE_B
        )
        c_start = start_a if REPLACE else start_a + start_b
        length = len_a if REPLACE else len_a + len_b
        end = tl.minimum(c_start + length, last)
        active = (global_rows >= cursor) & (global_rows < end)
        _copy_packed_tile(
            ValuesA,
            ValuesB,
            ValuesC,
            user,
            global_rows - c_start,
            active,
            start_a,
            len_a,
            start_b,
            len_b,
            D,
            STRIDE_A,
            STRIDE_B,
            STRIDE_C,
            DENSE_STRIDE,
            PREFIX,
            DENSE_A,
            DENSE_B,
            REPLACE,
            SPLIT,
            BLOCK_D,
            COPY_LAYOUT,
            VECTOR,
            USE_BUFFER,
        )
        cursor = end


def launch_tlx_jagged_packing(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    values_c: torch.Tensor,
    offsets_a: torch.Tensor | None,
    offsets_b: torch.Tensor | None,
    *,
    max_seq_len: int,
    batch_size: int,
    dense_size: int = 0,
    dense_stride: int = 0,
    n_prefix: int = 0,
    is_dense_a: bool = False,
    is_dense_b: bool = False,
    is_replace: bool = False,
    split: bool = False,
    schedule: str = "auto",
    block_rows: int = 32,
    num_warps: int = 4,
) -> None:
    """Copy per-user sequences without CPU reads of dynamic offsets or metadata."""
    if schedule not in ("auto", "compact", "rect", "strided"):
        raise ValueError(f"Unknown jagged packing schedule: {schedule}")
    total_rows, d = values_c.shape
    if total_rows == 0 or batch_size == 0 or d == 0:
        return
    if schedule == "auto":
        schedule = "strided" if d <= 128 and not split else "compact"
    strides = (values_a.stride(-2), values_b.stride(-2), values_c.stride(0))
    alignment = math.gcd(d, *strides, dense_stride)
    # CDNA buffer loads are at most 128 bits per lane.
    vector = min(16 // values_c.element_size(), alignment & -alignment)
    use_buffer = (max_seq_len + block_rows) * max(
        strides
    ) * values_c.element_size() < 2**31
    common = {
        "D": d,
        "STRIDE_A": strides[0],
        "STRIDE_B": strides[1],
        "STRIDE_C": strides[2],
        "DENSE_SIZE": dense_size,
        "DENSE_STRIDE": dense_stride,
        "PREFIX": n_prefix,
        "DENSE_A": is_dense_a,
        "DENSE_B": is_dense_b,
        "REPLACE": is_replace,
        "SPLIT": split,
        "BLOCK_ROWS": block_rows,
        "BLOCK_D": triton.next_power_of_2(d),
        "COPY_LAYOUT": _packing_layout(
            block_rows, triton.next_power_of_2(d), num_warps, vector
        ),
        "VECTOR": vector,
        "USE_BUFFER": use_buffer,
        "num_warps": num_warps,
        "num_stages": 1,
    }
    arguments = (values_a, values_b, values_c, offsets_a, offsets_b)
    if schedule == "compact":
        kernel = cast(Any, _tlx_pack_compact)
        kernel[(triton.cdiv(total_rows, block_rows),)](
            *arguments, T=total_rows, B=batch_size, **common
        )
    else:
        tiles = triton.cdiv(max_seq_len, block_rows)
        if schedule == "strided":
            tiles = min(tiles, max(1, triton.cdiv(total_rows, batch_size * block_rows)))
        kernel = cast(Any, _tlx_pack_rect)
        kernel[(tiles, batch_size)](*arguments, ROW_STEP=tiles * block_rows, **common)
