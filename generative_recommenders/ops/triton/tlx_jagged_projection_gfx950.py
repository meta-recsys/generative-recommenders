# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Any, cast

import torch
import triton
import triton.language as tl
from generative_recommenders.ops.triton.tlx_jagged_gfx950 import _packing_layout, tlx

_TLX: Any = tlx


def can_use_tlx_jagged_projection(
    phase: str, offsets: torch.Tensor, *values: torch.Tensor
) -> bool:
    if os.environ.get(f"ONEFLOW_TLX_JAGGED_PROJECTION_{phase}", "1") != "1":
        return False
    if tlx is None or torch.version.hip is None or not values:
        return False
    if not all(
        hasattr(tlx, name)
        for name in (
            "layout",
            "require_layout",
            "buffer_load",
            "buffer_store",
            "buffer_load_to_local",
            "amd_mfma_layout",
            "dot_operand_layout",
            "zeros",
            "amd_scheduled_mfma",
            "slice_layout",
            "local_alloc",
            "local_view",
            "local_load",
            "async_load",
            "async_load_commit_group",
            "async_load_wait_group",
            "warp_pipeline_stage",
            "padded_shared_layout_encoding",
            "dtype_of",
        )
    ):
        return False
    if not hasattr(getattr(tlx, "padded_shared_layout_encoding", None), "with_bases"):
        return False
    first = values[0]
    if (
        not first.is_cuda
        or first.ndim not in (2, 3)
        or first.shape[-1] == 0
        or first.dtype not in (torch.float16, torch.bfloat16)
    ):
        return False
    if (
        offsets.ndim != 1
        or offsets.device != first.device
        or offsets.dtype not in (torch.int32, torch.int64)
    ):
        return False
    if not all(
        value.device == first.device
        and value.dtype == first.dtype
        and value.ndim in (2, 3)
        and value.stride(-1) == 1
        and all(stride >= 0 for stride in value.stride())
        for value in values
    ):
        return False
    return (
        torch.cuda.get_device_properties(first.device).gcnArchName.split(":")[0]
        == "gfx950"
    )


def _column_layout(rows: int, cols: int, warps: int, vector: int) -> object:
    if tlx is None:
        raise RuntimeError("TLX is required for jagged projection")
    threads_m = min(64, rows // vector)
    threads_k = min(cols, warps * 64 // threads_m)
    duplicates = warps * 64 // (threads_m * threads_k)
    return _TLX.layout(
        shape=(
            (threads_m, threads_k, duplicates),
            (vector, rows // (threads_m * vector), cols // threads_k),
        ),
        stride=((vector * cols, 1, 0), (cols, threads_m * vector * cols, threads_k)),
    )


def _vector(*alignment: int) -> int:
    divisor = math.gcd(*alignment)
    return min(8, divisor & -divisor)


def _striped_copy_layout(
    rows: int,
    cols: int,
    warps: int,
    vector: int,
    column: bool,
    high_lane_row: bool,
) -> object:
    fast, slow = (rows, cols) if column else (cols, rows)
    fast_stride, slow_stride = (cols, 1) if column else (1, cols)
    lane_fast = min(64, fast // vector)
    lane_slow = min(slow, 64 // lane_fast)
    warp_slow = min(warps, slow // lane_slow)
    repeats = slow // (lane_slow * warp_slow)
    lane_stride = warp_slow * (repeats if high_lane_row else 1)
    repeat_stride = warp_slow * (1 if high_lane_row else lane_slow)
    duplicates = warps * 64 // (lane_fast * lane_slow * warp_slow)
    thread_shape: tuple[int, ...] = (lane_fast, lane_slow, warp_slow, duplicates)
    thread_stride: tuple[int, ...] = (
        vector * fast_stride,
        lane_stride * slow_stride,
        slow_stride,
        0,
    )
    if column and high_lane_row and lane_slow >= 4:
        thread_shape = (lane_fast, 2, lane_slow // 4, 2, warp_slow, duplicates)
        thread_stride = (
            vector * fast_stride,
            (lane_slow // 2) * lane_stride * slow_stride,
            2 * lane_stride * slow_stride,
            lane_stride * slow_stride,
            slow_stride,
            0,
        )
    return _TLX.layout(
        shape=((thread_shape), (vector, fast // (lane_fast * vector), repeats)),
        stride=(
            thread_stride,
            (
                fast_stride,
                lane_fast * vector * fast_stride,
                repeat_stride * slow_stride,
            ),
        ),
    )


def _mode_bits(shape: tuple[int, ...], stride: tuple[int, ...]) -> list[int]:
    return [
        step << bit
        for extent, step in zip(shape, stride)
        for bit in range(extent.bit_length() - 1)
        if step
    ]


def _shared_copy_layout(copy: Any, shape: tuple[int, int], padding: int) -> object:
    values = _mode_bits(copy.value_shape, copy.value_stride)
    threads = _mode_bits(copy.thread_shape, copy.thread_stride)
    vector_bits = copy.value_shape[0].bit_length() - 1
    offsets = values[:vector_bits] + threads + values[vector_bits:]
    bases = [(offset // shape[1], offset % shape[1]) for offset in offsets]
    # Preserve each wave's contiguous copy window in the permuted LDS map.
    return _TLX.padded_shared_layout_encoding.with_bases([(512, padding)], bases, shape)


@lru_cache(maxsize=256)
def _dot_layouts(
    block_m: int,
    block_n: int,
    block_k: int,
    warps: int,
    vector_a: int,
    vector_b: int,
    vector_out: int,
    column_a: bool,
    column_b: bool,
    instruction: int,
    num_buffers: int,
    register_class: str,
    layout_mode: str,
    warp_pipeline: bool,
) -> dict:
    if tlx is None:
        raise RuntimeError("TLX is required for jagged projection")
    copy_a = (
        _column_layout(block_m, block_k, warps, vector_a)
        if column_a
        else _packing_layout(block_m, block_k, warps, vector_a)
    )
    copy_b = (
        _column_layout(block_k, block_n, warps, vector_b)
        if column_b
        else _packing_layout(block_k, block_n, warps, vector_b)
    )
    shared_a = _TLX.padded_shared_layout_encoding.with_identity_for(
        [(512, 16)], (block_m, block_k), order=(0, 1) if column_a else (1, 0)
    )
    shared_b = _TLX.padded_shared_layout_encoding.with_identity_for(
        [(512, 16)], (block_k, block_n), order=(0, 1) if column_b else (1, 0)
    )
    if layout_mode == "mfma":
        copy_a = _striped_copy_layout(
            block_m, block_k, warps, vector_a, column_a, False
        )
        copy_b = _striped_copy_layout(block_k, block_n, warps, vector_b, column_b, True)
        shared_a = _shared_copy_layout(copy_a, (block_m, block_k), 8)
        shared_b = _shared_copy_layout(
            copy_b, (block_k, block_n), 8 if column_b else max(8, block_n // 8)
        )
    return {
        "LAYOUT_A": copy_a,
        "LAYOUT_B": copy_b,
        "MMA_LAYOUT": _TLX.amd_mfma_layout(
            version=4,
            instr_shape=(instruction, instruction, 32 if instruction == 16 else 16),
            transposed=True,
            warps_per_cta=(1, warps) if layout_mode == "mfma" else (warps // 2, 2),
        ),
        "STORE_LAYOUT": _packing_layout(block_m, block_n, warps, vector_out),
        "LDS_LAYOUT_A": shared_a,
        "LDS_LAYOUT_B": shared_b,
        "NUM_BUFFERS": max(2, num_buffers),
        "ACCUMULATOR_REG": register_class,
        "WARP_PIPELINE": warp_pipeline,
        "VECTOR_A": vector_a,
        "VECTOR_B": vector_b,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": warps,
        "num_stages": 1,
        "matrix_instr_nonkdim": instruction,
    }


@triton.jit
def _mfma(
    a,
    b,
    accumulator,
    ACCUMULATOR_REG: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    STAGED: tl.constexpr,
):
    # JIT tensor parameters need their hardware encodings restored in the callee.
    a = _TLX.require_layout(
        a, _TLX.dot_operand_layout(0, MMA_LAYOUT, k_width=8), pin=False
    )
    b = _TLX.require_layout(
        b, _TLX.dot_operand_layout(1, MMA_LAYOUT, k_width=8), pin=False
    )
    accumulator = _TLX.require_layout(accumulator, MMA_LAYOUT, pin=False)
    if STAGED:
        with _TLX.warp_pipeline_stage("mfma", priority=0):
            result = _TLX.amd_scheduled_mfma(
                a,
                b,
                accumulator,
                accumulator_role="transient"
                if ACCUMULATOR_REG == "vgpr"
                else "persistent",
            )
    else:
        result = _TLX.amd_scheduled_mfma(
            a,
            b,
            accumulator,
            accumulator_role="transient" if ACCUMULATOR_REG == "vgpr" else "persistent",
        )
    return result


@triton.jit
def _remap_program(pid, total):
    aligned = (total // 64) * 64
    if pid >= aligned:
        return pid
    xcd = pid % 8
    local = pid // 8
    return (local // 8) * 64 + xcd * 8 + local % 8


@triton.jit
def _prefetch_segment(
    A,
    B,
    rows,
    cols,
    m_bound,
    n_bound,
    k_first,
    k_last,
    smem_a,
    smem_b,
    slot,
    STRIDE_AM: tl.constexpr,
    STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    STRIDE_BN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
    STAGED: tl.constexpr,
):
    reduction = k_first + tl.arange(0, BLOCK_K)
    # The launcher bounds each user's relative byte offsets below 2**31.
    # User bases remain scalar i64 pointers; copy offsets stay in i32 registers.
    a_offsets = (
        rows.to(tl.int32)[:, None] * STRIDE_AM
        + reduction.to(tl.int32)[None, :] * STRIDE_AK
    )
    b_offsets = (
        reduction.to(tl.int32)[:, None] * STRIDE_BK
        + cols.to(tl.int32)[None, :] * STRIDE_BN
    )
    mask_a = _TLX.require_layout(
        ((rows >= 0) & (rows < m_bound))[:, None] & (reduction < k_last)[None, :],
        LAYOUT_A,
    )
    mask_b = _TLX.require_layout(
        (reduction < k_last)[:, None] & (cols < n_bound)[None, :], LAYOUT_B
    )
    zero_a = _TLX.require_layout(
        tl.full((rows.shape[0], BLOCK_K), 0, _TLX.dtype_of(A)), LAYOUT_A
    )
    zero_b = _TLX.require_layout(
        tl.full((BLOCK_K, cols.shape[0]), 0, _TLX.dtype_of(B)), LAYOUT_B
    )
    destination_a = _TLX.local_view(smem_a, slot)
    destination_b = _TLX.local_view(smem_b, slot)
    a_offsets = _TLX.require_layout(a_offsets.to(tl.int32), LAYOUT_A)
    b_offsets = _TLX.require_layout(b_offsets.to(tl.int32), LAYOUT_B)
    if STAGED:
        with _TLX.warp_pipeline_stage("mem", priority=1):
            token_a = _TLX.buffer_load_to_local(
                destination_a, A, a_offsets, mask=mask_a, other=zero_a
            )
            _TLX.async_load_commit_group([token_a])
            token_b = _TLX.buffer_load_to_local(
                destination_b, B, b_offsets, mask=mask_b, other=zero_b
            )
            _TLX.async_load_commit_group([token_b])
    else:
        token_a = _TLX.buffer_load_to_local(
            destination_a, A, a_offsets, mask=mask_a, other=zero_a
        )
        _TLX.async_load_commit_group([token_a])
        token_b = _TLX.buffer_load_to_local(
            destination_b, B, b_offsets, mask=mask_b, other=zero_b
        )
        _TLX.async_load_commit_group([token_b])


@triton.jit
def _load_dot(smem_a, smem_b, slot, DOT_A: tl.constexpr, DOT_B: tl.constexpr):
    a = _TLX.local_load(_TLX.local_view(smem_a, slot), layout=DOT_A, relaxed=True)
    b = _TLX.local_load(_TLX.local_view(smem_b, slot), layout=DOT_B, relaxed=True)
    return a, b


@triton.jit
def _segment_dot_pipeline(
    A,
    B,
    rows,
    cols,
    m_bound,
    n_bound,
    k_first,
    k_last,
    STRIDE_AM: tl.constexpr,
    STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    STRIDE_BN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    LDS_LAYOUT_A: tl.constexpr,
    LDS_LAYOUT_B: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    ACCUMULATOR_REG: tl.constexpr,
    STAGED_LOOP: tl.constexpr,
):
    dot_a: tl.constexpr = _TLX.dot_operand_layout(0, MMA_LAYOUT, k_width=8)
    dot_b: tl.constexpr = _TLX.dot_operand_layout(1, MMA_LAYOUT, k_width=8)
    smem_a = _TLX.local_alloc(
        (BLOCK_M, BLOCK_K), _TLX.dtype_of(A), NUM_BUFFERS, layout=LDS_LAYOUT_A
    )
    smem_b = _TLX.local_alloc(
        (BLOCK_K, BLOCK_N), _TLX.dtype_of(B), NUM_BUFFERS, layout=LDS_LAYOUT_B
    )
    # Zero-filled extra tiles make short and empty sequences obey the same pipeline.
    tiles = tl.maximum(tl.cdiv(k_last - k_first, BLOCK_K), NUM_BUFFERS).to(tl.int32)
    for initial_buffer in tl.static_range(NUM_BUFFERS):
        _prefetch_segment(
            A,
            B,
            rows,
            cols,
            m_bound,
            n_bound,
            k_first + initial_buffer * BLOCK_K,
            k_last,
            smem_a,
            smem_b,
            initial_buffer,
            STRIDE_AM,
            STRIDE_AK,
            STRIDE_BK,
            STRIDE_BN,
            BLOCK_K,
            LAYOUT_A,
            LAYOUT_B,
            False,
        )
    _TLX.async_load_wait_group((NUM_BUFFERS - 1) * 2)
    a, b = _load_dot(smem_a, smem_b, 0, dot_a, dot_b)
    # Triton TR005: retain FP32 accumulation until the output store.
    accumulator = _TLX.zeros((BLOCK_M, BLOCK_N), tl.float32, layout=MMA_LAYOUT)
    for tile in tl.range(0, tiles - NUM_BUFFERS):
        current_buffer = tile % NUM_BUFFERS
        next_slot = (tile + 1) % NUM_BUFFERS
        accumulator = _mfma(a, b, accumulator, ACCUMULATOR_REG, MMA_LAYOUT, STAGED_LOOP)
        _TLX.async_load_wait_group((NUM_BUFFERS - 2) * 2)
        _prefetch_segment(
            A,
            B,
            rows,
            cols,
            m_bound,
            n_bound,
            k_first + (tile + NUM_BUFFERS) * BLOCK_K,
            k_last,
            smem_a,
            smem_b,
            current_buffer,
            STRIDE_AM,
            STRIDE_AK,
            STRIDE_BK,
            STRIDE_BN,
            BLOCK_K,
            LAYOUT_A,
            LAYOUT_B,
            STAGED_LOOP,
        )
        # Layout-conversion synchronization must remain outside pipeline stages.
        a, b = _load_dot(smem_a, smem_b, next_slot, dot_a, dot_b)
    accumulator = _mfma(a, b, accumulator, ACCUMULATOR_REG, MMA_LAYOUT, False)
    _TLX.async_load_wait_group(0)
    for last in tl.static_range(NUM_BUFFERS - 1):
        drain_buffer = (tiles - (NUM_BUFFERS - 1) + last) % NUM_BUFFERS
        a, b = _load_dot(smem_a, smem_b, drain_buffer, dot_a, dot_b)
        accumulator = _mfma(a, b, accumulator, ACCUMULATOR_REG, MMA_LAYOUT, False)
    return accumulator


@triton.jit
def _segment_dot_register(
    A,
    B,
    rows,
    cols,
    m_bound,
    n_bound,
    k_first,
    k_last,
    STRIDE_AM: tl.constexpr,
    STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    STRIDE_BN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    VECTOR_A: tl.constexpr,
    VECTOR_B: tl.constexpr,
    USE_BUFFER: tl.constexpr,
    ACCUMULATOR_REG: tl.constexpr,
):
    dot_a: tl.constexpr = _TLX.dot_operand_layout(0, MMA_LAYOUT, k_width=8)
    dot_b: tl.constexpr = _TLX.dot_operand_layout(1, MMA_LAYOUT, k_width=8)
    # Triton TR005: retain FP32 accumulation until the output store.
    accumulator = _TLX.zeros((BLOCK_M, BLOCK_N), tl.float32, layout=MMA_LAYOUT)
    for tile in tl.range(k_first, k_last, BLOCK_K):
        reduction = tile + tl.arange(0, BLOCK_K)
        a_offsets = (
            rows.to(tl.int64)[:, None] * STRIDE_AM
            + reduction.to(tl.int64)[None, :] * STRIDE_AK
        )
        b_offsets = (
            reduction.to(tl.int64)[:, None] * STRIDE_BK
            + cols.to(tl.int64)[None, :] * STRIDE_BN
        )
        mask_a = ((rows >= 0) & (rows < m_bound))[:, None] & (reduction < k_last)[
            None, :
        ]
        mask_b = (reduction < k_last)[:, None] & (cols < n_bound)[None, :]
        if USE_BUFFER:
            a_offsets = _TLX.require_layout(a_offsets.to(tl.int32), LAYOUT_A)
            b_offsets = _TLX.require_layout(b_offsets.to(tl.int32), LAYOUT_B)
            a = _TLX.buffer_load(A, a_offsets, mask_a, other=0, contiguity=VECTOR_A)
            b = _TLX.buffer_load(B, b_offsets, mask_b, other=0, contiguity=VECTOR_B)
        else:
            a = tl.load(_TLX.require_layout(A + a_offsets, LAYOUT_A), mask_a, other=0)
            b = tl.load(_TLX.require_layout(B + b_offsets, LAYOUT_B), mask_b, other=0)
        a = _TLX.require_layout(a, dot_a, pin=False)
        b = _TLX.require_layout(b, dot_b, pin=False)
        accumulator = _mfma(a, b, accumulator, ACCUMULATOR_REG, MMA_LAYOUT, False)
    return accumulator


@triton.jit
def _segment_dot(
    A,
    B,
    rows,
    cols,
    m_bound,
    n_bound,
    k_first,
    k_last,
    STRIDE_AM: tl.constexpr,
    STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    STRIDE_BN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    VECTOR_A: tl.constexpr,
    VECTOR_B: tl.constexpr,
    USE_BUFFER: tl.constexpr,
    PIPELINE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    LDS_LAYOUT_A: tl.constexpr,
    LDS_LAYOUT_B: tl.constexpr,
    ACCUMULATOR_REG: tl.constexpr,
    STAGED_LOOP: tl.constexpr,
):
    if PIPELINE:
        return _segment_dot_pipeline(
            A,
            B,
            rows,
            cols,
            m_bound,
            n_bound,
            k_first,
            k_last,
            STRIDE_AM,
            STRIDE_AK,
            STRIDE_BK,
            STRIDE_BN,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            LAYOUT_A,
            LAYOUT_B,
            MMA_LAYOUT,
            LDS_LAYOUT_A,
            LDS_LAYOUT_B,
            NUM_BUFFERS,
            ACCUMULATOR_REG,
            STAGED_LOOP,
        )
    return _segment_dot_register(
        A,
        B,
        rows,
        cols,
        m_bound,
        n_bound,
        k_first,
        k_last,
        STRIDE_AM,
        STRIDE_AK,
        STRIDE_BK,
        STRIDE_BN,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        LAYOUT_A,
        LAYOUT_B,
        MMA_LAYOUT,
        VECTOR_A,
        VECTOR_B,
        USE_BUFFER,
        ACCUMULATOR_REG,
    )


@triton.jit
def _store_matrix(
    Out,
    value,
    rows,
    cols,
    m_bound,
    n_bound,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    STORE_LAYOUT: tl.constexpr,
    USE_BUFFER: tl.constexpr,
):
    mask = ((rows >= 0) & (rows < m_bound))[:, None] & (cols < n_bound)[None, :]
    value = _TLX.require_layout(value, MMA_LAYOUT, pin=False)
    value = _TLX.require_layout(value.to(Out.dtype.element_ty), STORE_LAYOUT)
    mask = _TLX.require_layout(mask, STORE_LAYOUT)
    if USE_BUFFER:
        offsets = (
            rows.to(tl.int32)[:, None] * STRIDE_M
            + cols.to(tl.int32)[None, :] * STRIDE_N
        )
        offsets = _TLX.require_layout(offsets, STORE_LAYOUT)
        _TLX.buffer_store(value, Out, offsets, mask)
    else:
        offsets = (
            rows.to(tl.int64)[:, None] * STRIDE_M
            + cols.to(tl.int64)[None, :] * STRIDE_N
        )
        pointers = _TLX.require_layout(Out + offsets, STORE_LAYOUT)
        tl.store(pointers, value, mask)


@triton.jit(do_not_specialize=["B"])
def _projection_tile_offsets(
    Offsets,
    TileOffsets,
    B,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    users = tl.arange(0, BLOCK)
    starts = tl.load(Offsets + users, users < B, other=0)
    ends = tl.load(Offsets + users + 1, users < B, other=0)
    counts = tl.cdiv(ends - starts, BLOCK_M).to(tl.int32)
    prefixes = tl.cumsum(counts)
    # Beta lowers scans in a blocked layout; the plan scatter uses TLX copies.
    tl.store(TileOffsets + users + 1, prefixes, users < B)
    tl.store(TileOffsets, 0)


@triton.jit(do_not_specialize=["ROW_TILES"])
def _projection_tile_plan(
    TileOffsets,
    Plan,
    ROW_TILES,
    BLOCK: tl.constexpr,
    COPY_LAYOUT: tl.constexpr,
):
    user = tl.program_id(0)
    start = tl.load(TileOffsets + user)
    end = tl.load(TileOffsets + user + 1)
    rows = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    indices = _TLX.require_layout(start + rows, COPY_LAYOUT)
    codes = _TLX.require_layout(user * ROW_TILES + rows, COPY_LAYOUT)
    _TLX.buffer_store(codes, Plan, indices, indices < end)


@triton.jit(do_not_specialize=["B", "ROW_TILES", "GRID_TILES"])
def _jagged_projection(
    Offsets,
    X,
    Weight,
    Bias,
    Out,
    TileOffsets,
    Plan,
    B,
    K: tl.constexpr,
    N: tl.constexpr,
    ROW_TILES,
    GRID_TILES,
    COMPACT: tl.constexpr,
    STRIDE_X: tl.constexpr,
    STRIDE_WB: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    STRIDE_WN: tl.constexpr,
    STRIDE_BIAS: tl.constexpr,
    STRIDE_OUT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    STORE_LAYOUT: tl.constexpr,
    VECTOR_A: tl.constexpr,
    VECTOR_B: tl.constexpr,
    USE_BUFFER: tl.constexpr,
    PIPELINE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    LDS_LAYOUT_A: tl.constexpr,
    LDS_LAYOUT_B: tl.constexpr,
    ACCUMULATOR_REG: tl.constexpr,
    WARP_PIPELINE: tl.constexpr,
):
    pid = _remap_program(tl.program_id(0), GRID_TILES * tl.cdiv(N, BLOCK_N))
    column_tile = pid % tl.cdiv(N, BLOCK_N)
    row_tile = pid // tl.cdiv(N, BLOCK_N)
    if COMPACT:
        if row_tile >= tl.load(TileOffsets + B):
            return
        code = tl.load(Plan + row_tile)
        user = code // ROW_TILES
        first_row = (code % ROW_TILES) * BLOCK_M
    else:
        user = row_tile // ROW_TILES
        first_row = (row_tile % ROW_TILES) * BLOCK_M
    start = tl.load(Offsets + user).to(tl.int64)
    length = tl.load(Offsets + user + 1).to(tl.int64) - start
    if first_row >= length:
        return
    columns = column_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    if USE_BUFFER:
        length = length.to(tl.int32)
    rows = first_row + tl.arange(0, BLOCK_M)
    result = _segment_dot(
        X + start * STRIDE_X,
        Weight + user.to(tl.int64) * STRIDE_WB,
        rows,
        columns,
        length,
        N,
        0,
        K,
        STRIDE_X,
        1,
        STRIDE_WK,
        STRIDE_WN,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        LAYOUT_A,
        LAYOUT_B,
        MMA_LAYOUT,
        VECTOR_A,
        VECTOR_B,
        USE_BUFFER,
        PIPELINE,
        NUM_BUFFERS,
        LDS_LAYOUT_A,
        LDS_LAYOUT_B,
        ACCUMULATOR_REG,
        WARP_PIPELINE and (K + BLOCK_K - 1) // BLOCK_K >= NUM_BUFFERS + 2,
    )
    if HAS_BIAS:
        bias = tl.load(
            Bias + user.to(tl.int64) * STRIDE_BIAS + columns, columns < N, other=0
        )
        bias = _TLX.require_layout(
            bias.to(tl.float32), _TLX.slice_layout(MMA_LAYOUT, 0), pin=False
        )
        result += bias[None, :]
    _store_matrix(
        Out + start * STRIDE_OUT,
        result,
        rows,
        columns,
        length,
        N,
        STRIDE_OUT,
        1,
        MMA_LAYOUT,
        STORE_LAYOUT,
        USE_BUFFER,
    )


@triton.jit(do_not_specialize=["B", "STRIDE_PART"])
def _jagged_weight_grad(
    Offsets,
    X,
    DY,
    DW,
    B,
    K: tl.constexpr,
    N: tl.constexpr,
    PARTS: tl.constexpr,
    STRIDE_X: tl.constexpr,
    STRIDE_DY: tl.constexpr,
    STRIDE_PART,
    STRIDE_WB: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LAYOUT_A: tl.constexpr,
    LAYOUT_B: tl.constexpr,
    MMA_LAYOUT: tl.constexpr,
    STORE_LAYOUT: tl.constexpr,
    VECTOR_A: tl.constexpr,
    VECTOR_B: tl.constexpr,
    USE_BUFFER: tl.constexpr,
    PIPELINE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    LDS_LAYOUT_A: tl.constexpr,
    LDS_LAYOUT_B: tl.constexpr,
    ACCUMULATOR_REG: tl.constexpr,
    WARP_PIPELINE: tl.constexpr,
):
    matrix_tiles: tl.constexpr = tl.cdiv(K, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    pid = _remap_program(tl.program_id(0), B * matrix_tiles * PARTS)
    part = pid % PARTS
    tile = pid // PARTS
    user = tile // matrix_tiles
    tile %= matrix_tiles
    start = tl.load(Offsets + user).to(tl.int64)
    length = tl.load(Offsets + user + 1).to(tl.int64) - start
    part_rows = tl.cdiv(tl.cdiv(length, BLOCK_K), PARTS) * BLOCK_K
    begin = part * part_rows
    end = tl.minimum(begin + part_rows, length)
    group = tile // (4 * tl.cdiv(N, BLOCK_N))
    first_m = group * 4
    group_m = tl.minimum(tl.cdiv(K, BLOCK_M) - first_m, 4)
    local = tile % (4 * tl.cdiv(N, BLOCK_N))
    rows = (first_m + local % group_m) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = (local // group_m) * BLOCK_N + tl.arange(0, BLOCK_N)
    result = _segment_dot(
        X + start * STRIDE_X,
        DY + start * STRIDE_DY,
        rows,
        cols,
        K,
        N,
        begin,
        end,
        1,
        STRIDE_X,
        STRIDE_DY,
        1,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        LAYOUT_A,
        LAYOUT_B,
        MMA_LAYOUT,
        VECTOR_A,
        VECTOR_B,
        USE_BUFFER,
        PIPELINE,
        NUM_BUFFERS,
        LDS_LAYOUT_A,
        LDS_LAYOUT_B,
        ACCUMULATOR_REG,
        WARP_PIPELINE,
    )
    output = DW + part.to(tl.int64) * STRIDE_PART + user.to(tl.int64) * STRIDE_WB
    _store_matrix(
        output,
        result,
        rows,
        cols,
        K,
        N,
        STRIDE_WK,
        1,
        MMA_LAYOUT,
        STORE_LAYOUT,
        USE_BUFFER,
    )


@triton.jit
def _jagged_bias_grad(
    Offsets,
    DY,
    DB,
    N: tl.constexpr,
    STRIDE_DY: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    COPY_LAYOUT: tl.constexpr,
    VECTOR: tl.constexpr,
    USE_BUFFER: tl.constexpr,
):
    user = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    start = tl.load(Offsets + user).to(tl.int64)
    length = tl.load(Offsets + user + 1).to(tl.int64) - start
    pointer = DY + start * STRIDE_DY
    accumulator = tl.zeros((BLOCK_N,), tl.float32)
    for tile in tl.range(0, length, BLOCK_ROWS):
        rows = tile + tl.arange(0, BLOCK_ROWS)
        offsets = rows.to(tl.int64)[:, None] * STRIDE_DY + columns[None, :]
        mask = (rows < length)[:, None] & (columns < N)[None, :]
        if USE_BUFFER:
            offsets = _TLX.require_layout(offsets.to(tl.int32), COPY_LAYOUT)
            value = _TLX.buffer_load(pointer, offsets, mask, other=0, contiguity=VECTOR)
        else:
            value = tl.load(
                _TLX.require_layout(pointer + offsets, COPY_LAYOUT), mask, other=0
            )
        accumulator += tl.sum(value.to(tl.float32), 0)
    tl.store(
        DB + user.to(tl.int64) * N + columns,
        accumulator.to(DB.dtype.element_ty),
        columns < N,
    )


@triton.jit(do_not_specialize=["ELEMENTS"])
def _reduce_weight_grad(
    Partial,
    DW,
    ELEMENTS,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
    COPY_LAYOUT: tl.constexpr,
    VECTOR: tl.constexpr,
    USE_BUFFER: tl.constexpr,
):
    indices = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = indices < ELEMENTS
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for part in tl.static_range(PARTS):
        pointer = Partial + part * tl.full((), ELEMENTS, tl.int64)
        if USE_BUFFER:
            offsets = _TLX.require_layout(indices.to(tl.int32), COPY_LAYOUT)
            value = _TLX.buffer_load(pointer, offsets, mask, other=0, contiguity=VECTOR)
        else:
            value = tl.load(
                _TLX.require_layout(pointer + indices, COPY_LAYOUT), mask, other=0
            )
        accumulator += value
    tl.store(DW + indices, accumulator.to(DW.dtype.element_ty), mask)


def _projection_schedule(
    rows: int,
    batch: int,
    row_tiles: int,
    transpose: bool,
    output_features: int,
    num_warps: int | None,
    grid_mode: str,
    warp_pipeline: bool | None,
) -> tuple[int, str, bool]:
    narrow_dx = transpose and output_features <= 256 and rows >= 65536
    if num_warps is None:
        num_warps = 4 if transpose or rows < 65536 else 8
    if grid_mode == "auto":
        grid_mode = (
            "compact"
            if (not transpose or narrow_dx)
            and rows >= 65536
            and batch <= 4096
            and batch * row_tiles < 2**31
            else "rectangular"
        )
    if warp_pipeline is None:
        # All five trace steps favor the compact plain loop for narrow dX.
        warp_pipeline = transpose and not narrow_dx
    return num_warps, grid_mode, warp_pipeline


def launch_tlx_jagged_projection(
    offsets: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    *,
    max_seq_len: int,
    transpose: bool = False,
    block_m: int = 128,
    block_n: int = 256,
    block_k: int = 32,
    num_warps: int | None = None,
    instruction: int = 32,
    num_buffers: int = 3,
    register_class: str = "vgpr",
    layout_mode: str = "mfma",
    grid_mode: str = "auto",
    warp_pipeline: bool | None = None,
) -> None:
    row_tiles = triton.cdiv(max_seq_len, block_m)
    num_warps, grid_mode, warp_pipeline = _projection_schedule(
        x.shape[0],
        weight.shape[0],
        row_tiles,
        transpose,
        out.shape[1],
        num_warps,
        grid_mode,
        warp_pipeline,
    )
    if num_warps not in (4, 8):
        raise ValueError("Use four or eight wavefronts for jagged MFMA")
    if layout_mode not in ("identity", "mfma"):
        raise ValueError("Use identity or mfma copy layouts")
    if grid_mode not in ("rectangular", "compact"):
        raise ValueError("Use a rectangular or compact projection grid")
    if num_buffers not in (0, 2, 3) or register_class not in ("agpr", "vgpr"):
        raise ValueError("Use 0, 2 or 3 LDS buffers and agpr or vgpr accumulation")
    if out.numel() == 0:
        return
    reduction, n = x.shape[1], out.shape[1]
    stride_wk, stride_wn = (
        (weight.stride(2), weight.stride(1))
        if transpose
        else (weight.stride(1), weight.stride(2))
    )
    vector_a = _vector(reduction, x.stride(0))
    vector_b = _vector(
        reduction if transpose else n,
        weight.stride(0),
        stride_wn if transpose else stride_wk,
    )
    layouts = _dot_layouts(
        block_m,
        block_n,
        block_k,
        num_warps,
        vector_a,
        vector_b,
        _vector(n, out.stride(0)),
        False,
        transpose,
        instruction,
        num_buffers,
        register_class,
        layout_mode,
        warp_pipeline,
    )
    largest = max(
        (max_seq_len + block_m) * max(x.stride(0), out.stride(0)),
        reduction * stride_wk + n * stride_wn,
    )
    use_buffer = largest * x.element_size() < 2**31 and min(vector_a, vector_b) >= 2
    batch = weight.shape[0]
    grid_tiles = batch * row_tiles
    tile_offsets: torch.Tensor | int = 0
    plan: torch.Tensor | int = 0
    if grid_mode == "compact":
        if batch > 4096 or grid_tiles >= 2**31:
            raise ValueError(
                "The compact plan requires at most 4096 users and i32 tile codes"
            )
        grid_tiles = triton.cdiv(x.shape[0], block_m) + batch
        # Citrine C3: build the tile schedule on the input GPU each invocation.
        tile_offsets = torch.empty((batch + 1,), dtype=torch.int32, device=x.device)
        plan = torch.empty((grid_tiles,), dtype=torch.int32, device=x.device)
        block = triton.next_power_of_2(batch)
        kernel = cast(Any, _projection_tile_offsets)
        kernel[(1,)](
            offsets,
            tile_offsets,
            B=batch,
            BLOCK_M=block_m,
            BLOCK=block,
            num_warps=4,
        )
        block = min(256, triton.next_power_of_2(row_tiles))
        kernel = cast(Any, _projection_tile_plan)
        kernel[(batch, triton.cdiv(row_tiles, block))](
            tile_offsets,
            plan,
            ROW_TILES=row_tiles,
            BLOCK=block,
            COPY_LAYOUT=_packing_layout(1, block, 4, 1),
            num_warps=4,
        )
    kernel = cast(Any, _jagged_projection)
    kernel[(grid_tiles * triton.cdiv(n, block_n),)](
        offsets,
        x,
        weight,
        bias,
        out,
        tile_offsets,
        plan,
        B=batch,
        K=reduction,
        N=n,
        ROW_TILES=row_tiles,
        GRID_TILES=grid_tiles,
        COMPACT=grid_mode == "compact",
        STRIDE_X=x.stride(0),
        STRIDE_WB=weight.stride(0),
        STRIDE_WK=stride_wk,
        STRIDE_WN=stride_wn,
        STRIDE_BIAS=bias.stride(0) if bias is not None else 0,
        STRIDE_OUT=out.stride(0),
        HAS_BIAS=bias is not None,
        USE_BUFFER=use_buffer,
        PIPELINE=num_buffers > 0 and use_buffer,
        **layouts,
    )


def launch_tlx_jagged_weight_grad(
    offsets: torch.Tensor,
    x: torch.Tensor,
    dy: torch.Tensor,
    dw: torch.Tensor,
    db: torch.Tensor,
    *,
    max_seq_len: int,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
    instruction: int = 32,
    parts: int = 1,
    num_buffers: int = 2,
    register_class: str = "vgpr",
    layout_mode: str = "identity",
    warp_pipeline: bool = True,
) -> None:
    if num_warps not in (4, 8):
        raise ValueError("Use four or eight wavefronts for jagged MFMA")
    if layout_mode not in ("identity", "mfma"):
        raise ValueError("Use identity or mfma copy layouts")
    if num_buffers not in (0, 2, 3) or register_class not in ("agpr", "vgpr"):
        raise ValueError("Use 0, 2 or 3 LDS buffers and agpr or vgpr accumulation")
    if not dw.is_contiguous() or not db.is_contiguous():
        raise ValueError("Weight and bias gradient outputs must be contiguous")
    if parts not in (1, 2, 4, 8):
        raise ValueError("Weight gradient partitions must be 1, 2, 4 or 8")
    batch, k, n = dw.shape
    if batch == 0 or k == 0 or n == 0:
        return
    vector_a, vector_b = _vector(k, x.stride(0)), _vector(n, dy.stride(0))
    layouts = _dot_layouts(
        block_m,
        block_n,
        block_k,
        num_warps,
        vector_a,
        vector_b,
        _vector(n),
        True,
        False,
        instruction,
        num_buffers,
        register_class,
        layout_mode,
        warp_pipeline,
    )
    use_buffer = (max_seq_len + block_k) * max(
        x.stride(0), dy.stride(0)
    ) * x.element_size() < 2**31 and min(vector_a, vector_b) >= 2
    # Citrine C3: allocate FP32 partial gradients directly on the input device.
    partial = (
        torch.empty((parts, batch, k, n), dtype=torch.float32, device=x.device)
        if parts > 1
        else dw
    )
    matrix_tiles = triton.cdiv(k, block_m) * triton.cdiv(n, block_n)
    kernel = cast(Any, _jagged_weight_grad)
    kernel[(batch * matrix_tiles * parts,)](
        offsets,
        x,
        dy,
        partial,
        B=batch,
        K=k,
        N=n,
        PARTS=parts,
        STRIDE_X=x.stride(0),
        STRIDE_DY=dy.stride(0),
        STRIDE_PART=batch * k * n if parts > 1 else 0,
        STRIDE_WB=k * n,
        STRIDE_WK=n,
        USE_BUFFER=use_buffer,
        PIPELINE=num_buffers > 0 and use_buffer,
        **layouts,
    )
    if parts > 1:
        elements = dw.numel()
        vector = min(4, elements & -elements)
        layout = _TLX.layout(
            shape=((256,), (vector, 4096 // (256 * vector))),
            stride=((vector,), (1, 256 * vector)),
        )
        kernel = cast(Any, _reduce_weight_grad)
        kernel[(triton.cdiv(elements, 4096),)](
            partial,
            dw,
            ELEMENTS=elements,
            PARTS=parts,
            BLOCK=4096,
            COPY_LAYOUT=layout,
            VECTOR=vector,
            USE_BUFFER=elements * 4 < 2**31,
            num_warps=4,
        )
    kernel = cast(Any, _jagged_bias_grad)
    kernel[(batch, triton.cdiv(n, 128))](
        offsets,
        dy,
        db,
        N=n,
        STRIDE_DY=dy.stride(0),
        BLOCK_ROWS=64,
        BLOCK_N=128,
        COPY_LAYOUT=_packing_layout(64, 128, 4, vector_b),
        VECTOR=vector_b,
        USE_BUFFER=use_buffer,
        num_warps=4,
    )
