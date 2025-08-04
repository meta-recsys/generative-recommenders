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

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    # suppress type checking errors
    tlx = None

    HAS_TLX = False

from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)

from triton.language.extra.libdevice import (  # @manual=//triton:triton
    fast_dividef,
    fast_expf,
)
from triton.tools.tensor_descriptor import TensorDescriptor

try:
    from generative_recommenders.ops.triton.fb.triton_attention_utils import acc_dq
except ImportError:
    from generative_recommenders.ops.triton.triton_attention_utils import acc_dq


def _host_descriptor_pre_hook(nargs):
    if not isinstance(nargs["Q"], TensorDescriptor):
        return
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_D_Q = nargs["BLOCK_D_Q"]
    BLOCK_D_V = nargs["BLOCK_D_V"]
    if "USE_TLX" in nargs and nargs["USE_TLX"]:
        BLOCK_M = BLOCK_M // nargs["NUM_MMA_GROUPS"]
    nargs["Q"].block_shape = [BLOCK_M, BLOCK_D_Q]
    nargs["V"].block_shape = [BLOCK_N, BLOCK_D_V]
    nargs["K"].block_shape = [BLOCK_N, BLOCK_D_Q]


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
        for BLOCK_M in [32, 64, 128]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "waves_per_eu": 0,
                                        "kpack": 2,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                )
                            )
    else:
        configs = [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=4,
                num_warps=4,
                pre_hook=_host_descriptor_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
                pre_hook=_host_descriptor_pre_hook,
            ),
        ]

        # Add 'USE_TLX' : False, 'NUM_BUFFERS': 1, 'NUM_MMA_WARPS_PER_GROUP': 1, 'NUM_MMA_GROUPS': 1 to non-TLX configs
        for config in configs:
            if not config.kwargs.get("USE_TLX", False):
                config.kwargs["USE_TLX"] = False
                config.kwargs["NUM_BUFFERS"] = 1
                config.kwargs["NUM_MMA_WARPS_PER_GROUP"] = 1
                config.kwargs["NUM_MMA_GROUPS"] = 1

        # Add TLX configs if TLX is available
        if HAS_TLX:
            if torch.cuda.get_device_capability()[0] == 9:
                # H100 configs
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": 128,
                            "BLOCK_N": 64,
                            "USE_TLX": True,
                            "NUM_BUFFERS": 2,
                            "NUM_MMA_WARPS_PER_GROUP": 4,
                            "NUM_MMA_GROUPS": 2,
                        },
                        num_stages=0,
                        num_warps=4,
                        pre_hook=_host_descriptor_pre_hook,
                    ),
                )

    return configs


@triton.jit
def _hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    q,
    K,
    V,
    K_block_ptr,
    V_block_ptr,
    offset_kh,
    offset_vh,
    seq_start,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = None
    qk = None
    if ENABLE_TMA:
        k = K.load(
            [(seq_start + start_n).to(tl.int32), offset_kh.to(tl.int32)],
        )
        # tma can only be loaded in one order, use trans afterwards
        qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask or (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or (
            offs_m[:, None] == 0 and offs_n[None, :] < max_ids
        )
    scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
    v = None
    if ENABLE_TMA:
        v = V.load(
            [(seq_start + start_n).to(tl.int32), offset_vh.to(tl.int32)],
        )
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    H,
    DimQ,
    DimV,
    workspace_ptr,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    off_z,
    off_h,
    pid,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        Q_block_ptr = None
        K_block_ptr = None
        V_block_ptr = None
        if not ENABLE_TMA:
            if IS_DELTA_Q:
                Q_block_ptr = tl.make_block_ptr(
                    base=Q + off_h * stride_qh + off_z * DeltaSize * stride_qm,
                    shape=(DeltaSize, BLOCK_D_Q),
                    strides=(stride_qm, 1),
                    offsets=(start_m_delta, 0),
                    block_shape=(BLOCK_M, BLOCK_D_Q),
                    order=(1, 0),
                )
            else:
                Q_block_ptr = tl.make_block_ptr(
                    base=Q + off_h * stride_qh + seq_start * stride_qm,
                    shape=(seq_len, BLOCK_D_Q),
                    strides=(stride_qm, 1),
                    offsets=(start_m, 0),
                    block_shape=(BLOCK_M, BLOCK_D_Q),
                    order=(1, 0),
                )
            q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

            K_block_ptr = tl.make_block_ptr(
                base=K + off_h * stride_kh + seq_start * stride_kn,
                shape=(BLOCK_D_Q, seq_len),
                strides=(1, stride_kn),
                offsets=(0, 0),
                block_shape=(BLOCK_D_Q, BLOCK_N),
                order=(0, 1),
            )
            V_block_ptr = tl.make_block_ptr(
                base=V + off_h * stride_vh + seq_start * stride_vn,
                shape=(seq_len, BLOCK_D_V),
                strides=(stride_vn, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_N, BLOCK_D_V),
                order=(1, 0),
            )
        else:
            if IS_DELTA_Q:
                q = Q.load(
                    [
                        (off_z * DeltaSize + start_m_delta).to(tl.int32),
                        (off_h * stride_qh).to(tl.int32),
                    ]
                )
            else:
                q = Q.load(
                    [
                        (seq_start + start_m).to(tl.int32),
                        (off_h * stride_qh).to(tl.int32),
                    ]
                )

        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if HAS_MULTIPLE_TARGETS:
            uih_end = seq_len - n_targets
        else:
            uih_end = seq_len
        if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
            # uih_end must be larger than start_m
            low = 0
            high = seq_len
        else:
            low = 0
            high = start_m + BLOCK_M
            if HAS_MAX_ATTN_LEN:
                if start_m > uih_end:
                    low = uih_end - max_attn_len
                else:
                    low = start_m - max_attn_len
                if HAS_CONTEXTUAL_SEQ_LEN:
                    low = low if low > contextual_seq_len else 0
                else:
                    low = low if low > 0 else 0
            if HAS_MULTIPLE_TARGETS:
                uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len - n_targets

        if low > 0:
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, low))
                V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        end_n = low
        for start_n in range(low, high, BLOCK_N):
            acc += _hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n + start_n,
                q=q,
                K=K,
                V=V,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                offset_kh=off_h * stride_kh,
                offset_vh=off_h * stride_vh,
                seq_start=seq_start,
                n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_N=BLOCK_N,
                ENABLE_TMA=ENABLE_TMA,
            )
            if not ENABLE_TMA:
                K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            end_n += BLOCK_N

        if HAS_MULTIPLE_TARGETS:
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - end_n).to(tl.int32)
                if not ENABLE_TMA:
                    K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                    V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(
                    low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_delta,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_delta,
                        q=q,
                        K=K,
                        V=V,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        offset_kh=off_h * stride_kh,
                        offset_vh=off_h * stride_vh,
                        seq_start=seq_start,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        contextual_seq_len=contextual_seq_len,
                        max_attn_len=max_attn_len,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_D_Q=BLOCK_D_Q,
                        BLOCK_D_V=BLOCK_D_V,
                        BLOCK_N=BLOCK_N,
                        ENABLE_TMA=ENABLE_TMA,
                    )
                    if not ENABLE_TMA:
                        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        # Don't use TMA in Jagged case since we don't want to overwrite
        # the output of another sequence
        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.jit
def _hstu_attn_fwd_compute_main_loop_tlx(  # noqa C901
    low,
    high,
    seq_len,
    offs_m,
    offs_n,
    acc,
    q_tiles,
    k_tiles,
    v_tiles,
    q_fulls,
    k_fulls,
    v_fulls,
    k_empties,
    v_empties,
    v_dtype,
    n_targets,
    alpha,
    end_n,
    loop_trip_cnt,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    cid: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    WAIT_FOR_Q: tl.constexpr,
):
    if WAIT_FOR_Q:
        # wait for the Q buffer to be populated by the producer
        q_full = tlx.local_view(q_fulls, cid)
        tlx.barrier_wait(q_full, 0)

    q_tile = tlx.local_view(q_tiles, cid)

    for start in tl.range(low + BLOCK_N, high, BLOCK_N, num_stages=0):
        buf_id = loop_trip_cnt % NUM_BUFFERS
        # buffers in a row share the same phase
        kv_phase = (loop_trip_cnt // NUM_BUFFERS) % 2

        start_n = tl.multiple_of(start, BLOCK_N)
        offs_n_start = offs_n
        offs_n = offs_n_start + start_n

        # wait for the K buffer to be populated by the producer
        k_full = tlx.local_view(k_fulls, buf_id)
        tlx.barrier_wait(k_full, kv_phase)
        k_tile = tlx.local_view(k_tiles, buf_id)

        # tma can only be loaded in one order, use trans afterwards
        k_tile = tlx.local_trans(k_tile)
        # second
        qk = tlx.async_dot(q_tile, k_tile)
        # wait for the MMA using to complete
        qk = tlx.async_dot_wait(0, qk)
        # release the K buffer
        k_empty = tlx.local_view(k_empties, buf_id)
        tlx.barrier_arrive(k_empty, 1)

        qk = qk * alpha

        invalid_mask = offs_m[:, None] == offs_n[None, :]
        max_ids = seq_len
        if HAS_MULTIPLE_TARGETS:
            max_ids = max_ids - n_targets
            offs_m = tl.where(
                offs_m < max_ids,
                offs_m,
                max_ids,
            )
            offs_n = tl.where(
                offs_n < max_ids,
                offs_n,
                max_ids,
            )
        offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
        invalid_mask = invalid_mask or (offs_m_minus_n > 0)
        if HAS_MAX_ATTN_LEN:
            invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask = invalid_mask or (
                offs_m[:, None] == 0 and offs_n[None, :] < max_ids
            )
        scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
        silu = silu.to(v_dtype)

        # wait for the V buffer to be populated by the producer
        v_full = tlx.local_view(v_fulls, buf_id)
        tlx.barrier_wait(v_full, kv_phase)
        v_tile = tlx.local_view(v_tiles, buf_id)
        acc = tlx.async_dot(silu, v_tile, acc)
        # wait for the MMA using to complete
        acc = tlx.async_dot_wait(0, acc)
        # release the V buffer
        v_empty = tlx.local_view(v_empties, buf_id)
        tlx.barrier_arrive(v_empty, 1)

        end_n += BLOCK_N

        # increment loop trip counts
        loop_trip_cnt += 1

    return acc, end_n, loop_trip_cnt


@triton.jit
def _hstu_attn_fwd_compute_main_loop_tlx_pipelined(  # noqa C901
    low,
    high,
    seq_len,
    offs_m,
    offs_n,
    acc,
    q_tiles,
    k_tiles,
    v_tiles,
    q_fulls,
    k_fulls,
    v_fulls,
    k_empties,
    v_empties,
    v_dtype,
    n_targets,
    alpha,
    end_n,
    loop_trip_cnt,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    cid: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    WAIT_FOR_Q: tl.constexpr,
):
    if WAIT_FOR_Q:
        # wait for the Q buffer to be populated by the producer
        q_full = tlx.local_view(q_fulls, cid)
        tlx.barrier_wait(q_full, 0)
    q_tile = tlx.local_view(q_tiles, cid)

    # wait for the K buffer to be populated by the producer
    k_buf_id = loop_trip_cnt % NUM_BUFFERS
    # buffers in a row share the same phase
    k_phase = (loop_trip_cnt // NUM_BUFFERS) % 2

    k_full = tlx.local_view(k_fulls, k_buf_id)
    tlx.barrier_wait(k_full, k_phase)
    k_tile = tlx.local_view(k_tiles, k_buf_id)

    # tma can only be loaded in one order, use trans afterwards
    k_tile = tlx.local_trans(k_tile)

    # Pingpong
    if cid == 0:
        # Consumer 0 waits for Consumer 1 to reach synchronization point at barrier 9.
        tlx.named_barrier_wait(9, 256)
    else:
        # Consumer 1 signals its arrival at barrier 9.
        tlx.named_barrier_arrive(9, 256)
        # Then waits at barrier 10 until Consumer 0 finishes issuing its async_dot.
        tlx.named_barrier_wait(10, 256)

    qk = tlx.async_dot(q_tile, k_tile)

    if cid == 0:
        # After issuing async_dot, Consumer 0 signals barrier 10 to unblock Consumer 1.
        tlx.named_barrier_arrive(10, 256)

    # wait for the MMA using to complete
    qk = tlx.async_dot_wait(0, qk)
    # release the K buffer
    k_empty = tlx.local_view(k_empties, k_buf_id)
    tlx.barrier_arrive(k_empty, 1)

    qk = qk * alpha

    start_n = tl.multiple_of(low, BLOCK_N)
    offs_n_start = offs_n
    offs_n = offs_n_start + start_n

    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    invalid_mask = invalid_mask or (offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or (
            offs_m[:, None] == 0 and offs_n[None, :] < max_ids
        )
    scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
    silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
    silu = silu.to(v_dtype)

    loop_trip_cnt += 1

    for start in tl.range(low + BLOCK_N, high, BLOCK_N, num_stages=0):
        start_n = tl.multiple_of(start, BLOCK_N)
        offs_n = offs_n_start + start_n

        k_buf_id = loop_trip_cnt % NUM_BUFFERS
        # buffers in a row share the same phase
        k_phase = k_phase ^ (k_buf_id == 0)

        # wait for the K buffer to be populated by the producer
        k_full = tlx.local_view(k_fulls, k_buf_id)
        tlx.barrier_wait(k_full, k_phase)
        k_tile = tlx.local_view(k_tiles, k_buf_id)

        # tma can only be loaded in one order, use trans afterwards
        k_tile = tlx.local_trans(k_tile)

        qk = tlx.async_dot(q_tile, k_tile)
        # wait for the MMA using to complete
        prev_silu = silu

        v_buf_id = (loop_trip_cnt - 1) % NUM_BUFFERS
        # v_phase = v_phase ^ (v_buf_id == 0)
        v_phase = ((loop_trip_cnt - 1) // NUM_BUFFERS) % 2
        v_full = tlx.local_view(v_fulls, v_buf_id)
        tlx.barrier_wait(v_full, v_phase)
        v_tile = tlx.local_view(v_tiles, v_buf_id)
        acc = tlx.async_dot(prev_silu, v_tile, acc)
        qk = tlx.async_dot_wait(1, qk)

        # release the K buffer
        k_empty = tlx.local_view(k_empties, k_buf_id)
        tlx.barrier_arrive(k_empty, 1)

        qk = qk * alpha
        invalid_mask = offs_m[:, None] == offs_n[None, :]
        max_ids = seq_len
        if HAS_MULTIPLE_TARGETS:
            max_ids = max_ids - n_targets
            offs_m = tl.where(
                offs_m < max_ids,
                offs_m,
                max_ids,
            )
            offs_n = tl.where(
                offs_n < max_ids,
                offs_n,
                max_ids,
            )
        offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
        invalid_mask = invalid_mask or (offs_m_minus_n > 0)
        if HAS_MAX_ATTN_LEN:
            invalid_mask = invalid_mask and offs_m_minus_n <= max_attn_len
        if HAS_CONTEXTUAL_SEQ_LEN:
            invalid_mask = invalid_mask or (
                offs_m[:, None] == 0 and offs_n[None, :] < max_ids
            )
        scale = tl.where(invalid_mask, (1.0 / MAX_SEQ_LEN), 0.0)
        silu = fast_dividef(qk, 1.0 + fast_expf(-qk)) * scale
        silu = silu.to(v_dtype)

        acc = tlx.async_dot_wait(0, acc)
        # release the V buffer
        v_empty = tlx.local_view(v_empties, v_buf_id)
        tlx.barrier_arrive(v_empty, 1)

        end_n += BLOCK_N

        # increment loop trip counts
        loop_trip_cnt += 1
        # v_buf_id = loop_trip_cnt % NUM_BUFFERS
        # v_phase = (loop_trip_cnt // NUM_BUFFERS) % 2

    # wait for the V buffer to be populated by the producer
    v_buf_id = (loop_trip_cnt - 1) % NUM_BUFFERS
    v_phase = ((loop_trip_cnt - 1) // NUM_BUFFERS) % 2
    v_full = tlx.local_view(v_fulls, v_buf_id)
    # tlx.barrier_wait(v_full, v_buf_id)
    v_tile = tlx.local_view(v_tiles, v_buf_id)
    tlx.barrier_wait(v_full, v_phase)
    acc = tlx.async_dot(silu, v_tile, acc)
    acc = tlx.async_dot_wait(0, acc)
    # release the V buffer
    v_empty = tlx.local_view(v_empties, v_buf_id)
    tlx.barrier_arrive(v_empty, 1)

    return acc, end_n, loop_trip_cnt


@triton.jit
def _hstu_attn_fwd_load_K_or_V(
    K,
    k_tiles,
    k_empties,
    k_fulls,
    buf_id,
    k_phase,
    start_n,
    seq_start,
    offset_kh,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # wait for the K buffer to be released by the consumer
    k_empty = tlx.local_view(k_empties, buf_id)
    tlx.barrier_wait(k_empty, k_phase)
    # load K
    k_full = tlx.local_view(k_fulls, buf_id)
    k_tile = tlx.local_view(k_tiles, buf_id)
    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * BLOCK_D_Q)  # float16
    tlx.async_descriptor_load(
        K,
        k_tile,
        [(seq_start + start_n).to(tl.int32), offset_kh.to(tl.int32)],
        k_full,
    )


@triton.jit
def _hstu_attn_fwd_load_Q(
    Q,
    q_tiles,
    q_fulls,
    cid,
    off_z,
    off_h,
    stride_qh,
    start_m,
    seq_start,
    DeltaSize,
    IS_DELTA_Q: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    q_full = tlx.local_view(q_fulls, cid)
    tlx.barrier_expect_bytes(q_full, 2 * BLOCK_M * BLOCK_D_Q)  # float16
    q_tile = tlx.local_view(q_tiles, cid)
    seq_offset = start_m + cid * BLOCK_M
    if IS_DELTA_Q:
        tlx.async_descriptor_load(
            Q,
            q_tile,
            [
                (off_z * DeltaSize + start_m).to(tl.int32),
                (off_h * stride_qh).to(tl.int32),
            ],
            q_full,
        )
    else:
        tlx.async_descriptor_load(
            Q,
            q_tile,
            [
                (seq_start + seq_offset).to(tl.int32),
                (off_h * stride_qh).to(tl.int32),
            ],
            q_full,
        )


@triton.jit
def _hstu_attn_fwd_caculate_range(
    seq_len,
    start_m,
    n_targets,
    contextual_seq_len,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if HAS_MULTIPLE_TARGETS:
        uih_end = seq_len - n_targets
    else:
        uih_end = seq_len

    if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
        # uih_end must be larger than start_m
        low = 0
        high = seq_len
    else:
        low = 0
        high = start_m + BLOCK_M
        if HAS_MAX_ATTN_LEN:
            if start_m > uih_end:
                low = uih_end - max_attn_len
            else:
                low = start_m - max_attn_len
            if HAS_CONTEXTUAL_SEQ_LEN:
                low = low if low > contextual_seq_len else 0
            else:
                low = low if low > 0 else 0
        if HAS_MULTIPLE_TARGETS:
            uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
            if uih_end < start_m:
                high = seq_len - n_targets

    return low, high, uih_end


@triton.jit
def _hstu_attn_fwd_load_Q_K_V(
    Q,
    K,
    V,
    q_tiles,
    k_tiles,
    v_tiles,
    q_fulls,
    k_fulls,
    v_fulls,
    k_empties,
    v_empties,
    stride_qh,
    stride_kh,
    stride_vh,
    contextual_seq_len,
    max_attn_len,
    DeltaSize,
    off_z,
    off_h,
    start_m,
    seq_start,
    seq_len,
    n_targets,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    # load q: it will stay in SRAM throughout
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    _hstu_attn_fwd_load_Q(
        Q=Q,
        q_tiles=q_tiles,
        q_fulls=q_fulls,
        cid=0,
        off_z=off_z,
        off_h=off_h,
        stride_qh=stride_qh,
        start_m=start_m,
        seq_start=seq_start,
        DeltaSize=DeltaSize,
        IS_DELTA_Q=IS_DELTA_Q,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_M=BLOCK_M_SPLIT,
    )

    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    offset_kh = off_h * stride_kh
    offset_vh = off_h * stride_vh

    low, high, uih_end = _hstu_attn_fwd_caculate_range(
        seq_len,
        start_m,
        n_targets,
        contextual_seq_len,
        max_attn_len,
        HAS_MULTIPLE_TARGETS,
        HAS_CONTEXTUAL_SEQ_LEN,
        HAS_MAX_ATTN_LEN,
        BLOCK_M,
        BLOCK_N,
    )

    kv_phase = 0
    loop_trip_cnt = 0

    # pyre-ignore[58]
    buf_id = loop_trip_cnt % NUM_BUFFERS
    # buffers in a row share the same phase
    kv_phase = kv_phase ^ (buf_id == 0)

    start_n = tl.multiple_of(low, BLOCK_N)

    _hstu_attn_fwd_load_K_or_V(
        K,
        k_tiles,
        k_empties,
        k_fulls,
        buf_id,
        kv_phase,
        start_n,
        seq_start,
        offset_kh,
        BLOCK_D_Q,
        BLOCK_N,
    )

    for cid in tl.range(1, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS - 1):
        _hstu_attn_fwd_load_Q(
            Q,
            q_tiles,
            q_fulls,
            cid,
            off_z,
            off_h,
            stride_qh,
            start_m,
            seq_start,
            DeltaSize,
            IS_DELTA_Q,
            BLOCK_D_Q,
            BLOCK_M_SPLIT,
        )

    _hstu_attn_fwd_load_K_or_V(
        V,
        v_tiles,
        v_empties,
        v_fulls,
        buf_id,
        kv_phase,
        start_n,
        seq_start,
        offset_vh,
        BLOCK_D_V,
        BLOCK_N,
    )

    loop_trip_cnt += 1

    for start in range(low + BLOCK_N, high, BLOCK_N):
        # pyre-ignore[58]
        buf_id = loop_trip_cnt % NUM_BUFFERS
        # buffers in a row share the same phase
        kv_phase = kv_phase ^ (buf_id == 0)

        start_n = tl.multiple_of(start, BLOCK_N)

        _hstu_attn_fwd_load_K_or_V(
            K,
            k_tiles,
            k_empties,
            k_fulls,
            buf_id,
            kv_phase,
            start_n,
            seq_start,
            offset_kh,
            BLOCK_D_Q,
            BLOCK_N,
        )

        _hstu_attn_fwd_load_K_or_V(
            V,
            v_tiles,
            v_empties,
            v_fulls,
            buf_id,
            kv_phase,
            start_n,
            seq_start,
            offset_vh,
            BLOCK_D_V,
            BLOCK_N,
        )

        # increment loop trip counts
        loop_trip_cnt += 1

    # pyre-ignore[61]
    if uih_end < start_m:
        low_delta = start_m
        high_delta = start_m + BLOCK_M
        for start_delta in tl.range(low_delta, high_delta, BLOCK_N, num_stages=0):
            # pyre-ignore[58]
            buf_id = loop_trip_cnt % NUM_BUFFERS
            # buffers in a row share the same phase
            kv_phase = kv_phase ^ (buf_id == 0)

            start_n = tl.multiple_of(start_delta, BLOCK_N)

            _hstu_attn_fwd_load_K_or_V(
                K,
                k_tiles,
                k_empties,
                k_fulls,
                buf_id,
                kv_phase,
                start_n,
                seq_start,
                offset_kh,
                BLOCK_D_Q,
                BLOCK_N,
            )

            _hstu_attn_fwd_load_K_or_V(
                V,
                v_tiles,
                v_empties,
                v_fulls,
                buf_id,
                kv_phase,
                start_n,
                seq_start,
                offset_vh,
                BLOCK_D_V,
                BLOCK_N,
            )

            # increment loop trip counts
            loop_trip_cnt += 1


@triton.jit
def _hstu_attn_fwd_compute_tlx(  # noqa C901
    Q,
    K,
    V,
    H,
    DimQ,
    DimV,
    seq_offsets,
    num_targets,
    Out,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    off_z,
    off_h,
    pid,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)

    if IS_DELTA_Q:
        start_m = pid * BLOCK_M
        start_m = (start_m + seq_len - DeltaSize).to(tl.int32)
    else:
        start_m = pid * BLOCK_M

    if start_m >= seq_len:
        return

    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None

    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS
    # allocate buffers
    q_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_D_Q), tlx.dtype_of(Q), NUM_MMA_GROUPS
    )
    k_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D_Q), tlx.dtype_of(K), NUM_BUFFERS)
    v_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), NUM_BUFFERS)

    # allocate barriers
    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS, arrive_count=1)
    k_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    k_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    v_empties = tlx.alloc_barriers(
        num_barriers=NUM_BUFFERS, arrive_count=NUM_MMA_GROUPS
    )
    v_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        # producer group
        with tlx.async_task("default"):
            _hstu_attn_fwd_load_Q_K_V(
                Q=Q,
                K=K,
                V=V,
                q_tiles=q_tiles,
                k_tiles=k_tiles,
                v_tiles=v_tiles,
                q_fulls=q_fulls,
                k_fulls=k_fulls,
                v_fulls=v_fulls,
                k_empties=k_empties,
                v_empties=v_empties,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                DeltaSize=DeltaSize,
                off_z=off_z,
                off_h=off_h,
                start_m=start_m,
                seq_start=seq_start,
                seq_len=seq_len,
                n_targets=n_targets,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                IS_DELTA_Q=IS_DELTA_Q,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                NUM_BUFFERS=NUM_BUFFERS,
                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            )

        # consumer groups
        with tlx.async_task(
            num_warps=NUM_MMA_WARPS_PER_GROUP, registers=232, replicate=NUM_MMA_GROUPS
        ):
            cid = tlx.async_task_replica_id()
            acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_D_V], dtype=tl.float32)
            # initialize offsets
            offs_m = start_m + tl.arange(0, BLOCK_M_SPLIT) + cid * BLOCK_M_SPLIT
            offs_n = tl.arange(0, BLOCK_N)

            low, high, uih_end = _hstu_attn_fwd_caculate_range(
                seq_len,
                start_m,
                n_targets,
                contextual_seq_len,
                max_attn_len,
                HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN,
                BLOCK_M,
                BLOCK_N,
            )

            end_n = low
            loop_trip_cnt = 0

            acc, end_n, loop_trip_cnt = _hstu_attn_fwd_compute_main_loop_tlx_pipelined(
                low=low,
                high=high,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n,
                acc=acc,
                q_tiles=q_tiles,
                k_tiles=k_tiles,
                v_tiles=v_tiles,
                q_fulls=q_fulls,
                k_fulls=k_fulls,
                v_fulls=v_fulls,
                k_empties=k_empties,
                v_empties=v_empties,
                v_dtype=tlx.dtype_of(V),
                n_targets=n_targets,
                alpha=alpha,
                end_n=end_n,
                loop_trip_cnt=loop_trip_cnt,
                max_attn_len=max_attn_len,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                cid=cid,
                BLOCK_N=BLOCK_N,
                NUM_BUFFERS=NUM_BUFFERS,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                WAIT_FOR_Q=1,
            )

            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                acc, end_n, loop_trip_cnt = _hstu_attn_fwd_compute_main_loop_tlx(
                    low=low_delta,
                    high=high_delta,
                    seq_len=seq_len,
                    offs_m=offs_m,
                    offs_n=offs_n,
                    acc=acc,
                    q_tiles=q_tiles,
                    k_tiles=k_tiles,
                    v_tiles=v_tiles,
                    q_fulls=q_fulls,
                    k_fulls=k_fulls,
                    v_fulls=v_fulls,
                    k_empties=k_empties,
                    v_empties=v_empties,
                    v_dtype=tlx.dtype_of(V),
                    n_targets=n_targets,
                    alpha=alpha,
                    end_n=end_n,
                    loop_trip_cnt=loop_trip_cnt,
                    max_attn_len=max_attn_len,
                    HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                    HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                    HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                    cid=cid,
                    BLOCK_N=BLOCK_N,
                    NUM_BUFFERS=NUM_BUFFERS,
                    MAX_SEQ_LEN=MAX_SEQ_LEN,
                    WAIT_FOR_Q=0,
                )

            # Don't use TMA in Jagged case since we don't want to overwrite
            # the output of another sequence
            if IS_DELTA_Q:
                start_m_delta = pid * BLOCK_M + cid * BLOCK_M_SPLIT
                offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M_SPLIT)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
                out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
                tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
            else:
                # rematerialize offsets to save registers
                start_m = pid * BLOCK_M + cid * BLOCK_M_SPLIT
                offs_m = start_m + tl.arange(0, BLOCK_M_SPLIT)
                offs_v_d = tl.arange(0, BLOCK_D_V)
                off_o = Out + seq_start * stride_om + off_h * stride_oh
                out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
                tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_TLX: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    if USE_TLX:
        _hstu_attn_fwd_compute_tlx(
            Q=Q,
            K=K,
            V=V,
            H=H,
            DimQ=DimQ,
            DimV=DimV,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_BUFFERS=NUM_BUFFERS,
            NUM_MMA_WARPS_PER_GROUP=NUM_MMA_WARPS_PER_GROUP,
            NUM_MMA_GROUPS=NUM_MMA_GROUPS,
        )
    else:
        _hstu_attn_fwd_compute(
            Q=Q,
            K=K,
            V=V,
            H=H,
            DimQ=DimQ,
            DimV=DimV,
            workspace_ptr=workspace_ptr,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qm=stride_qm,
            stride_qh=stride_qh,
            stride_kn=stride_kn,
            stride_kh=stride_kh,
            stride_vn=stride_vn,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ENABLE_TMA=ENABLE_TMA,
            TMA_DESC_SIZE=TMA_DESC_SIZE,
        )


@triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _hstu_attn_fwd_persistent(  # noqa C901
    Q,
    K,
    V,
    workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    contextual_seq_len,
    max_attn_len,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_TLX: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,  #
    NUM_MMA_WARPS_PER_GROUP: tl.constexpr,  #
    NUM_MMA_GROUPS: tl.constexpr,  #
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
):
    n_tile_num = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    for _ in range(0, tiles_per_sm):
        pid = (total_tiles - tile_idx - 1) // (Z * H)
        off_hz = (total_tiles - tile_idx - 1) % (Z * H)
        off_z = off_hz // H
        off_h = off_hz % H
        _hstu_attn_fwd_compute(
            Q=Q,
            K=K,
            V=V,
            H=H,
            DimQ=DimQ,
            DimV=DimV,
            workspace_ptr=workspace_ptr,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qm=stride_qm,
            stride_qh=stride_qh,
            stride_kn=stride_kn,
            stride_kh=stride_kh,
            stride_vn=stride_vn,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ENABLE_TMA=ENABLE_TMA,
            TMA_DESC_SIZE=TMA_DESC_SIZE,
        )
        tile_idx += num_progs


@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    do_ptrs,
    device_desc_q,
    device_desc_do,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    max_ids,
    contextual_seq_len,
    max_attn_len,
    LOCK,
    off_h,
    stride_qh,
    stride_doh,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(
            pos_offs_m > 0,
            pos_offs_m,
            0,
        )
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(
            pos_offs_m < max_ids,
            pos_offs_m,
            max_ids,
        )
    if ENABLE_TMA:
        q = device_desc_q.load(
            [start_m, (off_h * stride_qh).to(tl.int32)],
        )
        q_trans = tl.trans(q)
    else:
        q_trans = tl.load(
            q_ptrs_trans + start_m * stride_qm,
            mask=mask_m[None, :],
            other=0.0,
        )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
    invalid_mask_trans = invalid_mask_trans or (pos_offs_m_minus_n > 0)
    if HAS_MAX_ATTN_LEN:
        invalid_mask_trans = invalid_mask_trans and pos_offs_m_minus_n <= max_attn_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask_trans = invalid_mask_trans or (
            pos_offs_m[None, :] == 0 and pos_offs_n[:, None] < max_ids
        )
    silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    if ENABLE_TMA:
        do = device_desc_do.load(
            [start_m, (off_h * stride_doh).to(tl.int32)],
        )
    else:
        do = tl.load(
            do_ptrs + start_m * stride_dom,
            mask=mask_m[:, None],
            other=0.0,
        )
    dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = (
        dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)

    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    acc_dq(
        dq_ptrs_trans=dq_ptrs_trans,
        start_m=start_m,
        stride_dqm=stride_dqm,
        k=k,
        dqk_trans=dqk_trans,
        alpha=alpha,
        mask_m=mask_m,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        LOCK=LOCK,
        BLOCK_M=BLOCK_M,
        ATOMIC_ADD=ATOMIC_ADD,
        ALLOW_TF32=ALLOW_TF32,
    )
    return dk, dv


@triton.jit
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    n_targets,
    contextual_seq_len,
    max_attn_len,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
    device_desc_q,
    device_desc_k,
    device_desc_v,
    device_desc_do,
    device_desc_dk,
    device_desc_dv,
    LOCK,
    off_h,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dkh,
    stride_dvh,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    MAX_SEQ_LEN,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
):
    if HAS_MULTIPLE_TARGETS:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high + n_targets < seq_len else seq_len
        else:
            high = seq_len
    else:
        low = start_n
        if HAS_MAX_ATTN_LEN:
            high = start_n + max_attn_len + BLOCK_N
            high = high if high < seq_len else seq_len
        else:
            high = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
        if low < contextual_block_end:
            low = contextual_block_end

    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    if ENABLE_TMA:
        q_ptrs_trans = None
        do_ptrs = None
        k = device_desc_k.load(
            [start_n, (off_h * stride_kh).to(tl.int32)],
        )
        v = device_desc_v.load(
            [start_n, (off_h * stride_vh).to(tl.int32)],
        )
    else:
        mask_n = offs_n < seq_len
        q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
        do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
        v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    # loop over rows
    if HAS_CONTEXTUAL_SEQ_LEN:
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                do_ptrs=do_ptrs,
                device_desc_q=device_desc_q,
                device_desc_do=device_desc_do,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_doh=stride_doh,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                ATOMIC_ADD=ATOMIC_ADD,
                ENABLE_TMA=ENABLE_TMA,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
            )
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            do_ptrs=do_ptrs,
            device_desc_q=device_desc_q,
            device_desc_do=device_desc_do,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            max_ids=max_ids,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            LOCK=LOCK,
            off_h=off_h,
            stride_qh=stride_qh,
            stride_doh=stride_doh,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_M=BLOCK_M,
            ATOMIC_ADD=ATOMIC_ADD,
            ENABLE_TMA=ENABLE_TMA,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
        )
    # write-back
    dk = dk * alpha
    if ENABLE_TMA:
        device_desc_dv.store(
            [start_n, (off_h * stride_dvh).to(tl.int32)],
            dv.to(k.dtype),
        )
        device_desc_dk.store(
            [start_n, (off_h * stride_dkh).to(tl.int32)],
            dk.to(k.dtype),
        )
    else:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])  # pyre-ignore[61]
        tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])  # pyre-ignore[61]


def _bwd_pre_hook(nargs):
    nargs["DQ"].zero_()
    if nargs["SEQUENCE_PARALLEL"] is True:
        nargs["LOCK"].zero_()


def _get_bw_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64, 128]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2, 4]:
                                for sp in [True, False]:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": BLOCK_M,
                                                "BLOCK_N": BLOCK_N,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "waves_per_eu": waves_per_eu,
                                                "SEQUENCE_PARALLEL": sp,
                                                "UNROLL": 1,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                            pre_hook=_bwd_pre_hook,
                                        )
                                    )
        return configs

    configs = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=3,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 4},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
    ]
    if torch.cuda.is_available() and torch.version.cuda < "12.8":
        configs += [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=3,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=1,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
                num_stages=2,
                num_warps=4,
                pre_hook=_bwd_pre_hook,
            ),
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 128,
                    "SEQUENCE_PARALLEL": False,
                    "UNROLL": 2,
                },
                num_stages=2,
                num_warps=8,
                pre_hook=_bwd_pre_hook,
            ),
        ]
    else:
        print("WARNING: temporarily disabled some autotune configs for CUDA 12.8+")
    return configs


@triton_autotune(
    configs=_get_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
    ],
)
@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    tma_workspace_ptr,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    contextual_seq_len,
    max_attn_len,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
    ENABLE_TMA: tl.constexpr,
    TMA_DESC_SIZE: tl.constexpr,
    ENABLE_BUFFER_OPS_ASSUMES: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    if ENABLE_BUFFER_OPS_ASSUMES:
        tl.assume(off_hz >= 0)
        tl.assume(off_z >= 0)
        tl.assume(off_h >= 0)
        tl.assume(seq_start >= 0)
        tl.assume(stride_qm >= 0)
        tl.assume(stride_qh >= 0)
        tl.assume(stride_kn >= 0)
        tl.assume(stride_kh >= 0)
        tl.assume(stride_vn >= 0)
        tl.assume(stride_vh >= 0)
        tl.assume(stride_dom >= 0)
        tl.assume(stride_doh >= 0)
        tl.assume(stride_dqm >= 0)
        tl.assume(stride_dqh >= 0)
        tl.assume(stride_dkn >= 0)
        tl.assume(stride_dkh >= 0)
        tl.assume(stride_dvn >= 0)
        tl.assume(stride_dvh >= 0)

    # offset pointers for batch/head
    Q = Q + seq_start * stride_qm
    K = K + seq_start * stride_kn
    V = V + seq_start * stride_vn
    DOut = DOut + seq_start * stride_dom
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn
    DV = DV + seq_start * stride_dvn
    device_desc_q = None
    device_desc_k = None
    device_desc_v = None
    device_desc_do = None
    device_desc_dk = None
    device_desc_dv = None
    if ENABLE_TMA:
        device_desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[seq_len, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_M, BLOCK_D_Q],
        )
        device_desc_do = tl.make_tensor_descriptor(
            DOut,
            shape=[seq_len, H * DimV],
            strides=[H * DimV, 1],
            block_shape=[BLOCK_M, BLOCK_D_V],
        )
        device_desc_k = tl.make_tensor_descriptor(
            K,
            shape=[seq_len, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_N, BLOCK_D_Q],
        )
        device_desc_dk = tl.make_tensor_descriptor(
            DK,
            shape=[seq_len, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=[BLOCK_N, BLOCK_D_Q],
        )
        device_desc_v = tl.make_tensor_descriptor(
            V,
            shape=[seq_len, H * DimV],
            strides=[H * DimV, 1],
            block_shape=[BLOCK_N, BLOCK_D_V],
        )
        device_desc_dv = tl.make_tensor_descriptor(
            DV,
            shape=[seq_len, H * DimV],
            strides=[H * DimV, 1],
            block_shape=[BLOCK_N, BLOCK_D_V],
        )
    else:
        Q += off_h * stride_qh
        K += off_h * stride_kh
        V += off_h * stride_vh
        DOut += off_h * stride_doh
        DK += off_h * stride_dkh
        DV += off_h * stride_dvh
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            contextual_seq_len=contextual_seq_len,
            max_attn_len=max_attn_len,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            device_desc_q=device_desc_q,
            device_desc_k=device_desc_k,
            device_desc_v=device_desc_v,
            device_desc_do=device_desc_do,
            device_desc_dk=device_desc_dk,
            device_desc_dv=device_desc_dv,
            LOCK=LOCK,
            off_h=off_h,
            stride_qh=stride_qh,
            stride_kh=stride_kh,
            stride_vh=stride_vh,
            stride_doh=stride_doh,
            stride_dkh=stride_dkh,
            stride_dvh=stride_dvh,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=True,
            ENABLE_TMA=ENABLE_TMA,
        )
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len=seq_len,
                n_targets=n_targets,
                contextual_seq_len=contextual_seq_len,
                max_attn_len=max_attn_len,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                device_desc_q=device_desc_q,
                device_desc_k=device_desc_k,
                device_desc_v=device_desc_v,
                device_desc_do=device_desc_do,
                device_desc_dk=device_desc_dk,
                device_desc_dv=device_desc_dv,
                LOCK=LOCK,
                off_h=off_h,
                stride_qh=stride_qh,
                stride_kh=stride_kh,
                stride_vh=stride_vh,
                stride_doh=stride_doh,
                stride_dkh=stride_dkh,
                stride_dvh=stride_dvh,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
                ENABLE_TMA=ENABLE_TMA,
            )


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
) -> torch.Tensor:
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    TMA_DESC_SIZE = 128
    workspace = None
    desc_q = q
    desc_k = k
    desc_v = v

    if enable_tma:
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            q,
            shape=[L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[L, H * DimV],
            strides=[H * DimV, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device="cuda")

    # pyre-ignore [6]
    triton.set_allocator(alloc_fn)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _hstu_attn_fwd[grid](
        Q=desc_q,
        K=desc_k,
        V=desc_v,
        workspace_ptr=workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )
    return out


def triton_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
    enable_tma: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    grid = lambda meta: (  # noqa E731
        Z * H,
        (triton.cdiv(N, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    TMA_DESC_SIZE = 128
    tma_workspace = None

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device="cuda")

    # pyre-ignore [6]
    triton.set_allocator(alloc_fn)

    # Enable BufferOps on AMD
    ENABLE_BUFFER_OPS_ASSUMES = torch.version.hip is not None
    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        tma_workspace_ptr=tma_workspace,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len > 0,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
        ENABLE_BUFFER_OPS_ASSUMES=ENABLE_BUFFER_OPS_ASSUMES,
    )

    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        sort_by_length: bool,
        enable_tma: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
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
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
                enable_tma=ctx.enable_tma,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
            )


@torch.fx.wrap
def triton_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        sort_by_length,
        enable_tma,
    )


@torch.fx.wrap
def triton_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    enable_tma: bool = False,
) -> torch.Tensor:
    Z = seq_offsets.size(0) - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    DELTA_L, H, DimQ = delta_q.shape
    DeltaSize = DELTA_L // Z
    L, _, DimV = v.shape
    out = torch.empty((DELTA_L, H, DimV), dtype=delta_q.dtype, device=delta_q.device)

    TMA_DESC_SIZE = 128
    desc_q = delta_q
    desc_k = k
    desc_v = v

    if enable_tma:
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(
            delta_q,
            shape=[DELTA_L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )
        desc_v = TensorDescriptor(
            v,
            shape=[L, H * DimV],
            strides=[H * DimV, 1],
            block_shape=dummy_block,
        )
        desc_k = TensorDescriptor(
            k,
            shape=[L, H * DimQ],
            strides=[H * DimQ, 1],
            block_shape=dummy_block,
        )

    def alloc_fn(size: int, align: int, stream: Optional[int]):
        assert align == TMA_DESC_SIZE
        return torch.empty(size, dtype=torch.int8, device="cuda")

    # pyre-ignore [6]
    triton.set_allocator(alloc_fn)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(DeltaSize, meta["BLOCK_M"]),
        Z * H,
    )

    has_contextual_seq_len = contextual_seq_len > 0
    has_max_attn_len = max_attn_len > 0
    _hstu_attn_fwd[grid](
        Q=desc_q,
        K=desc_k,
        V=desc_v,
        workspace_ptr=None,
        sort_by_length_indices=None,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=delta_q.stride(0),
        stride_qh=delta_q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=DeltaSize,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        IS_DELTA_Q=True,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
        HAS_SORT_BY_LENGTH_INDICES=False,
        ENABLE_TMA=enable_tma,
        TMA_DESC_SIZE=TMA_DESC_SIZE,
    )
    return out
