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

# pyre-unsafe

from typing import List, Optional
import numpy as np

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    # @manual=//triton:triton
    from triton.language.extra.cuda.libdevice import fast_dividef
except ImportError:
    # pyre-ignore: Undefined import [21]
    # @manual=//triton:triton
    from triton.language.math import fast_dividef


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64]:
                for num_stages in [0, 1]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2]:
                                configs.append(
                                    triton.Config(
                                        {
                                            "BLOCK_M": BLOCK_M,
                                            "BLOCK_N": BLOCK_N,
                                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                            "waves_per_eu": waves_per_eu,
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
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
        ]
    return configs


@triton.jit
def _ragged_hstu_attn_fwd_one_block_tma(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    mask_m,
    mask_n,
    q,
    K_desc_ptr,
    V_desc_ptr,
    offset_h,
    seq_start,
    n_targets,
    ts_1_ptrs,
    ts_0,
    TW,
    PW,
    alpha,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    max_attn_len,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    bias_ptrs,
    attn_scale,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_ATTN_SCALE: tl.constexpr,
    HAS_MAX_ATTN_LEN: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl._experimental_descriptor_load(
        K_desc_ptr,
        [(seq_start + start_n).to(tl.int32), offset_h.to(tl.int32)],
        [BLOCK_N, BLOCK_D_Q],
        tl.bfloat16,
    )
    qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
    #qk = tl.dot(k, tl.trans(q), allow_tf32=ALLOW_TF32) * alpha
    #qk = tl.trans(qk)
    #k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    #qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == "lower_triangular":
            offs_m = tl.where(
                offs_m < seq_len - n_targets,
                offs_m,
                seq_len - n_targets,
            )
            offs_n = tl.where(
                offs_n < seq_len - n_targets,
                offs_n,
                seq_len - n_targets,
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            offs_m = tl.where(offs_m > n_targets - 1, offs_m, n_targets - 1)
            offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1)
    offs_n_minus_m = offs_n[None, :] - offs_m[:, None]
    if HAS_MAX_ATTN_LEN:
        if INVALID_MASK_TYPE == "lower_triangular":
            invalid_mask = invalid_mask or (
                offs_n_minus_m < 0 and offs_n_minus_m >= -max_attn_len
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            invalid_mask = invalid_mask or (
                offs_n_minus_m > 0 and offs_n_minus_m <= max_attn_len
            )
    else:
        if INVALID_MASK_TYPE == "lower_triangular":
            invalid_mask = invalid_mask or offs_n_minus_m < 0
        elif INVALID_MASK_TYPE == "upper_triangular":
            invalid_mask = invalid_mask or offs_n_minus_m > 0
    if ATTN_BIAS_TYPE == "fused":
        attn_bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_1 = tl.load(ts_1_ptrs + start_n, mask=mask_n)
            else:
                ts_1 = tl.load(ts_1_ptrs + start_n + 1, mask=mask_n)
            ts = ts_0[:, None] - ts_1[None, :]
            ts = ts + time_delta
            ts = tl.where(ts > 1e-6, ts, 1e-6)
            ts = ts * (1.0 / time_bucket_incr)
            if BUCKET_FN == "log":
                ts = tl.log(ts)
            elif BUCKET_FN == "sqrt":
                ts = tl.sqrt(ts)
            ts = ts * (1.0 / time_bucket_div)
            ts = ts.to(tl.int32)
            ts = tl.where(ts > 0, ts, 0)
            ts = tl.where(ts < num_buckets, ts, num_buckets)
            ts_w = tl.load(
                TW + ts,
                mask=mask_m[:, None] and mask_n[None, :],
            )
            attn_bias = attn_bias + ts_w
        if USE_POS_BIAS:
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n_minus_m + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(
                    offs_pos_w < 2 * max_pos_ind - 2,
                    offs_pos_w,
                    2 * max_pos_ind - 2,
                )
            else:
                offs_pos_w = offs_n_minus_m + MAX_SEQ_LEN - 1
            pos_w = tl.load(
                PW + offs_pos_w,
                mask=mask_m[:, None] and mask_n[None, :],
            )
            attn_bias = attn_bias + pos_w
        qk = qk + attn_bias
    elif ATTN_BIAS_TYPE == "separate":
        attn_bias = tl.load(
            bias_ptrs + start_n,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        qk = qk + attn_bias
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    if HAS_ATTN_SCALE:
        silu = silu * attn_scale[:, None]

    #v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    v = tl._experimental_descriptor_load(
        V_desc_ptr,
        [(seq_start + start_n).to(tl.int32), offset_h.to(tl.int32)],
        [BLOCK_N, BLOCK_D_V],
        tl.bfloat16,
    )
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.autotune(
    configs=_get_fw_configs(),
    key=[
        "Z",
        "H",
        "MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _ragged_hstu_attn_fwd_tma(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    delta_x_offsets,
    num_targets,
    Scale,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_sz,
    stride_sm,
    stride_ts,
    stride_om,
    stride_oh,
    alpha,
    Z,
    H,
    MAX_SEQ_LEN,
    DimQ,
    DimV,
    DeltaSize,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_ATTN_SCALE: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    max_attn_len,
    HAS_MAX_ATTN_LEN: tl.constexpr,
):
    # M_CTX == N_CTX
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    seq_start = tl.load(seq_offsets + off_z)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = tl.program_id(0) * BLOCK_M
        delta_start = tl.load(delta_x_offsets + off_z * DeltaSize)
        start_m = (start_m_delta + delta_start - seq_start).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = tl.program_id(0) * BLOCK_M
    if start_m >= seq_len:
        return
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None

    # initialize offsets
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
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
    #K_block_ptr = tl.make_block_ptr(
    #    base=K + off_h * stride_kh + seq_start * stride_kn,
    #    shape=(BLOCK_D_Q, seq_len),
    #    strides=(1, stride_kn),
    #    offsets=(0, 0),
    #    block_shape=(BLOCK_D_Q, BLOCK_N),
    #    order=(0, 1),
    #)
    #V_block_ptr = tl.make_block_ptr(
    #    base=V + off_h * stride_vh + seq_start * stride_vn,
    #    shape=(seq_len, BLOCK_D_V),
    #    strides=(stride_vn, 1),
    #    offsets=(0, 0),
    #    block_shape=(BLOCK_N, BLOCK_D_V),
    #    order=(1, 0),
    #)
    mask_m = offs_m < seq_len
    if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS:
        ts_0_ptrs = TS + off_z * stride_ts + offs_m
        ts_1_ptrs = TS + off_z * stride_ts + offs_n
        if CAUSAL:
            ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
        else:
            ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
    elif ATTN_BIAS_TYPE == "separate":
        seq2_start = tl.load(seq2_offsets + off_z)
        bias_start = seq2_start * H + off_h * seq_len * seq_len
        off_bias = offs_m[:, None] * seq_len + offs_n[None, :]
        bias_ptrs = Bias + bias_start + off_bias

    if HAS_ATTN_SCALE:
        scale_ptrs = Scale + off_z * stride_sz
        attn_scale = tl.load(scale_ptrs + offs_m * stride_sm, mask=offs_m < seq_len)

    # convert q to tma
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
    if INVALID_MASK_TYPE == "lower_triangular":
        if HAS_MULTIPLE_TARGETS:
            if HAS_MAX_ATTN_LEN:
                start_m_index = seq_len - n_targets if start_m > seq_len - n_targets else start_m
                low = start_m_index - max_attn_len
                low = low if low > 0 else 0
            else:
                low = 0
            uih_end = (seq_len - n_targets + BLOCK_N - 1) // BLOCK_N * BLOCK_N
            if uih_end < start_m:
                high = seq_len - n_targets
            else:
                high = start_m + BLOCK_M
        else:
            if HAS_MAX_ATTN_LEN:
                low = start_m - max_attn_len
                low = low if low > 0 else 0
            else:
                low = 0
            high = start_m + BLOCK_M
    elif INVALID_MASK_TYPE == "upper_triangular":
        low = start_m
        high = seq_len

    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [K], dtype=tl.int32, is_pure=False, pack=1)
    tl.inline_asm_elementwise("fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg", "=r, l",
                              [V], dtype=tl.int32, is_pure=False, pack=1)

    # pyre-ignore[61]
    for start_n in range(low, high, BLOCK_N):
        cur_offs_n = offs_n + start_n
        mask_n = cur_offs_n < seq_len
        acc += _ragged_hstu_attn_fwd_one_block_tma(
            start_n=start_n,
            seq_len=seq_len,
            offs_m=offs_m,
            offs_n=cur_offs_n,
            mask_m=mask_m,
            mask_n=mask_n,
            q=q,
            #K_block_ptr=K_block_ptr,
            K_desc_ptr=K,
            #V_block_ptr=V_block_ptr,
            V_desc_ptr=V,
            offset_h=off_h * stride_vh,
            seq_start=seq_start,
            # pyre-ignore[61]
            n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
            ts_1_ptrs=(
                # pyre-ignore[61]
                ts_1_ptrs
                if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS
                else None
            ),
            # pyre-ignore[61]
            ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
            TW=TW,
            PW=PW,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            num_buckets=num_buckets,
            max_pos_ind=max_pos_ind,
            max_attn_len=max_attn_len,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            # pyre-ignore[61]
            bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
            # pyre-ignore[61]
            attn_scale=attn_scale if HAS_ATTN_SCALE else None,
            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
            CAUSAL=CAUSAL,
            BUCKET_FN=BUCKET_FN,
            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
            USE_TIME_BIAS=USE_TIME_BIAS,
            USE_POS_BIAS=USE_POS_BIAS,
            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_ATTN_SCALE=HAS_ATTN_SCALE,
            HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        #K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        #V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if HAS_MULTIPLE_TARGETS and INVALID_MASK_TYPE == "lower_triangular":
        # pyre-ignore[61]
        if uih_end < start_m:
            low_delta = start_m
            high_delta = start_m + BLOCK_M
            #offset = (low_delta - uih_end).to(tl.int32)  # pyre-ignore [61]
            #K_block_ptr = tl.advance(K_block_ptr, (0, offset))
            #V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
            for start_delta in range(low_delta, high_delta, BLOCK_N):
                cur_offs_n = offs_n + start_delta
                mask_n = cur_offs_n < seq_len
                acc += _ragged_hstu_attn_fwd_one_block_tma(
                    start_n=start_delta,
                    seq_len=seq_len,
                    offs_m=offs_m,
                    offs_n=cur_offs_n,
                    mask_m=mask_m,
                    mask_n=mask_n,
                    q=q,
                    #K_block_ptr=K_block_ptr,
                    K_desc_ptr=K,
                    #V_block_ptr=V_block_ptr,
                    V_desc_ptr=V,
                    offset_h=off_h * stride_vh,
                    seq_start=seq_start,
                    n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                    ts_1_ptrs=(
                        # pyre-ignore[61]
                        ts_1_ptrs
                        if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS
                        else None
                    ),
                    ts_0=(
                        # pyre-ignore[61]
                        ts_0
                        if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS
                        else None
                    ),
                    TW=TW,
                    PW=PW,
                    alpha=alpha,
                    MAX_SEQ_LEN=MAX_SEQ_LEN,
                    num_buckets=num_buckets,
                    max_pos_ind=max_pos_ind,
                    max_attn_len=max_attn_len,
                    time_bucket_incr=time_bucket_incr,
                    time_bucket_div=time_bucket_div,
                    time_delta=time_delta,
                    # pyre-ignore[61]
                    bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                    # pyre-ignore[61]
                    attn_scale=attn_scale if HAS_ATTN_SCALE else None,
                    INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                    CAUSAL=CAUSAL,
                    BUCKET_FN=BUCKET_FN,
                    ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                    USE_TIME_BIAS=USE_TIME_BIAS,
                    USE_POS_BIAS=USE_POS_BIAS,
                    HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                    HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                    HAS_ATTN_SCALE=HAS_ATTN_SCALE,
                    HAS_MAX_ATTN_LEN=HAS_MAX_ATTN_LEN,
                    IS_DELTA_Q=IS_DELTA_Q,
                    ALLOW_TF32=ALLOW_TF32,
                    BLOCK_D_Q=BLOCK_D_Q,
                    BLOCK_D_V=BLOCK_D_V,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                #K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                #V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if IS_DELTA_Q:
        start_m_delta = tl.program_id(0) * BLOCK_M
        offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_D_V)
        off_o = (
            (off_z * DeltaSize + offs_m_delta[:, None]) * stride_om
            + off_h * stride_oh
            + offs_v_d[None, :]
        )
        out_ptrs = Out + off_o
        # todo: convert out to tma
        tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
    else:
        # rematerialize offsets to save registers
        start_m = tl.program_id(0) * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_D_V)
        off_o = (
            (seq_start + offs_m[:, None]) * stride_om
            + off_h * stride_oh
            + offs_v_d[None, :]
        )
        out_ptrs = Out + off_o
        tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


def triton_ragged_attention_tma(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_bias: Optional[torch.Tensor],
    seq2_offsets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: Optional[int],
) -> torch.Tensor:
    assert invalid_attn_mask_type in [
        "lower_triangular",
        "upper_triangular",
    ]
    Z = seq_offsets.numel() - 1
    L, H, DimQ = q.shape
    _, _, DimV = v.shape

    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_attn_bias = attn_bias is not None
    has_attn_scale = attn_scale is not None
    has_max_attn_len = max_attn_len is not None

    TMA_SIZE = 128
    BLOCK_N, BLOCK_D_V, BLOCK_D_Q = 64, DimV, DimQ
    desc_k = np.empty(TMA_SIZE, dtype=np.int8)
    desc_v = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        k.data_ptr(),
        L,
        H * DimQ,
        BLOCK_N,
        BLOCK_D_Q,
        k.element_size(),
        desc_k,
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        v.data_ptr(),
        L,
        H * DimV,
        BLOCK_N,
        BLOCK_D_V,
        v.element_size(),
        desc_v,
    )
    desc_k = torch.tensor(desc_k, device=v.device)
    desc_v = torch.tensor(desc_v, device=v.device)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    stride_sz = 0
    stride_sm = 0
    if attn_scale is not None:
        if attn_scale.dim() == 1:
            stride_sm = attn_scale.stride(0)
        else:
            stride_sz = attn_scale.stride(0)
            stride_sm = attn_scale.stride(1)

    _ragged_hstu_attn_fwd_tma[grid](
        Q=q,
        K=desc_k,
        V=desc_v,
        seq_offsets=seq_offsets,
        TS=None,
        TW=None,
        PW=None,
        Bias=attn_bias,
        seq2_offsets=seq2_offsets,
        delta_x_offsets=None,
        num_targets=num_targets,
        Scale=attn_scale,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_sz=stride_sz,
        stride_sm=stride_sm,
        stride_ts=None,
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        H=H,
        MAX_SEQ_LEN=N,
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        num_buckets=None,
        max_pos_ind=None,
        time_bucket_incr=None,
        time_bucket_div=None,
        time_delta=None,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=None,
        BUCKET_FN="none",
        ATTN_BIAS_TYPE="separate" if has_attn_bias else "none",
        USE_TIME_BIAS=False,
        USE_POS_BIAS=False,
        HAS_MAX_POS_IND=False,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        HAS_ATTN_SCALE=has_attn_scale,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        max_attn_len=max_attn_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
    )
    return out


def triton_ragged_attention_tma_relative_bias(
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    timestamps: torch.Tensor,
    ts_weights: torch.Tensor,
    pos_weights: torch.Tensor,
    causal: bool,
    num_buckets: int,
    time_bucket_fn: str,
    time_bucket_incr: float,
    time_bucket_div: float,
    time_delta: float,
    max_pos_ind: Optional[int],
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    relative_bias_type: str,
    max_attn_len: Optional[int],
) -> torch.Tensor:
    Z = timestamps.size(0)
    N = timestamps.size(1) - 1
    has_attn_scale = attn_scale is not None
    has_multiple_targets = num_targets is not None
    has_max_pos_id = max_pos_ind is not None
    has_max_attn_len = max_attn_len is not None
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)

    TMA_SIZE = 128
    BLOCK_D_V, BLOCK_D_Q = DimV, DimQ
    desc_k = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
    desc_v = torch.empty((TMA_SIZE), device="cuda", dtype=torch.int8)
    '''
    desc_k = np.empty(TMA_SIZE, dtype=np.int8)
    desc_v = np.empty(TMA_SIZE, dtype=np.int8)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        k.data_ptr(),
        L,
        H * DimQ,
        BLOCK_N,
        BLOCK_D_Q,
        k.element_size(),
        desc_k,
    )
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
        v.data_ptr(),
        L,
        H * DimV,
        BLOCK_N,
        BLOCK_D_V,
        v.element_size(),
        desc_v,
    )
    desc_k = torch.tensor(desc_k, device=v.device)
    desc_v = torch.tensor(desc_v, device=v.device)
    '''

    def grid2(META):
        nonlocal desc_k
        nonlocal desc_v
        #a_buf = torch.empty(TMA_SIZE, dtype=torch.int8)
        k_buf = torch.empty_like(desc_k, device="cpu")
        v_buf = torch.empty_like(desc_v, device="cpu")
        #desc_a = desc_a.numpy()  # if start with cuda, will need cpu() here
        #desc_b = desc_b.numpy()
        #desc_c = desc_c.numpy()
        #print("enter grid2", META['BLOCK_M'], META['BLOCK_K'])
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(k.data_ptr(), L, H * DimQ, META['BLOCK_N'], BLOCK_D_Q, k.element_size(),
                                                                  k_buf.numpy())
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(v.data_ptr(), L, H * DimV, META['BLOCK_N'], BLOCK_D_V, v.element_size(),
                                                                  v_buf.numpy())
        #desc_a = torch.tensor(desc_a, device="cuda")
        #desc_b = torch.tensor(desc_b, device="cuda")
        #desc_c = torch.tensor(desc_c, device="cuda")
        desc_k.copy_(k_buf)
        desc_v.copy_(v_buf)
        return (triton.cdiv(N, META["BLOCK_M"]), Z * H, 1)

    stride_sz = 0
    stride_sm = 0
    if attn_scale is not None:
        if attn_scale.dim() == 1:
            stride_sm = attn_scale.stride(0)
        else:
            stride_sz = attn_scale.stride(0)
            stride_sm = attn_scale.stride(1)
    use_time_bias = relative_bias_type == "TIME" or relative_bias_type == "ALL"
    use_pos_bias = relative_bias_type == "POSITION" or relative_bias_type == "ALL"

    _ragged_hstu_attn_fwd_tma[grid2](
        Q=q,
        K=desc_k,
        V=desc_v,
        seq_offsets=seq_offsets,
        TS=timestamps,
        TW=ts_weights,
        PW=pos_weights,
        Bias=None,
        seq2_offsets=None,
        delta_x_offsets=None,
        num_targets=num_targets,
        Scale=attn_scale,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_sz=stride_sz,
        stride_sm=stride_sm,
        stride_ts=timestamps.stride(0),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        H=H,
        MAX_SEQ_LEN=N,
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        num_buckets=num_buckets,
        max_pos_ind=max_pos_ind,
        time_bucket_incr=time_bucket_incr,
        time_bucket_div=time_bucket_div,
        time_delta=time_delta,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=causal,
        BUCKET_FN=time_bucket_fn,
        ATTN_BIAS_TYPE="fused",
        USE_TIME_BIAS=use_time_bias,
        USE_POS_BIAS=use_pos_bias,
        HAS_MAX_POS_IND=has_max_pos_id,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        HAS_ATTN_SCALE=has_attn_scale,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        max_attn_len=max_attn_len,
        HAS_MAX_ATTN_LEN=has_max_attn_len,
    )
    return out
