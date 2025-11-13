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
    autotune_max_seq_len,
    fine_grained_autotune_max_seq_len,
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.utils import is_sm100


def _triton_concat_2D_jagged_internal(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    values_out: torch.Tensor,
    max_seq_len: int,
    B: int,
    offsets_a: Optional[torch.Tensor],
    offsets_b: Optional[torch.Tensor],
    D: int,
    dense_size: int,
    stride_dense_batch: int,
    n_prefix: int,
    is_dense_a: bool,
    is_dense_b: bool,
    is_replace: bool,
    BLOCK_D: int,
) -> None:
    if n_prefix != 0:
        if is_sm100():

            def grid(meta):
                return (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

            concat_2D_jagged_jagged_w_prefix_multirow[grid](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                Out=values_out,
                D=D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_od=values_out.stride(0),
                n_prefix_from_B=n_prefix,
                BLOCK_D=BLOCK_D,
            )
        else:
            concat_2D_jagged_jagged_w_prefix[(max_seq_len, B)](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                Out=values_out,
                D=D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_od=values_out.stride(0),
                n_prefix_from_B=n_prefix,
                BLOCK_D=BLOCK_D,  # pyre-ignore[6]
            )
    else:
        if is_sm100():

            def grid(meta):
                return (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

            concat_2D_jagged_multirow[grid](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                DenseSize=dense_size,
                Out=values_out,
                D=D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_dense_batch=stride_dense_batch,
                stride_od=values_out.stride(0),
                IS_DENSE_A=is_dense_a,  # pyre-ignore[6]
                IS_DENSE_B=is_dense_b,  # pyre-ignore[6]
                BLOCK_D=BLOCK_D,  # pyre-ignore[6]
                IS_REPLACE=is_replace,  # pyre-ignore[6]
            )
        else:
            concat_2D_jagged[(max_seq_len, B)](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                DenseSize=dense_size,
                Out=values_out,
                D=D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_dense_batch=stride_dense_batch,
                stride_od=values_out.stride(0),
                IS_DENSE_A=is_dense_a,  # pyre-ignore[6]
                IS_DENSE_B=is_dense_b,  # pyre-ignore[6]
                BLOCK_D=BLOCK_D,  # pyre-ignore[6]
                IS_REPLACE=is_replace,  # pyre-ignore[6]
            )


def _get_split_concat_2d_jagged_multirow_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [1, 2, 4, 8]:
        for num_warps in [1, 2, 4]:
            configs.append(
                triton.Config(
                    {"BLOCK_N": BLOCK_N},
                    num_warps=num_warps,
                )
            )
    return configs


def _get_bmm_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128, 256]:
            for BLOCK_K in [32, 64]:
                for num_stages in [3, 5]:
                    for num_warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton_autotune(
    configs=_get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K", "ELEMENTWISE", "HAS_BIAS"],
)
@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ELEMENTWISE: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """

    off_n = tl.program_id(0)
    off_m = tl.program_id(1).to(tl.int64)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += (seq_start + start_m) * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < (seq_len - start_m)) & ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        if ELEMENTWISE:
            Bias += (seq_start + start_m) * stride_bias_b
            bias_ptrs = Bias + offs_m[:, None] * stride_bias_b + offs_n[None, :]
            bias = tl.load(
                bias_ptrs,
                mask=(offs_m[:, None] < (seq_len - start_m)) & (offs_n[None, :] < N),
                other=0.0,
            )
            accumulator += bias.to(tl.float32)
        else:
            bias_ptrs = Bias + off_b.to(tl.int64) * stride_bias_b + offs_n
            bias = tl.load(bias_ptrs, mask=offs_n < N)
            accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    Out += start_m * stride_om
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < (seq_len - start_m)) & (offs_n[None, :] < N),
    )


def _get_bmm_reduce_sum_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128]:
            for BLOCK_K in [64, 128]:
                for num_stages in [3, 4]:
                    for num_warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton_autotune(
    configs=_get_bmm_reduce_sum_configs(),
    key=["M", "N", "AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _jagged_jagged_bmm_reduce_sum(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    ReduceOut,
    M,
    N,
    AUTOTUNE_MAX_SEQ_LEN,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    REDUCE_JAGGEDB: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Jagged
    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N), and Out has shape (B, M, N)
    """

    off_m = tl.program_id(0).to(tl.int64)
    off_n = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b.to(tl.int64) * stride_ob
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    Out += start_m * stride_om
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if REDUCE_JAGGEDB:
        out_reduce_ptrs = (
            ReduceOut + off_b.to(tl.int64) * stride_orb + offs_n * stride_orn
        )
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(
            out_ptrs,
            out,
            mask=(offs_m[:, None] < (M - start_m)) & (offs_n[None, :] < N),
        )
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(
                    out_reduce_ptrs,  # pyre-ignore [61]
                    acc_reduce.to(ReduceOut.dtype.element_ty),
                    mask=(offs_n < N),
                )
        return

    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + (start_m + offs_m)[:, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n[None, :]

    for k in range(0, seq_len, BLOCK_K):
        jg_a = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < (M - start_m)) & ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b.to(tl.float32), axis=0)

        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=(offs_m[:, None] < (M - start_m)) & (offs_n[None, :] < N),
    )
    if REDUCE_JAGGEDB:
        if off_m == 0:
            tl.store(
                out_reduce_ptrs,  # pyre-ignore [61]
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )


class _JaggedDenseBmmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
    ):
        jagged = switch_to_contiguous_if_needed(jagged)
        L, D = jagged.shape
        B, _, K = dense.shape
        bmm_out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(K, meta["BLOCK_N"]),
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            B,
        )

        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=jagged,
            Dense=dense,
            Bias=0,
            Out=bmm_out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            N=K,
            K=D,
            stride_jm=jagged.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(1),
            stride_dn=dense.stride(2),
            stride_bias_b=0,
            stride_om=bmm_out.stride(0),
            HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            ELEMENTWISE=False,
        )

        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.B = B
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.D = D
        return bmm_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_bmm_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor]:
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        d_dense = torch.empty_like(dense)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.D, meta["BLOCK_N"]),
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            ctx.B,
        )
        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=d_bmm_out,
            Dense=dense,
            Bias=None,
            Out=d_jagged,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            N=ctx.D,
            K=ctx.K,
            stride_jm=d_bmm_out.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(2),
            stride_dn=dense.stride(1),
            stride_bias_b=0,
            stride_om=d_jagged.stride(0),
            HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            ELEMENTWISE=False,
        )

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.D, meta["BLOCK_M"]),
            triton.cdiv(ctx.K, meta["BLOCK_N"]),
            ctx.B,
        )
        _jagged_jagged_bmm_reduce_sum[grid](
            seq_offsets=seq_offsets,
            JaggedA=jagged,
            JaggedB=d_bmm_out,
            Out=d_dense,
            ReduceOut=None,
            M=ctx.D,
            N=ctx.K,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            stride_ak=jagged.stride(0),
            stride_bk=d_bmm_out.stride(0),
            stride_ob=d_dense.stride(0),
            stride_om=d_dense.stride(1),
            stride_on=d_dense.stride(2),
            stride_orb=0,
            stride_orn=0,
            REDUCE_JAGGEDB=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return None, None, d_jagged, d_dense


def _get_jagged_dense_broadcast_add_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [16, 32, 64]:
        for num_stages in [1, 2]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_N": BLOCK_N,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


@triton_autotune(
    configs=_get_jagged_dense_broadcast_add_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def jagged_dense_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    stride_jn,
    stride_db,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    Jagged += seq_start * stride_jn
    Dense += off_b * stride_db
    Out += seq_start * stride_on
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_n[:, None] * stride_jn + offs_d[None, :]
    dense_ptrs = Dense + offs_d
    out_ptrs = Out + offs_n[:, None] * stride_jn + offs_d[None, :]
    for d in range(0, D, BLOCK_D):
        jg = tl.load(
            jagged_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_n[:, None] < seq_len) and (d + offs_d)[None, :] < D,
        )
        dn = tl.load(dense_ptrs, mask=d + offs_d < D)
        out = jg + dn[None, :]
        tl.store(
            out_ptrs,
            out,
            mask=(offs_n[:, None] < seq_len) and (d + offs_d)[None, :] < D,
        )
        dense_ptrs += BLOCK_D
        jagged_ptrs += BLOCK_D
        out_ptrs += BLOCK_D


@triton.jit
def jagged_reduce_sum(
    seq_offsets,
    Jagged,
    Out,
    D,
    stride_jn,
    stride_ob,
    BLOCK_D: tl.constexpr,
):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_D
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    Jagged += seq_start * stride_jn
    Out += off_b * stride_ob
    offs_d = off_d + tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_d
    out_ptrs = Out + offs_d
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _ in range(0, seq_len):
        jg = tl.load(
            jagged_ptrs,
            mask=offs_d < D,
        )
        accumulator += jg
        jagged_ptrs += stride_jn
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=offs_d < D,
    )


class _JaggedDenseBroadcastAddFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
    ):
        jagged = switch_to_contiguous_if_needed(jagged)
        dense = switch_to_contiguous_if_needed(dense)
        L, D = jagged.shape
        B, _ = dense.shape
        out = torch.empty_like(jagged)

        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
        )
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        jagged_dense_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=jagged,
            Dense=dense,
            Out=out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            D=D,
            stride_jn=jagged.stride(0),
            stride_db=dense.stride(0),
            stride_on=out.stride(0),
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(seq_offsets)
        ctx.max_seq_len = max_seq_len
        ctx.B = B
        ctx.D = D
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor]:
        seq_offsets = ctx.saved_tensors[0]
        d_dense = torch.empty((ctx.B, ctx.D), device=d_out.device, dtype=d_out.dtype)
        BLOCK_D = triton.next_power_of_2(ctx.D) if ctx.D < 64 else 64
        jagged_reduce_sum[(ctx.B, triton.cdiv(ctx.D, BLOCK_D))](
            seq_offsets=seq_offsets,
            Jagged=d_out,
            Out=d_dense,
            D=ctx.D,
            stride_jn=d_out.stride(0),
            stride_ob=d_dense.stride(0),
            BLOCK_D=BLOCK_D,
        )
        return None, None, d_out, d_dense


def triton_jagged_dense_bmm_add_fwd(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    elementwise: bool = False,
) -> Tuple[torch.Tensor, int, int, int]:
    jagged = switch_to_contiguous_if_needed(jagged)
    bias = switch_to_contiguous_if_needed(bias)
    L, K = jagged.shape
    B, _, N = dense.shape
    out = torch.empty((L, N), dtype=jagged.dtype, device=jagged.device)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(max_seq_len, meta["BLOCK_M"]),
        B,
    )

    jagged_dense_bmm_broadcast_add_kernel[grid](
        seq_offsets=seq_offsets,
        Jagged=jagged,
        Dense=dense,
        Bias=bias,
        Out=out,
        AUTOTUNE_MAX_SEQ_LEN=fine_grained_autotune_max_seq_len(max_seq_len),
        N=N,
        K=K,
        stride_jm=jagged.stride(0),
        stride_db=dense.stride(0),
        stride_dk=dense.stride(1),
        stride_dn=dense.stride(2),
        stride_bias_b=bias.stride(0),
        stride_om=out.stride(0),
        HAS_BIAS=True,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        ELEMENTWISE=elementwise,
    )

    return out, B, K, N


def triton_jagged_dense_bmm_add_bwd_jagged(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    d_jagged: torch.Tensor,
    dense: torch.Tensor,
    d_out: torch.Tensor,
    K: int,
    B: int,
    N: int,
) -> torch.Tensor:
    grid = lambda meta: (  # noqa E731
        triton.cdiv(K, meta["BLOCK_N"]),
        triton.cdiv(max_seq_len, meta["BLOCK_M"]),
        B,
    )
    jagged_dense_bmm_broadcast_add_kernel[grid](
        seq_offsets=seq_offsets,
        Jagged=d_out,
        Dense=dense,
        Bias=None,
        Out=d_jagged,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        N=K,
        K=N,
        stride_jm=d_out.stride(0),
        stride_db=dense.stride(0),
        stride_dk=dense.stride(2),
        stride_dn=dense.stride(1),
        stride_bias_b=0,
        stride_om=d_jagged.stride(0),
        HAS_BIAS=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        ELEMENTWISE=False,
    )

    return d_jagged


def triton_jagged_dense_bmm_add_bwd_dense_bias(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    d_dense: torch.Tensor,
    B: int,
    K: int,
    N: int,
    d_out: torch.Tensor,
    elementwise: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_bias = torch.empty((B, N), device=d_out.device, dtype=d_out.dtype)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(K, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
        B,
    )

    if elementwise:
        d_bias = d_out
        reduce_out = None
        stride_orb = 0
        stride_orn = 0
        reduce_jaggedb = False
    else:
        reduce_out = d_bias
        stride_orb = d_bias.stride(0)
        stride_orn = d_bias.stride(1)
        reduce_jaggedb = True

    _jagged_jagged_bmm_reduce_sum[grid](
        seq_offsets=seq_offsets,
        JaggedA=jagged,
        JaggedB=d_out,
        Out=d_dense,
        ReduceOut=reduce_out,
        M=K,
        N=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        stride_ak=jagged.stride(0),
        stride_bk=d_out.stride(0),
        stride_ob=d_dense.stride(0),
        stride_om=d_dense.stride(1),
        stride_on=d_dense.stride(2),
        stride_orb=stride_orb,
        stride_orn=stride_orn,
        REDUCE_JAGGEDB=reduce_jaggedb,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    return d_dense, d_bias


class _JaggedDenseBmmAddFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
        bias: torch.Tensor,
        elementwise: bool = False,
    ):
        out, B, K, N = triton_jagged_dense_bmm_add_fwd(
            max_seq_len, seq_offsets, jagged, dense, bias, elementwise
        )

        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.B = B
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.N = N
        ctx.elementwise = elementwise
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = triton_jagged_dense_bmm_add_bwd_jagged(
            ctx.max_seq_len,
            seq_offsets,
            torch.empty_like(jagged),
            dense,
            d_out,
            ctx.K,
            ctx.B,
            ctx.N,
        )
        d_dense, d_bias = triton_jagged_dense_bmm_add_bwd_dense_bias(
            ctx.max_seq_len,
            seq_offsets,
            jagged,
            torch.empty_like(dense),
            ctx.B,
            ctx.K,
            ctx.N,
            d_out,
            ctx.elementwise,
        )

        return None, None, d_jagged, d_dense, d_bias, None


@triton.jit
def concat_2D_jagged_w_prefix(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    n_prefix_from_B,  # nonzero is not supported when IS_REPLACE=True
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b

    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return

    offs_d = tl.arange(0, BLOCK_D)
    if IS_REPLACE:
        out_seq_start = seq_start_a + off_n
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        out_seq_start = seq_start_a + seq_start_b + off_n
        out_seq_b_start = seq_len_a + n_prefix_from_B

    out_ptrs = Out + out_seq_start.to(tl.int64) * stride_od + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_from_B:
        off_a = off_n - n_prefix_from_B
        if IS_DENSE_A:
            in_ptrs = (
                ValuesA
                + off_a.to(tl.int64) * stride_ad
                + off_z.to(tl.int64) * stride_dense_batch
                + offs_d
            )
        else:
            in_ptrs = ValuesA + (off_a + seq_start_a).to(tl.int64) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_from_B
        if off_n < n_prefix_from_B:
            off_b += out_seq_b_start - n_prefix_from_B
        if IS_DENSE_B:
            in_ptrs = (
                ValuesB
                + off_b.to(tl.int64) * stride_bd
                + off_z.to(tl.int64) * stride_dense_batch
                + offs_d
            )
        else:
            in_ptrs = ValuesB + (off_b + seq_start_b).to(tl.int64) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def concat_2D_jagged(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    concat_2D_jagged_w_prefix(
        OffsetsA,
        ValuesA,
        OffsetsB,
        ValuesB,
        DenseSize,
        Out,
        D,
        stride_ad,
        stride_bd,
        stride_dense_batch,
        stride_od,
        0,
        IS_DENSE_A,
        IS_DENSE_B,
        BLOCK_D,
        IS_REPLACE,
    )


@triton.jit
def concat_2D_jagged_jagged_w_prefix(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_od,
    n_prefix_from_B,
    BLOCK_D: tl.constexpr,
):
    concat_2D_jagged_w_prefix(
        OffsetsA,
        ValuesA,
        OffsetsB,
        ValuesB,
        0,
        Out,
        D,
        stride_ad,
        stride_bd,
        0,
        stride_od,
        n_prefix_from_B,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=BLOCK_D,
        IS_REPLACE=False,
    )


@triton.jit
def split_2D_jagged_w_prefix(
    JaggedIn,
    DenseSize,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return

    if IS_REPLACE:
        seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_to_B

    offs_d = tl.arange(0, BLOCK_D)
    in_ptrs = JaggedIn + (seq_start + off_n).to(tl.int64) * stride_id + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_to_B:
        off_a = off_n - n_prefix_to_B
        out_ptrs = OutA + (off_a + seq_start_a).to(tl.int64) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_to_B
        if off_n < n_prefix_to_B:
            off_b += out_seq_b_start - n_prefix_to_B
        out_ptrs = OutB + (off_b + seq_start_b).to(tl.int64) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def split_2D_jagged(
    JaggedIn,
    DenseSize,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    split_2D_jagged_w_prefix(
        JaggedIn,
        DenseSize,
        OffsetsA,
        OffsetsB,
        OutA,
        OutB,
        D,
        stride_id,
        stride_ad,
        stride_bd,
        0,
        IS_DENSE_A,
        IS_DENSE_B,
        BLOCK_D,
        IS_REPLACE,
    )


@triton.jit
def split_2D_jagged_jagged_w_prefix(
    JaggedIn,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    BLOCK_D: tl.constexpr,
):
    split_2D_jagged_w_prefix(
        JaggedIn,
        0,
        OffsetsA,
        OffsetsB,
        OutA,
        OutB,
        D,
        stride_id,
        stride_ad,
        stride_bd,
        n_prefix_to_B,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=BLOCK_D,
        IS_REPLACE=False,
    )


def _triton_split_2D_jagged_internal(
    jagged_in: torch.Tensor,
    max_seq_len: int,
    B: int,
    offsets_a: Optional[torch.Tensor],
    offsets_b: Optional[torch.Tensor],
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    D: int,
    dense_size: int,
    n_prefix: int,
    is_dense_a: bool,
    is_dense_b: bool,
    is_replace: bool,
    BLOCK_D: int,
) -> None:
    if n_prefix != 0:
        if is_sm100():

            def grid(meta):
                return (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

            split_2D_jagged_jagged_w_prefix_multirow[grid](
                JaggedIn=jagged_in,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=out_a,
                OutB=out_b,
                D=D,
                stride_id=jagged_in.stride(0),
                stride_ad=out_a.stride(0),
                stride_bd=out_b.stride(0),
                n_prefix_to_B=n_prefix,
                BLOCK_D=BLOCK_D,
            )
        else:
            split_2D_jagged_jagged_w_prefix[(max_seq_len, B)](
                JaggedIn=jagged_in,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=out_a,
                OutB=out_b,
                D=D,
                stride_id=jagged_in.stride(0),
                stride_ad=out_a.stride(0),
                stride_bd=out_b.stride(0),
                n_prefix_to_B=n_prefix,
                BLOCK_D=BLOCK_D,  # pyre-ignore[6]
            )
    else:
        if is_sm100():

            def grid(meta):
                return (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

            split_2D_jagged_multirow[grid](
                JaggedIn=jagged_in,
                DenseSize=dense_size,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=out_a,
                OutB=out_b,
                D=D,
                stride_id=jagged_in.stride(0),
                stride_ad=out_a.stride(0),
                stride_bd=out_b.stride(0),
                IS_DENSE_A=is_dense_a,  # pyre-ignore[6]
                IS_DENSE_B=is_dense_b,  # pyre-ignore[6]
                BLOCK_D=BLOCK_D,  # pyre-ignore[6]
                IS_REPLACE=is_replace,  # pyre-ignore[6]
            )
        else:
            split_2D_jagged[(max_seq_len, B)](
                JaggedIn=jagged_in,
                DenseSize=dense_size,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=out_a,
                OutB=out_b,
                D=D,
                stride_id=jagged_in.stride(0),
                stride_ad=out_a.stride(0),
                stride_bd=out_b.stride(0),
                IS_DENSE_A=is_dense_a,  # pyre-ignore[6]
                IS_DENSE_B=is_dense_b,  # pyre-ignore[6]
                BLOCK_D=BLOCK_D,  # pyre-ignore[6]
                IS_REPLACE=is_replace,  # pyre-ignore[6]
            )


class _Concat2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        values_a: torch.Tensor,
        values_b: torch.Tensor,
        offsets_a: Optional[torch.Tensor] = None,
        offsets_b: Optional[torch.Tensor] = None,
        is_replace: bool = False,
        n_prefix_from_right: int = 0,
    ):
        values_a = switch_to_contiguous_if_needed(values_a)
        values_b = switch_to_contiguous_if_needed(values_b)
        is_dense_a = offsets_a is None
        is_dense_b = offsets_b is None
        dense_size: int = 0
        if is_dense_a:
            assert offsets_b is not None
            B, dense_size, D = values_a.shape
            seq_len_a = dense_size * B
            seq_len_b, _ = values_b.shape
            device = values_b.device
            dtype = values_b.dtype
            stride_dense_batch = values_a.stride(0)
        elif is_dense_b:
            assert offsets_a is not None
            B, dense_size, D = values_b.shape
            seq_len_a, _ = values_a.shape
            seq_len_b = dense_size * B
            device = values_a.device
            dtype = values_a.dtype
            stride_dense_batch = values_b.stride(0)
        else:
            assert offsets_a is not None and offsets_b is not None
            B = offsets_a.shape[0] - 1
            seq_len_a, D = values_a.shape
            seq_len_b, _ = values_b.shape
            device = values_a.device
            dtype = values_a.dtype
            stride_dense_batch = 0

        BLOCK_D = triton.next_power_of_2(D)
        if is_replace:
            values_out = torch.empty_like(values_a)
        else:
            values_out = torch.empty(
                (seq_len_a + seq_len_b, D), device=device, dtype=dtype
            )
        _triton_concat_2D_jagged_internal(
            values_a=values_a,
            values_b=values_b,
            values_out=values_out,
            max_seq_len=max_seq_len,
            B=B,
            offsets_a=offsets_a,
            offsets_b=offsets_b,
            D=D,
            dense_size=dense_size,
            stride_dense_batch=stride_dense_batch,
            n_prefix=n_prefix_from_right,
            is_dense_a=is_dense_a,
            is_dense_b=is_dense_b,
            is_replace=is_replace,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.seq_len_a = seq_len_a
        ctx.seq_len_b = seq_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.dense_size = dense_size
        ctx.is_replace = is_replace
        ctx.n_prefix_from_right = n_prefix_from_right
        return values_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, torch.Tensor, torch.Tensor, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        is_dense_a, is_dense_b, is_replace = (
            ctx.is_dense_a,
            ctx.is_dense_b,
            ctx.is_replace,
        )
        dense_size = ctx.dense_size
        if is_dense_a:
            B = offsets_b.shape[0] - 1
        else:
            B = offsets_a.shape[0] - 1
        _, D = d_out.shape
        BLOCK_D = triton.next_power_of_2(D)
        values_a = torch.zeros(
            (ctx.seq_len_a, D), device=d_out.device, dtype=d_out.dtype
        )
        values_b = torch.empty(
            (ctx.seq_len_b, D), device=d_out.device, dtype=d_out.dtype
        )
        _triton_split_2D_jagged_internal(
            jagged_in=d_out,
            max_seq_len=ctx.max_seq_len,
            B=B,
            offsets_a=offsets_a,
            offsets_b=offsets_b,
            out_a=values_a,
            out_b=values_b,
            D=D,
            dense_size=dense_size,
            n_prefix=ctx.n_prefix_from_right,
            is_dense_a=is_dense_a,
            is_dense_b=is_dense_b,
            is_replace=is_replace,
            BLOCK_D=BLOCK_D,
        )

        if is_dense_a:
            values_a = values_a.reshape((B, dense_size, D))
        elif is_dense_b:
            values_b = values_b.reshape((B, dense_size, D))
        return None, values_a, values_b, None, None, None, None


class _Split2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values: torch.Tensor,
        max_seq_len: int,
        offsets_a: Optional[torch.Tensor] = None,
        offsets_b: Optional[torch.Tensor] = None,
        dense_size: int = 0,
        n_prefix_to_right: int = 0,
        seq_len_a: Optional[int] = None,
        seq_len_b: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = switch_to_contiguous_if_needed(values)
        is_dense_a: bool = offsets_a is None
        is_dense_b: bool = offsets_b is None
        if is_dense_a:
            L, _ = values.shape
            assert offsets_b is not None
            B = offsets_b.shape[0] - 1
            seq_len_a = dense_size * B
            seq_len_b = L - seq_len_a
            offsets_a = offsets_b.new_empty(0)
        elif is_dense_b:
            L, _ = values.shape
            assert offsets_a is not None
            B = offsets_a.shape[0] - 1
            seq_len_b = dense_size * B
            seq_len_a = L - seq_len_b
            offsets_b = offsets_a.new_empty(0)
        else:
            assert offsets_a is not None and offsets_b is not None
            B = offsets_a.shape[0] - 1

            # Select the last offset item using torch.index_select instead of
            # "int(offsets_a[-1].item())" so that it won't cause "Cannot cast
            # FakeTensor to python number" error for AOTI.
            if torch.compiler.is_compiling():
                offsets_a_last_idx = torch.tensor(offsets_a.size(0) - 1).to(
                    offsets_a.device, non_blocking=True
                )
                offsets_b_last_idx = torch.tensor(offsets_b.size(0) - 1).to(
                    offsets_b.device, non_blocking=True
                )
                if seq_len_a is None:
                    seq_len_a = offsets_a.index_select(dim=0, index=offsets_a_last_idx)
                if seq_len_b is None:
                    seq_len_b = offsets_b.index_select(dim=0, index=offsets_b_last_idx)
            else:
                if seq_len_a is None:
                    seq_len_a = int(offsets_a[-1].item())
                if seq_len_b is None:
                    seq_len_b = int(offsets_b[-1].item())
        _, D = values.shape
        BLOCK_D = triton.next_power_of_2(D)
        # pyre-ignore[6] Incompatible parameter type
        values_a = torch.empty((seq_len_a, D), device=values.device, dtype=values.dtype)
        # pyre-ignore[6] Incompatible parameter type
        values_b = torch.empty((seq_len_b, D), device=values.device, dtype=values.dtype)
        _triton_split_2D_jagged_internal(
            jagged_in=values,
            max_seq_len=max_seq_len,
            B=B,
            offsets_a=offsets_a,
            offsets_b=offsets_b,
            out_a=values_a,
            out_b=values_b,
            D=D,
            dense_size=dense_size,
            n_prefix=n_prefix_to_right,
            is_dense_a=is_dense_a,
            is_dense_b=is_dense_b,
            is_replace=False,
            BLOCK_D=BLOCK_D,
        )
        if is_dense_a:
            values_a = values_a.reshape(B, dense_size, D)
        if is_dense_b:
            values_b = values_b.reshape(B, dense_size, D)
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.seq_len_a = seq_len_a
        ctx.seq_len_b = seq_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.dense_size = dense_size
        ctx.B = B
        ctx.D = D
        ctx.n_prefix_to_right = n_prefix_to_right
        return values_a, values_b

    @staticmethod
    def backward(
        ctx, *d_values
    ) -> Tuple[torch.Tensor, None, None, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        is_dense_a, is_dense_b = ctx.is_dense_a, ctx.is_dense_b
        values_a, values_b = d_values
        if is_dense_a:
            stride_dense_batch = values_a.stride(0)
        elif is_dense_b:
            stride_dense_batch = values_b.stride(0)
        else:
            stride_dense_batch = 0

        BLOCK_D = triton.next_power_of_2(ctx.D)
        dvalues = torch.empty(
            (ctx.seq_len_a + ctx.seq_len_b, ctx.D),
            device=values_a.device,
            dtype=values_b.dtype,
        )
        _triton_concat_2D_jagged_internal(
            values_a=values_a,
            values_b=values_b,
            values_out=dvalues,
            max_seq_len=ctx.max_seq_len,
            B=ctx.B,
            offsets_a=offsets_a,
            offsets_b=offsets_b,
            D=ctx.D,
            dense_size=ctx.dense_size,
            stride_dense_batch=stride_dense_batch,
            n_prefix=ctx.n_prefix_to_right,
            is_dense_a=is_dense_a,
            is_dense_b=is_dense_b,
            is_replace=False,
            BLOCK_D=BLOCK_D,
        )

        return dvalues, None, None, None, None, None, None, None


@torch.jit.unused
@torch.fx.wrap
def triton_jagged_dense_bmm_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    elementwise: bool = False,
) -> torch.Tensor:
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N) or (sum_B(M_i), N) depending on Elementwise, and Out has shape (sum_B(M_i), N)
    """
    return _JaggedDenseBmmAddFunction.apply(
        max_seq_len, seq_offsets, jagged, dense, bias, elementwise
    )


@torch.fx.wrap
def triton_concat_2D_jagged(
    max_seq_len: int,
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    is_replace: bool = False,
    n_prefix_from_right: int = 0,
) -> torch.Tensor:
    return _Concat2DJaggedFunction.apply(
        max_seq_len,
        values_a,
        values_b,
        offsets_a,
        offsets_b,
        is_replace,
        n_prefix_from_right,
    )


@torch.fx.wrap
def triton_concat_2D_jagged_jagged(
    max_seq_len_left: int,
    offsets_left: torch.Tensor,
    values_left: torch.Tensor,
    max_seq_len_right: int,
    offsets_right: torch.Tensor,
    values_right: torch.Tensor,
    is_replace: bool,
    n_prefix_from_right: int,
) -> torch.Tensor:
    return triton_concat_2D_jagged(
        max_seq_len=max_seq_len_left + max_seq_len_right,
        values_a=values_left,
        values_b=values_right,
        offsets_a=offsets_left,
        offsets_b=offsets_right,
        is_replace=is_replace,
        n_prefix_from_right=n_prefix_from_right,
    )


@torch.fx.wrap
def triton_concat_2D_dense_jagged(
    jagged_max_seq_len: int,
    jagged_offsets: torch.Tensor,
    jagged_values: torch.Tensor,
    dense_values: torch.Tensor,
) -> torch.Tensor:
    B, dense_size, D = dense_values.size()
    max_seq_len = jagged_max_seq_len + dense_size
    return triton_concat_2D_jagged(
        max_seq_len=max_seq_len,
        values_a=dense_values,
        values_b=jagged_values,
        offsets_a=None,
        offsets_b=jagged_offsets,
    )


def triton_jagged_dense_bmm(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    return _JaggedDenseBmmFunction.apply(max_seq_len, seq_offsets, jagged, dense)


@torch.jit.unused
def triton_split_2D_jagged(
    values: torch.Tensor,
    max_seq_len: int,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    dense_size: int = 0,
    n_prefix_to_right: int = 0,
    seq_len_a: Optional[int] = None,
    seq_len_b: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _Split2DJaggedFunction.apply(
        values,
        max_seq_len,
        offsets_a,
        offsets_b,
        dense_size,
        n_prefix_to_right,
        seq_len_a,
        seq_len_b,
    )


@triton.jit
def concat_2D_jagged_w_prefix_multirow(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    n_prefix_from_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_block_n = tl.program_id(0)

    if IS_DENSE_A:
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b

    if IS_REPLACE:
        seq_len = seq_len_a
        out_seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_len = seq_len_a + seq_len_b
        out_seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_from_B

    start_n = off_block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    if start_n >= seq_len:
        return
    valid_mask = offs_n < seq_len

    out_ptrs = (
        Out
        + (out_seq_start + offs_n[:, None]).to(tl.int64) * stride_od
        + offs_d[None, :]
    )

    to_a_mask = (offs_n < out_seq_b_start) & (offs_n >= n_prefix_from_B) & valid_mask
    to_b_mask = ~to_a_mask & valid_mask

    off_a = offs_n - n_prefix_from_B
    if IS_DENSE_A:
        in_a_ptrs = (
            ValuesA
            + off_a[:, None].to(tl.int64) * stride_ad
            + off_z.to(tl.int64) * stride_dense_batch
            + offs_d[None, :]
        )
    else:
        in_a_ptrs = (
            ValuesA
            + (off_a[:, None] + seq_start_a).to(tl.int64) * stride_ad
            + offs_d[None, :]
        )

    v_a = tl.load(in_a_ptrs, mask=to_a_mask[:, None] & (offs_d[None, :] < D), other=0.0)
    tl.store(out_ptrs, v_a, mask=to_a_mask[:, None] & (offs_d[None, :] < D))

    prefix_mask = offs_n < n_prefix_from_B

    off_b = tl.where(prefix_mask, offs_n, offs_n - out_seq_b_start + n_prefix_from_B)
    if IS_DENSE_B:
        in_b_ptrs = (
            ValuesB
            + off_b[:, None].to(tl.int64) * stride_bd
            + off_z.to(tl.int64) * stride_dense_batch
            + offs_d[None, :]
        )
    else:
        in_b_ptrs = (
            ValuesB
            + (off_b[:, None] + seq_start_b).to(tl.int64) * stride_bd
            + offs_d[None, :]
        )

    v_b = tl.load(in_b_ptrs, mask=to_b_mask[:, None] & (offs_d[None, :] < D), other=0.0)
    tl.store(out_ptrs, v_b, mask=to_b_mask[:, None] & (offs_d[None, :] < D))


@triton_autotune(
    configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def concat_2D_jagged_multirow(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    concat_2D_jagged_w_prefix_multirow(
        OffsetsA,
        ValuesA,
        OffsetsB,
        ValuesB,
        DenseSize,
        Out,
        D,
        stride_ad,
        stride_bd,
        stride_dense_batch,
        stride_od,
        0,
        IS_DENSE_A,
        IS_DENSE_B,
        BLOCK_D,
        BLOCK_N,
        IS_REPLACE,
    )


@triton_autotune(
    configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def concat_2D_jagged_jagged_w_prefix_multirow(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_od,
    n_prefix_from_B,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    concat_2D_jagged_w_prefix_multirow(
        OffsetsA,
        ValuesA,
        OffsetsB,
        ValuesB,
        0,
        Out,
        D,
        stride_ad,
        stride_bd,
        0,
        stride_od,
        n_prefix_from_B,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        IS_REPLACE=False,
    )


@triton.jit
def split_2D_jagged_w_prefix_multirow(
    JaggedIn,
    DenseSize,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_block_n = tl.program_id(0)

    if IS_DENSE_A:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b

    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b

    if IS_REPLACE:
        seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_to_B

    start_n = off_block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    if start_n >= seq_len:
        return
    valid_mask = offs_n < seq_len

    in_ptrs = (
        JaggedIn
        + (seq_start + offs_n[:, None]).to(tl.int64) * stride_id
        + offs_d[None, :]
    )

    v = tl.load(in_ptrs, mask=valid_mask[:, None] & (offs_d[None, :] < D), other=0.0)

    to_a_mask = (offs_n < out_seq_b_start) & (offs_n >= n_prefix_to_B) & valid_mask
    to_b_mask = ~to_a_mask & valid_mask

    off_a = offs_n - n_prefix_to_B
    out_a_ptrs = (
        OutA + (off_a[:, None] + seq_start_a).to(tl.int64) * stride_ad + offs_d[None, :]
    )
    tl.store(out_a_ptrs, v, mask=to_a_mask[:, None] & (offs_d[None, :] < D))

    prefix_mask = offs_n < n_prefix_to_B

    off_b = tl.where(prefix_mask, offs_n, offs_n - out_seq_b_start + n_prefix_to_B)
    out_b_ptrs = (
        OutB + (off_b[:, None] + seq_start_b).to(tl.int64) * stride_bd + offs_d[None, :]
    )
    tl.store(out_b_ptrs, v, mask=to_b_mask[:, None] & (offs_d[None, :] < D))


@triton_autotune(
    configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def split_2D_jagged_multirow(
    JaggedIn,
    DenseSize,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    split_2D_jagged_w_prefix_multirow(
        JaggedIn,
        DenseSize,
        OffsetsA,
        OffsetsB,
        OutA,
        OutB,
        D,
        stride_id,
        stride_ad,
        stride_bd,
        0,
        IS_DENSE_A,
        IS_DENSE_B,
        BLOCK_D,
        BLOCK_N,
        IS_REPLACE,
    )


@triton_autotune(
    configs=_get_split_concat_2d_jagged_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def split_2D_jagged_jagged_w_prefix_multirow(
    JaggedIn,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    split_2D_jagged_w_prefix_multirow(
        JaggedIn,
        0,
        OffsetsA,
        OffsetsB,
        OutA,
        OutB,
        D,
        stride_id,
        stride_ad,
        stride_bd,
        n_prefix_to_B,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        IS_REPLACE=False,
    )


def triton_jagged_dense_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    return _JaggedDenseBroadcastAddFunction.apply(
        max_seq_len, seq_offsets, jagged, dense
    )
