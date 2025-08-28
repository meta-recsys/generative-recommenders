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


def _get_concat_split_2d_jagged_multirow_configs() -> List[triton.Config]:
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


@triton_autotune(
    configs=_get_concat_split_2d_jagged_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _concat_2D_jagged(
    ValuesA,
    ValuesB,
    OffsetsA,
    OffsetsB,
    MaxLenA,
    MaxLenB,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_od,
    n_prefix_from_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_z = tl.program_id(1)
    block_n = tl.program_id(0)

    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b

    start_n = block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_mask = offs_n < seq_len

    out_seq_start = seq_start_a + seq_start_b + offs_n
    out_ptrs = Out + out_seq_start[:, None].to(tl.int64) * stride_od + offs_d[None, :]

    from_prefix_b_mask = (offs_n < n_prefix_from_B) & valid_mask
    from_a_mask = (
        (offs_n >= n_prefix_from_B)
        & (offs_n < seq_len_a + n_prefix_from_B)
        & valid_mask
    )
    from_suffix_b_mask = (offs_n >= seq_len_a + n_prefix_from_B) & valid_mask

    in_b1_ptrs = (
        ValuesB
        + (offs_n[:, None] + seq_start_b).to(tl.int64) * stride_bd
        + offs_d[None, :]
    )
    v_b1 = tl.load(
        in_b1_ptrs, mask=from_prefix_b_mask[:, None] & (offs_d[None, :] < D), other=0.0
    )
    tl.store(out_ptrs, v_b1, mask=from_prefix_b_mask[:, None] & (offs_d[None, :] < D))

    off_a = offs_n - n_prefix_from_B
    in_a_ptrs = (
        ValuesA
        + (off_a[:, None] + seq_start_a).to(tl.int64) * stride_ad
        + offs_d[None, :]
    )
    v_a = tl.load(
        in_a_ptrs, mask=from_a_mask[:, None] & (offs_d[None, :] < D), other=0.0
    )
    tl.store(out_ptrs, v_a, mask=from_a_mask[:, None] & (offs_d[None, :] < D))

    off_b = offs_n - seq_len_a
    in_b2_ptrs = (
        ValuesB
        + (off_b[:, None] + seq_start_b).to(tl.int64) * stride_bd
        + offs_d[None, :]
    )
    v_b2 = tl.load(
        in_b2_ptrs, mask=from_suffix_b_mask[:, None] & (offs_d[None, :] < D), other=0.0
    )
    tl.store(out_ptrs, v_b2, mask=from_suffix_b_mask[:, None] & (offs_d[None, :] < D))


@triton_autotune(
    configs=_get_concat_split_2d_jagged_multirow_configs(),
    key=["BLOCK_D"],
)
@triton.jit
def _split_2D_jagged(
    JaggedIn,
    OffsetsA,
    OffsetsB,
    MaxLenA,
    MaxLenB,
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
):
    off_z = tl.program_id(1)
    block_n = tl.program_id(0)

    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    seq_start = seq_start_a + seq_start_b

    start_n = block_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_mask = offs_n < seq_len

    in_ptrs = (
        JaggedIn
        + (seq_start + offs_n[:, None]).to(tl.int64) * stride_id
        + offs_d[None, :]
    )

    v = tl.load(in_ptrs, mask=valid_mask[:, None] & (offs_d[None, :] < D), other=0.0)

    to_prefix_b_mask = (offs_n < n_prefix_to_B) & valid_mask
    to_a_mask = (
        (offs_n >= n_prefix_to_B) & (offs_n < seq_len_a + n_prefix_to_B) & valid_mask
    )
    to_suffix_b_mask = (offs_n >= seq_len_a + n_prefix_to_B) & valid_mask

    out_b1_ptrs = (
        OutB
        + (offs_n[:, None] + seq_start_b).to(tl.int64) * stride_bd
        + offs_d[None, :]
    )
    tl.store(out_b1_ptrs, v, mask=to_prefix_b_mask[:, None] & (offs_d[None, :] < D))

    off_a = offs_n - n_prefix_to_B
    out_a_ptrs = (
        OutA + (off_a[:, None] + seq_start_a).to(tl.int64) * stride_ad + offs_d[None, :]
    )
    tl.store(out_a_ptrs, v, mask=to_a_mask[:, None] & (offs_d[None, :] < D))

    off_b = offs_n - seq_len_a
    out_b2_ptrs = (
        OutB + (off_b[:, None] + seq_start_b).to(tl.int64) * stride_bd + offs_d[None, :]
    )
    tl.store(out_b2_ptrs, v, mask=to_suffix_b_mask[:, None] & (offs_d[None, :] < D))


class _Concat2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        values_a: torch.Tensor,
        values_b: torch.Tensor,
        max_len_a: Optional[int],
        max_len_b: Optional[int],
        offsets_a: Optional[torch.Tensor],
        offsets_b: Optional[torch.Tensor],
        n_prefix_from_B: int,
    ):
        values_a = switch_to_contiguous_if_needed(values_a)
        values_b = switch_to_contiguous_if_needed(values_b)
        is_dense_a = offsets_a is None
        is_dense_b = offsets_b is None
        total_len_a, D = values_a.shape
        total_len_b, _ = values_b.shape
        if is_dense_a:
            assert max_len_a is not None
            B = total_len_a // max_len_a
        else:
            assert offsets_a is not None
            B = offsets_a.shape[0] - 1
        if is_dense_b:
            assert max_len_b is not None
            B = total_len_b // max_len_b
        else:
            assert offsets_b is not None
            B = offsets_b.shape[0] - 1
        total_seq_len = total_len_a + total_len_b
        BLOCK_D = triton.next_power_of_2(D)
        values_out = torch.empty(
            (total_seq_len, D), device=values_a.device, dtype=values_a.dtype
        )

        def get_grid(meta):
            return (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

        _concat_2D_jagged[get_grid](
            ValuesA=values_a,
            ValuesB=values_b,
            OffsetsA=offsets_a,
            OffsetsB=offsets_b,
            MaxLenA=max_len_a,
            MaxLenB=max_len_b,
            Out=values_out,
            D=D,
            stride_ad=values_a.stride(-2),
            stride_bd=values_b.stride(-2),
            stride_od=values_out.stride(-2),
            n_prefix_from_B=n_prefix_from_B,
            # pyre-ignore[6]
            IS_DENSE_A=is_dense_a,
            # pyre-ignore[6]
            IS_DENSE_B=is_dense_b,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.total_len_a = total_len_a
        ctx.total_len_b = total_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.max_len_a = max_len_a
        ctx.max_len_b = max_len_b
        ctx.B = B
        ctx.n_prefix_from_B = n_prefix_from_B
        return values_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, torch.Tensor, torch.Tensor, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        _, D = d_out.shape
        BLOCK_D = triton.next_power_of_2(D)
        d_values_a = torch.zeros(
            (ctx.total_len_a, D), device=d_out.device, dtype=d_out.dtype
        )
        d_values_b = torch.empty(
            (ctx.total_len_b, D), device=d_out.device, dtype=d_out.dtype
        )

        def get_grid(meta):
            return (triton.cdiv(ctx.max_seq_len, meta["BLOCK_N"]), ctx.B)

        _split_2D_jagged[get_grid](
            JaggedIn=d_out,
            OffsetsA=offsets_a,
            OffsetsB=offsets_b,
            MaxLenA=ctx.max_len_a,
            MaxLenB=ctx.max_len_b,
            OutA=d_values_a,
            OutB=d_values_b,
            D=D,
            stride_id=d_out.stride(-2),
            stride_ad=d_values_a.stride(-2),
            stride_bd=d_values_b.stride(-2),
            n_prefix_to_B=ctx.n_prefix_from_B,
            BLOCK_D=BLOCK_D,
            IS_DENSE_A=ctx.is_dense_a,
            IS_DENSE_B=ctx.is_dense_b,
        )
        return None, d_values_a, d_values_b, None, None, None, None, None


class _Split2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        values: torch.Tensor,
        total_len_left: Optional[int],
        total_len_right: Optional[int],
        max_len_a: Optional[int],
        max_len_b: Optional[int],
        offsets_a: Optional[torch.Tensor],
        offsets_b: Optional[torch.Tensor],
        n_prefix_to_B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = switch_to_contiguous_if_needed(values)
        is_dense_a: bool = offsets_a is None
        is_dense_b: bool = offsets_b is None
        total_seq_len, D = values.shape
        if is_dense_a:
            assert is_dense_b is False
            assert offsets_b is not None
            assert max_len_a is not None
            B = offsets_b.shape[0] - 1
            total_len_a = max_len_a * B
            total_len_b = total_seq_len - total_len_a
        elif is_dense_b:
            assert is_dense_a is False
            assert offsets_a is not None
            assert max_len_b is not None
            B = offsets_a.shape[0] - 1
            total_len_b = max_len_b * B
            total_len_a = total_seq_len - total_len_b
        else:
            assert offsets_a is not None and offsets_b is not None
            B = offsets_a.shape[0] - 1
            if total_len_left is not None and total_len_right is not None:
                assert total_len_left + total_len_right == total_seq_len
                total_len_a = total_len_left
                total_len_b = total_len_right
            else:
                total_len_a = int(offsets_a[-1].item())
                total_len_b = values.size(0) - total_len_a
        _, D = values.shape
        BLOCK_D = triton.next_power_of_2(D)
        values_a = torch.empty(
            (total_len_a, D), device=values.device, dtype=values.dtype
        )
        values_b = torch.empty(
            (total_len_b, D), device=values.device, dtype=values.dtype
        )

        def get_grid(meta):
            return (triton.cdiv(max_seq_len, meta["BLOCK_N"]), B)

        _split_2D_jagged[get_grid](
            JaggedIn=values,
            OffsetsA=offsets_a,
            OffsetsB=offsets_b,
            MaxLenA=max_len_a,
            MaxLenB=max_len_b,
            OutA=values_a,
            OutB=values_b,
            D=D,
            stride_id=values.stride(0),
            stride_ad=values_a.stride(0),
            stride_bd=values_b.stride(0),
            n_prefix_to_B=n_prefix_to_B,
            # pyre-ignore[6]
            IS_DENSE_A=is_dense_a,
            # pyre-ignore[6]
            IS_DENSE_B=is_dense_b,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.total_seq_len = total_seq_len
        ctx.max_len_a = max_len_a
        ctx.max_len_b = max_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.B = B
        ctx.D = D
        ctx.n_prefix_to_B = n_prefix_to_B
        return values_a, values_b

    @staticmethod
    def backward(
        ctx, *d_values
    ) -> Tuple[None, torch.Tensor, None, None, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        d_values_a, d_values_b = d_values
        BLOCK_D = triton.next_power_of_2(ctx.D)
        d_jagged_in = torch.empty(
            (ctx.total_seq_len, ctx.D),
            device=d_values_a.device,
            dtype=d_values_a.dtype,
        )

        def get_grid(meta):
            return (triton.cdiv(ctx.max_seq_len, meta["BLOCK_N"]), ctx.B)

        _concat_2D_jagged[get_grid](
            ValuesA=d_values_a,
            ValuesB=d_values_b,
            OffsetsA=offsets_a,
            OffsetsB=offsets_b,
            MaxLenA=ctx.max_len_a,
            MaxLenB=ctx.max_len_b,
            Out=d_jagged_in,
            D=ctx.D,
            stride_ad=d_values_a.stride(-2),
            stride_bd=d_values_b.stride(-2),
            stride_od=d_jagged_in.stride(-2),
            n_prefix_from_B=ctx.n_prefix_to_B,
            IS_DENSE_A=ctx.is_dense_a,
            IS_DENSE_B=ctx.is_dense_b,
            BLOCK_D=BLOCK_D,
        )

        return None, d_jagged_in, None, None, None, None, None, None, None


@torch.fx.wrap
def triton_concat_2D_jagged(
    max_seq_len: int,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
    n_prefix_from_right: int = 0,
) -> torch.Tensor:
    return _Concat2DJaggedFunction.apply(
        max_seq_len,
        values_left,
        values_right,
        max_len_left,
        max_len_right,
        offsets_left,
        offsets_right,
        n_prefix_from_right,
    )


@torch.fx.wrap
def triton_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int],
    total_len_right: Optional[int],
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
    n_prefix_to_right: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _Split2DJaggedFunction.apply(
        max_seq_len,
        values,
        total_len_left,
        total_len_right,
        max_len_left,
        max_len_right,
        offsets_left,
        offsets_right,
        n_prefix_to_right,
    )
