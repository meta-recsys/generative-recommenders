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

from typing import Optional

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    next_power_of_2,
    should_trigger_eager_impl,
)
from generative_recommenders.ops.pytorch.pt_jagged import (
    pytorch_replace_last_n_with_jagged,
)
from generative_recommenders.ops.pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
)
from generative_recommenders.ops.triton.triton_jagged import concat_2D_jagged
from generative_recommenders.ops.triton_aot.types import triton_aot


concat_2D_jagged = triton_aot(
    annotations={
        "DenseSize": "i32",
        "D": "i32",
        "stride_ad": "i32",
        "stride_bd": "i32",
        "stride_dense_batch": "i32",
        "stride_od": "i32",
    },
    # pyrefly: ignore [bad-argument-type]
)(concat_2D_jagged)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_concat_2D_jagged(
    max_seq_len: int,
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    is_replace: bool = False,
) -> torch.Tensor:
    is_dense_a = offsets_a is None
    is_dense_b = offsets_b is None

    dense_size: int = 0
    if is_dense_a:
        B, dense_size, D = values_a.size()
        offsets_b = fx_unwrap_optional_tensor(offsets_b)
        jagged_seq_len, _ = values_b.shape
        values_out = torch.empty(
            (dense_size * B + jagged_seq_len, D),
            device=values_b.device,
            dtype=values_b.dtype,
        )
        offsets_a = offsets_b.new_empty(0)
        stride_dense_batch = values_a.stride(0)
    elif is_dense_b:
        B, dense_size, D = values_b.size()
        offsets_a = fx_unwrap_optional_tensor(offsets_a)
        jagged_seq_len, _ = values_a.shape
        values_out = torch.empty(
            (jagged_seq_len + dense_size * B, D),
            device=values_a.device,
            dtype=values_a.dtype,
        )
        offsets_b = offsets_a.new_empty(0)
        stride_dense_batch = values_b.stride(0)
    else:
        offsets_a = fx_unwrap_optional_tensor(offsets_a)
        offsets_b = fx_unwrap_optional_tensor(offsets_b)
        B = offsets_a.size(0) - 1
        seq_len_a, D = values_a.shape
        seq_len_b, _ = values_b.shape
        if is_replace:
            values_out = torch.empty_like(values_a)
        else:
            values_out = torch.empty(
                (seq_len_a + seq_len_b, D), device=values_a.device, dtype=values_a.dtype
            )
        stride_dense_batch = 0

    # Make sure offsets are alignted on 16-byte to match AOTT spec
    if (
        offsets_a is not None
        and (offsets_a.storage_offset() * offsets_a.element_size()) % 16 != 0
    ):
        offsets_a = offsets_a.clone()
    if (
        offsets_b is not None
        and (offsets_b.storage_offset() * offsets_b.element_size()) % 16 != 0
    ):
        offsets_b = offsets_b.clone()

    BLOCK_D = next_power_of_2(D)

    grid = (max_seq_len, B)
    # pyrefly: ignore [not-callable]
    concat_2D_jagged[grid](
        OffsetsA=offsets_a,
        ValuesA=values_a,
        OffsetsB=offsets_b,
        ValuesB=values_b,
        DenseSize=dense_size,
        Out=values_out,
        D=D,
        stride_ad=(values_a.stride(1) if is_dense_a else values_a.stride(0)),
        stride_bd=(values_b.stride(1) if is_dense_b else values_b.stride(0)),
        stride_dense_batch=stride_dense_batch,
        stride_od=values_out.stride(0),
        # pyrefly: ignore [bad-argument-type]
        IS_DENSE_A=is_dense_a,
        # pyrefly: ignore [bad-argument-type]
        IS_DENSE_B=is_dense_b,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_D=BLOCK_D,
        # pyrefly: ignore [bad-argument-type]
        IS_REPLACE=is_replace,
    )
    return values_out


@torch.fx.wrap
# "aot_triton_kernel_wrapper_" is a pre-defined prefix for
# AOT-T triton kernel wrapper functions. This is required for
# AOT-T backend to recognize and trace correctly for ops transformation.
def aot_triton_kernel_wrapper_concat_2D_jagged(
    max_seq_len: int,
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    is_replace: bool = False,
) -> torch.Tensor:
    if should_trigger_eager_impl():
        if is_replace:
            assert offsets_a is not None and offsets_b is not None
            return pytorch_replace_last_n_with_jagged(
                max_seq_len_left=max_seq_len,
                offsets_left=offsets_a,
                values_left=values_a,
                offsets_right=offsets_b,
                values_right=values_b,
            )
        return pytorch_concat_2D_jagged(
            values_left=values_a,
            values_right=values_b,
            max_len_left=max_seq_len if offsets_a is None else None,
            max_len_right=max_seq_len if offsets_b is None else None,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
        )
    else:
        return _triton_aot_concat_2D_jagged(
            max_seq_len=max_seq_len,
            values_a=values_a,
            values_b=values_b,
            offsets_a=offsets_a,
            offsets_b=offsets_b,
            is_replace=is_replace,
        )
