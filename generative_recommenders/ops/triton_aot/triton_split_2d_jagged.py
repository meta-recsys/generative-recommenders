# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Optional, Tuple

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    next_power_of_2,
    should_trigger_eager_impl,
)
from generative_recommenders.ops.pytorch.pt_jagged_tensors import (
    pytorch_split_2D_jagged,
)
from generative_recommenders.ops.triton.triton_jagged import split_2D_jagged
from generative_recommenders.ops.triton_aot.types import triton_aot


split_2D_jagged = triton_aot(
    annotations={
        "DenseSize": "i32",
        "D": ("i32", 16),
        "stride_id": ("i32", 16),
        "stride_ad": ("i32", 16),
        "stride_bd": ("i32", 16),
    },
    # pyrefly: ignore [bad-argument-type]
)(split_2D_jagged)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_split_2D_jagged(
    values: torch.Tensor,
    max_seq_len: int,
    offsets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    dense_size: int = 0,
    is_dense_a: bool = False,
    is_dense_b: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, D = values.shape
    BLOCK_D = next_power_of_2(D)

    if is_dense_a:
        L, _ = values.shape
        B = offsets_b.size(0) - 1
        seq_len_a = dense_size * B
        seq_len_b = L - seq_len_a
    elif is_dense_b:
        L, _ = values.shape
        B = offsets_a.size(0) - 1
        seq_len_b = dense_size * B
        seq_len_a = L - seq_len_b
    else:
        B = offsets_a.size(0) - 1
        seq_len_a = int(offsets_a[-1].item())
        seq_len_b = int(offsets_b[-1].item())

    values_a = torch.empty((seq_len_a, D), device=values.device, dtype=values.dtype)
    values_b = torch.empty((seq_len_b, D), device=values.device, dtype=values.dtype)

    grid = (max_seq_len, B)
    # pyre-ignore[29]: TritonAOT.__getitem__ is callable at runtime
    split_2D_jagged[grid](
        JaggedIn=values,
        DenseSize=dense_size,
        OffsetsA=offsets_a,
        OffsetsB=offsets_b,
        OutA=values_a,
        OutB=values_b,
        D=D,
        stride_id=values.stride(0),
        stride_ad=values_a.stride(0),
        stride_bd=values_b.stride(0),
        # pyrefly: ignore [bad-argument-type]
        IS_DENSE_A=is_dense_a,
        # pyrefly: ignore [bad-argument-type]
        IS_DENSE_B=is_dense_b,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_D=BLOCK_D,
        # pyrefly: ignore [bad-argument-type]
        IS_REPLACE=False,
    )

    if is_dense_a:
        values_a = values_a.reshape(B, dense_size, D)
    if is_dense_b:
        values_b = values_b.reshape(B, dense_size, D)

    return values_a, values_b


@torch.fx.wrap
def aot_triton_kernel_wrapper_split_2D_jagged(
    values: torch.Tensor,
    max_seq_len: int,
    offsets_a: Optional[torch.Tensor] = None,
    offsets_b: Optional[torch.Tensor] = None,
    dense_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if should_trigger_eager_impl():
        assert offsets_a is not None and offsets_b is not None, (
            "Eager fallback requires both offsets_a and offsets_b"
        )
        return pytorch_split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            max_len_left=None,
            max_len_right=None,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
        )
    else:
        is_dense_a: bool = offsets_a is None
        is_dense_b: bool = offsets_b is None
        resolved_offsets_a: torch.Tensor = values.new_empty(0)
        resolved_offsets_b: torch.Tensor = values.new_empty(0)
        if is_dense_a:
            resolved_offsets_b = fx_unwrap_optional_tensor(offsets_b)
            resolved_offsets_a = resolved_offsets_b.new_empty(0)
        elif is_dense_b:
            resolved_offsets_a = fx_unwrap_optional_tensor(offsets_a)
            resolved_offsets_b = resolved_offsets_a.new_empty(0)
        else:
            resolved_offsets_a = fx_unwrap_optional_tensor(offsets_a)
            resolved_offsets_b = fx_unwrap_optional_tensor(offsets_b)

        return _triton_aot_split_2D_jagged(
            values=values,
            max_seq_len=max_seq_len,
            offsets_a=resolved_offsets_a,
            offsets_b=resolved_offsets_b,
            dense_size=dense_size,
            is_dense_a=is_dense_a,
            is_dense_b=is_dense_b,
        )
