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

#!/usr/bin/env python3

from typing import Optional

import torch
from generative_recommenders.common import (
    autotune_max_seq_len,
    BACKEND_ALLOW_TF32,
    cdiv,
    prev_power_of_2,
    should_trigger_eager_impl,
)
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)
from generative_recommenders.ops.triton.triton_hstu_attention import _hstu_attn_fwd
from generative_recommenders.ops.triton_aot.types import triton_aot


for _config in _hstu_attn_fwd.configs:
    if isinstance(_config.kwargs.get("USE_TLX"), bool):
        _config.kwargs["USE_TLX"] = int(_config.kwargs["USE_TLX"])


_hstu_attn_fwd = triton_aot(
    annotations={
        "stride_qm": ("i32", 16),
        "stride_qh": ("i32", 16),
        "stride_kn": ("i32", 16),
        "stride_kh": ("i32", 16),
        "stride_vn": ("i32", 16),
        "stride_vh": ("i32", 16),
        "stride_om": ("i32", 16),
        "stride_oh": ("i32", 16),
        "contextual_seq_len": "i32",
        "max_attn_len": "i32",
        "Z": "i32",
        "AUTOTUNE_Z": "i32",
        "H": "i32",
        "MAX_SEQ_LEN": "i32",
        "AUTOTUNE_MAX_SEQ_LEN": "i32",
        "DimQ": "i32",
        "DimV": "i32",
        "DeltaSize": "i32",
        "workspace_ptr": "*i8",
        "sort_by_length_indices": "*i64",
    }
)(_hstu_attn_fwd)


def _check_common_args(
    invalid_attn_mask_type: str,
    attn_scale: Optional[torch.Tensor],
    full_attn_size: int,
    num_softmax_heads: int,
) -> None:
    assert invalid_attn_mask_type in ("causal", "lower_triangular"), (
        f"unsupported invalid_attn_mask_type: {invalid_attn_mask_type}"
    )
    assert attn_scale is None, "attn_scale is not implemented for AOT-T HSTU MHA"
    assert full_attn_size == 0, "full_attn_size is not implemented for AOT-T HSTU MHA"
    assert num_softmax_heads == 0, (
        "num_softmax_heads is not implemented for AOT-T HSTU MHA"
    )


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_ragged_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int = 0,
    allow_tf32: bool = BACKEND_ALLOW_TF32,
) -> torch.Tensor:
    assert invalid_attn_mask_type in ("causal", "lower_triangular"), (
        f"unsupported invalid_attn_mask_type: {invalid_attn_mask_type}"
    )
    assert attn_scale is None, "attn_scale is not implemented for AOT-T HSTU MHA"
    assert full_attn_size == 0, "full_attn_size is not implemented for AOT-T HSTU MHA"
    assert num_softmax_heads == 0, (
        "num_softmax_heads is not implemented for AOT-T HSTU MHA"
    )
    Z = seq_offsets.numel() - 1
    L, H, DimQ = q.shape
    DimV = v.shape[2]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seq_offsets = seq_offsets.contiguous()

    out = torch.empty_like(v)
    if L == 0:
        return out
    workspace = torch.empty(0, dtype=torch.int8, device=q.device)
    sort_by_length_indices = torch.empty(
        0, dtype=torch.int64, device=seq_offsets.device
    )

    grid = lambda meta: (  # noqa E731
        cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )
    # pyrefly: ignore [not-callable]
    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
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
        contextual_seq_len=contextual_seq_len,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=prev_power_of_2(Z),
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        IS_DELTA_Q=False,
        ALLOW_TF32=allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len > 0,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        HAS_SORT_BY_LENGTH_INDICES=False,
        ENABLE_TMA=False,
        TMA_DESC_SIZE=128,
    )
    return out


@torch.fx.wrap
def aot_triton_kernel_wrapper_ragged_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    full_attn_size: int,
    num_softmax_heads: int,
    allow_tf32: bool = BACKEND_ALLOW_TF32,
) -> torch.Tensor:
    _check_common_args(
        invalid_attn_mask_type=invalid_attn_mask_type,
        attn_scale=attn_scale,
        full_attn_size=full_attn_size,
        num_softmax_heads=num_softmax_heads,
    )
    if should_trigger_eager_impl():
        return pytorch_hstu_mha(
            max_seq_len=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            attn_scale=attn_scale,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=full_attn_size,
        )
    return _triton_aot_ragged_hstu_mha(
        N=N,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        invalid_attn_mask_type=invalid_attn_mask_type,
        num_targets=num_targets,
        attn_scale=attn_scale,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        full_attn_size=full_attn_size,
        num_softmax_heads=num_softmax_heads,
        allow_tf32=allow_tf32,
    )


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_x_offsets: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    full_attn_size: int,
    allow_tf32: bool = BACKEND_ALLOW_TF32,
) -> torch.Tensor:
    assert attn_scale is None, "attn_scale is not implemented for AOT-T HSTU MHA"
    assert full_attn_size == 0, "full_attn_size is not implemented for AOT-T HSTU MHA"
    Z = seq_offsets.size(0) - 1
    DELTA_L, H, DimQ = delta_q.shape
    DeltaSize = DELTA_L // Z
    DimV = v.shape[2]

    delta_q = delta_q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seq_offsets = seq_offsets.contiguous()

    out = torch.empty((DELTA_L, H, DimV), dtype=delta_q.dtype, device=delta_q.device)
    if DELTA_L == 0:
        return out
    workspace = torch.empty(0, dtype=torch.int8, device=delta_q.device)
    sort_by_length_indices = torch.empty(
        0, dtype=torch.int64, device=seq_offsets.device
    )

    grid = lambda meta: (  # noqa E731
        cdiv(DeltaSize, meta["BLOCK_M"]),
        Z * H,
    )
    # pyrefly: ignore [not-callable]
    _hstu_attn_fwd[grid](
        Q=delta_q,
        K=k,
        V=v,
        workspace_ptr=workspace,
        sort_by_length_indices=sort_by_length_indices,
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
        contextual_seq_len=0,
        max_attn_len=max_attn_len,
        Z=Z,
        AUTOTUNE_Z=prev_power_of_2(Z),
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=DeltaSize,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        IS_DELTA_Q=True,
        ALLOW_TF32=allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=False,
        HAS_MAX_ATTN_LEN=max_attn_len > 0,
        HAS_SORT_BY_LENGTH_INDICES=False,
        ENABLE_TMA=False,
        TMA_DESC_SIZE=128,
    )
    return out


@torch.fx.wrap
def aot_triton_kernel_wrapper_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_x_offsets: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor],
    max_attn_len: int,
    full_attn_size: int,
) -> torch.Tensor:
    _check_common_args(
        invalid_attn_mask_type="causal",
        attn_scale=attn_scale,
        full_attn_size=full_attn_size,
        num_softmax_heads=0,
    )
    if should_trigger_eager_impl():
        return pytorch_cached_hstu_mha(
            max_seq_len=N,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=0,
        )
    return _triton_aot_cached_hstu_mha(
        N=N,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        delta_x_offsets=delta_x_offsets,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        attn_scale=attn_scale,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
    )
