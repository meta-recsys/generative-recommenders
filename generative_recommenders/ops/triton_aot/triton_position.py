# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Optional

import torch
from generative_recommenders.common import (
    cdiv,
    fx_unwrap_optional_tensor,
    next_power_of_2,
    prev_power_of_2,
    should_trigger_eager_impl,
)
from generative_recommenders.ops.pytorch.pt_position import (
    pytorch_add_timestamp_positional_embeddings,
)
from generative_recommenders.ops.triton.triton_position import (
    _add_timestamp_position_embeddings_kernel,
)
from generative_recommenders.ops.triton_aot.types import triton_aot


_add_timestamp_position_embeddings_kernel = triton_aot(
    annotations={
        "SeqEmb": ("*bf16", 16),
        "Offsets": ("*i64", 16),
        "Lengths": ("*i64", 16),
        "PosEmb": ("*fp32", 16),
        "TsEmb": ("*fp32", 16),
        "Out": ("*bf16", 16),
        "TS": ("*i64", 16),
        "PosInds": ("*i32", 16),
        "TsInds": ("*i32", 16),
        "NumTargets": ("*i64", 16),
        "AUTOTUNE_MAX_SEQ_LEN": "i32",
        "D": "i32",
        "num_time_buckets": "i32",
        "time_bucket_increments": "fp32",
        "time_bucket_scale": "fp32",
        "time_delta": "i32",
        "max_contextual_seq_len": "i32",
        "max_pos_ind": "i32",
        "stride_sn": ("i32", 16),
        "stride_pn": ("i32", 16),
        "stride_tn": ("i32", 16),
        "stride_on": ("i32", 16),
    },
)(_add_timestamp_position_embeddings_kernel)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_position(
    seq_embeddings: torch.Tensor,
    seq_offsets: torch.Tensor,
    pos_embeddings: torch.Tensor,
    ts_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    max_seq_len: int,
    max_contextual_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
) -> torch.Tensor:
    has_multiple_targets = num_targets is not None
    if not has_multiple_targets:
        num_targets_resolved = torch.empty(
            0, dtype=torch.int64, device=seq_embeddings.device
        )
    else:
        num_targets_resolved = fx_unwrap_optional_tensor(num_targets).to(torch.int64)

    seq_embeddings = seq_embeddings.contiguous()
    pos_embeddings = pos_embeddings.contiguous()
    ts_embeddings = ts_embeddings.contiguous()

    max_pos_ind = pos_embeddings.shape[0]
    B = seq_lengths.shape[0]

    N, D = seq_embeddings.shape
    out = torch.empty_like(seq_embeddings)

    timestamps = timestamps.contiguous()
    ts_inds = torch.empty((N,), device=timestamps.device, dtype=torch.int32)
    pos_inds = torch.empty((N,), device=timestamps.device, dtype=torch.int32)

    autotune_max_seq_len = prev_power_of_2(max_seq_len)
    BLOCK_D = next_power_of_2(D) if D < 64 else 64

    grid = lambda meta: (  # noqa E731
        B,
        cdiv(max_seq_len, meta["BLOCK_N"]),
    )
    # pyrefly: ignore [not-callable]
    _add_timestamp_position_embeddings_kernel[grid](
        SeqEmb=seq_embeddings,
        Offsets=seq_offsets,
        Lengths=seq_lengths,
        PosEmb=pos_embeddings,
        TsEmb=ts_embeddings,
        Out=out,
        TS=timestamps,
        PosInds=pos_inds,
        TsInds=ts_inds,
        NumTargets=num_targets_resolved,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len,
        D=D,
        num_time_buckets=2048,
        time_bucket_increments=60.0,
        time_bucket_scale=1.0,
        time_delta=0,
        max_contextual_seq_len=max_contextual_seq_len,
        max_pos_ind=max_pos_ind,
        stride_sn=seq_embeddings.stride(0),
        stride_pn=pos_embeddings.stride(0),
        stride_tn=ts_embeddings.stride(0),
        stride_on=out.stride(0),
        TRAINING=False,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        INTERLEAVE_TARGETS=interleave_targets,
        TIME_BUCKET_FN=time_bucket_fn,
        BLOCK_D=BLOCK_D,
    )

    return out


@torch.fx.wrap
# "aot_triton_kernel_wrapper_" is a pre-defined prefix for
# AOT-T triton kernel wrapper functions. This is required for
# AOT-T backend to recognize and trace correctly for ops transformation.
def aot_triton_kernel_wrapper_position(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str,
) -> torch.Tensor:
    seq_embeddings = seq_embeddings * alpha
    if should_trigger_eager_impl():
        return pytorch_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    else:
        return _triton_aot_position(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
