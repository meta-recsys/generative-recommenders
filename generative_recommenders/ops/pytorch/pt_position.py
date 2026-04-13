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

# pyre-strict

from typing import Optional

import torch
from generative_recommenders.common import (
    fx_unwrap_optional_tensor,
    jagged_to_padded_dense,
)

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def torch_arange(end: int, device: torch.device) -> torch.Tensor:
    return torch.arange(end, device=device)


@torch.fx.wrap
def _get_col_indices(
    max_seq_len: int,
    max_contextual_seq_len: int,
    max_pos_ind: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    B = seq_lengths.size(0)
    col_indices = torch.arange(max_seq_len, device=seq_lengths.device).expand(
        B, max_seq_len
    )
    if num_targets is not None:
        if interleave_targets:
            high_inds = seq_lengths - fx_unwrap_optional_tensor(num_targets) * 2
        else:
            high_inds = seq_lengths - fx_unwrap_optional_tensor(num_targets)
        col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))
        col_indices = high_inds.view(-1, 1) - col_indices
    else:
        col_indices = seq_lengths.view(-1, 1) - col_indices
    col_indices = col_indices + max_contextual_seq_len
    col_indices = torch.clamp(col_indices, max=max_pos_ind - 1)
    if max_contextual_seq_len > 0:
        col_indices[:, :max_contextual_seq_len] = torch.arange(
            0,
            max_contextual_seq_len,
            device=col_indices.device,
            dtype=col_indices.dtype,
        ).view(1, -1)
    return col_indices


def pytorch_add_timestamp_positional_embeddings(
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
    max_pos_ind = pos_embeddings.size(0)
    # position encoding
    pos_inds = _get_col_indices(
        max_seq_len=max_seq_len,
        max_contextual_seq_len=max_contextual_seq_len,
        max_pos_ind=max_pos_ind,
        seq_lengths=seq_lengths,
        num_targets=num_targets,
        interleave_targets=interleave_targets,
    )
    B, _ = pos_inds.shape
    # timestamp encoding
    num_time_buckets = ts_embeddings.size(1) - 1
    time_bucket_increments = 60.0
    time_bucket_divisor = 1.0
    time_delta = 0
    timestamps = jagged_to_padded_dense(
        values=timestamps.unsqueeze(-1),
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    ).squeeze(-1)
    query_time = torch.gather(
        timestamps, dim=1, index=(seq_lengths - 1).unsqueeze(1).clamp(min=0)
    )
    ts = query_time - timestamps
    ts = ts + time_delta
    ts = ts.clamp(min=1e-6) / time_bucket_increments
    if time_bucket_fn == "log":
        ts = torch.log(ts)
    else:
        ts = torch.sqrt(ts)
    ts = (ts / time_bucket_divisor).clamp(min=0).int()
    ts = torch.clamp(
        ts,
        min=0,
        max=num_time_buckets,
    )
    position_embeddings = torch.index_select(
        pos_embeddings, 0, pos_inds.reshape(-1)
    ).view(B, max_seq_len, -1)
    time_embeddings = torch.index_select(ts_embeddings, 0, ts.reshape(-1)).view(
        B, max_seq_len, -1
    )
    padded_emb = torch.ops.fbgemm.jagged_to_padded_dense(
        values=seq_embeddings,
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    summed = padded_emb + (time_embeddings + position_embeddings).to(
        seq_embeddings.dtype
    )
    result, _ = torch.ops.fbgemm.dense_to_jagged(
        summed, [seq_offsets], seq_embeddings.shape[0]
    )
    return result
