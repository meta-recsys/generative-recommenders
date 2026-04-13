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

from typing import Optional, Tuple

import torch
from generative_recommenders.common import fx_arange

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def _concat_2D_jagged_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = max_len_left + max_len_right
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    padded_left = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_left,
        offsets=[offsets_left],
        max_lengths=[max_len_left],
        padding_value=0.0,
    )
    padded_right = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_right,
        offsets=[offsets_right],
        max_lengths=[max_len_right],
        padding_value=0.0,
    )
    concatted_dense = torch.cat([padded_left, padded_right], dim=1)
    mask = fx_arange(max_seq_len, device=offsets_left.device).view(1, -1)
    mask = torch.logical_or(
        mask < lengths_left.view(-1, 1),
        torch.logical_and(
            mask >= max_len_left,
            mask < max_len_left + lengths_right.view(-1, 1),
        ),
    )
    return concatted_dense.flatten(0, 1)[mask.view(-1), :]


@torch.fx.wrap
def pytorch_concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> torch.Tensor:
    if offsets_left is None:
        assert max_len_left is not None
        # Handle 3D dense input [B, S, D] -> reshape to 2D [B*S, D]
        if values_left.dim() == 3:
            B, dense_size, D = values_left.size()
            values_left = values_left.reshape(B * dense_size, D)
            max_len_left = dense_size
        B = values_left.shape[0] // max_len_left
        offsets_left_non_optional = max_len_left * torch.arange(
            B + 1, device=values_left.device
        )
    else:
        offsets_left_non_optional = offsets_left
    if offsets_right is None:
        assert max_len_right is not None
        # Handle 3D dense input [B, S, D] -> reshape to 2D [B*S, D]
        if values_right.dim() == 3:
            B, dense_size, D = values_right.size()
            values_right = values_right.reshape(B * dense_size, D)
            max_len_right = dense_size
        B = values_right.shape[0] // max_len_right
        offsets_right_non_optional = max_len_right * torch.arange(
            B + 1, device=values_left.device
        )
    else:
        offsets_right_non_optional = offsets_right
    max_len_left = (
        int(
            (offsets_left_non_optional[1:] - offsets_left_non_optional[:-1])
            .max()
            .item()
        )
        if max_len_left is None
        else max_len_left
    )
    max_len_right = (
        int(
            (offsets_right_non_optional[1:] - offsets_right_non_optional[:-1])
            .max()
            .item()
        )
        if max_len_right is None
        else max_len_right
    )
    return _concat_2D_jagged_jagged(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def _split_2D_jagged_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = offsets_left + offsets_right
    padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values,
        offsets=[offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    ).flatten(0, 1)
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    mask = fx_arange(max_seq_len, device=values.device).view(1, -1)
    mask_left = mask < lengths_left.view(-1, 1)
    mask_right = torch.logical_and(
        mask >= lengths_left.view(-1, 1),
        mask < (lengths_left + lengths_right).view(-1, 1),
    )
    return padded_values[mask_left.view(-1), :], padded_values[mask_right.view(-1), :]


@torch.fx.wrap
def pytorch_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets_left is None:
        assert max_len_left is not None
        assert offsets_right is not None
        offsets_left_non_optional = max_len_left * torch.arange(
            offsets_right.shape[0], device=values.device
        )
    else:
        offsets_left_non_optional = offsets_left
    if offsets_right is None:
        assert max_len_right is not None
        assert offsets_left is not None
        offsets_right_non_optional = max_len_right * torch.arange(
            offsets_left.shape[0], device=values.device
        )
    else:
        offsets_right_non_optional = offsets_right
    return _split_2D_jagged_jagged(
        max_seq_len=max_seq_len,
        values=values,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def pytorch_hstu_split_l2_embeddings(
    max_seq_len: int,
    x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_offsets = prefix_offsets + l2_offsets
    x_lengths = x_offsets[1:] - x_offsets[:-1]
    padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
        values=x,
        offsets=[x_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    ).flatten(0, 1)
    prefix_lengths = prefix_offsets[1:] - prefix_offsets[:-1]
    mask = fx_arange(max_seq_len, device=x_offsets.device).view(1, -1)
    mask_prefix = torch.logical_and(
        mask >= contextual_seq_len,
        mask < prefix_lengths.view(-1, 1) + contextual_seq_len,
    )
    mask_l2 = torch.logical_or(
        mask < contextual_seq_len,
        torch.logical_and(
            mask >= prefix_lengths.view(-1, 1) + contextual_seq_len,
            mask < x_lengths.view(-1, 1),
        ),
    )
    return padded_x[mask_prefix.view(-1), :], padded_x[mask_l2.view(-1), :]


def pytorch_hstu_concat_l2_embeddings(
    max_prefix_len: int,
    prefix_x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    max_l2_len: int,
    l2_x: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> torch.Tensor:
    padded_prefix_x = torch.ops.fbgemm.jagged_to_padded_dense(
        values=prefix_x,
        offsets=[prefix_offsets],
        max_lengths=[max_prefix_len],
        padding_value=0.0,
    )
    padded_l2_x = torch.ops.fbgemm.jagged_to_padded_dense(
        values=l2_x,
        offsets=[l2_offsets],
        max_lengths=[max_l2_len],
        padding_value=0.0,
    )
    padded_x = torch.cat(
        [
            padded_l2_x[:, 0:contextual_seq_len, :],
            padded_prefix_x,
            padded_l2_x[:, contextual_seq_len:, :],
        ],
        dim=1,
    )
    mask = fx_arange(max_prefix_len + max_l2_len, device=prefix_x.device).view(1, -1)
    prefix_lengths = prefix_offsets[1:] - prefix_offsets[:-1]
    l2_lengths = l2_offsets[1:] - l2_offsets[:-1]
    mask = torch.logical_or(
        mask < prefix_lengths.view(-1, 1) + contextual_seq_len,
        torch.logical_and(
            mask >= max_prefix_len + contextual_seq_len,
            mask < max_prefix_len + l2_lengths.view(-1, 1),
        ),
    )
    return padded_x.flatten(0, 1)[mask.view(-1), :]
