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

from typing import Tuple

import torch


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def pytorch_jagged_dense_bmm(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    dtype = jagged.dtype
    jagged = jagged.to(torch.float32)
    dense = dense.to(torch.float32)
    padded_jagged = torch.ops.fbgemm.jagged_to_padded_dense(
        values=jagged,
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    bmm_out = torch.bmm(padded_jagged, dense)
    jagged_bmm_out = torch.ops.fbgemm.dense_to_jagged(
        bmm_out, [seq_offsets], total_L=jagged.shape[0]
    )[0]
    jagged_bmm_out = jagged_bmm_out.to(dtype)
    return jagged_bmm_out


def pytorch_jagged_dense_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
) -> torch.Tensor:
    dtype = jagged.dtype
    jagged = jagged.to(torch.float32)
    dense = dense.to(torch.float32)
    padded_jagged = torch.ops.fbgemm.jagged_to_padded_dense(
        values=jagged,
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    out = padded_jagged + dense.unsqueeze(1)
    jagged_out = torch.ops.fbgemm.dense_to_jagged(
        out, [seq_offsets], total_L=jagged.shape[0]
    )[0]
    jagged_out = jagged_out.to(dtype)
    return jagged_out


def pytorch_jagged_dense_bmm_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    elementwise: bool = False,
) -> torch.Tensor:
    dtype = jagged.dtype
    jagged = jagged.to(torch.float32)
    dense = dense.to(torch.float32)
    padded_jagged = torch.ops.fbgemm.jagged_to_padded_dense(
        values=jagged,
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    bmm_out = torch.bmm(padded_jagged, dense)

    if elementwise:
        jagged_out = (
            torch.ops.fbgemm.dense_to_jagged(
                bmm_out, [seq_offsets], total_L=jagged.shape[0]
            )[0]
            + bias
        )
    else:
        jagged_out = torch.ops.fbgemm.dense_to_jagged(
            bmm_out + bias.unsqueeze(1), [seq_offsets], total_L=jagged.shape[0]
        )[0]

    jagged_out = jagged_out.to(dtype)
    return jagged_out


@torch.fx.wrap
def _arange(len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(len, device=device)


def pytorch_concat_2D_dense_jagged(
    jagged_max_seq_len: int,
    jagged_offsets: torch.Tensor,
    jagged_values: torch.Tensor,
    dense_values: torch.Tensor,
) -> torch.Tensor:
    B, dense_size, D = dense_values.size()
    jagged_dense = torch.ops.fbgemm.jagged_to_padded_dense(
        values=jagged_values,
        offsets=[jagged_offsets],
        max_lengths=[jagged_max_seq_len],
        padding_value=0.0,
    )
    concatted_dense = torch.cat([dense_values, jagged_dense], dim=1)
    concatted_offsets = (
        dense_size * _arange(B + 1, device=jagged_offsets.device) + jagged_offsets
    )
    return torch.ops.fbgemm.dense_to_jagged(
        concatted_dense,
        [concatted_offsets],
        total_L=jagged_values.shape[0] + dense_size * B,
    )[0]


def pytorch_concat_2D_jagged_jagged(
    max_seq_len_left: int,
    offsets_left: torch.Tensor,
    values_left: torch.Tensor,
    max_seq_len_right: int,
    offsets_right: torch.Tensor,
    values_right: torch.Tensor,
    is_replace: bool = False,
    n_prefix_from_right: int = 0,
) -> torch.Tensor:
    # is_replace with n_prefix_from_right != 0 is not supported yet (neither in triton)
    if is_replace:
        return pytorch_replace_last_n_with_jagged(
            max_seq_len_left,
            offsets_left,
            values_left,
            offsets_right,
            values_right,
        )

    lengths_a = offsets_left[1:] - offsets_left[:-1]
    lengths_b = offsets_right[1:] - offsets_right[:-1]

    # Compute output offsets via cumsum (no dynamic shapes).
    output_lengths = lengths_a + lengths_b
    output_offsets = torch.nn.functional.pad(
        torch.cumsum(output_lengths, dim=0), (1, 0)
    )

    total_len = values_left.shape[0] + values_right.shape[0]
    positions = torch.arange(total_len, device=values_left.device)
    batch_idx = torch.searchsorted(output_offsets[1:], positions, right=True)
    local_pos = positions - output_offsets[batch_idx]

    per_batch_lengths_a = lengths_a[batch_idx]

    # Classify each output position into prefix / left / suffix.
    is_prefix = local_pos < n_prefix_from_right
    is_left = (local_pos >= n_prefix_from_right) & (
        local_pos < n_prefix_from_right + per_batch_lengths_a
    )

    # Source indices (clamped — invalid lanes masked out by torch.where).
    left_idx = (offsets_left[batch_idx] + (local_pos - n_prefix_from_right)).clamp(
        min=0, max=values_left.shape[0] - 1
    )
    right_prefix_idx = offsets_right[batch_idx] + local_pos
    right_suffix_idx = offsets_right[batch_idx] + (local_pos - per_batch_lengths_a)
    right_idx = torch.where(is_prefix, right_prefix_idx, right_suffix_idx).clamp(
        min=0, max=values_right.shape[0] - 1
    )

    left_values = values_left.index_select(0, left_idx)
    right_values = values_right.index_select(0, right_idx)

    return torch.where(is_left.unsqueeze(-1), left_values, right_values)


def pytorch_jagged_remove_first_or_last_1D(
    values: torch.Tensor,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.view(-1, 1)
    shrunk_lengths = lengths - 1
    k_lengths = torch.stack([shrunk_lengths, torch.ones_like(lengths)], dim=1).view(-1)
    q_lengths = torch.stack([torch.ones_like(lengths), shrunk_lengths], dim=1).view(-1)
    all_indices = torch.arange(
        start=0, end=q_lengths.numel(), device=values.device
    ).reshape(-1, 2)
    q_indices, k_indices = all_indices[:, 1], all_indices[:, 0]
    values_no_first, _ = torch.ops.fbgemm.jagged_index_select(
        values, q_lengths, q_indices
    )
    values_no_last, _ = torch.ops.fbgemm.jagged_index_select(
        values, k_lengths, k_indices
    )
    return values_no_first.squeeze(), values_no_last.squeeze()


@torch.fx.wrap
def fx_apply_mask(
    tensor: torch.Tensor, mask: torch.Tensor, fill_value: torch.Tensor
) -> torch.Tensor:
    tensor[mask] = fill_value
    return tensor


def pytorch_replace_last_n_with_jagged(
    max_seq_len_left: int,
    offsets_left: torch.Tensor,
    values_left: torch.Tensor,
    offsets_right: torch.Tensor,
    values_right: torch.Tensor,
) -> torch.Tensor:
    lengths_a = offsets_left[1:] - offsets_left[:-1]
    lengths_b = offsets_right[1:] - offsets_right[:-1]

    total_len = values_left.shape[0]
    positions = torch.arange(total_len, device=values_left.device)
    batch_idx = torch.searchsorted(offsets_left[1:], positions, right=True)
    local_pos = positions - offsets_left[batch_idx]

    # Positions >= (lengths_a - lengths_b) within each batch are in the replace zone.
    threshold = lengths_a[batch_idx] - lengths_b[batch_idx]
    in_replace_zone = local_pos >= threshold

    # Index into values_right (clamped — non-replace positions masked by torch.where).
    right_idx = (offsets_right[batch_idx] + (local_pos - threshold)).clamp(min=0)
    right_values = values_right.index_select(0, right_idx)
    return torch.where(in_replace_zone.unsqueeze(-1), right_values, values_left)
