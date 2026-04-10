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
    _, D = values_left.shape
    max_seq_len = max_seq_len_left + max_seq_len_right
    B = offsets_left.shape[0] - 1

    lengths_a = offsets_left[1:] - offsets_left[:-1]
    lengths_b = offsets_right[1:] - offsets_right[:-1]
    dense_a = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_left,
        offsets=[offsets_left],
        max_lengths=[max_seq_len_left],
        padding_value=0.0,
    )
    dense_b = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_right,
        offsets=[offsets_right],
        max_lengths=[max_seq_len_right],
        padding_value=0.0,
    )
    dense_b_prefix, dense_b_suffix = torch.split(
        dense_b, [n_prefix_from_right, max_seq_len_right - n_prefix_from_right], dim=1
    )
    dense = torch.cat([dense_b_prefix, dense_a, dense_b_suffix], dim=1)
    mask = _arange(max_seq_len, device=offsets_left.device).expand(B, max_seq_len)
    mask = torch.logical_or(
        mask < lengths_a.view(B, 1) + n_prefix_from_right,
        torch.logical_and(
            mask >= max_seq_len_left + n_prefix_from_right,
            mask < max_seq_len_left + lengths_b.view(B, 1),
        ),
    )
    return dense.view(-1, D)[mask.view(-1), :]


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
    B = offsets_left.shape[0] - 1
    lengths_a = offsets_left[1:] - offsets_left[:-1]
    lengths_b = offsets_right[1:] - offsets_right[:-1]
    dense_a = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_left,
        offsets=[offsets_left],
        max_lengths=[max_seq_len_left],
        padding_value=0.0,
    )
    raw_mask = torch.arange(max_seq_len_left, device=offsets_left.device).expand(
        B, max_seq_len_left
    )
    mask = torch.logical_and(
        raw_mask >= (lengths_a - lengths_b).unsqueeze(1),
        raw_mask < lengths_a.unsqueeze(1),
    )
    dense_a = fx_apply_mask(dense_a, mask, values_right)
    jagged_a = torch.ops.fbgemm.dense_to_jagged(
        dense_a,
        [offsets_left],
    )[0]
    return jagged_a
