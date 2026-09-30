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

"""PyTorch reference implementation of HSTU Ultra attention."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.fx.wrap
def get_hstu_ultra_valid_attn_mask(
    device: torch.device,
    max_seq_len: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
) -> torch.Tensor:
    """Build the causal HSTU Ultra semi-local attention mask.

    Rows outside the trailing ``full_attn_size`` positions attend to at most
    ``max_attn_len`` preceding positions. The trailing rows retain full causal
    attention, matching the HSTU Ultra L1 layout.
    """
    ids = torch.arange(max_seq_len, device=device).view(1, max_seq_len)
    max_ids = seq_lengths.view(-1, 1, 1)
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(ids, max=max_ids)
        batch_size = num_targets.shape[0]
        row_ids = ids.view(batch_size, max_seq_len, 1).expand(
            batch_size, max_seq_len, max_seq_len
        )
        col_ids = ids.view(batch_size, 1, max_seq_len).expand(
            batch_size, max_seq_len, max_seq_len
        )
    else:
        row_ids = ids.view(max_seq_len, 1).expand(max_seq_len, max_seq_len)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, max_seq_len, max_seq_len)
        col_ids = col_ids.view(1, max_seq_len, max_seq_len)

    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(max_seq_len, device=device, dtype=torch.bool).view(
        1, max_seq_len, max_seq_len
    )
    if max_attn_len > 0:
        valid_attn_mask = torch.logical_or(
            valid_attn_mask,
            torch.logical_and(row_col_dist > 0, row_col_dist <= max_attn_len),
        )
        if full_attn_size > 0:
            full_attn_mask = torch.logical_and(
                row_col_dist > 0,
                row_ids >= max_ids - full_attn_size,
            )
            valid_attn_mask = torch.logical_or(valid_attn_mask, full_attn_mask)
    else:
        valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)

    row_pos = torch.arange(max_seq_len, device=device).view(1, max_seq_len, 1)
    col_pos = torch.arange(max_seq_len, device=device).view(1, 1, max_seq_len)
    valid_positions = torch.logical_and(
        row_pos < seq_lengths.view(-1, 1, 1),
        col_pos < seq_lengths.view(-1, 1, 1),
    )
    return torch.logical_and(valid_attn_mask, valid_positions)


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, attention_dim = q.shape
    value_dim = v.shape[2]
    batch_size = seq_offsets.numel() - 1

    def pad(x: torch.Tensor, dim: int) -> torch.Tensor:
        return (
            torch.ops.fbgemm.jagged_to_padded_dense(
                values=x.reshape(num_tokens, num_heads * dim),
                offsets=[seq_offsets],
                max_lengths=[max_seq_len],
                padding_value=0.0,
            )
            .view(batch_size, max_seq_len, num_heads, dim)
            .transpose(1, 2)
        )

    return pad(q, attention_dim), pad(k, attention_dim), pad(v, value_dim)


@torch.fx.wrap
def pytorch_hstu_ultra_attention(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    dropout_pr: float = 0.0,
    training: bool = False,
    max_attn_len: int = 0,
    full_attn_size: int = 0,
) -> torch.Tensor:
    """Run the dense PyTorch reference for HSTU Ultra attention."""
    num_tokens, num_heads, _ = q.shape
    value_dim = v.shape[2]
    padded_q, padded_k, padded_v = _pad_qkv(q, k, v, seq_offsets, max_seq_len)
    qk_attn = torch.einsum("bhxa,bhya->bhxy", padded_q, padded_k) * alpha

    if attn_scale is not None:
        if attn_scale.ndim > 0:
            attn_scale = (
                torch.ops.fbgemm.jagged_to_padded_dense(
                    values=attn_scale.unsqueeze(-1),
                    offsets=[seq_offsets],
                    max_lengths=[max_seq_len],
                    padding_value=0.0,
                )
                .unsqueeze(1)
                .to(qk_attn.dtype)
            )
        qk_attn = F.silu(qk_attn) * attn_scale
    else:
        qk_attn = F.silu(qk_attn) / max_seq_len

    valid_attn_mask = get_hstu_ultra_valid_attn_mask(
        device=q.device,
        max_seq_len=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, padded_v)
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),
        [seq_offsets],
        num_tokens,
    )[0].view(num_tokens, num_heads, value_dim)
