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
import torch.nn.functional as F


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return valid_attn_mask


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = _pad_qkv(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
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

        # pyrefly: ignore [unsupported-operation]
        qk_attn = F.silu(qk_attn) * attn_scale
    else:
        qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )
    # raise NotImplementedError(valid_attn_mask[0, :, :].to(torch.int32))
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    _, _, V = v.shape
    B = seq_offsets.size(0) - 1
    delta_size = L // B
    delta_q = delta_q.view(B, -1, H, D).transpose(1, 2)
    full_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )
    full_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(-1, H * V),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, V)
        .transpose(1, 2)
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    full_valid_attn_mask = _get_valid_attn_mask(
        device=delta_q.device,
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = torch.arange(max_seq_len, device=delta_q.device).view(1, -1)
    mask = torch.logical_and(
        mask >= (seq_lengths - delta_size).view(-1, 1),
        mask < seq_lengths.view(-1, 1),
    )
    valid_attn_mask = (
        full_valid_attn_mask.expand(B, -1, -1)
        .flatten(0, 1)[mask.view(-1), :]
        .view(-1, delta_size, max_seq_len)
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)
    return attn_output.transpose(1, 2).reshape(-1, H, V)


@torch.fx.wrap
def _get_valid_cross_attn_mask(
    device: torch.device,
    max_seq_len_q: int,
    max_seq_len_kv: int,
    seq_lengths_q: torch.Tensor,
    seq_lengths_kv: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    targets_in_kv: bool = True,
) -> torch.Tensor:
    """
    Returns a [B, max_seq_len_q, max_seq_len_kv] mask. With causal=True, the
    last query of each sequence is aligned with the last key; with num_targets,
    the target queries (the last num_targets of each Q sequence) attend to all
    non-target keys.
    """
    col_ids = torch.arange(0, max_seq_len_q, device=device).view(1, max_seq_len_q, 1)
    row_ids = torch.arange(0, max_seq_len_kv, device=device).view(1, 1, max_seq_len_kv)
    in_boundary_valid_attn_mask = torch.logical_and(
        row_ids < seq_lengths_kv.view(-1, 1, 1), col_ids < seq_lengths_q.view(-1, 1, 1)
    )
    if not causal:
        return in_boundary_valid_attn_mask
    if num_targets is None:
        delta_col_ids = seq_lengths_kv - seq_lengths_q
        shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
        return torch.logical_and(
            in_boundary_valid_attn_mask, shifted_col_ids >= row_ids
        )
    uih_lengths_q = seq_lengths_q - num_targets
    delta_col_ids = seq_lengths_kv - uih_lengths_q
    if targets_in_kv:
        delta_col_ids = delta_col_ids - num_targets
    shifted_col_ids = col_ids + delta_col_ids.view(-1, 1, 1)
    causal_mask = torch.logical_and(
        col_ids < uih_lengths_q.view(-1, 1, 1), shifted_col_ids >= row_ids
    )
    target_mask = col_ids >= uih_lengths_q.view(-1, 1, 1)
    if targets_in_kv:
        target_mask = torch.logical_and(
            target_mask,
            torch.logical_or(
                row_ids < (seq_lengths_kv - num_targets).view(-1, 1, 1),
                shifted_col_ids == row_ids,
            ),
        )
    return torch.logical_and(
        in_boundary_valid_attn_mask, torch.logical_or(causal_mask, target_mask)
    )


def _pad_cross_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    seq_offsets_kv: torch.Tensor,
    max_seq_len_q: int,
    max_seq_len_kv: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_seq_len_q, H_q, D = q.shape
    total_seq_len_kv, H_kv, _ = k.shape
    batch_size = seq_offsets_q.shape[0] - 1
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(total_seq_len_q, H_q * D),
            offsets=[seq_offsets_q],
            max_lengths=[max_seq_len_q],
            padding_value=0.0,
        )
        .view(batch_size, max_seq_len_q, H_q, D)
        .transpose(1, 2)
    )  # [B, H_q, N_q, D]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(total_seq_len_kv, H_kv * D),
            offsets=[seq_offsets_kv],
            max_lengths=[max_seq_len_kv],
            padding_value=0.0,
        )
        .view(batch_size, max_seq_len_kv, H_kv, D)
        .transpose(1, 2)
    )  # [B, H_kv, N_kv, D]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(total_seq_len_kv, H_kv * V),
            offsets=[seq_offsets_kv],
            max_lengths=[max_seq_len_kv],
            padding_value=0.0,
        )
        .view(batch_size, max_seq_len_kv, H_kv, V)
        .transpose(1, 2)
    )  # [B, H_kv, N_kv, V]
    # Grouped-query attention: each KV head serves H_q // H_kv consecutive Q heads.
    if H_q > H_kv:
        assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
        gqa_ratio = H_q // H_kv
        padded_k = padded_k.repeat_interleave(gqa_ratio, dim=1)
        padded_v = padded_v.repeat_interleave(gqa_ratio, dim=1)
    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_cross_mha(
    max_seq_len_q: int,
    max_seq_len_kv: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    seq_offsets_kv: torch.Tensor,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    targets_in_kv: bool = False,
) -> torch.Tensor:
    """
    Reference HSTU cross-attention over jagged Q [total_q, H_q, D] and K/V
    [total_kv, H_kv, D|V] with separate sequence offsets. The first
    num_softmax_heads heads use softmax, the rest use SiLU scaled by
    attn_scale (a scalar, or one value per query token).
    """
    total_seq_len_q, H, _ = q.shape
    assert num_softmax_heads <= H
    V = v.shape[2]
    q, k, v = _pad_cross_qkv(
        q, k, v, seq_offsets_q, seq_offsets_kv, max_seq_len_q, max_seq_len_kv
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q.float(), k.float()) * alpha
    qk_attn_softmax = qk_attn[:, :num_softmax_heads, :, :]
    qk_attn_silu = qk_attn[:, num_softmax_heads:, :, :]
    if attn_scale.ndim > 0:
        attn_scale = (
            torch.ops.fbgemm.jagged_to_padded_dense(
                values=attn_scale.unsqueeze(-1),
                offsets=[seq_offsets_q],
                max_lengths=[max_seq_len_q],
                padding_value=0.0,
            )
            .unsqueeze(1)
            .to(q.dtype)
        )
    valid_attn_mask = _get_valid_cross_attn_mask(
        device=q.device,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_kv=max_seq_len_kv,
        seq_lengths_q=seq_offsets_q[1:] - seq_offsets_q[:-1],
        seq_lengths_kv=seq_offsets_kv[1:] - seq_offsets_kv[:-1],
        num_targets=num_targets,
        causal=causal,
        targets_in_kv=targets_in_kv,
    ).unsqueeze(1)
    qk_max = torch.max(qk_attn_softmax, dim=-1, keepdim=True)[0]
    qk_attn_softmax = qk_attn_softmax - qk_max
    qk_attn_softmax = torch.where(valid_attn_mask, qk_attn_softmax, -float("inf"))
    qk_attn_softmax = torch.exp(qk_attn_softmax)
    qk_attn_softmax = qk_attn_softmax / (
        torch.sum(qk_attn_softmax, dim=-1, keepdim=True) + 1e-6
    )
    qk_attn_softmax = torch.where(valid_attn_mask, qk_attn_softmax, 0.0)
    qk_attn_softmax = qk_attn_softmax.to(q.dtype)
    qk_attn_silu = qk_attn_silu.to(q.dtype)
    qk_attn_silu = F.silu(qk_attn_silu) * attn_scale
    qk_attn_silu = torch.where(valid_attn_mask, qk_attn_silu, 0.0)
    qk_attn = torch.cat((qk_attn_softmax, qk_attn_silu), dim=1)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N_q, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N_q, H, V]->[B, N_q, H * V]
        [seq_offsets_q],
        total_seq_len_q,
    )[0].view(total_seq_len_q, H, V)
