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
from generative_recommenders.ops.triton.triton_addmm import (
    maybe_triton_addmm_fwd,
    triton_addmm_bwd,
    triton_addmm_fwd,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_hstu_attention_bwd,
    triton_hstu_attention_fwd,
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
    triton_weighted_layer_norm_fwd,
)
from torch.nn import functional as F


class _HSTUPreprocessAndAttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore [14]
    def forward(
        ctx,  # pyre-ignore [2]
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        norm_eps: float,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,
        uvqk_bias: torch.Tensor,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        attn_alpha: float,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        sort_by_length: bool,
        enable_tma: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed_x, x_mean, x_rstd, BLOCK_D = triton_weighted_layer_norm_fwd(
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            eps=norm_eps,
        )
        uvqk = maybe_triton_addmm_fwd(
            x=normed_x, w=uvqk_weight, y=uvqk_bias
        ).contiguous()
        u, v, q, k = uvqk.split(
            [
                hidden_dim * num_heads,
                hidden_dim * num_heads,
                attn_dim * num_heads,
                attn_dim * num_heads,
            ],
            dim=1,
        )
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        silu_u = F.silu(u)
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        out = triton_hstu_attention_fwd(
            N=max_seq_len,
            alpha=attn_alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=enable_tma,
        )
        # update ctx
        saved_tensors = [
            x,
            norm_weight,
            norm_bias,
            x_mean,
            x_rstd,
            uvqk_weight,
            seq_offsets,
        ]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        if recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
        else:
            saved_tensors.append(uvqk)
        if sort_by_length:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.attn_alpha = attn_alpha
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
        ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.enable_tma = enable_tma
        return silu_u, out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx,  # pyre-ignore[2]
        dsilu_u: torch.Tensor,
        dout: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,  # d_x
        torch.Tensor,  # d_norm_weight
        torch.Tensor,  # d_norm_bias
        None,
        None,
        None,
        None,
        torch.Tensor,  # d_uvqk_weight
        torch.Tensor,  # d_uvqk_bias
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets = (
            ctx.saved_tensors[:7]
        )
        idx = 7
        if ctx.has_multiple_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        else:
            num_targets = None
        if ctx.recompute_normed_x_in_backward:
            normed_x, _, _, _ = triton_weighted_layer_norm_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
            )
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        if ctx.recompute_uvqk_in_backward:
            uvqk_bias = ctx.saved_tensors[idx]
            uvqk = maybe_triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=uvqk_bias)
            idx += 1
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
        if ctx.sort_by_length:
            sort_by_length_indices = ctx.saved_tensors[idx]
        else:
            sort_by_length_indices = None

        duvqk = torch.empty_like(uvqk)
        du, dv, dq, dk = duvqk.split(
            [
                ctx.hidden_dim * ctx.num_heads,
                ctx.hidden_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
            ],
            dim=1,
        )
        u, v, q, k = uvqk.split(
            [
                ctx.hidden_dim * ctx.num_heads,
                ctx.hidden_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
                ctx.attn_dim * ctx.num_heads,
            ],
            dim=1,
        )
        q = q.view(-1, ctx.num_heads, ctx.attn_dim)
        k = k.view(-1, ctx.num_heads, ctx.attn_dim)
        v = v.view(-1, ctx.num_heads, ctx.hidden_dim)
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        # Note: the two operations below update duvqk in place
        (
            _dq,
            _dk,
            _dv,
        ) = triton_hstu_attention_bwd(
            dout=dout,
            q=q,
            k=k,
            v=v,
            dq=dq,
            dk=dk,
            dv=dv,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            N=ctx.max_seq_len,
            max_attn_len=ctx.max_attn_len,
            alpha=ctx.attn_alpha,
            contextual_seq_len=ctx.contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
            enable_tma=ctx.enable_tma,
        )
        if dq.data_ptr() != _dq.data_ptr():
            dq.copy_(_dq)
        if dk.data_ptr() != _dk.data_ptr():
            dk.copy_(_dk)
        if dv.data_ptr() != _dv.data_ptr():
            dv.copy_(_dv)
        torch.ops.aten.silu_backward(dsilu_u, u, grad_input=du)
        d_normed_x, d_uvqk_weight, d_uvqk_bias = triton_addmm_bwd(
            x=normed_x,
            w=uvqk_weight,
            dz=duvqk,
            is_y_1d=ctx.uvqk_bias_1d,
        )
        d_x, d_norm_weight, d_norm_bias = triton_weighted_layer_norm_bwd(
            dy=d_normed_x,
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            mean=x_mean,
            rstd=x_rstd,
            learnable=True,
            eps=ctx.norm_eps,
            BLOCK_D=ctx.norm_BLOCK_D,
        )
        # pyre-ignore[7]
        return (
            d_x,
            d_norm_weight,
            d_norm_bias,
            None,
            None,
            None,
            None,
            d_uvqk_weight,
            d_uvqk_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_hstu_preprocess_and_attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    recompute_uvqk_in_backward: bool = False,
    recompute_normed_x_in_backward: bool = False,
    sort_by_length: bool = False,
    enable_tma: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _HSTUPreprocessAndAttentionFunction.apply(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        max_seq_len,
        seq_offsets,
        attn_alpha,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        recompute_uvqk_in_backward,
        recompute_normed_x_in_backward,
        sort_by_length,
        enable_tma,
    )
