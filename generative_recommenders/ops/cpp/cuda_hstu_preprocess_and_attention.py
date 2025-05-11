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

torch.ops.load_library(
    "//generative_recommenders/ops/cpp/hstu_attention:hstu_flash_attention"
)

from generative_recommenders.ops.triton.triton_addmm import (
    triton_addmm_bwd,
    triton_addmm_fwd,
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
        alpha: float,
        invalid_attn_mask_type: str,
        num_targets: Optional[torch.Tensor],
        attn_scale: Optional[torch.Tensor] = None,
        recompute_uvqk_in_backward: bool = False,
        recompute_normed_x_in_backward: bool = False,
        contextual_seq_len: int = 0,
        sort_by_length: bool = False,
        max_attn_len: Optional[int] = None,
        full_attn_size: Optional[int] = None,
        silu_u: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_attn_len = max_attn_len or 0
        full_attn_size = full_attn_size or 0

        normed_x, x_mean, x_rstd, BLOCK_D, num_warps = triton_weighted_layer_norm_fwd(
            x=x,
            weight=norm_weight,
            bias=norm_bias,
            eps=norm_eps,
        )
        uvqk = triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=uvqk_bias).contiguous()
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
        if silu_u:
            u = F.silu(u)
        out = torch.ops.hstu.hstu_mha_fwd(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            True,  # causal
            num_targets,
            attn_scale,
            max_attn_len,
            full_attn_size,
            contextual_seq_len,
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            0,  # sm_margin
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
        if attn_scale is not None:
            saved_tensors.append(attn_scale)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        if recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
        else:
            saved_tensors.append(uvqk)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.has_attn_scale = attn_scale is not None
        ctx.max_seq_len = max_seq_len
        ctx.max_attn_len = max_attn_len
        ctx.full_attn_size = full_attn_size
        ctx.recompute_normed_x_in_backward = recompute_normed_x_in_backward
        ctx.recompute_uvqk_in_backward = recompute_uvqk_in_backward
        ctx.hidden_dim = hidden_dim
        ctx.attn_dim = attn_dim
        ctx.num_heads = num_heads
        ctx.uvqk_bias_1d = uvqk_bias.dim() == 1
        ctx.norm_eps = norm_eps
        ctx.norm_BLOCK_D = BLOCK_D
        ctx.norm_num_warps = num_warps
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.silu_u = silu_u
        return u, out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx,  # pyre-ignore[2]
        _du: torch.Tensor,
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
        if ctx.has_attn_scale:
            attn_scale = ctx.saved_tensors[idx]
            idx += 1
        else:
            attn_scale = None
        if ctx.recompute_normed_x_in_backward:
            normed_x, _, _, _, _ = triton_weighted_layer_norm_fwd(
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
            uvqk = triton_addmm_fwd(x=normed_x, w=uvqk_weight, y=uvqk_bias)
            idx += 1
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1

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
        _dq, _dk, _dv = torch.ops.hstu.hstu_mha_bwd(
            ctx.max_seq_len,
            ctx.alpha,
            dout,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            seq_offsets,
            True,  # causal
            num_targets,
            attn_scale,
            ctx.max_attn_len,
            ctx.full_attn_size,
            ctx.contextual_seq_len,
            ctx.sort_by_length,
            False,  # deterministic
            0,  # sm_margin
        )
        if dq.data_ptr() != _dq.data_ptr():
            dq.copy_(_dq)
        if dk.data_ptr() != _dk.data_ptr():
            dk.copy_(_dk)
        if dv.data_ptr() != _dv.data_ptr():
            dv.copy_(_dv)
        if ctx.silu_u:
            torch.ops.aten.silu_backward(_du, u, grad_input=du)
        else:
            if du.data_ptr() != _du.data_ptr():
                du.copy_(_du)
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
            num_warps=ctx.norm_num_warps,
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
            None,
            None,
            None,
        )


def cuda_hstu_preprocess_and_attention(
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
    alpha: float,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_scale: Optional[torch.Tensor] = None,
    recompute_uvqk_in_backward: bool = False,
    recompute_normed_x_in_backward: bool = False,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    max_attn_len: Optional[int] = None,
    full_attn_size: Optional[int] = None,
    silu_u: bool = True,
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
        alpha,
        invalid_attn_mask_type,
        num_targets,
        attn_scale,
        recompute_uvqk_in_backward,
        recompute_normed_x_in_backward,
        contextual_seq_len,
        sort_by_length,
        max_attn_len,
        full_attn_size,
        silu_u,
    )
