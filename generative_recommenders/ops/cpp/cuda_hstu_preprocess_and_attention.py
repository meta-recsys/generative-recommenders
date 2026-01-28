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
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
)
from generative_recommenders.ops.utils import is_sm100_plus
from torch.nn import functional as F

try:
    from generative_recommenders.fb.ultra.ops.fp8.fp8_addmm import (
        fp8_rowwise_quantize_addmm,
    )
    from generative_recommenders.fb.ultra.ops.fp8.layer_norm_quantization import (
        triton_weighted_layer_norm_quantization_fwd,
    )
    from hammer.ops.triton.triton_apply_rope import (
        triton_apply_rope_bwd,
        triton_apply_rope_fwd,
    )

    if is_sm100_plus():
        torch.ops.load_library(
            "//generative_recommenders/fb/ultra/ops/blackwell/hstu_attention:hstu_flash_attention"
        )
    torch.ops.load_library(
        "//generative_recommenders/ops/cpp/hstu_attention:hstu_flash_attention"
    )
except ImportError:
    pass


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
        rotary_weights: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None,
        attn_scale: Optional[torch.Tensor] = None,
        recompute_uvqk_in_backward: bool = False,
        recompute_normed_x_in_backward: bool = False,
        contextual_seq_len: int = 0,
        sort_by_length: bool = False,
        max_attn_len: Optional[int] = None,
        full_attn_size: Optional[int] = None,
        silu_u: bool = True,
        fp8_in_addmm_fwd: bool = False,
        num_softmax_heads: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_attn_len = max_attn_len or 0
        full_attn_size = full_attn_size or 0
        normed_x, x_mean, x_rstd, BLOCK_D, x_scale, normed_x_fp8 = (
            triton_weighted_layer_norm_quantization_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                quantize_output=fp8_in_addmm_fwd,
            )
        )
        # When silu_u is False and we want to recompute in backward, we split the weight
        # for u and vqk separately during training to compute them independently.
        # This avoids needing to clone u (which would otherwise keep the whole uvqk alive).
        if not silu_u and recompute_uvqk_in_backward:
            # Split the weights/biases to compute u and vqk separately
            u_weight, vqk_weight = uvqk_weight.split(
                [
                    hidden_dim * num_heads,
                    hidden_dim * num_heads
                    + attn_dim * num_heads
                    + attn_dim * num_heads,
                ],
                dim=1,
            )
            u_bias, vqk_bias = uvqk_bias.split(
                [
                    hidden_dim * num_heads,
                    hidden_dim * num_heads
                    + attn_dim * num_heads
                    + attn_dim * num_heads,
                ],
                dim=0,
            )
            if fp8_in_addmm_fwd:
                assert x_scale is not None and normed_x_fp8 is not None
                u = fp8_rowwise_quantize_addmm(
                    x=normed_x,
                    x_fp8=normed_x_fp8,
                    w=u_weight,
                    y=u_bias,
                    x_scale=x_scale,
                    custom_kernel=False,
                    is_inference=False,
                ).contiguous()
                vqk = fp8_rowwise_quantize_addmm(
                    x=normed_x,
                    x_fp8=normed_x_fp8,
                    w=vqk_weight,
                    y=vqk_bias,
                    x_scale=x_scale,
                    custom_kernel=False,
                    is_inference=False,
                ).contiguous()
            else:
                u = maybe_triton_addmm_fwd(normed_x, u_weight, u_bias).contiguous()
                vqk = maybe_triton_addmm_fwd(
                    normed_x, vqk_weight, vqk_bias
                ).contiguous()
            v, q, k = vqk.split(
                [
                    hidden_dim * num_heads,
                    attn_dim * num_heads,
                    attn_dim * num_heads,
                ],
                dim=1,
            )
            # uvqk is not used since we split the computation, but we need it
            # for saving in case recompute_uvqk_in_backward is False in a
            # different code path. Set to None to satisfy type checker.
            uvqk = None
        else:
            if fp8_in_addmm_fwd:
                assert x_scale is not None and normed_x_fp8 is not None
                uvqk = fp8_rowwise_quantize_addmm(
                    x=normed_x,
                    x_fp8=normed_x_fp8,
                    w=uvqk_weight,
                    y=uvqk_bias,
                    x_scale=x_scale,
                    custom_kernel=False,
                    is_inference=False,
                ).contiguous()
            else:
                uvqk = maybe_triton_addmm_fwd(
                    normed_x, uvqk_weight, uvqk_bias
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
            if silu_u:
                u = F.silu(u)
        if rotary_weights is not None:
            q_cos_weights = rotary_weights[0]
            q_sin_weights = rotary_weights[1]
            k_cos_weights = rotary_weights[2]
            k_sin_weights = rotary_weights[3]
            _q = triton_apply_rope_fwd(
                x=q.view(-1, num_heads, attn_dim),
                N=max_seq_len,
                seq_offsets=seq_offsets,
                cos_rope=q_cos_weights,
                sin_rope=q_sin_weights,
            ).view(-1, num_heads * attn_dim)
            _k = triton_apply_rope_fwd(
                x=k.view(-1, num_heads, attn_dim),
                N=max_seq_len,
                seq_offsets=seq_offsets,
                cos_rope=k_cos_weights,
                sin_rope=k_sin_weights,
            ).view(-1, num_heads * attn_dim)
            if q.data_ptr() != _q.data_ptr():
                q.copy_(_q)
            if k.data_ptr() != _k.data_ptr():
                k.copy_(_k)
        q = q.view(-1, num_heads, attn_dim)
        k = k.view(-1, num_heads, attn_dim)
        v = v.view(-1, num_heads, hidden_dim)
        if is_sm100_plus():
            out = torch.ops.bw_hstu.bw_hstu_mha_fwd(
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
                max_seq_len,  # max_q_len,
                None,  # seq_offsets_q,
                None,  # max_seq_len_tensor,
                None,  # contextual_seq_len_tensor,
                None,  # max_attn_len_tensor,
                None,  # min_full_attn_seq_len_tensor,
                1,  # num_groups
            )
            softmax_lse = None
        else:
            out, softmax_lse = torch.ops.hstu.hstu_mha_fwd(
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
                0,  # max_q_len,
                None,  # seq_offsets_q,
                num_softmax_heads,  # num_softmax_heads,
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
            out,
        ]
        if num_softmax_heads > 0:
            saved_tensors.append(softmax_lse)
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if attn_scale is not None:
            saved_tensors.append(attn_scale)
        if not recompute_normed_x_in_backward:
            saved_tensors.append(normed_x)
        if recompute_uvqk_in_backward:
            saved_tensors.append(uvqk_bias)
            if fp8_in_addmm_fwd:
                saved_tensors.append(x_scale)  # pyre-ignore
                saved_tensors.append(normed_x_fp8)  # pyre-ignore
        else:
            saved_tensors.append(uvqk)
        if rotary_weights is not None:
            saved_tensors.append(rotary_weights[0])
            saved_tensors.append(rotary_weights[1])
            saved_tensors.append(rotary_weights[2])
            saved_tensors.append(rotary_weights[3])
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.has_rotary_weights = rotary_weights is not None
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
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        ctx.silu_u = silu_u
        ctx.fp8_in_addmm_fwd = fp8_in_addmm_fwd
        ctx.num_softmax_heads = num_softmax_heads
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
        None,
        None,
        None,
    ]:
        x, norm_weight, norm_bias, x_mean, x_rstd, uvqk_weight, seq_offsets, out = (
            ctx.saved_tensors[:8]
        )
        idx = 8
        if ctx.num_softmax_heads > 0:
            softmax_lse = ctx.saved_tensors[idx]
            idx += 1
        else:
            softmax_lse = None
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
            normed_x, _, _, _, _, _ = triton_weighted_layer_norm_quantization_fwd(
                x=x,
                weight=norm_weight,
                bias=norm_bias,
                eps=ctx.norm_eps,
                mean=x_mean,
                rstd=x_rstd,
                quantize_output=ctx.fp8_in_addmm_fwd,
            )
        else:
            normed_x = ctx.saved_tensors[idx]
            idx += 1
        if ctx.recompute_uvqk_in_backward:
            uvqk_bias = ctx.saved_tensors[idx]
            idx += 1
            if not ctx.silu_u:
                # When silu_u is False, we only recompute vqk (not u)
                # Split the weights/biases to extract vqk portion
                _, vqk_weight = uvqk_weight.split(
                    [
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.hidden_dim * ctx.num_heads
                        + ctx.attn_dim * ctx.num_heads
                        + ctx.attn_dim * ctx.num_heads,
                    ],
                    dim=1,
                )
                _, vqk_bias = uvqk_bias.split(
                    [
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.hidden_dim * ctx.num_heads
                        + ctx.attn_dim * ctx.num_heads
                        + ctx.attn_dim * ctx.num_heads,
                    ],
                    dim=0,
                )
                if ctx.fp8_in_addmm_fwd:
                    x_scale, normed_x_fp8 = ctx.saved_tensors[idx : idx + 2]
                    vqk = fp8_rowwise_quantize_addmm(
                        x=normed_x,
                        x_fp8=normed_x_fp8,
                        w=vqk_weight,
                        y=vqk_bias,
                        x_scale=x_scale,
                        custom_kernel=False,
                        is_inference=False,
                    )
                    idx += 2
                else:
                    vqk = maybe_triton_addmm_fwd(
                        normed_x, vqk_weight, vqk_bias
                    ).contiguous()
                # Split vqk into v, q, k components
                v, q, k = vqk.split(
                    [
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                    ],
                    dim=1,
                )
                u = None
            else:
                # When silu_u is True, we recompute uvqk (all components)
                if ctx.fp8_in_addmm_fwd:
                    x_scale, normed_x_fp8 = ctx.saved_tensors[idx : idx + 2]
                    uvqk = fp8_rowwise_quantize_addmm(
                        x=normed_x,
                        x_fp8=normed_x_fp8,
                        w=uvqk_weight,
                        y=uvqk_bias,
                        x_scale=x_scale,
                        custom_kernel=False,
                        is_inference=False,
                    )
                    idx += 2
                else:
                    uvqk = maybe_triton_addmm_fwd(
                        normed_x, uvqk_weight, uvqk_bias
                    ).contiguous()
                # Split uvqk into u, v, q, k components
                u, v, q, k = uvqk.split(
                    [
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.hidden_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                        ctx.attn_dim * ctx.num_heads,
                    ],
                    dim=1,
                )
        else:
            uvqk = ctx.saved_tensors[idx]
            idx += 1
            # Split saved uvqk into u, v, q, k components
            u, v, q, k = uvqk.split(
                [
                    ctx.hidden_dim * ctx.num_heads,
                    ctx.hidden_dim * ctx.num_heads,
                    ctx.attn_dim * ctx.num_heads,
                    ctx.attn_dim * ctx.num_heads,
                ],
                dim=1,
            )
        if ctx.has_rotary_weights:
            q_cos_weights, q_sin_weights, k_cos_weights, k_sin_weights = (
                ctx.saved_tensors[idx : idx + 4]
            )
            idx += 4
        else:
            q_cos_weights, q_sin_weights, k_cos_weights, k_sin_weights = (
                None,
                None,
                None,
                None,
            )

        duvqk = torch.empty(
            [
                x.size(0),
                ctx.hidden_dim * ctx.num_heads * 2 + ctx.attn_dim * ctx.num_heads * 2,
            ],
            device=x.device,
            dtype=x.dtype,
        )
        du, dv, dq, dk = duvqk.split(
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
        if (
            ctx.recompute_uvqk_in_backward and ctx.has_rotary_weights
        ):  # recompute ROPE on qk
            q = triton_apply_rope_fwd(
                x=q,
                N=ctx.max_seq_len,
                seq_offsets=seq_offsets,
                cos_rope=q_cos_weights,
                sin_rope=q_sin_weights,
            )
            k = triton_apply_rope_fwd(
                x=k,
                N=ctx.max_seq_len,
                seq_offsets=seq_offsets,
                cos_rope=k_cos_weights,
                sin_rope=k_sin_weights,
            )
        dq = dq.view(-1, ctx.num_heads, ctx.attn_dim)
        dk = dk.view(-1, ctx.num_heads, ctx.attn_dim)
        dv = dv.view(-1, ctx.num_heads, ctx.hidden_dim)
        # Note: the two operations below update duvqk in place
        if is_sm100_plus():
            _dq, _dk, _dv = torch.ops.bw_hstu.bw_hstu_mha_bwd(
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
                ctx.max_seq_len,  # max_q_len,
                None,  # seq_offsets_q,
                None,  # max_seq_len_tensor,
                None,  # contextual_seq_len_tensor,
                None,  # max_attn_len_tensor,
                None,  # min_full_attn_seq_len_tensor,
                1,  # num_groups
            )
        else:
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
                out,
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
                0,  # max_q_len,
                None,  # seq_offsets_q,
                ctx.num_softmax_heads,  # num_softmax_heads,
                softmax_lse,
            )
        if ctx.has_rotary_weights:
            _dq = triton_apply_rope_bwd(
                grad=_dq,
                N=ctx.max_seq_len,
                seq_offsets=seq_offsets,
                cos_rope=q_cos_weights,
                sin_rope=q_sin_weights,
            )
            _dk = triton_apply_rope_bwd(
                grad=_dk,
                N=ctx.max_seq_len,
                seq_offsets=seq_offsets,
                cos_rope=k_cos_weights,
                sin_rope=k_sin_weights,
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
            None,
            None,
            None,
        )
