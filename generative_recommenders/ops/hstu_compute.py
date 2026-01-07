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
from generative_recommenders.ops.layer_norm import layer_norm
from generative_recommenders.ops.mm import addmm
from generative_recommenders.ops.pytorch.pt_hstu_linear import (
    pytorch_hstu_compute_output,
)

try:
    from hammer.ops.triton.cc.addmm.triton_cc_addmm import triton_cc_addmm
    from hammer.ops.triton.cc.group_norm_mul_dropout.triton_cc_group_norm_mul_dropout import (
        triton_cc_group_norm_mul_dropout_wrapper,
    )
    from hammer.ops.triton.cc.layer_norm_mul_dropout.triton_cc_layer_norm_mul_dropout import (
        triton_cc_layer_norm_mul_dropout_wrapper,
    )
except ImportError:
    pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.hstu_attention import hstu_mha
from generative_recommenders.ops.triton.triton_hstu_linear import (
    triton_hstu_compute_output,
)
from generative_recommenders.ops.triton.triton_hstu_preprocess_and_attention import (
    triton_hstu_preprocess_and_attention,
)
from torch.fx._symbolic_trace import is_fx_tracing


def hstu_compute_uqvk(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normed_x = layer_norm(
        x,
        weight=norm_weight,
        bias=norm_bias,
        eps=norm_eps,
        kernel=kernel,
    )
    # NOTE: for AMD training, we go with torch.addmm instead of the triton
    # version before Triton on AMD achieves on-par perf with NV GPU.
    if torch.version.hip and kernel == HammerKernel.TRITON:
        uvqk = torch.addmm(uvqk_bias, normed_x, uvqk_weight)
    else:
        uvqk = addmm(uvqk_bias, normed_x, uvqk_weight, kernel)
    u, v, q, k = torch.split(
        uvqk,
        [
            hidden_dim * num_heads,
            hidden_dim * num_heads,
            attn_dim * num_heads,
            attn_dim * num_heads,
        ],
        dim=1,
    )
    u = F.silu(u)
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    return u, q, k, v


def hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    output_weight: torch.Tensor,
    num_heads: int,
    linear_dim: int,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool,
    group_norm: bool,
    recompute_y_in_backward: bool,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        return triton_hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            output_weight=output_weight,
            eps=norm_eps,
            dropout_ratio=dropout_ratio,
            training=training,
            concat_ux=concat_ux,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=linear_dim,
            seed=None,
            recompute_y_in_backward=recompute_y_in_backward,
        )
    elif kernel == HammerKernel.TRITON_CC:
        if group_norm:
            y = triton_cc_group_norm_mul_dropout_wrapper(
                x=attn,
                u=u,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                dropout_ratio=dropout_ratio,
                training=training,
                concat_ux=concat_ux,
                num_heads=num_heads,
                linear_dim=linear_dim,
            )
        else:
            y = triton_cc_layer_norm_mul_dropout_wrapper(
                x=attn,
                u=u,
                weight=norm_weight,
                bias=norm_bias,
                eps=norm_eps,
                dropout_ratio=dropout_ratio,
                training=training,
                concat_ux=concat_ux,
            )
        return triton_cc_addmm(x, y, output_weight)
    else:
        return pytorch_hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            output_weight=output_weight,
            eps=norm_eps,
            dropout_ratio=dropout_ratio,
            training=training,
            concat_ux=concat_ux,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=linear_dim,
        )


def hstu_preprocess_and_attention(
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
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    recompute_uvqk_in_backward: bool,
    recompute_normed_x_in_backward: bool,
    sort_by_length: bool,
    prefill: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(x.dim() == 2, "x must be 2-D")
        torch._assert(
            x.shape[1] == uvqk_weight.shape[0],
            "x.shape[1] must equal uvqk_weight.shape[0]",
        )
        torch._assert(
            uvqk_weight.shape[1] == 2 * num_heads * (hidden_dim + attn_dim),
            "uvqk_weight.shape[1] must equal 2 * num_heads * (hidden_dim + attn_dim)",
        )
        torch._assert(causal is True, "only causal attention is supported.")
    if kernel == HammerKernel.TRITON and prefill is False:
        u, attn_output = triton_hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=num_heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            attn_alpha=attn_alpha,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            sort_by_length=sort_by_length,
        )
        attn_output = attn_output.view(-1, hidden_dim * num_heads)
        k = None
        v = None
    else:
        u, q, k, v = hstu_compute_uqvk(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=num_heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            kernel=kernel,
        )
        attn_output = hstu_mha(
            max_seq_len=max_seq_len,
            alpha=attn_alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length=sort_by_length,
            kernel=kernel,
        ).view(-1, hidden_dim * num_heads)
    return u, attn_output, k, v
