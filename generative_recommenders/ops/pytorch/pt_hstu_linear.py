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

import torch
import torch.nn.functional as F


def pytorch_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    u = u.to(torch.float32)
    if group_norm:
        if silu_u:
            u = F.silu(u)
            u = u.to(torch.float32)
        y = u * F.group_norm(
            x.view(-1, num_heads, linear_dim),
            num_groups=num_heads,
            weight=weight.to(torch.float32),
            bias=bias.to(torch.float32),
            eps=eps,
        ).view(-1, num_heads * linear_dim)
        if concat_u and concat_x:
            y = torch.cat([u, x, y], dim=1)
    else:
        mul_u = u
        if mul_u_activation_type == "sigmoid":
            mul_u = torch.sigmoid(u)
        elif mul_u_activation_type == "silu":
            mul_u = F.silu(u)
        y = mul_u * F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=weight.to(torch.float32),
            bias=bias.to(torch.float32),
            eps=eps,
        )
        if concat_u:
            if silu_u:
                u = F.silu(u)
            if concat_x:
                y = torch.cat([u, x, y], dim=1)
            else:
                y = torch.cat([u, y], dim=1)
        elif concat_x:
            y = torch.cat([x, y], dim=1)
    y = F.dropout(
        y,
        p=dropout_ratio,
        training=training,
    )
    return y.to(dtype)


def pytorch_hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    output_weight: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool = False,
    concat_u: bool = False,
    concat_x: bool = False,
    mul_u_activation_type: str = "none",
    group_norm: bool = False,
    num_heads: int = 1,
    linear_dim: int = -1,
) -> torch.Tensor:
    dtype = x.dtype
    y = pytorch_norm_mul_dropout(
        x=attn,
        u=u,
        weight=norm_weight,
        bias=norm_bias,
        eps=eps,
        dropout_ratio=dropout_ratio,
        training=training,
        silu_u=silu_u,
        concat_u=concat_u,
        concat_x=concat_x,
        mul_u_activation_type=mul_u_activation_type,
        group_norm=group_norm,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )
    return torch.addmm(x, y, output_weight.to(x.dtype)).to(dtype)
