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

from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.pytorch.pt_jagged import pytorch_jagged_dense_bmm_add
from generative_recommenders.ops.pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
    pytorch_hstu_concat_l2_embeddings,
    pytorch_hstu_split_l2_embeddings,
    pytorch_split_2D_jagged,
)

from generative_recommenders.ops.triton.triton_jagged import triton_jagged_dense_bmm_add
from generative_recommenders.ops.triton.triton_jagged_tensors import (
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from torch.fx._symbolic_trace import is_fx_tracing

try:
    from hammer.ops.triton.cc.jagged_dense_bmm.triton_cc_jagged_dense_bmm import (
        triton_cc_jagged_dense_bmm,
    )
except ImportError:
    pass


torch.fx.wrap("triton_concat_2D_jagged")
torch.fx.wrap("triton_split_2D_jagged")


def concat_2D_jagged(
    max_seq_len: int,
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if not is_fx_tracing():
        torch._assert(values_left.dim() == 2, "values_left must be 2D")
        torch._assert(values_right.dim() == 2, "values_right must be 2D")
        torch._assert(
            values_right.shape[1] == values_left.shape[1],
            f"values_left shape[1] must be equal to values_right shape[1] {values_left.shape[1]} vs {values_right.shape[1]}",
        )
    if kernel == HammerKernel.TRITON:
        return triton_concat_2D_jagged(
            max_seq_len=max_seq_len,
            values_left=values_left,
            values_right=values_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )
    else:
        return pytorch_concat_2D_jagged(
            values_left=values_left,
            values_right=values_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )


def split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int] = None,
    total_len_right: Optional[int] = None,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not is_fx_tracing():
        torch._assert(values.dim() == 2, "values must be 2D")
        torch._assert(
            offsets_left is not None or offsets_right is not None,
            "offsets_left and offsets_right cannot be None at the same time",
        )
        if offsets_left is None:
            torch._assert(
                max_len_left is not None,
                "max_len_left must be provided when offsets_left is None",
            )
        if offsets_right is None:
            torch._assert(
                max_len_right is not None,
                "max_len_right must be provided when offsets_right is None",
            )
        if offsets_left is not None and offsets_right is not None:
            torch._assert(
                offsets_left.shape[0] == offsets_right.shape[0],
                "offsets_left shape[0] must be equal to offsets_right shape[0]",
            )
    if kernel == HammerKernel.TRITON:
        return triton_split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            total_len_left=total_len_left,
            total_len_right=total_len_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )
    else:
        return pytorch_split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )


def hstu_split_l2_embeddings(
    max_seq_len: int,
    x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kernel == HammerKernel.TRITON:
        return triton_split_2D_jagged(
            max_seq_len=max_seq_len,
            values=x,
            total_len_right=None,
            total_len_left=None,
            max_len_left=None,
            max_len_right=None,
            offsets_left=prefix_offsets,
            offsets_right=l2_offsets,
            n_prefix_to_right=contextual_seq_len,
        )
    else:
        return pytorch_hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            prefix_offsets=prefix_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
        )


def hstu_concat_l2_embeddings(
    max_prefix_len: int,
    prefix_x: torch.Tensor,
    prefix_offsets: torch.Tensor,
    max_l2_len: int,
    l2_x: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        return triton_concat_2D_jagged(
            max_seq_len=max_prefix_len + max_l2_len,
            values_left=prefix_x,
            values_right=l2_x,
            max_len_left=max_prefix_len,
            max_len_right=max_l2_len,
            offsets_left=prefix_offsets,
            offsets_right=l2_offsets,
            n_prefix_from_right=contextual_seq_len,
        )
    else:
        return pytorch_hstu_concat_l2_embeddings(
            contextual_seq_len=contextual_seq_len,
            max_prefix_len=max_prefix_len,
            prefix_x=prefix_x,
            prefix_offsets=prefix_offsets,
            max_l2_len=max_l2_len,
            l2_x=l2_x,
            l2_offsets=l2_offsets,
        )


def jagged_dense_bmm_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Computing out = jagged x dense + bias
    jagged has shape (sum_B(M_i), K), dense has shape (B, K, N), and bias has shape (B, N)
    out has shape (sum_B(M_i), N)
    """
    if not is_fx_tracing():
        _, K = jagged.shape
        B, _, N = dense.shape
        torch._assert(dense.shape[1] == K, "wrong dense shape[1]")
        torch._assert(seq_offsets.shape[0] == B + 1, "wrong seq_offsets shape[0]")
        torch._assert(bias.shape[0] == B, "wrong bias shape[0]")
        torch._assert(bias.shape[1] == N, "wrong bias shape[1]")
    if kernel == HammerKernel.TRITON:
        return triton_jagged_dense_bmm_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
            elementwise=False,
        )
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_jagged_dense_bmm(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
        )
    else:
        return pytorch_jagged_dense_bmm_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
        )
