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

from typing import Optional

import torch
from generative_recommenders.common import HammerKernel, switch_to_contiguous_if_needed
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_cached_hstu_mha,
    triton_hstu_mha,
)
from hammer.v2.ops.triton.template.tlx_bw_hstu_attention import tlx_bw_hstu_mha_wrapper

try:
    from hammer.ops.triton.cc.hstu_attention.triton_cc_hstu_attention import (
        triton_cc_hstu_mha,
    )
except:
    from generative_recommenders.ops.triton.triton_hstu_attention import (
        triton_hstu_mha as triton_cc_hstu_mha,
    )
from torch.fx._symbolic_trace import is_fx_tracing


def hstu_mha(
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
    sort_by_length: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    _, H, _ = q.shape
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(q.dim() == 3, "q must be 3-D")
        torch._assert(k.shape == q.shape, "k must be the same shape as q")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")
        torch._assert(causal, "only support causal attention")

    if kernel in [HammerKernel.TRITON, HammerKernel.TLX, HammerKernel.TRITON_CC]:
        if not is_fx_tracing() and kernel == HammerKernel.TRITON:
            torch._assert(q.is_cuda, "q must be CUDA tensor")
            torch._assert(k.is_cuda, "k must be CUDA tensor")
            torch._assert(v.is_cuda, "v must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")
            torch._assert(
                min_full_attn_seq_len == 0, "min_full_attn_seq_len not implemented"
            )
        assert attn_scale is None, "attn_scale not implemented"
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        v = switch_to_contiguous_if_needed(v)
        seq_offsets = seq_offsets.contiguous()

    if kernel == HammerKernel.TRITON:
        return triton_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length=sort_by_length,
            enable_tma=enable_tma,
        )
    elif kernel == HammerKernel.TLX:
        return tlx_bw_hstu_mha_wrapper(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            attn_scale=torch.tensor(1.0 / max_seq_len).to(q.device),
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length=sort_by_length,
        )
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
    else:
        return pytorch_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            dropout_pr=dropout_pr,
            training=training,
            num_targets=num_targets,
            attn_scale=attn_scale,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
        )


def delta_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    B = seq_offsets.size(0) - 1
    DeltaSize = L // B
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(delta_q.dim() == 3, "delta_q must be 3-D")
        torch._assert(L % B == 0, "delta_q must be padded")
        torch._assert(k.dim() == 3, "k must be 3-D")
        torch._assert(k.shape[1] == H, "wrong k shape[1]")
        torch._assert(k.shape[2] == D, "wrong k shape[2]")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")
    if kernel in [HammerKernel.TRITON, HammerKernel.TRITON_CC]:
        if not is_fx_tracing() and kernel == HammerKernel.TRITON:
            torch._assert(delta_q.is_cuda, "q must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            if num_targets is not None:
                torch._assert(num_targets.is_cuda, "num_targets must be CUDA tensor")
        seq_offsets = seq_offsets.contiguous()
        delta_q = switch_to_contiguous_if_needed(delta_q)
        k = switch_to_contiguous_if_needed(k)
        v = switch_to_contiguous_if_needed(v)

    if kernel == HammerKernel.TRITON:
        return triton_cached_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            enable_tma=enable_tma,
        )
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            is_delta_q=True,
            delta_size=DeltaSize,
        )
    else:
        return pytorch_cached_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
