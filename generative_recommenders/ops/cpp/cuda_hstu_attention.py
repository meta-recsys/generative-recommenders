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

# pyre-strict

from typing import Optional

import torch

try:
    # We need to import the CUDA kernels after importing torch
    import hstu._C  # pyre-ignore [21]
except:
    pass
try:
    torch.ops.load_library(
        "//generative_recommenders/ops/cpp/hstu_attention:hstu_flash_attention"
    )
except:
    pass


def cuda_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: Optional[torch.Tensor] = None,
    causal: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    min_full_attn_seq_len: int = 0,
    contextual_seq_len: int = 0,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    deterministic: bool = False,
    sm_margin: int = 0,
    max_q_len: int = 0,
    seq_offsets_q: Optional[torch.Tensor] = None,
    num_softmax_heads: int = 0,
    training: bool = True,
) -> torch.Tensor:
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
    """
    return torch.ops.hstu.hstu_mha(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        attn_scale,
        max_attn_len,
        min_full_attn_seq_len,
        contextual_seq_len,
        q_descale,
        k_descale,
        v_descale,
        sort_by_length,
        deterministic,
        sm_margin,
        max_q_len,
        seq_offsets_q,
        num_softmax_heads,
        training,
    )
