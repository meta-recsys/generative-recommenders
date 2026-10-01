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

"""PyTorch reference configurations and attention for HSTU Ultra."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import torch
from generative_recommenders.common import HammerKernel, switch_to_contiguous_if_needed
from generative_recommenders.ops.pytorch.pt_hstu_ultra import (
    get_hstu_ultra_valid_attn_mask,
    pytorch_hstu_ultra_attention,
)
from generative_recommenders.ops.triton.triton_hstu_attention import triton_hstu_mha

__all__ = [
    "get_hstu_ultra_valid_attn_mask",
    "hstu_ultra_mha",
    "hstu_ultra_attention_configs",
    "HSTUUltraAttentionConfig",
    "pytorch_hstu_ultra_mha",
]

HSTU_ULTRA_CONFIG_NAMES = ("hstu_ultra_l1", "hstu_ultra_post_cross")


@dataclass(frozen=True)
class HSTUUltraAttentionConfig:
    """Model-derived configuration for an HSTU Ultra attention layer."""

    name: str
    description: str
    heads: int
    attention_dim: int
    value_dim: int
    max_uih_length: int
    max_targets: int
    max_attn_len: int
    full_attn_size: int
    default_sequence_lengths: tuple[int, ...]

    @property
    def max_sequence_length(self) -> int:
        return self.max_uih_length + self.max_targets


def hstu_ultra_attention_configs() -> dict[str, HSTUUltraAttentionConfig]:
    """Return the supported HSTU Ultra self-attention configurations."""
    return {
        "hstu_ultra_l1": HSTUUltraAttentionConfig(
            name="hstu_ultra_l1",
            description="L1 causal self-attention with a semi-local window",
            heads=4,
            attention_dim=128,
            value_dim=128,
            max_uih_length=16384 - 30,
            max_targets=0,
            max_attn_len=256,
            full_attn_size=256,
            default_sequence_lengths=(512, 1024, 2048),
        ),
        "hstu_ultra_post_cross": HSTUUltraAttentionConfig(
            name="hstu_ultra_post_cross",
            description="Post-cross causal self-attention with target tokens",
            heads=4,
            attention_dim=128,
            value_dim=128,
            max_uih_length=1024,
            max_targets=512,
            max_attn_len=0,
            full_attn_size=0,
            default_sequence_lengths=(768, 1024, 1536),
        ),
    }


def _register_fbgemm_ops() -> None:
    try:
        importlib.import_module("fbgemm_gpu")
    except ModuleNotFoundError as error:
        if error.name != "fbgemm_gpu":
            raise
        # fbsource links these operators through BUCK; OSS installs this package.


def _validate_inputs(
    config: HSTUUltraAttentionConfig,
    max_seq_len: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None,
) -> None:
    if max_seq_len <= 0 or max_seq_len > config.max_sequence_length:
        raise ValueError(f"max_seq_len must be in [1, {config.max_sequence_length}]")
    if q.ndim != 3 or k.shape != q.shape:
        raise ValueError("q and k must have the same three-dimensional shape")
    if v.ndim != 3 or v.shape[:2] != q.shape[:2]:
        raise ValueError("v must match q in token and head dimensions")
    if q.shape[1:] != (config.heads, config.attention_dim):
        raise ValueError("q and k do not match the configured heads and dimension")
    if v.shape[2] != config.value_dim:
        raise ValueError("v does not match the configured value dimension")
    batch_size = seq_offsets.numel() - 1
    if config.max_targets == 0 and num_targets is not None:
        raise ValueError(f"{config.name} does not use target tokens")
    if config.max_targets > 0 and (
        num_targets is None or num_targets.shape != (batch_size,)
    ):
        raise ValueError(f"{config.name} requires one target count per sequence")


def pytorch_hstu_ultra_mha(
    config: HSTUUltraAttentionConfig,
    max_seq_len: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
    dropout_pr: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Run an HSTU Ultra self-attention configuration with PyTorch."""
    _register_fbgemm_ops()
    _validate_inputs(config, max_seq_len, q, k, v, seq_offsets, num_targets)
    return pytorch_hstu_ultra_attention(
        max_seq_len=max_seq_len,
        alpha=1.0 / config.attention_dim,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        dropout_pr=dropout_pr,
        training=training,
        num_targets=num_targets,
        max_attn_len=config.max_attn_len,
        full_attn_size=config.full_attn_size,
    )


def hstu_ultra_mha(
    config: HSTUUltraAttentionConfig,
    max_seq_len: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: torch.Tensor | None = None,
    dropout_pr: float = 0.0,
    training: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """Run a named HSTU Ultra configuration with PyTorch or Triton."""
    if kernel == HammerKernel.PYTORCH:
        return pytorch_hstu_ultra_mha(
            config=config,
            max_seq_len=max_seq_len,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            dropout_pr=dropout_pr,
            training=training,
        )
    if kernel != HammerKernel.TRITON:
        raise ValueError(f"unsupported HSTU Ultra kernel: {kernel}")

    _register_fbgemm_ops()
    _validate_inputs(config, max_seq_len, q, k, v, seq_offsets, num_targets)
    torch._assert(q.is_cuda, "q must be a CUDA tensor for Triton")
    torch._assert(k.is_cuda, "k must be a CUDA tensor for Triton")
    torch._assert(v.is_cuda, "v must be a CUDA tensor for Triton")
    torch._assert(seq_offsets.is_cuda, "seq_offsets must be a CUDA tensor for Triton")
    torch._assert(dropout_pr == 0.0, "dropout is not implemented for Triton")
    return triton_hstu_mha(
        N=max_seq_len,
        alpha=1.0 / config.attention_dim,
        q=switch_to_contiguous_if_needed(q),
        k=switch_to_contiguous_if_needed(k),
        v=switch_to_contiguous_if_needed(v),
        seq_offsets=seq_offsets.contiguous(),
        num_targets=num_targets,
        max_attn_len=config.max_attn_len,
        min_full_attn_seq_len=config.full_attn_size,
    )
