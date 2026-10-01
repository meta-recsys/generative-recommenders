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

"""HSTU Ultra layer-stack configurations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.ops.hstu_ultra import (
    hstu_ultra_attention_configs,
    HSTUUltraAttentionConfig,
)

__all__ = [
    "HSTUUltraStack",
    "HSTUUltraStackConfig",
    "hstu_ultra_stack_configs",
]


@dataclass(frozen=True)
class HSTUUltraStackConfig:
    """Configuration for a homogeneous HSTU Ultra layer stack."""

    name: str
    description: str
    attention: HSTUUltraAttentionConfig
    embedding_dim: int
    num_layers: int
    output_dropout_ratio: float = 0.0
    fp8_addmm_fwd: bool = False


def hstu_ultra_stack_configs(
    fp8_addmm_fwd: bool = False,
) -> dict[str, HSTUUltraStackConfig]:
    """Return model-derived semi-local and full-causal Ultra configurations."""
    attention_configs = hstu_ultra_attention_configs()
    return {
        "hstu_ultra_semi_local": HSTUUltraStackConfig(
            name="hstu_ultra_semi_local",
            description="Eight-layer HSTU Ultra semi-local stack",
            attention=attention_configs["hstu_ultra_semi_local"],
            embedding_dim=512,
            num_layers=8,
            fp8_addmm_fwd=fp8_addmm_fwd,
        ),
        "hstu_ultra": HSTUUltraStackConfig(
            name="hstu_ultra",
            description="Twelve-layer target-aware HSTU Ultra stack",
            attention=attention_configs["hstu_ultra"],
            embedding_dim=512,
            num_layers=12,
            fp8_addmm_fwd=fp8_addmm_fwd,
        ),
    }


class HSTUUltraStack(STUStack):
    """A stack of complete HSTU layers using an HSTU Ultra configuration."""

    def __init__(
        self,
        config: HSTUUltraStackConfig,
        is_inference: bool = False,
        kernel: HammerKernel = HammerKernel.TRITON,
    ) -> None:
        attention = config.attention
        layers: list[STU] = [
            STULayer(
                config=STULayerConfig(
                    embedding_dim=config.embedding_dim,
                    num_heads=attention.heads,
                    hidden_dim=attention.value_dim,
                    attention_dim=attention.attention_dim,
                    output_dropout_ratio=config.output_dropout_ratio,
                    causal=True,
                    target_aware=attention.max_targets > 0,
                    max_attn_len=attention.max_attn_len,
                    attn_alpha=1.0 / attention.attention_dim,
                    sort_by_length=True,
                    fp8_addmm_fwd=config.fp8_addmm_fwd,
                    min_full_attn_seq_len=attention.full_attn_size,
                ),
                is_inference=is_inference,
            )
            for _ in range(config.num_layers)
        ]
        super().__init__(stu_list=layers, is_inference=is_inference)
        self.config: HSTUUltraStackConfig = config
        self.set_hammer_kernel(kernel)

    @torch.jit.unused
    def prepare_fp8_weights(self) -> None:
        """Cache row-wise quantized projection weights for inference."""
        for layer in self._stu_layers:
            assert isinstance(layer, STULayer)
            layer.prepare_fp8_weights()

    @torch.jit.unused
    def clear_fp8_weight_cache(self) -> None:
        """Discard cached FP8 projection weights."""
        for layer in self._stu_layers:
            assert isinstance(layer, STULayer)
            layer.clear_fp8_weight_cache()
