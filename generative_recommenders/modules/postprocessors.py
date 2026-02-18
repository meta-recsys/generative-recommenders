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

from abc import abstractmethod
from typing import Dict, List, Tuple

import torch
from generative_recommenders.common import HammerModule, init_mlp_weights_optional_bias


@torch.fx.wrap
def _cast_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype != dtype:
        return t.to(dtype)
    return t


class OutputPostprocessor(HammerModule):
    """An abstract class for post-processing user embeddings after HSTU layers."""

    @abstractmethod
    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            seq_embeddings: (L, D)
            seq_timestamps: (L, )
            seq_payloads: str-keyed tensors. Implementation specific.

        Returns:
            postprocessed seq_embeddings, (L, D)
        """
        pass


class L2NormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with l2 norm."""

    def __init__(self, is_inference: bool = False) -> None:
        super().__init__(is_inference=is_inference)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return seq_embeddings / torch.linalg.norm(
            seq_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)


class LayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with layer norm."""

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._layer_norm: torch.nn.Module = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim], eps=eps
        )

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # pyre-fixme[6]: For 1st argument expected `dtype` but got `Union[dtype,
        #  Tensor, Module]`.
        return self._layer_norm(seq_embeddings.to(self._layer_norm.weight.dtype))


@torch.fx.wrap
def _unsqueeze_if_needed(t: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() == 3:
        return t.unsqueeze(0)
    return t


class TimestampLayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with timestamp-based MLP -> layer norm."""

    def __init__(
        self,
        embedding_dim: int,
        time_duration_features: List[Tuple[int, int]],
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._layer_norm: torch.nn.Module = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim], eps=eps
        )
        self.register_buffer(
            "_period_units",
            torch.Tensor([f[0] for f in time_duration_features]).view(1, -1),
        )
        self.register_buffer(
            "_units_per_period",
            torch.Tensor([f[1] for f in time_duration_features]).view(1, -1),
        )
        self._time_feature_combiner: torch.nn.Module = torch.nn.Linear(
            embedding_dim + 2 * len(time_duration_features),
            embedding_dim,
        ).apply(init_mlp_weights_optional_bias)

    def _concat_time_features(
        self,
        combined_embeddings: torch.Tensor,
        timestamps: torch.Tensor,  # [B] or [B, D]
    ) -> torch.Tensor:
        # concat time representation to combined embeddings
        period_units = self._period_units
        units_per_period = self._units_per_period

        timestamps = timestamps.unsqueeze(-1)
        period_units = _unsqueeze_if_needed(period_units, combined_embeddings)
        units_per_period = _unsqueeze_if_needed(
            units_per_period, combined_embeddings
        ).float()
        # Compute time features in float32 to avoid bf16 precision loss through
        # discontinuous floor/remainder ops, matching Inductor fusion behavior.
        _units_elapsed_type: torch.dtype = combined_embeddings.dtype
        _units_since_epoch = torch.div(
            timestamps.float(), period_units.float(), rounding_mode="floor"
        )  # [sum(N_i), num_time_features] or [B, N, num_time_features]
        _units_elapsed = (
            (torch.remainder(_units_since_epoch, units_per_period) / units_per_period)
            * 2
            * 3.14
        )
        _units_elapsed = torch.view_as_real(
            torch.polar(
                _cast_dtype(torch.ones_like(_units_elapsed), torch.float32),
                _cast_dtype(_units_elapsed, torch.float32),
            )
        ).flatten(
            -2, -1
        )  # [sum(N_i), num_time_features * 2] or [B, N, num_time_features * 2]
        _units_elapsed = _cast_dtype(_units_elapsed, _units_elapsed_type)
        combined_embeddings = torch.cat([combined_embeddings, _units_elapsed], dim=-1)
        return combined_embeddings

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        user_embeddings = self._time_feature_combiner(
            self._concat_time_features(seq_embeddings, timestamps=seq_timestamps).to(
                self._time_feature_combiner.weight.dtype  # pyre-fixme[6]: For 1st argument expected `dtype` but got `Union[dtype,
                #  Tensor, Module]`.
            )
        )
        return self._layer_norm(user_embeddings)
