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
import abc
from typing import Optional

import torch
from generative_recommenders.common import HammerModule, init_mlp_weights_optional_bias
from generative_recommenders.ops.jagged_tensors import jagged_dense_bmm_broadcast_add
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from libfb.py.pyre import none_throws


class ContextualizedMLP(HammerModule):
    @abc.abstractmethod
    def forward(
        self,
        max_seq_len: int,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            seq_embeddings: (L, D)
            seq_offsets: (B + 1,)
            max_seq_len: int
            contextual_embeddings: (B, D')
        """
        pass


class SimpleContextualizedMLP(ContextualizedMLP):
    def __init__(
        self,
        sequential_input_dim: int,
        sequential_output_dim: int,
        hidden_dim: int,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=sequential_input_dim,
                out_features=hidden_dim,
            ),
            SwishLayerNorm(hidden_dim, is_inference=is_inference),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=sequential_output_dim,
            ),
            LayerNorm(sequential_output_dim),
        ).apply(init_mlp_weights_optional_bias)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self._mlp(seq_embeddings)


class ParameterizedContextualizedMLP(ContextualizedMLP):
    def __init__(
        self,
        contextual_embedding_dim: int,
        sequential_input_dim: int,
        sequential_output_dim: int,
        hidden_dim: int,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._sequential_input_dim: int = sequential_input_dim
        self._sequential_output_dim: int = sequential_output_dim

        self._dense_features_compress: torch.nn.Module = torch.nn.Linear(
            in_features=contextual_embedding_dim,
            out_features=hidden_dim,
        ).apply(init_mlp_weights_optional_bias)

        self._attn_raw_weights: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=sequential_input_dim * sequential_output_dim,
            ),
        ).apply(init_mlp_weights_optional_bias)

        self._attn_weights_norm: torch.nn.Module = torch.nn.LayerNorm(
            [sequential_input_dim, sequential_output_dim]
        )

        self._res_weights: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            SwishLayerNorm(hidden_dim),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=sequential_output_dim,
            ),
        ).apply(init_mlp_weights_optional_bias)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shared_input = self._dense_features_compress(none_throws(contextual_embeddings))
        attn_weights = self._attn_weights_norm(
            self._attn_raw_weights(shared_input).reshape(
                -1, self._sequential_input_dim, self._sequential_output_dim
            )
        )
        return jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=seq_embeddings,
            dense=attn_weights.to(seq_embeddings.dtype),
            bias=self._res_weights(shared_input),
            kernel=self.hammer_kernel(),
        )
