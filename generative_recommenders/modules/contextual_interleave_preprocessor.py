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

from math import sqrt
from typing import Callable, Dict, Optional, Tuple

import torch
from generative_recommenders.common import fx_unwrap_optional_tensor
from generative_recommenders.modules.action_encoder import ActionEncoder
from generative_recommenders.modules.content_encoder import ContentEncoder
from generative_recommenders.modules.contextualize_mlps import (
    ContextualizedMLP,
    ParameterizedContextualizedMLP,
)
from generative_recommenders.modules.preprocessors import (
    get_contextual_input_embeddings,
    InputPreprocessor,
)
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged


class ContextualInterleavePreprocessor(InputPreprocessor):
    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        content_encoder: ContentEncoder,
        content_contextualize_mlp_fn: Callable[
            [int, int, int, bool], ContextualizedMLP
        ],
        action_encoder: ActionEncoder,
        action_contextualize_mlp_fn: Callable[[int, int, int, bool], ContextualizedMLP],
        pmlp_contextual_dropout_ratio: float = 0.0,
        enable_interleaving: bool = False,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._input_embedding_dim: int = input_embedding_dim
        self._output_embedding_dim: int = output_embedding_dim
        self._contextual_feature_to_max_length: Dict[str, int] = (
            contextual_feature_to_max_length
        )
        self._max_contextual_seq_len: int = sum(
            contextual_feature_to_max_length.values()
        )
        self._contextual_feature_to_min_uih_length: Dict[str, int] = (
            contextual_feature_to_min_uih_length
        )
        std = 1.0 * sqrt(2.0 / float(input_embedding_dim + output_embedding_dim))
        self._batched_contextual_linear_weights = torch.nn.Parameter(
            torch.empty(
                (
                    self._max_contextual_seq_len,
                    input_embedding_dim,
                    output_embedding_dim,
                )
            ).normal_(0.0, std)
        )
        self._pmlp_contextual_dropout_ratio: float = pmlp_contextual_dropout_ratio
        self._batched_contextual_linear_bias = torch.nn.Parameter(
            torch.empty((self._max_contextual_seq_len, 1, output_embedding_dim)).fill_(
                0.0
            )
        )
        contextual_embedding_dim: int = (
            self._max_contextual_seq_len * input_embedding_dim
        )
        self._content_encoder: ContentEncoder = content_encoder
        self._content_embedding_mlp: ContextualizedMLP = content_contextualize_mlp_fn(
            self._content_encoder.output_embedding_dim,
            output_embedding_dim,
            contextual_embedding_dim,
            is_inference,
        )
        self._action_encoder: ActionEncoder = action_encoder
        self._action_embedding_mlp: ContextualizedMLP = action_contextualize_mlp_fn(
            self._action_encoder.output_embedding_dim,
            output_embedding_dim,
            contextual_embedding_dim,
            is_inference,
        )
        self._enable_interleaving: bool = enable_interleaving

    def combine_embeddings(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        content_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor,
        contextual_embeddings: Optional[torch.Tensor],
        num_targets: torch.Tensor,
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self._enable_interleaving:
            output_seq_timestamps = seq_timestamps.repeat_interleave(2)
            output_seq_embeddings = torch.stack(
                [content_embeddings, action_embeddings], dim=1
            ).reshape(-1, self._output_embedding_dim)
            if self.interleave_targets():
                output_seq_lengths = seq_lengths * 2
                output_max_seq_len = (max_uih_len + max_targets) * 2
                output_num_targets = num_targets * 2
                output_total_uih_len = total_uih_len * 2
                output_total_targets = total_targets * 2
            else:
                seq_lengths_by_2 = seq_lengths * 2
                output_seq_lengths = seq_lengths_by_2 - num_targets
                output_max_seq_len = 2 * max_uih_len + max_targets
                indices = torch.arange(
                    2 * (max_uih_len + max_targets), device=seq_lengths.device
                ).view(1, -1)
                valid_mask = torch.logical_and(
                    indices < seq_lengths_by_2.view(-1, 1),
                    torch.logical_or(
                        indices < (output_seq_lengths - num_targets).view(-1, 1),
                        torch.remainder(indices, 2) == 0,
                    ),
                )
                jagged_valid_mask = (
                    torch.ops.fbgemm.dense_to_jagged(
                        valid_mask.int().unsqueeze(-1),
                        [
                            torch.ops.fbgemm.asynchronous_complete_cumsum(
                                seq_lengths_by_2
                            )
                        ],
                    )[0]
                    .to(torch.bool)
                    .squeeze(1)
                )
                output_seq_embeddings = output_seq_embeddings[jagged_valid_mask]
                output_seq_timestamps = output_seq_timestamps[jagged_valid_mask]
                output_num_targets = num_targets
                output_total_uih_len = total_uih_len * 2
                output_total_targets = total_targets
        else:
            output_max_seq_len = max_uih_len + max_targets
            output_seq_lengths = seq_lengths
            output_num_targets = num_targets
            output_seq_timestamps = seq_timestamps
            output_seq_embeddings = content_embeddings + action_embeddings
            output_total_uih_len = total_uih_len
            output_total_targets = total_targets

        # concat contextual embeddings
        output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            output_seq_lengths
        )
        if self._max_contextual_seq_len > 0:
            output_seq_embeddings = concat_2D_jagged(
                max_seq_len=self._max_contextual_seq_len + output_max_seq_len,
                values_left=fx_unwrap_optional_tensor(contextual_embeddings).reshape(
                    -1, self._output_embedding_dim
                ),
                values_right=output_seq_embeddings,
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.hammer_kernel(),
            )
            output_seq_timestamps = concat_2D_jagged(
                max_seq_len=self._max_contextual_seq_len + output_max_seq_len,
                values_left=torch.zeros(
                    (output_seq_lengths.size(0) * self._max_contextual_seq_len, 1),
                    dtype=output_seq_timestamps.dtype,
                    device=output_seq_timestamps.device,
                ),
                values_right=output_seq_timestamps.unsqueeze(-1),
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.hammer_kernel(),
            ).squeeze(-1)
            output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
            output_total_uih_len = (
                output_total_uih_len
                + self._max_contextual_seq_len * output_seq_lengths.size(0)
            )
            output_seq_lengths = output_seq_lengths + self._max_contextual_seq_len
            output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                output_seq_lengths
            )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        )

    def forward(  # noqa C901
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        max_seq_len = max_uih_len + max_targets
        with torch.autocast(
            seq_embeddings.device.type,
            dtype=torch.bfloat16,
            enabled=(not self.is_inference and self._training_dtype == torch.bfloat16),
        ):
            # get contextual_embeddings
            contextual_embeddings: Optional[torch.Tensor] = None
            pmlp_contextual_embeddings: Optional[torch.Tensor] = None
            if self._max_contextual_seq_len > 0:
                contextual_input_embeddings = get_contextual_input_embeddings(
                    seq_lengths=seq_lengths,
                    seq_payloads=seq_payloads,
                    contextual_feature_to_max_length=self._contextual_feature_to_max_length,
                    contextual_feature_to_min_uih_length=self._contextual_feature_to_min_uih_length,
                    dtype=seq_embeddings.dtype,
                )
                if isinstance(
                    self._action_embedding_mlp, ParameterizedContextualizedMLP
                ) or isinstance(
                    self._action_embedding_mlp, ParameterizedContextualizedMLP
                ):
                    pmlp_contextual_embeddings = torch.nn.functional.dropout(
                        contextual_input_embeddings,
                        p=self._pmlp_contextual_dropout_ratio,
                        training=self.training,
                    )
                contextual_embeddings = torch.baddbmm(
                    self._batched_contextual_linear_bias.to(
                        contextual_input_embeddings.dtype
                    ),
                    contextual_input_embeddings.view(
                        -1, self._max_contextual_seq_len, self._input_embedding_dim
                    ).transpose(0, 1),
                    self._batched_contextual_linear_weights.to(
                        contextual_input_embeddings.dtype
                    ),
                ).transpose(0, 1)

            # content embeddings
            seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
            target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)
            uih_offsets = seq_offsets - target_offsets
            content_embeddings = self._content_encoder(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                uih_offsets=uih_offsets,
                target_offsets=target_offsets,
                seq_embeddings=seq_embeddings,
                seq_payloads=seq_payloads,
            )
            content_embeddings = self._content_embedding_mlp(
                seq_embeddings=content_embeddings,
                seq_offsets=seq_offsets,
                max_seq_len=max_seq_len,
                contextual_embeddings=pmlp_contextual_embeddings,
            )

            # action embeddings
            action_embeddings = self._action_encoder(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                uih_offsets=uih_offsets,
                target_offsets=target_offsets,
                seq_embeddings=seq_embeddings,
                seq_payloads=seq_payloads,
            ).to(seq_embeddings.dtype)
            action_embeddings = self._action_embedding_mlp(
                seq_embeddings=action_embeddings,
                seq_offsets=seq_offsets,
                max_seq_len=max_seq_len,
                contextual_embeddings=pmlp_contextual_embeddings,
            )

            (
                output_max_seq_len,
                output_total_uih_len,
                output_total_targets,
                output_seq_lengths,
                output_seq_offsets,
                output_seq_timestamps,
                output_seq_embeddings,
                output_num_targets,
            ) = self.combine_embeddings(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_lengths=seq_lengths,
                seq_timestamps=seq_timestamps,
                content_embeddings=content_embeddings,
                action_embeddings=action_embeddings,
                contextual_embeddings=contextual_embeddings,
                num_targets=num_targets,
            )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            seq_payloads,
        )

    def interleave_targets(self) -> bool:
        return self.is_train and self._enable_interleaving
