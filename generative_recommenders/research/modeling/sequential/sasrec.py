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

# pyre-unsafe

"""
Implements SASRec (Self-Attentive Sequential Recommendation, https://arxiv.org/abs/1808.09781, ICDM'18).

Compared with the original paper which used BCE loss, this implementation is modified so that
we can utilize a Sampled Softmax loss proposed in Revisiting Neural Retrieval on Accelerators
(https://arxiv.org/abs/2306.04039, KDD'23) and Turning Dross Into Gold Loss: is BERT4Rec really
better than SASRec? (https://arxiv.org/abs/2309.07602, RecSys'23), where the authors showed
sampled softmax loss to significantly improved SASRec model quality.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders.research.modeling.sequential.utils import (
    get_current_embeddings,
)
from generative_recommenders.research.modeling.similarity_module import (
    SequentialEncoderWithLearnedSimilarityModule,
)
from generative_recommenders.research.rails.similarities.module import SimilarityModule


class StandardAttentionFF(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation_fn: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        assert activation_fn == "relu" or activation_fn == "gelu", (
            f"Invalid activation_fn {activation_fn}"
        )

        self._conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            torch.nn.GELU() if activation_fn == "gelu" else torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=embedding_dim,
                kernel_size=1,
            ),
            torch.nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Conv1D requires (B, D, N)
        return self._conv1d(inputs.transpose(-1, -2)).transpose(-1, -2) + inputs


class SASRec(SequentialEncoderWithLearnedSimilarityModule):
    """
    Implements SASRec (Self-Attentive Sequential Recommendation, https://arxiv.org/abs/1808.09781, ICDM'18).

    Compared with the original paper which used BCE loss, this implementation is modified so that
    we can utilize a Sampled Softmax loss proposed in Revisiting Neural Retrieval on Accelerators
    (https://arxiv.org/abs/2306.04039, KDD'23) and Turning Dross Into Gold Loss: is BERT4Rec really
    better than SASRec? (https://arxiv.org/abs/2309.07602, RecSys'23), where the authors showed
    sampled softmax loss to significantly improved SASRec model quality.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_hidden_dim: int,
        ffn_activation_fn: str,
        ffn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: SimilarityModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        activation_checkpoint: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)

        self._embedding_module: EmbeddingModule = embedding_module
        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len + max_output_len
        self._input_features_preproc: InputFeaturesPreprocessorModule = (
            input_features_preproc_module
        )
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._activation_checkpoint: bool = activation_checkpoint
        self._verbose: bool = verbose

        self.attention_layers = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._ffn_hidden_dim: int = ffn_hidden_dim
        self._ffn_activation_fn: str = ffn_activation_fn
        self._ffn_dropout_rate: float = ffn_dropout_rate

        for _ in range(num_blocks):
            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    embed_dim=self._embedding_dim,
                    num_heads=num_heads,
                    dropout=ffn_dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layers.append(
                StandardAttentionFF(
                    embedding_dim=self._embedding_dim,
                    hidden_dim=ffn_hidden_dim,
                    activation_fn=ffn_activation_fn,
                    dropout_rate=self._ffn_dropout_rate,
                )
            )

        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (self._max_sequence_length, self._max_sequence_length),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self.reset_state()

    def reset_state(self) -> None:
        for name, params in self.named_parameters():
            if (
                "_input_features_preproc" in name
                or "_embedding_module" in name
                or "_output_postproc" in name
            ):
                if self._verbose:
                    print(f"Skipping initialization for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(
                        f"Initialize {name} as xavier normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"SASRec-d{self._item_embedding_dim}-b{self._num_blocks}-h{self._num_heads}"
            + "-"
            + self._input_features_preproc.debug_str()
            + "-"
            + self._output_postproc.debug_str()
            + f"-ffn{self._ffn_hidden_dim}-{self._ffn_activation_fn}-d{self._ffn_dropout_rate}"
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )

    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        Q = F.layer_norm(
            user_embeddings,
            normalized_shape=(self._embedding_dim,),
            eps=1e-8,
        )
        mha_outputs, _ = self.attention_layers[i](
            query=Q,
            key=user_embeddings,
            value=user_embeddings,
            attn_mask=self._attn_mask,
        )
        user_embeddings = self.forward_layers[i](
            F.layer_norm(
                Q + mha_outputs,
                normalized_shape=(self._embedding_dim,),
                eps=1e-8,
            )
        )
        user_embeddings *= valid_mask
        return user_embeddings

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            past_ids: (B, N,) x int

        Returns:
            (B, N, D,) x float
        """
        past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )

        for i in range(len(self.attention_layers)):
            if self._activation_checkpoint:
                user_embeddings = torch.utils.checkpoint.checkpoint(
                    self._run_one_layer,
                    i,
                    user_embeddings,
                    valid_mask,
                    use_reentrant=False,
                )
            else:
                user_embeddings = self._run_one_layer(i, user_embeddings, valid_mask)

        return self._output_postproc(user_embeddings)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            past_ids: [B, N] x int64 where the latest engaged ids come first. In
                particular, [:, 0] should correspond to the last engaged values.
            past_ratings: [B, N] x int64.
            past_timestamps: [B, N] x int64.

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths,
            past_ids,
            past_embeddings,
            past_payloads,
        )
        return encoded_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,  # [B, N] x int64
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded_seq_embeddings = self.generate_user_embeddings(
            past_lengths, past_ids, past_embeddings, past_payloads
        )  # [B, N, D]
        return get_current_embeddings(
            lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings
        )

    def predict(
        self,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
        past_timestamps: torch.Tensor,
        next_timestamps: torch.Tensor,
        target_ids: torch.Tensor,
        batch_id: Optional[int] = None,
    ) -> torch.Tensor:
        return self.interaction(  # pyre-ignore [29]
            self.encode(
                past_ids,
                past_ratings,
                past_timestamps,
                next_timestamps,  # pyre-ignore [6]
            ),
            target_ids,
        )  # [B, X]
