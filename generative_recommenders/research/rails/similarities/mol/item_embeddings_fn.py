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

"""
Defines functions to generate item-side embeddings for MoL.

Forked from bailuding/rails @ 664fdb9.
"""

from typing import Callable, Dict, Tuple

import torch
from generative_recommenders.research.rails.similarities.mol.embeddings_fn import (
    MoLEmbeddingsFn,
)


def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


class RecoMoLItemEmbeddingsFn(MoLEmbeddingsFn):
    """
    Generates P_X query-side embeddings for MoL based on input embeddings and other
    optional tensors for recommendation models. Tested for sequential retrieval
    scenarios.
    """

    def __init__(
        self,
        item_embedding_dim: int,
        item_dot_product_groups: int,
        dot_product_dimension: int,
        dot_product_l2_norm: bool,
        proj_fn: Callable[[int, int], torch.nn.Module],
        eps: float,
    ) -> None:
        super().__init__()

        self._item_emb_based_dot_product_groups: int = item_dot_product_groups
        self._item_emb_proj_module: torch.nn.Module = proj_fn(
            item_embedding_dim,
            dot_product_dimension * self._item_emb_based_dot_product_groups,
        )
        self._dot_product_dimension: int = dot_product_dimension
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        self._eps: float = eps

    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, item_embedding_dim,) x float where B is the batch size.
            kwargs: str-keyed tensors. Implementation-specific.

        Returns:
            Tuple of (
                (B, item_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed aux_losses,
            ).
        """
        split_item_embeddings = self._item_emb_proj_module(input_embeddings).reshape(
            input_embeddings.size()[:-1]
            + (
                self._item_emb_based_dot_product_groups,
                self._dot_product_dimension,
            )
        )

        if self._dot_product_l2_norm:
            split_item_embeddings = split_item_embeddings / torch.clamp(
                torch.linalg.norm(
                    split_item_embeddings,
                    ord=None,
                    dim=-1,
                    keepdim=True,
                ),
                min=self._eps,
            )
        return split_item_embeddings, {}
