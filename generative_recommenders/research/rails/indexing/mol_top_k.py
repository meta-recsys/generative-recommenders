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
Defines exact- and approximate- Top-K modules for Mixture-of-Logits (MoL),
discussed in Retrieval with Learned Similarities (https://arxiv.org/abs/2407.15462).

Forked from bailuding/rails @ 664fdb9.
"""

from typing import Tuple

import torch
from generative_recommenders.research.rails.indexing.candidate_index import TopKModule
from generative_recommenders.research.rails.similarities.mol.similarity_fn import (
    MoLSimilarity,
)


class MoLTopKModule(TopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        flatten_item_ids_and_embeddings: bool,
        keep_component_level_item_embeddings: bool,
        component_level_item_embeddings_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Args:
            mol_module: MoLSimilarity.
            item_embeddings: (1, X, D) if mol_module._apply_item_embeddings_fn is True,
                (1, X, P_X, D_P) otherwise.
            item_ids: (1, X,) representing the item ids.
            flatten_item_ids_and_embeddings: bool. If true, do not keep the extra (1,)
                dimension at size(0).
            keep_component_level_item_embeddings: bool. If true, keep P_x component-level
                embeddings in `self._mol_item_embeddings` for downstream applications.
            component_level_item_embeddings_dtype: torch.dtype. If set, the dtype
                to keep component-level item embeddings in. By default we use bfloat16.
        """
        super().__init__()

        self._mol_module: MoLSimilarity = mol_module
        self._item_embeddings: torch.Tensor = (
            item_embeddings
            if not flatten_item_ids_and_embeddings
            else item_embeddings.squeeze(0)
        )

        if keep_component_level_item_embeddings:
            self._mol_item_embeddings: torch.Tensor = (
                mol_module.get_item_component_embeddings(
                    (
                        self._item_embeddings.squeeze(0)
                        if not flatten_item_ids_and_embeddings
                        else self._item_embeddings
                    ),
                    decoupled_inference=True,
                )[0]  # (X, D) -> (X, P_X, D_P)
            ).to(component_level_item_embeddings_dtype)

        self._item_ids: torch.Tensor = (
            item_ids if not flatten_item_ids_and_embeddings else item_ids.squeeze(0)
        )

    @property
    def mol_module(self) -> MoLSimilarity:
        return self._mol_module


class MoLBruteForceTopK(MoLTopKModule):
    def __init__(
        self,
        mol_module: MoLSimilarity,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        super().__init__(
            mol_module=mol_module,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            flatten_item_ids_and_embeddings=False,
            keep_component_level_item_embeddings=False,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, X, D) if mol_module._apply_query_embeddings_fn is True,
                (B, X, P_Q, D_P) otherwise.
            k: int. final top-k to return.
            sorted: bool. whether to sort final top-k results or not.
            **kwargs: Implementation-specific keys/values.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        # (B, X,)
        all_logits, _ = self.mol_module(
            query_embeddings,
            self._item_embeddings,
            **kwargs,
        )
        top_k_logits, top_k_indices = torch.topk(
            all_logits,
            dim=1,
            k=k,
            sorted=sorted,
            largest=True,
        )  # (B, k,)
        return top_k_logits, self._item_ids.squeeze(0)[top_k_indices]
