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

import abc
from typing import Optional

import torch
from generative_recommenders.research.rails.similarities.module import SimilarityModule


class SequentialEncoderWithLearnedSimilarityModule(torch.nn.Module):
    """
    Interface enabling using various similarity functions (besides inner products)
    as part of a sequential encoder/decoder.

    See rails/ for more details.
    """

    def __init__(
        self,
        ndp_module: SimilarityModule,
    ) -> None:
        super().__init__()

        self._ndp_module: SimilarityModule = ndp_module

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def similarity_fn(
        self,
        query_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
        item_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        torch._assert(
            len(query_embeddings.size()) == 2, "len(query_embeddings.size()) must be 2"
        )
        torch._assert(len(item_ids.size()) == 2, "len(item_ids.size()) must be 2")
        if item_embeddings is None:
            item_embeddings = self.get_item_embeddings(item_ids)  # pyre-ignore [29]
        torch._assert(
            len(item_embeddings.size()) == 3, "len(item_embeddings.size()) must be 3"
        )

        return self._ndp_module(
            query_embeddings=query_embeddings,  # (B, query_embedding_dim)
            item_embeddings=item_embeddings,  # (1/B, X, item_embedding_dim)
            item_ids=item_ids,
            **kwargs,
        )
