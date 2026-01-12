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

from typing import Optional, Tuple

import torch
from generative_recommenders.research.rails.indexing.candidate_index import TopKModule


class CandidateIndex(object):
    def __init__(
        self,
        ids: torch.Tensor,
        embeddings: torch.Tensor,
        invalid_ids: Optional[torch.Tensor] = None,
        debug_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._ids: torch.Tensor = ids
        self._embeddings: torch.Tensor = embeddings
        self._invalid_ids: Optional[torch.Tensor] = invalid_ids
        self._debug_path: Optional[str] = debug_path

    @property
    def ids(self) -> torch.Tensor:
        """
        Returns:
            (1, X) or (B, X), where valid ids are positive integers.
        """
        return self._ids

    @property
    def num_objects(self) -> int:
        return self._ids.size(1)

    @property
    def embeddings(self) -> torch.Tensor:
        """
        Returns:
            (1, X, D) or (B, X, D) with the same shape as `ids'.
        """
        return self._embeddings

    def filter_invalid_ids(
        self,
        invalid_ids: torch.Tensor,
    ) -> "CandidateIndex":
        """
        Filters invalid_ids (batch dimension dependent) from the current index.

        Args:
            invalid_ids: (B, N) x int64.

        Returns:
            CandidateIndex with invalid_ids filtered.
        """
        X = self._ids.size(1)
        if self._ids.size(0) == 1:
            # ((1, X, 1) == (B, 1, N)) -> (B, X)
            invalid_mask, _ = (self._ids.unsqueeze(2) == invalid_ids.unsqueeze(1)).max(
                dim=2
            )
            lengths = (~invalid_mask).int().sum(-1)  # (B,)
            valid_1d_mask = (~invalid_mask).view(-1)
            B: int = lengths.size(0)
            D: int = self._embeddings.size(-1)
            jagged_ids = self._ids.expand(B, -1).reshape(-1)[valid_1d_mask]
            jagged_embeddings = self._embeddings.expand(B, -1, -1).reshape(-1, D)[
                valid_1d_mask
            ]
            X_prime: int = lengths.max(-1)[0].item()
            jagged_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            return CandidateIndex(
                ids=torch.ops.fbgemm.jagged_to_padded_dense(
                    values=jagged_ids.unsqueeze(-1),
                    offsets=[jagged_offsets],
                    max_lengths=[X_prime],
                    padding_value=0,
                ).squeeze(-1),
                embeddings=torch.ops.fbgemm.jagged_to_padded_dense(
                    values=jagged_embeddings,
                    offsets=[jagged_offsets],
                    max_lengths=[X_prime],
                    padding_value=0.0,
                ),
                debug_path=self._debug_path,
            )
        else:
            assert self._invalid_ids == None
            return CandidateIndex(
                ids=self.ids,
                embeddings=self.embeddings,
                invalid_ids=invalid_ids,
                debug_path=self._debug_path,
            )

    def get_top_k_outputs(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        top_k_module: TopKModule,
        invalid_ids: Optional[torch.Tensor],
        r: int = 1,
        return_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Gets top-k outputs specified by `policy_fn', while filtering out
        invalid ids per row as specified by `invalid_ids'.

        Args:
            k: int. top k to return.
            policy_fn: lambda that takes in item-side embeddings (B, X, D,) and user-side
                embeddings (B * r, ...), and returns predictions (unnormalized logits)
                of shape (B * r, X,).
            invalid_ids: (B * r, N_0) x int64. The list of ids (if > 0) to filter from
                results if present. Expect N_0 to be a small constant.
            return_embeddings: bool if we should additionally return embeddings for the
                top k results.

        Returns:
            A tuple of (top_k_ids, top_k_prs, top_k_embeddings) of shape (B * r, k, ...).
        """
        B: int = query_embeddings.size(0)
        max_num_invalid_ids = 0
        if invalid_ids is not None:
            max_num_invalid_ids = invalid_ids.size(1)

        k_prime = min(k + max_num_invalid_ids, self.num_objects)
        top_k_prime_scores, top_k_prime_ids = top_k_module(
            query_embeddings=query_embeddings, k=k_prime
        )
        # Masks out invalid items rowwise.
        if invalid_ids is not None:
            id_is_valid = ~(
                (top_k_prime_ids.unsqueeze(2) == invalid_ids.unsqueeze(1)).max(2)[0]
            )  # [B, K + N_0]
            id_is_valid = torch.logical_and(
                id_is_valid, torch.cumsum(id_is_valid.int(), dim=1) <= k
            )
            # [[1, 0, 1, 0], [0, 1, 1, 1]], k=2 -> [[0, 2], [1, 2]]
            top_k_rowwise_offsets = torch.nonzero(id_is_valid, as_tuple=True)[1].view(
                -1, k
            )
            top_k_scores = torch.gather(
                top_k_prime_scores, dim=1, index=top_k_rowwise_offsets
            )
            top_k_ids = torch.gather(
                top_k_prime_ids, dim=1, index=top_k_rowwise_offsets
            )
        else:
            top_k_scores = top_k_prime_scores
            top_k_ids = top_k_prime_ids

        # TODO: this should be decoupled from candidate_index.
        if return_embeddings:
            raise ValueError("return_embeddings not supported yet.")
        else:
            top_k_embeddings = None
        return top_k_ids, top_k_scores, top_k_embeddings

    def apply_object_filter(self) -> "CandidateIndex":
        """
        Applies general per batch filters.
        """
        raise NotImplementedError("not implemented.")
