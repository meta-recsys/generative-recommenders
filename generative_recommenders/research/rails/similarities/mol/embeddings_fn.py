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
Defines interface for generating query- and item-side embeddings for MoL.

Forked from bailuding/rails @ 664fdb9.
"""

import abc
from typing import Dict, Tuple

import torch


class MoLEmbeddingsFn(torch.nn.Module):
    """
    Generates K_Q query-side (K_I item-side) embeddings for MoL based on
    input embeddings and other optional implementation-specific tensors.
    """

    @abc.abstractmethod
    def forward(
        self,
        input_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, ...) x float where B is the batch size.
            kwargs: implementation-specific.

        Returns:
            Tuple of (
                (B, query_dot_product_groups/item_dot_product_groups, dot_product_embedding_dim) x float,
                str-keyed auxiliary losses.
            ).
        """
        pass
