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

import torch
from generative_recommenders.research.rails.indexing.candidate_index import TopKModule
from generative_recommenders.research.rails.indexing.mips_top_k import (
    MIPSBruteForceTopK,
)
from generative_recommenders.research.rails.indexing.mol_top_k import MoLBruteForceTopK


def get_top_k_module(
    top_k_method: str,
    model: torch.nn.Module,
    item_embeddings: torch.Tensor,
    item_ids: torch.Tensor,
) -> TopKModule:
    if top_k_method == "MIPSBruteForceTopK":
        top_k_module = MIPSBruteForceTopK(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    elif top_k_method == "MoLBruteForceTopK":
        top_k_module = MoLBruteForceTopK(  # pyre-ignore [20]
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    else:
        raise ValueError(f"Invalid top-k method {top_k_method}")
    return top_k_module
