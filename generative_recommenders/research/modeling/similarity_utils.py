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

from typing import List, Optional, Tuple

import gin
import torch
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
    DotProductSimilarity,
)
from generative_recommenders.research.rails.similarities.layers import SwiGLU
from generative_recommenders.research.rails.similarities.mol.item_embeddings_fn import (
    RecoMoLItemEmbeddingsFn,
)
from generative_recommenders.research.rails.similarities.mol.query_embeddings_fn import (
    RecoMoLQueryEmbeddingsFn,
)
from generative_recommenders.research.rails.similarities.mol.similarity_fn import (
    MoLSimilarity,
    SoftmaxDropoutCombiner,
)


def init_mlp_xavier_weights_zero_bias(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0.0)


@gin.configurable
def create_mol_interaction_module(
    query_embedding_dim: int,
    item_embedding_dim: int,
    dot_product_dimension: int,
    query_dot_product_groups: int,
    item_dot_product_groups: int,
    temperature: float,
    query_dropout_rate: float,
    query_hidden_dim: int,
    item_dropout_rate: float,
    item_hidden_dim: int,
    gating_query_hidden_dim: int,
    gating_qi_hidden_dim: int,
    gating_item_hidden_dim: int,
    softmax_dropout_rate: float,
    bf16_training: bool,
    gating_query_fn: bool = True,
    gating_item_fn: bool = True,
    dot_product_l2_norm: bool = True,
    query_nonlinearity: str = "geglu",
    item_nonlinearity: str = "geglu",
    uid_dropout_rate: float = 0.5,
    uid_embedding_hash_sizes: Optional[List[int]] = None,
    uid_embedding_level_dropout: bool = False,
    gating_combination_type: str = "glu_silu",
    gating_item_dropout_rate: float = 0.0,
    gating_qi_dropout_rate: float = 0.0,
    eps: float = 1e-6,
) -> Tuple[MoLSimilarity, str]:
    """
    Gin wrapper for creating MoL learned similarity.
    """
    mol_module = MoLSimilarity(
        query_embedding_dim=query_embedding_dim,
        item_embedding_dim=item_embedding_dim,
        dot_product_dimension=dot_product_dimension,
        query_dot_product_groups=query_dot_product_groups,
        item_dot_product_groups=item_dot_product_groups,
        temperature=temperature,
        dot_product_l2_norm=dot_product_l2_norm,
        query_embeddings_fn=RecoMoLQueryEmbeddingsFn(
            query_embedding_dim=query_embedding_dim,
            query_dot_product_groups=query_dot_product_groups,
            dot_product_dimension=dot_product_dimension,
            dot_product_l2_norm=dot_product_l2_norm,
            proj_fn=lambda input_dim, output_dim: (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=query_dropout_rate),
                    SwiGLU(
                        in_features=input_dim,
                        out_features=query_hidden_dim,
                    ),
                    torch.nn.Linear(
                        in_features=query_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
            eps=eps,
        ),
        item_embeddings_fn=RecoMoLItemEmbeddingsFn(
            item_embedding_dim=item_embedding_dim,
            item_dot_product_groups=item_dot_product_groups,
            dot_product_dimension=dot_product_dimension,
            dot_product_l2_norm=dot_product_l2_norm,
            proj_fn=lambda input_dim, output_dim: (
                torch.nn.Sequential(
                    torch.nn.Dropout(p=item_dropout_rate),
                    SwiGLU(in_features=input_dim, out_features=item_hidden_dim),
                    torch.nn.Linear(
                        in_features=item_hidden_dim,
                        out_features=output_dim,
                    ),
                ).apply(init_mlp_xavier_weights_zero_bias)
            ),
            eps=eps,
        ),
        gating_query_only_partial_fn=lambda input_dim, output_dim: (  # pyre-ignore [6]
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=gating_query_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=gating_query_hidden_dim,
                    out_features=output_dim,
                    bias=False,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if gating_query_fn
            else None
        ),
        gating_item_only_partial_fn=lambda input_dim, output_dim: (  # pyre-ignore [6]
            torch.nn.Sequential(
                torch.nn.Dropout(p=gating_item_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=gating_item_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=gating_item_hidden_dim,
                    out_features=output_dim,
                    bias=False,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if gating_item_fn
            else None
        ),
        gating_qi_partial_fn=lambda input_dim, output_dim: (  # pyre-ignore [6]
            torch.nn.Sequential(
                torch.nn.Dropout(p=gating_qi_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=gating_qi_hidden_dim,
                ),
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=gating_qi_hidden_dim,
                    out_features=output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
            if gating_qi_hidden_dim > 0
            else torch.nn.Sequential(
                torch.nn.Dropout(p=gating_qi_dropout_rate),
                torch.nn.Linear(
                    in_features=input_dim,
                    out_features=output_dim,
                ),
            ).apply(init_mlp_xavier_weights_zero_bias)
        ),
        gating_combination_type=gating_combination_type,
        gating_normalization_fn=lambda _: SoftmaxDropoutCombiner(
            dropout_rate=softmax_dropout_rate, eps=1e-6
        ),
        eps=eps,
        autocast_bf16=bf16_training,
    )
    interaction_module_debug_str = (
        f"MoL-{query_dot_product_groups}x{item_dot_product_groups}x{dot_product_dimension}"
        + f"-t{temperature}-d{softmax_dropout_rate}"
        + f"{'-l2' if dot_product_l2_norm else ''}"
        + f"-q{query_hidden_dim}d{query_dropout_rate}{query_nonlinearity}"
        + f"-i{item_hidden_dim}d{item_dropout_rate}{item_nonlinearity}"
        + (f"-gq{gating_query_hidden_dim}" if gating_query_fn else "")
        + (
            f"-gi{gating_item_hidden_dim}d{gating_item_dropout_rate}"
            if gating_item_fn
            else ""
        )
        + f"-gqi{gating_qi_hidden_dim}d{gating_qi_dropout_rate}-x-{gating_combination_type}"
    )
    return mol_module, interaction_module_debug_str


@gin.configurable
def get_similarity_function(
    module_type: str,
    query_embedding_dim: int,
    item_embedding_dim: int,
    bf16_training: bool = False,
    activation_checkpoint: bool = False,
) -> Tuple[torch.nn.Module, str]:
    if module_type == "DotProduct":
        interaction_module = DotProductSimilarity()
        interaction_module_debug_str = "DotProduct"
    elif module_type == "MoL":
        interaction_module, interaction_module_debug_str = (
            create_mol_interaction_module(
                query_embedding_dim=query_embedding_dim,
                item_embedding_dim=item_embedding_dim,
                bf16_training=bf16_training,
            )
        )
    else:
        raise ValueError(f"Unknown interaction_module_type {module_type}")
    return interaction_module, interaction_module_debug_str
