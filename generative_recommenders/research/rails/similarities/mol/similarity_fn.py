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
Implements MoL (Mixture-of-Logits) with load balancing regularization loss, as discussed in:
- Revisiting Neural Retrieval on Accelerators (https://arxiv.org/abs/2306.04039, KDD'23).
- Retrieval with Learned Similarities (https://arxiv.org/abs/2407.15462).

Forked from bailuding/rails @ 664fdb9.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from generative_recommenders.research.rails.similarities.module import SimilarityModule
from generative_recommenders.research.rails.similarities.mol.embeddings_fn import (
    MoLEmbeddingsFn,
)


@torch.compile(dynamic=True)
def _softmax_dropout_combiner_fn(
    x: torch.Tensor,
    y: torch.Tensor,
    dropout_pr: float,
    eps: float,
    training: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes (_softmax_dropout_fn(x) * y).sum(-1).
    """
    x = F.softmax(x, dim=-1)
    if dropout_pr > 0.0:
        x = F.dropout(x, p=dropout_pr, training=training)
        x = x / torch.clamp(x.sum(-1, keepdims=True), min=eps)  # pyre-ignore [19]
    return x, (x * y).sum(-1)


@torch.compile
def _load_balancing_mi_loss_fn(
    gating_prs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    See Retrieval with Learned Similarities (RAILS, https://arxiv.org/abs/2407.15462) for discussions.
    """
    B, X, E = gating_prs.size()
    expert_util_prs = gating_prs.view(B * X, E).sum(0, keepdim=False) / (1.0 * B * X)
    expert_util_entropy = -(expert_util_prs * torch.log(expert_util_prs + eps)).sum()
    per_example_expert_entropy = -(gating_prs * torch.log(gating_prs + eps)).sum() / (
        1.0 * B * X
    )
    return -expert_util_entropy + per_example_expert_entropy


class SoftmaxDropoutCombiner(torch.nn.Module):
    def __init__(
        self,
        dropout_rate: float,
        eps: float,
    ) -> None:
        super().__init__()

        self._dropout_rate: float = dropout_rate
        self._eps: float = eps

    def forward(
        self,
        gating_weights: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gating_prs, combined_logits = _softmax_dropout_combiner_fn(
            x=gating_weights,
            y=x,
            dropout_pr=self._dropout_rate,
            eps=self._eps,
            training=self.training,
        )

        aux_losses = {}
        if self.training:
            aux_losses["mi_loss"] = _load_balancing_mi_loss_fn(
                gating_prs, eps=self._eps
            )

        return combined_logits, aux_losses


class MoLGatingFn(torch.nn.Module):
    """
    Implements the gating function for MoL, used to compute pi_p(q, x) for a given (p, x) pair.
    """

    def __init__(
        self,
        num_logits: int,
        query_embedding_dim: int,
        item_embedding_dim: int,
        query_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        item_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        qi_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        combination_type: str,
        normalization_fn: Callable[[int], torch.nn.Module],
    ) -> None:
        super().__init__()

        self._query_only_partial_module: Optional[torch.nn.Module] = (
            query_only_partial_fn(query_embedding_dim, num_logits)
            if query_only_partial_fn
            else None
        )
        self._item_only_partial_module: Optional[torch.nn.Module] = (
            item_only_partial_fn(item_embedding_dim, num_logits)
            if item_only_partial_fn
            else None
        )
        self._qi_partial_module: Optional[torch.nn.Module] = (
            qi_partial_fn(
                num_logits,
                num_logits,
            )
            if qi_partial_fn is not None
            else None
        )
        if (
            self._query_only_partial_module is None
            and self._item_only_partial_module is None
            and self._qi_partial_module is None
        ):
            raise ValueError(
                "At least one of query_only_partial_fn, item_only_partial_fn, "
                "and qi_partial_fn must not be None."
            )
        self._num_logits: int = num_logits
        self._combination_type: str = combination_type
        self._normalization_fn: torch.nn.Module = normalization_fn(num_logits)

    def forward(
        self,
        logits: torch.Tensor,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            logits: (B, X, P_Q * P_X) x float;
            query_embeddings: (B, D) x float;
            item_embeddings: (1/B, X, D') x float;

        Returns:
            (B, X) x float, Dict[str, Tensor] representing auxiliary losses.
        """
        B, X, _ = logits.size()
        # [B, 1, F], [1/B, X, F], [B, X, F]
        query_partial_inputs, item_partial_inputs, qi_partial_inputs = None, None, None
        if self._query_only_partial_module is not None:
            query_partial_inputs = self._query_only_partial_module(
                query_embeddings
            ).unsqueeze(1)
        if self._item_only_partial_module is not None:
            item_partial_inputs = self._item_only_partial_module(item_embeddings)
        if self._qi_partial_module is not None:
            qi_partial_inputs = self._qi_partial_module(logits)

        if self._combination_type == "glu_silu":
            gating_inputs = (
                query_partial_inputs * item_partial_inputs + qi_partial_inputs
            )
            gating_weights = gating_inputs * F.sigmoid(gating_inputs)
        elif self._combination_type == "glu_silu_ln":
            gating_inputs = (
                query_partial_inputs * item_partial_inputs + qi_partial_inputs
            )
            gating_weights = gating_inputs * F.sigmoid(
                F.layer_norm(gating_inputs, normalized_shape=[self._num_logits])
            )
        elif self._combination_type == "none":
            gating_inputs = query_partial_inputs
            if gating_inputs is None:
                gating_inputs = item_partial_inputs
            elif item_partial_inputs is not None:
                gating_inputs += item_partial_inputs
            if gating_inputs is None:
                gating_inputs = qi_partial_inputs
            elif qi_partial_inputs is not None:
                gating_inputs += qi_partial_inputs
            gating_weights = gating_inputs
        else:
            raise ValueError(f"Unknown combination_type {self._combination_type}")

        return self._normalization_fn(gating_weights, logits)


class MoLSimilarity(SimilarityModule):
    def __init__(
        self,
        query_embedding_dim: int,
        item_embedding_dim: int,
        dot_product_dimension: int,
        query_dot_product_groups: int,
        item_dot_product_groups: int,
        temperature: float,
        dot_product_l2_norm: bool,
        query_embeddings_fn: MoLEmbeddingsFn,
        item_embeddings_fn: Optional[MoLEmbeddingsFn],
        gating_query_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        gating_item_only_partial_fn: Optional[Callable[[int, int], torch.nn.Module]],
        gating_qi_partial_fn: Optional[Callable[[int], torch.nn.Module]],
        gating_combination_type: str,
        gating_normalization_fn: Callable[[int], torch.nn.Module],
        eps: float,
        apply_query_embeddings_fn: bool = True,
        apply_item_embeddings_fn: bool = True,
        autocast_bf16: bool = False,
    ) -> None:
        """
        Args:
            apply_query_embeddings_fn: bool. If true, compute query_embeddings_fn
                to input during forward(). Otherwise, we assume the caller will
                invoke get_query_component_embeddings() separately before
                calling forward().
            apply_item_embeddings_fn: bool. If true, compute item_embeddings_fn
                to input during forward(). Otherwise, we assume the caller will
                invoke get_item_component_embeddings() separately before
                calling forward().
        """
        super().__init__()

        self._gating_fn: MoLGatingFn = MoLGatingFn(
            num_logits=query_dot_product_groups * item_dot_product_groups,
            query_embedding_dim=query_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            query_only_partial_fn=gating_query_only_partial_fn,
            item_only_partial_fn=gating_item_only_partial_fn,
            qi_partial_fn=gating_qi_partial_fn,  # pyre-ignore [6]
            combination_type=gating_combination_type,
            normalization_fn=gating_normalization_fn,
        )
        self._query_embeddings_fn: MoLEmbeddingsFn = query_embeddings_fn
        self._item_embeddings_fn: MoLEmbeddingsFn = (  # pyre-ignore [8]
            item_embeddings_fn
        )
        self._apply_query_embeddings_fn: bool = apply_query_embeddings_fn
        self._apply_item_embeddings_fn: bool = apply_item_embeddings_fn
        self._dot_product_l2_norm: bool = dot_product_l2_norm
        self._query_dot_product_groups: int = query_dot_product_groups
        self._item_dot_product_groups: int = item_dot_product_groups
        self._dot_product_dimension: int = dot_product_dimension
        self._temperature: float = temperature
        self._eps: float = eps
        self._autocast_bf16: bool = autocast_bf16

    def get_query_component_embeddings(
        self,
        input_embeddings: torch.Tensor,
        decoupled_inference: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (B, self._input_embedding_dim,) x float
                or (B, P_Q, self._dot_product_dimension) x float.
            decoupled_inference: bool. If true, the call represents an attempt to run
                forward() in decoupled mode at inference time (e.g., to pre-compute
                component-level query embeddings for filtering, etc.). We simulate
                the logic in forward() in this case (e.g., if forward() doesn't apply
                query_embeddings_fn, then this call won't either).
            kwargs: additional implementation-specific arguments.

        Returns:
            (B, query_dot_product_groups, dot_product_embedding_dim) x float.
        """
        if decoupled_inference and not self._apply_query_embeddings_fn:
            return input_embeddings, {}
        return self._query_embeddings_fn(input_embeddings, **kwargs)

    def get_item_component_embeddings(
        self,
        input_embeddings: torch.Tensor,
        decoupled_inference: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            input_embeddings: (..., self._input_embedding_dim,) x float
                or (..., P_X, self._dot_product_dimension) x float.
            decoupled_inference: bool. If true, the call represents an attempt to run
                forward() in decoupled mode at inference time (e.g., to pre-compute
                component-level item embeddings for filtering, etc.). We simulate
                the logic in forward() in this case (e.g., if forward() doesn't apply
                item_embeddings_fn, then this call won't either).
            kwargs: additional implementation-specific arguments.

        Returns:
            (..., item_dot_product_groups, dot_product_embedding_dim) x float.
        """
        if decoupled_inference and not self._apply_item_embeddings_fn:
            return input_embeddings, {}

        return self._item_embeddings_fn(input_embeddings, **kwargs)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            query_embeddings: (B, self._input_embedding_dim) x float or
                (B, P_Q, self._dot_product_dimension) x float (when query_embeddings_fn
                is applied externally).
            item_embeddings: (1/B, X, self._item_embedding_dim) x float or
                (1/B, X, P_X, self._dot_product_dimension) x float (when item_embeddings_fn
                is applied externally).
            kwargs: additional implementation-specific arguments.

        Returns:
            (B, X) x float, Dict[str, Tensor] representing auxiliary losses.
        """
        with torch.autocast(
            enabled=self._autocast_bf16, dtype=torch.bfloat16, device_type="cuda"
        ):
            B = query_embeddings.size(0)
            B_prime = item_embeddings.shape[0]  # 1 or B
            X = item_embeddings.shape[1]

            if self._apply_query_embeddings_fn:
                (
                    split_query_embeddings,
                    query_aux_losses,
                ) = self.get_query_component_embeddings(
                    query_embeddings,
                    **kwargs,
                )
            else:
                split_query_embeddings, query_aux_losses = query_embeddings, {}

            if self._apply_item_embeddings_fn:
                (
                    split_item_embeddings,
                    item_aux_losses,
                ) = self.get_item_component_embeddings(
                    input_embeddings=item_embeddings,
                    **kwargs,
                )
            else:
                split_item_embeddings, item_aux_losses = item_embeddings, {}

            if B_prime == 1:
                logits = torch.einsum(
                    "bnd,xmd->bxnm",
                    split_query_embeddings,
                    split_item_embeddings.squeeze(0),
                ).reshape(
                    B, X, self._query_dot_product_groups * self._item_dot_product_groups
                )
            else:
                logits = torch.einsum(
                    "bnd,bxmd->bxnm", split_query_embeddings, split_item_embeddings
                ).reshape(
                    B, X, self._query_dot_product_groups * self._item_dot_product_groups
                )

            gated_outputs, gating_aux_losses = self._gating_fn(
                logits=logits / self._temperature,  # [B, X, L]
                query_embeddings=query_embeddings,  # [B, D]
                item_embeddings=item_embeddings,  # [1/B, X, D']
            )
            return gated_outputs, {
                **gating_aux_losses,
                **query_aux_losses,
                **item_aux_losses,
            }
