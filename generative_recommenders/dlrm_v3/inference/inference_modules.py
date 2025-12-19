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
Inference modules for DLRMv3.

This module provides inference-specific components for the HSTU model,
including sparse inference modules and utilities for moving tensors between devices.
"""

from typing import Dict, Optional, Tuple

import torch
from generative_recommenders.modules.dlrm_hstu import (
    DlrmHSTU,
    DlrmHSTUConfig,
    SequenceEmbedding,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


IS_INFERENCE: bool = True


def set_is_inference(is_inference: bool = False) -> None:
    """
    Set the global inference mode flag.

    Args:
        is_inference: If True, model operates in inference mode (no labels/weights).
                     If False, model operates in training/eval mode with labels.
    """
    global IS_INFERENCE
    IS_INFERENCE = is_inference


def get_hstu_model(
    table_config,
    hstu_config: DlrmHSTUConfig,
    table_device: str = "meta",
    max_hash_size: Optional[int] = None,
    is_dense: bool = False,
) -> DlrmHSTU:
    """
    Create and initialize an HSTU model for inference.

    Args:
        table_config: Dictionary of embedding table configurations.
        hstu_config: HSTU model configuration object.
        table_device: Device to place embedding tables on ('meta', 'cpu', or 'cuda').
        max_hash_size: Optional maximum hash size to cap embedding table sizes.
        is_dense: If True, creates model for dense-only operations.

    Returns:
        Initialized DlrmHSTU model in eval mode.
    """
    if max_hash_size is not None:
        for t in table_config.values():
            t.num_embeddings = (
                max_hash_size if t.num_embeddings > max_hash_size else t.num_embeddings
            )
    model = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=IS_INFERENCE,
        is_dense=is_dense,
    )
    model.eval()
    model.recursive_setattr("_use_triton_cc", False)
    for _, module in model.named_modules():
        if isinstance(module, EmbeddingBagCollection) or isinstance(
            module, EmbeddingCollection
        ):
            module.to_empty(device=table_device)
    return model


class HSTUSparseInferenceModule(torch.nn.Module):
    """
    Module for sparse (embedding) inference operations.

    Handles embedding lookups and preprocessing for the HSTU model,
    running on CPU to handle large embedding tables.

    Args:
        table_config: Dictionary of embedding table configurations.
        hstu_config: HSTU model configuration object.
    """

    def __init__(
        self,
        table_config,
        hstu_config: DlrmHSTUConfig,
    ) -> None:
        super().__init__()
        self._hstu_model: DlrmHSTU = get_hstu_model(
            table_config,
            hstu_config,
            table_device="cpu",
        )

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        """
        Run sparse preprocessing and embedding lookups.

        Args:
            uih_features: User interaction history features as KeyedJaggedTensor.
            candidates_features: Candidate item features as KeyedJaggedTensor.

        Returns:
            Tuple containing:
                - seq_embeddings: Dictionary of sequence embeddings per feature.
                - payload_features: Dictionary of payload feature tensors.
                - max_uih_len: Maximum user interaction history length.
                - uih_seq_lengths: Tensor of UIH sequence lengths per batch item.
                - max_num_candidates: Maximum number of candidates.
                - num_candidates: Tensor of candidate counts per batch item.
        """
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = self._hstu_model.preprocess(
            uih_features=uih_features,
            candidates_features=candidates_features,
        )
        return (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )


def move_sparse_output_to_device(
    seq_embeddings: Dict[str, SequenceEmbedding],
    payload_features: Dict[str, torch.Tensor],
    uih_seq_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    device: torch.device,
) -> Tuple[
    Dict[str, SequenceEmbedding],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """
    Move sparse module outputs from CPU to the target device (typically GPU).

    Converts embeddings to bfloat16 for efficient GPU computation.

    Args:
        seq_embeddings: Dictionary of sequence embeddings to move.
        payload_features: Dictionary of payload features to move.
        uih_seq_lengths: UIH sequence lengths tensor to move.
        num_candidates: Number of candidates tensor to move.
        device: Target device (e.g., torch.device('cuda:0')).

    Returns:
        Tuple of moved tensors on the target device.
    """
    num_candidates = num_candidates.to(device)
    uih_seq_lengths = uih_seq_lengths.to(device)
    seq_embeddings = {
        k: SequenceEmbedding(
            lengths=seq_embeddings[k].lengths.to(device),
            embedding=seq_embeddings[k].embedding.to(device).to(torch.bfloat16),
        )
        for k in seq_embeddings.keys()
    }
    for k, v in payload_features.items():
        payload_features[k] = v.to(device)
    return seq_embeddings, payload_features, uih_seq_lengths, num_candidates
