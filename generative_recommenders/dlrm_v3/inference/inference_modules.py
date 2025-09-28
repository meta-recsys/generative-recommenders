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
    global IS_INFERENCE
    IS_INFERENCE = is_inference


def get_hstu_model(
    table_config,
    hstu_config: DlrmHSTUConfig,
    table_device: str = "meta",
    max_hash_size: Optional[int] = None,
    is_dense: bool = False,
) -> DlrmHSTU:
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
