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

# pyre-strict

"""
TorchScript-friendly wrapper for the HSTU sparse path (CPU embedding lookup).

``HSTUSparseScriptModule`` wraps :class:`HSTUSparseInferenceModule` and
flattens the ``Dict[str, SequenceEmbedding]`` output into the parallel
value/length dicts defined in :mod:`ts_types` so the boundary is composed
entirely of TorchScript-supported types.
"""

from typing import Dict, Tuple

import torch
from generative_recommenders.dlrm_v3.inference.inference_modules import (
    _NoCopyEmbeddingCollection,
    HSTUSparseInferenceModule,
)
from generative_recommenders.dlrm_v3.inference.ts_types import (
    flatten_seq_embeddings,
    SeqEmbLengths,
    SeqEmbValues,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class HSTUSparseScriptModule(torch.nn.Module):
    """Script-friendly sparse module.

    ``forward`` returns 5 tensors / dicts (no Python ``int`` scalars):

    1. ``seq_emb_values``   ``Dict[str, Tensor]`` -- jagged embedding values.
    2. ``seq_emb_lengths``  ``Dict[str, Tensor]`` -- per-feature lengths.
    3. ``payload_features`` ``Dict[str, Tensor]`` -- side features.
    4. ``uih_seq_lengths``  ``Tensor[B]``        -- UIH lengths.
    5. ``num_candidates``   ``Tensor[B]``        -- candidate counts.

    The dense module (or the C++ glue) recovers the ``int`` ``max_uih_len`` /
    ``max_num_candidates`` values from these tensors via ``.max().item()``.
    """

    def __init__(
        self,
        table_config: Dict[str, EmbeddingConfig],
        hstu_config: DlrmHSTUConfig,
        use_no_copy_embedding_collection: bool = True,
    ) -> None:
        super().__init__()
        self._sparse: HSTUSparseInferenceModule = HSTUSparseInferenceModule(
            table_config=table_config,
            hstu_config=hstu_config,
        )
        if use_no_copy_embedding_collection:
            # Re-class the existing EmbeddingCollection so TorchScript picks up
            # the no-copy ``forward`` override (matches the eager-only
            # ``ec_patched_forward_wo_embedding_copy`` monkey-patch).
            self._sparse._hstu_model._embedding_collection.__class__ = (
                _NoCopyEmbeddingCollection
            )

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        SeqEmbValues,
        SeqEmbLengths,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            seq_embeddings,
            payload_features,
            _max_uih_len,
            uih_seq_lengths,
            _max_num_candidates,
            num_candidates,
        ) = self._sparse(
            uih_features=uih_features,
            candidates_features=candidates_features,
        )
        seq_emb_values, seq_emb_lengths = flatten_seq_embeddings(seq_embeddings)
        return (
            seq_emb_values,
            seq_emb_lengths,
            payload_features,
            uih_seq_lengths,
            num_candidates,
        )
