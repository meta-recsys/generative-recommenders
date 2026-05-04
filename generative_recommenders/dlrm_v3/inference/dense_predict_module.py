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
TorchScript-friendly wrapper for the HSTU dense path (GPU transformer).

``HSTUDenseScriptModule`` accepts the *flattened* sparse-output dicts produced
by :class:`HSTUSparseScriptModule`, reconstructs ``Dict[str,
SequenceEmbedding]`` for the existing :meth:`DlrmHSTU.main_forward` and
returns a 3-tuple of ``(preds, labels, weights)`` -- the only fields the
predictor actually consumes.
"""

from typing import Dict

import torch
from generative_recommenders.dlrm_v3.inference.inference_modules import get_hstu_model
from generative_recommenders.dlrm_v3.inference.ts_types import (
    SeqEmbLengths,
    SeqEmbValues,
    unflatten_seq_embeddings,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig


class HSTUDenseScriptModule(torch.nn.Module):
    """Script-friendly dense module.

    The wrapper owns a dense-only :class:`DlrmHSTU` (no
    ``_embedding_collection``) and delegates to ``main_forward`` after
    reconstructing the ``SequenceEmbedding`` NamedTuple form.
    """

    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
    ) -> None:
        super().__init__()
        self._hstu_model: DlrmHSTU = get_hstu_model(
            table_config=table_config,
            hstu_config=hstu_config,
            table_device="cpu",
            is_dense=True,
        )

    def forward(
        self,
        seq_emb_values: SeqEmbValues,
        seq_emb_lengths: SeqEmbLengths,
        payload_features: Dict[str, torch.Tensor],
        uih_seq_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        # TorchScript supports ``int(tensor.item())`` on a 0-d tensor.
        max_uih_len: int = int(uih_seq_lengths.max().item())
        max_num_candidates: int = int(num_candidates.max().item())

        seq_embeddings = unflatten_seq_embeddings(seq_emb_values, seq_emb_lengths)

        (
            _,
            _,
            _,
            mt_target_preds,
            _mt_target_labels,
            _mt_target_weights,
        ) = self._hstu_model.main_forward(
            seq_embeddings=seq_embeddings,
            payload_features=payload_features,
            max_uih_len=max_uih_len,
            uih_seq_lengths=uih_seq_lengths,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates,
        )
        assert mt_target_preds is not None
        # Return just the predictions tensor; labels/weights are unused by
        # the predictor at inference time and would force ``Optional[Tensor]``
        # in the return type, which torch.jit.trace rejects ("Only tensors,
        # lists, tuples of tensors, or dictionary of tensors can be output
        # from traced functions").
        return mt_target_preds
