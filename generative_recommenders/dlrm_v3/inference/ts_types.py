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
TorchScript-friendly boundary types for the HSTU sparse <-> dense interface.

The eager path uses ``Dict[str, SequenceEmbedding]`` (a NamedTuple of
``lengths`` and ``embedding`` tensors). TorchScript supports ``NamedTuple`` but
does not script cleanly through ``Dict[str, NamedTuple]`` once the dict crosses
device boundaries. The packaged sparse / dense modules instead exchange two
parallel ``Dict[str, Tensor]`` dicts -- one of jagged values, one of lengths.

These helpers convert between the two representations so we can keep the
existing eager code unchanged while the scripted modules use only TS-friendly
types at their boundaries.
"""

from typing import Dict, Tuple

import torch
from generative_recommenders.modules.dlrm_hstu import SequenceEmbedding


# Per-feature jagged values (concatenated across batch, [L_total, table_dim]).
SeqEmbValues = Dict[str, torch.Tensor]
# Per-feature per-batch lengths ([B]).
SeqEmbLengths = Dict[str, torch.Tensor]


def flatten_seq_embeddings(
    seq_embeddings: Dict[str, SequenceEmbedding],
) -> Tuple[SeqEmbValues, SeqEmbLengths]:
    """Split ``Dict[str, SequenceEmbedding]`` into parallel value/length dicts.

    Lossless and zero-copy -- the returned tensors alias the inputs.
    """
    values: Dict[str, torch.Tensor] = {}
    lengths: Dict[str, torch.Tensor] = {}
    for k, v in seq_embeddings.items():
        values[k] = v.embedding
        lengths[k] = v.lengths
    return values, lengths


def unflatten_seq_embeddings(
    values: SeqEmbValues,
    lengths: SeqEmbLengths,
) -> Dict[str, SequenceEmbedding]:
    """Inverse of :func:`flatten_seq_embeddings`.

    Reconstructs ``Dict[str, SequenceEmbedding]`` for code paths (e.g.
    ``DlrmHSTU.main_forward``) that still consume the NamedTuple form.
    """
    out: Dict[str, SequenceEmbedding] = {}
    for k, val in values.items():
        out[k] = SequenceEmbedding(lengths=lengths[k], embedding=val)
    return out
