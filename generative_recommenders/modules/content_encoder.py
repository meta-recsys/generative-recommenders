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

#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Optional

import torch
from generative_recommenders.common import HammerModule
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged


class ContentEncoder(HammerModule):
    def __init__(
        self,
        input_embedding_dim: int,
        additional_content_features: Optional[Dict[str, int]] = None,
        target_enrich_features: Optional[Dict[str, int]] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._input_embedding_dim: int = input_embedding_dim
        self._additional_content_features: Dict[str, int] = (
            additional_content_features
            if additional_content_features is not None
            else {}
        )
        self._target_enrich_features: Dict[str, int] = (
            target_enrich_features if target_enrich_features is not None else {}
        )
        self._target_enrich_dummy_embeddings: torch.nn.ParameterDict = (
            torch.nn.ParameterDict(
                {
                    name: torch.nn.Parameter(
                        torch.empty((1, dim)).normal_(mean=0, std=0.1),
                    )
                    for name, dim in self._target_enrich_features.items()
                }
            )
        )

    @property
    def output_embedding_dim(self) -> int:
        return self._input_embedding_dim + sum(
            list(self._additional_content_features.values())
            + list(self._target_enrich_features.values())
        )

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        uih_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        content_embeddings_list: List[torch.Tensor] = [seq_embeddings]
        if len(self._additional_content_features) > 0:
            content_embeddings_list = content_embeddings_list + [
                (seq_payloads[x].to(seq_embeddings.dtype))
                for x in self._additional_content_features.keys()
            ]

        if self._target_enrich_dummy_embeddings:
            total_seq_len: int = seq_embeddings.size(0)
            for name, param in self._target_enrich_dummy_embeddings.items():
                enrich_embeddings_target = seq_payloads[name].to(seq_embeddings.dtype)
                total_targets: int = enrich_embeddings_target.size(0)
                total_uih_len: int = total_seq_len - total_targets
                enrich_embeddings_uih = param.tile(total_uih_len, 1).to(
                    seq_embeddings.dtype
                )
                enrich_embeddings = concat_2D_jagged(
                    max_seq_len=max_uih_len + max_targets,
                    values_left=enrich_embeddings_uih,
                    values_right=enrich_embeddings_target,
                    max_len_left=max_uih_len,
                    max_len_right=max_targets,
                    offsets_left=uih_offsets,
                    offsets_right=target_offsets,
                    kernel=self.hammer_kernel(),
                )
                content_embeddings_list.append(enrich_embeddings)

        if (
            len(self._target_enrich_features) == 0
            and len(self._additional_content_features) == 0
        ):
            return seq_embeddings
        else:
            content_embeddings = torch.cat(
                content_embeddings_list,
                dim=1,
            )
            return content_embeddings
