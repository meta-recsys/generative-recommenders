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

import logging
from typing import Dict, Optional, Tuple

import torch
from generative_recommenders.common import fx_unwrap_optional_tensor, HammerModule
from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder
from generative_recommenders.modules.postprocessors import (
    L2NormPostprocessor,
    OutputPostprocessor,
)
from generative_recommenders.modules.preprocessors import InputPreprocessor
from generative_recommenders.modules.stu import STU
from generative_recommenders.ops.jagged_tensors import split_2D_jagged
from torch.profiler import record_function

logger: logging.Logger = logging.getLogger(__name__)
torch.fx.wrap("len")

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def default_seq_payload(
    seq_payloads: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if seq_payloads is None:
        return {}
    else:
        return torch.jit._unwrap_optional(seq_payloads)


class HSTUTransducer(HammerModule):
    def __init__(
        self,
        stu_module: STU,
        input_preprocessor: InputPreprocessor,
        output_postprocessor: Optional[OutputPostprocessor] = None,
        input_dropout_ratio: float = 0.0,
        positional_encoder: Optional[HSTUPositionalEncoder] = None,
        is_inference: bool = True,
        return_full_embeddings: bool = False,
        listwise: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._stu_module = stu_module
        self._input_preprocessor: InputPreprocessor = input_preprocessor
        self._output_postprocessor: OutputPostprocessor = (
            output_postprocessor
            if output_postprocessor is not None
            else L2NormPostprocessor(is_inference=is_inference)
        )
        assert self._is_inference == self._input_preprocessor._is_inference, (
            f"input_preprocessor must have the same mode; self: {self._is_inference} vs input_preprocessor {self._input_preprocessor._is_inference}"
        )
        self._positional_encoder: Optional[HSTUPositionalEncoder] = positional_encoder
        self._input_dropout_ratio: float = input_dropout_ratio
        self._return_full_embeddings: bool = return_full_embeddings
        self._listwise_training: bool = listwise and self.is_train

        for name, m in self.named_modules():
            if "_stu_module" in name:
                continue
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, torch.nn.LayerNorm):
                if m.weight.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None and m.bias.dim() >= 2:
                    torch.nn.init.xavier_normal_(m.bias)

    def _preprocess(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        seq_payloads = default_seq_payload(seq_payloads)

        with record_function("hstu_input_preprocessor"):
            (
                output_max_seq_len,
                output_total_uih_len,
                output_total_targets,
                output_seq_lengths,
                output_seq_offsets,
                output_seq_timestamps,
                output_seq_embeddings,
                output_num_targets,
                output_seq_payloads,
            ) = self._input_preprocessor(
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_lengths=seq_lengths,
                seq_timestamps=seq_timestamps,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                seq_payloads=seq_payloads,
            )

        with record_function("hstu_positional_encoder"):
            if self._positional_encoder is not None:
                output_seq_embeddings = self._positional_encoder(
                    max_seq_len=output_max_seq_len,
                    seq_lengths=output_seq_lengths,
                    seq_offsets=output_seq_offsets,
                    seq_timestamps=output_seq_timestamps,
                    seq_embeddings=output_seq_embeddings,
                    num_targets=(
                        None if self._listwise_training else output_num_targets
                    ),
                )

        output_seq_embeddings = torch.nn.functional.dropout(
            output_seq_embeddings,
            p=self._input_dropout_ratio,
            training=self.training,
        )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            output_seq_payloads,
        )

    def _hstu_compute(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        with record_function("hstu"):
            seq_embeddings = self._stu_module(
                max_seq_len=max_seq_len,
                x=seq_embeddings,
                x_lengths=seq_lengths,
                x_offsets=seq_offsets,
                num_targets=(None if self._listwise_training else num_targets),
            )
        return seq_embeddings

    def _postprocess(
        self,
        max_seq_len: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        with record_function("hstu_output_postprocessor"):
            if self._return_full_embeddings:
                seq_embeddings = self._output_postprocessor(
                    seq_embeddings=seq_embeddings,
                    seq_timestamps=seq_timestamps,
                    seq_payloads=seq_payloads,
                )
            uih_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                seq_lengths - num_targets
            )
            candidates_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                num_targets
            )
            _, candidate_embeddings = split_2D_jagged(
                values=seq_embeddings,
                max_seq_len=max_seq_len,
                total_len_left=total_uih_len,
                total_len_right=total_targets,
                offsets_left=uih_offsets,
                offsets_right=candidates_offsets,
                kernel=self.hammer_kernel(),
            )
            interleave_targets: bool = self._input_preprocessor.interleave_targets()
            if interleave_targets:
                candidate_embeddings = candidate_embeddings.view(
                    -1, 2, candidate_embeddings.size(-1)
                )[:, 0, :]
            if not self._return_full_embeddings:
                _, candidate_timestamps = split_2D_jagged(
                    values=seq_timestamps.unsqueeze(-1),
                    max_seq_len=max_seq_len,
                    total_len_left=total_uih_len,
                    total_len_right=total_targets,
                    offsets_left=uih_offsets,
                    offsets_right=candidates_offsets,
                    kernel=self.hammer_kernel(),
                )
                candidate_timestamps = candidate_timestamps.squeeze(-1)
                if interleave_targets:
                    candidate_timestamps = candidate_timestamps.view(-1, 2)[:, 0]
                candidate_embeddings = self._output_postprocessor(
                    seq_embeddings=candidate_embeddings,
                    seq_timestamps=candidate_timestamps,
                    seq_payloads=seq_payloads,
                )

            return (
                seq_embeddings if self._return_full_embeddings else None,
                candidate_embeddings,
            )

    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        orig_dtype = seq_embeddings.dtype
        if not self._is_inference:
            seq_embeddings = seq_embeddings.to(self._training_dtype)

        (
            max_seq_len,
            total_uih_len,
            total_targets,
            seq_lengths,
            seq_offsets,
            seq_timestamps,
            seq_embeddings,
            num_targets,
            seq_payloads,
        ) = self._preprocess(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        encoded_embeddings = self._hstu_compute(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
        )

        encoded_embeddings, encoded_candidate_embeddings = self._postprocess(
            max_seq_len=max_seq_len,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_embeddings=encoded_embeddings,
            seq_timestamps=seq_timestamps,
            num_targets=num_targets,
            seq_payloads=seq_payloads,
        )

        if not self._is_inference:
            encoded_candidate_embeddings = encoded_candidate_embeddings.to(orig_dtype)
            if self._return_full_embeddings:
                encoded_embeddings = fx_unwrap_optional_tensor(encoded_embeddings).to(
                    orig_dtype
                )
        return (
            encoded_candidate_embeddings,
            encoded_embeddings,
        )
