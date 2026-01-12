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

import abc
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from generative_recommenders.common import HammerModule

logger: logging.Logger = logging.getLogger(__name__)


class MultitaskTaskType(IntEnum):
    BINARY_CLASSIFICATION = 0
    REGRESSION = 1


@dataclass
class TaskConfig:
    task_name: str
    task_weight: int
    task_type: MultitaskTaskType


class MultitaskModule(HammerModule):
    @abc.abstractmethod
    def forward(
        self,
        encoded_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        supervision_labels: Dict[str, torch.Tensor],
        supervision_weights: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Computes multi-task predictions.

        Args:
            encoded_user_embeddings: (L, D) x float.
            item_embeddings: (L, D) x float.
            supervision_labels: Dict[T, L] x float or int
            supervision_weights: Dict[T', L] x float or int, T' <= T
        Returns:
            (T, L) x float, predictions, labels, weights, losses
        """
        pass


def _compute_pred_and_logits(
    prediction_module: torch.nn.Module,
    encoded_user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    task_offsets: List[int],
    has_multiple_task_types: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mt_logits = prediction_module(encoded_user_embeddings * item_embeddings).transpose(
        0, 1
    )
    mt_preds_list: List[torch.Tensor] = []
    for task_type in MultitaskTaskType:
        logits = mt_logits[
            task_offsets[task_type] : task_offsets[task_type + 1],
            :,
        ]
        if task_offsets[task_type + 1] - task_offsets[task_type] > 0:
            if task_type == MultitaskTaskType.REGRESSION:
                mt_preds_list.append(logits)
            else:
                mt_preds_list.append(F.sigmoid(logits))
    if has_multiple_task_types:
        mt_preds: torch.Tensor = torch.concat(mt_preds_list, dim=0)
    else:
        mt_preds: torch.Tensor = mt_preds_list[0]

    return mt_preds, mt_logits


def _compute_labels_and_weights(
    supervision_labels: Dict[str, torch.Tensor],
    supervision_weights: Dict[str, torch.Tensor],
    task_configs: List[TaskConfig],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first_label: torch.Tensor = list(supervision_labels.values())[0]
    default_supervision_weight = torch.ones_like(
        first_label,
        dtype=dtype,
        device=device,
    )
    mt_lables_list: List[torch.Tensor] = []
    mt_weights_list: List[torch.Tensor] = []
    for task in task_configs:
        mt_lables_list.append(supervision_labels[task.task_name])
        mt_weights_list.append(
            supervision_weights.get(task.task_name, default_supervision_weight)
        )
    if len(task_configs) > 1:
        mt_labels = torch.stack(mt_lables_list, dim=0)
        mt_weights = torch.stack(mt_weights_list, dim=0)
    else:
        mt_labels = mt_lables_list[0].unsqueeze(0)
        mt_weights = mt_weights_list[0].unsqueeze(0)
    return mt_labels, mt_weights


def _compute_loss(
    task_offsets: List[int],
    causal_multitask_weights: float,
    mt_logits: torch.Tensor,
    mt_labels: torch.Tensor,
    mt_weights: torch.Tensor,
    has_multiple_task_types: bool,
) -> torch.Tensor:
    mt_losses_list: List[torch.Tensor] = []
    for task_type in MultitaskTaskType:
        if task_offsets[task_type + 1] - task_offsets[task_type] > 0:
            logits = mt_logits[
                task_offsets[task_type] : task_offsets[task_type + 1],
                :,
            ]
            labels = mt_labels[
                task_offsets[task_type] : task_offsets[task_type + 1],
                :,
            ]
            weights = mt_weights[
                task_offsets[task_type] : task_offsets[task_type + 1],
                :,
            ]
            if task_type == MultitaskTaskType.REGRESSION:
                mt_losses_list.append(
                    F.mse_loss(logits, labels, reduction="none") * weights
                )
            else:
                mt_losses_list.append(
                    F.binary_cross_entropy_with_logits(
                        input=logits, target=labels, reduction="none"
                    )
                    * weights
                )

    if has_multiple_task_types:
        mt_losses = torch.concat(mt_losses_list, dim=0)
    else:
        mt_losses = mt_losses_list[0]
    mt_losses = (
        mt_losses.sum(-1) / mt_weights.sum(-1).clamp(min=1.0) * causal_multitask_weights
    )
    return mt_losses


class DefaultMultitaskModule(MultitaskModule):
    def __init__(
        self,
        task_configs: List[TaskConfig],
        embedding_dim: int,
        prediction_fn: Callable[[int, int], torch.nn.Module],
        causal_multitask_weights: float,
        is_inference: bool,
    ) -> None:
        super().__init__(is_inference)
        assert sorted(task_configs, key=lambda x: x.task_type) == task_configs, (
            "task_configs must be sorted by task_type."
        )
        assert len(task_configs) > 0, "task_configs must be non-empty."
        self._task_configs: List[TaskConfig] = task_configs
        self._task_offsets: List[int] = [0] * (len(MultitaskTaskType) + 1)
        for task in self._task_configs:
            self._task_offsets[task.task_type + 1] += 1
        self._has_multiple_task_types: bool = self._task_offsets.count(0) < len(
            MultitaskTaskType
        )
        self._task_offsets[1:] = np.cumsum(self._task_offsets[1:]).tolist()
        self._causal_multitask_weights: float = causal_multitask_weights
        self._prediction_module: torch.nn.Module = prediction_fn(
            embedding_dim, len(task_configs)
        )

    def forward(
        self,
        encoded_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        supervision_labels: Dict[str, torch.Tensor],
        supervision_weights: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        orig_dtype = encoded_user_embeddings.dtype
        if not self._is_inference:
            encoded_user_embeddings = encoded_user_embeddings.to(self._training_dtype)
            item_embeddings = item_embeddings.to(self._training_dtype)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not self.is_inference and self._training_dtype == torch.bfloat16),
        ):
            mt_preds, mt_logits = _compute_pred_and_logits(
                prediction_module=self._prediction_module,
                encoded_user_embeddings=encoded_user_embeddings,
                item_embeddings=item_embeddings,
                task_offsets=self._task_offsets,
                has_multiple_task_types=self._has_multiple_task_types,
            )

        # losses are always computed in fp32
        mt_labels: Optional[torch.Tensor] = None
        mt_weights: Optional[torch.Tensor] = None
        mt_losses: Optional[torch.Tensor] = None
        if not self._is_inference:
            mt_labels, mt_weights = _compute_labels_and_weights(
                supervision_labels=supervision_labels,
                supervision_weights=supervision_weights,
                task_configs=self._task_configs,
                device=encoded_user_embeddings.device,
            )
            mt_losses = _compute_loss(
                task_offsets=self._task_offsets,
                causal_multitask_weights=self._causal_multitask_weights,
                mt_logits=mt_logits.to(mt_labels.dtype),
                mt_labels=mt_labels,
                mt_weights=mt_weights,
                has_multiple_task_types=self._has_multiple_task_types,
            )
            mt_preds = mt_preds.to(orig_dtype)

        return (
            mt_preds,
            mt_labels,
            mt_weights,
            mt_losses,
        )
