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

import unittest
from typing import Dict, List, Tuple

import torch
from generative_recommenders.common import gpu_unavailable, set_dev_mode
from generative_recommenders.modules.multitask_module import (
    DefaultMultitaskModule,
    MultitaskTaskType,
    TaskConfig,
)
from generative_recommenders.ops.layer_norm import SwishLayerNorm
from hypothesis import given, settings, strategies as st, Verbosity


_task_configs: List[List[TaskConfig]] = [
    [
        TaskConfig(
            task_name="is_click",
            task_weight=1,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
    ],
    [
        TaskConfig(
            task_name="vvp",
            task_weight=2,
            task_type=MultitaskTaskType.REGRESSION,
        ),
    ],
    [
        TaskConfig(
            task_name="is_click",
            task_weight=1,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_like",
            task_weight=2,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_follow",
            task_weight=4,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
    ],
    [
        TaskConfig(
            task_name="rating",
            task_weight=1,
            task_type=MultitaskTaskType.REGRESSION,
        ),
        TaskConfig(
            task_name="vvp",
            task_weight=2,
            task_type=MultitaskTaskType.REGRESSION,
        ),
    ],
    [
        TaskConfig(
            task_name="type_1",
            task_weight=2,
            task_type=MultitaskTaskType.REGRESSION,
        ),
    ],
    [
        TaskConfig(
            task_name="is_click",
            task_weight=1,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_like",
            task_weight=2,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_follow",
            task_weight=4,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="rating",
            task_weight=1,
            task_type=MultitaskTaskType.REGRESSION,
        ),
        TaskConfig(
            task_name="vvp",
            task_weight=2,
            task_type=MultitaskTaskType.REGRESSION,
        ),
    ],
    [
        TaskConfig(
            task_name="is_click",
            task_weight=1,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_like",
            task_weight=2,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="is_follow",
            task_weight=4,
            task_type=MultitaskTaskType.BINARY_CLASSIFICATION,
        ),
        TaskConfig(
            task_name="rating",
            task_weight=1,
            task_type=MultitaskTaskType.REGRESSION,
        ),
        TaskConfig(
            task_name="vvp",
            task_weight=2,
            task_type=MultitaskTaskType.REGRESSION,
        ),
    ],
]


def _get_random_supervision_labels_and_weights(
    num_examples: int,
    task_configs: List[TaskConfig],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    supervision_labels: Dict[str, torch.Tensor] = {}
    supervision_weights: Dict[str, torch.Tensor] = {}
    for task in task_configs:
        if task.task_type == MultitaskTaskType.REGRESSION:
            supervision_labels[task.task_name] = torch.randn(
                num_examples, device=device
            ).to(torch.float32)
        else:
            supervision_labels[task.task_name] = torch.randint(
                0,
                11,
                (num_examples,),
                device=device,
            ).to(torch.float32)

    return supervision_labels, supervision_weights


class MultiTaskModuleTest(unittest.TestCase):
    # pyre-ignore
    @given(
        task_config_idx=st.sampled_from(range(len(_task_configs))),
        training=st.booleans(),
        is_inference=st.booleans(),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=50, deadline=None)
    def test_default_multitask_module(
        self,
        task_config_idx: int,
        training: bool,
        is_inference: bool,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")

        L = 200
        embedding_dim = 64
        causal_multitask_weights = 0.3

        task_configs: List[TaskConfig] = _task_configs[task_config_idx]
        task_configs.sort(key=lambda x: x.task_type)
        multitask_module = DefaultMultitaskModule(
            task_configs=task_configs,
            embedding_dim=embedding_dim,
            prediction_fn=lambda in_dim, num_tasks: torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim, out_features=512),
                SwishLayerNorm(512),
                torch.nn.Linear(in_features=512, out_features=num_tasks),
            ),
            causal_multitask_weights=causal_multitask_weights,
            is_inference=is_inference,
        ).to(device)
        multitask_module.set_training_dtype(dtype)
        supervision_labels, supervision_weights = (
            _get_random_supervision_labels_and_weights(
                num_examples=L,
                task_configs=task_configs,
                device=device,
            )
        )
        encoded_user_embeddings = torch.rand(L, embedding_dim, device=device)
        item_embeddings = torch.rand(L, embedding_dim, device=device)

        (
            mt_preds,
            mt_labels,
            mt_weights,
            mt_losses,
        ) = multitask_module(
            encoded_user_embeddings=encoded_user_embeddings,
            item_embeddings=item_embeddings,
            supervision_labels=supervision_labels,
            supervision_weights=supervision_weights,
        )

        self.assertEqual(mt_preds.size(), (len(task_configs), L))
        if not is_inference:
            self.assertEqual(mt_labels.size(), (len(task_configs), L))
            self.assertEqual(mt_weights.size(), (len(task_configs), L))
            if training:
                self.assertEqual(mt_losses.size(), (len(task_configs),))
