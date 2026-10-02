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
import unittest

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import get_random_data
from generative_recommenders.dlrm_v3.inference.inference_modules import (
    get_hstu_model,
    set_is_inference,
)
from generative_recommenders.dlrm_v3.train.train_ranker import main


class DLRMV3TrainTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_e2e(self) -> None:
        main()

    @unittest.skipIf(*gpu_unavailable)
    def test_hstu_ultra_optimizer_step_and_eager_inference(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        self.addCleanup(set_is_inference, False)
        hstu_config = get_hstu_configs(
            "kuairand-1k",
            hstu_ultra_stack_name="hstu_ultra",
        )
        hstu_config.use_layer_norm_postprocessor = True
        table_config = get_embedding_table_config("kuairand-1k")
        min_rows = min(table.num_embeddings for table in table_config.values())
        uih_features, candidate_features = get_random_data(
            contexual_features=list(
                hstu_config.contextual_feature_to_max_length.keys()
            ),
            hstu_uih_keys=hstu_config.hstu_uih_feature_names,
            hstu_candidates_keys=hstu_config.hstu_candidate_feature_names,
            uih_max_seq_len=8,
            max_num_candidates=2,
            value_bound=max(2, min_rows),
        )
        uih_features = uih_features.to(device)
        candidate_features = candidate_features.to(device)

        set_is_inference(False)
        model = (
            get_hstu_model(
                table_config=table_config,
                hstu_config=hstu_config,
                table_device="cuda",
                max_hash_size=100,
            )
            .to(device=device)
            .train()
        )
        # The HSTU Ultra module tests cover the Triton kernels directly. Keep
        # this integration test on the PyTorch reference path so it focuses on
        # DLRM data flow, loss/backward, checkpoint state, and eager inference.
        model.set_hammer_kernel(HammerKernel.PYTORCH)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, foreach=True)
        optimizer.zero_grad()
        _, _, losses, predictions, labels, weights = model(
            uih_features, candidate_features
        )
        loss = torch.stack(list(losses.values())).sum()
        loss.backward()

        self.assertEqual(
            set(losses), {task.task_name for task in hstu_config.multitask_configs}
        )
        self.assertIsNotNone(predictions)
        self.assertIsNotNone(labels)
        self.assertIsNotNone(weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            )
        )
        optimizer.step()

        set_is_inference(True)
        inference_model = (
            get_hstu_model(
                table_config=table_config,
                hstu_config=hstu_config,
                table_device="cuda",
                max_hash_size=100,
            )
            .to(device=device)
            .eval()
        )
        inference_model.set_hammer_kernel(HammerKernel.PYTORCH)
        inference_model.load_state_dict(model.state_dict())
        with torch.no_grad():
            _, _, inference_losses, inference_predictions, _, _ = inference_model(
                uih_features, candidate_features
            )

        self.assertEqual(inference_losses, {})
        self.assertIsNotNone(inference_predictions)
        assert inference_predictions is not None
        self.assertTrue(torch.isfinite(inference_predictions).all().item())
