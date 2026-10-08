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

import dataclasses
import unittest
from typing import Tuple

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn, get_random_data
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

EMB_DIM = 64
MAX_UIH_LEN = 40
MAX_CANDIDATES = 6


def _small_model(kernel: HammerKernel) -> Tuple[DlrmHSTU, DlrmHSTUConfig]:
    cfg = get_hstu_configs("debug")
    cfg.hstu_attn_num_layers = 1
    cfg.hstu_embedding_table_dim = EMB_DIM
    cfg.hstu_input_dropout_ratio = 0.0
    cfg.hstu_linear_dropout_rate = 0.0
    tables = {
        name: dataclasses.replace(t, num_embeddings=1000, embedding_dim=EMB_DIM)
        for name, t in get_embedding_table_config("debug").items()
    }
    torch.manual_seed(0)
    model = DlrmHSTU(hstu_configs=cfg, embedding_tables=tables, is_inference=False)
    model._embedding_collection.to_empty(device="cuda")
    for p in model._embedding_collection.parameters():
        torch.nn.init.uniform_(p, -0.05, 0.05)
    model = model.cuda()
    model.set_hammer_kernel(kernel)
    return model, cfg


def _batch(cfg: DlrmHSTUConfig) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    contextual = list(cfg.contextual_feature_to_max_length)
    torch.manual_seed(1)
    samples = [
        get_random_data(
            contexual_features=contextual,
            hstu_uih_keys=cfg.hstu_uih_feature_names,
            hstu_candidates_keys=cfg.hstu_candidate_feature_names,
            uih_max_seq_len=MAX_UIH_LEN - 10,
            max_num_candidates=MAX_CANDIDATES - 2,
        )
        for _ in range(3)
    ]
    batch = collate_fn(samples)
    return (
        batch.uih_features_kjt.to(torch.device("cuda")),
        batch.candidates_features_kjt.to(torch.device("cuda")),
    )


def _with_length_per_key(kjt: KeyedJaggedTensor) -> KeyedJaggedTensor:
    return KeyedJaggedTensor(
        keys=kjt.keys(),
        values=kjt.values(),
        lengths=kjt.lengths(),
        length_per_key=kjt.length_per_key(),
    )


class DlrmHSTUStaticShapeTest(unittest.TestCase):
    def _check_static_matches_default(self, kernel: HammerKernel) -> None:
        model, cfg = _small_model(kernel)
        uih, cand = _batch(cfg)
        default = model(uih, cand)
        # Upper bounds larger than the batch's real maxima only add padding.
        static = model(
            _with_length_per_key(uih),
            _with_length_per_key(cand),
            max_uih_len=MAX_UIH_LEN,
            max_num_candidates=MAX_CANDIDATES,
        )
        for name in default[2]:
            torch.testing.assert_close(
                static[2][name], default[2][name], atol=2e-3, rtol=2e-3
            )
        for i in (3, 4, 5):
            torch.testing.assert_close(static[i], default[i], atol=2e-3, rtol=2e-3)

    @unittest.skipIf(*gpu_unavailable)
    def test_static_matches_default_pytorch(self) -> None:
        self._check_static_matches_default(HammerKernel.PYTORCH)

    @unittest.skipIf(*gpu_unavailable)
    def test_static_matches_default_triton(self) -> None:
        self._check_static_matches_default(HammerKernel.TRITON)

    @unittest.skipIf(*gpu_unavailable)
    def test_candidate_weights(self) -> None:
        model, cfg = _small_model(HammerKernel.PYTORCH)
        uih, cand = _batch(cfg)
        total = int(cand.lengths().view(len(cand.keys()), -1)[0].sum().item())
        default = model(uih, cand)
        ones = model(uih, cand, candidate_weights=torch.ones(total, device="cuda"))
        for name in default[2]:
            torch.testing.assert_close(ones[2][name], default[2][name])

        weights = torch.ones(total, device="cuda")
        weights[0] = 0.0
        weighted = model(uih, cand, candidate_weights=weights)
        # The zero-weight candidate is reported with weight 0 for every task.
        mt_weights = weighted[5]
        assert mt_weights is not None
        self.assertTrue(torch.all(mt_weights[:, 0] == 0))
        self.assertTrue(torch.all(mt_weights[:, 1:] == 1))
