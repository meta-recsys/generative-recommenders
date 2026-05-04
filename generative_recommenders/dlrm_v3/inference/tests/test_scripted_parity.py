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
Numerical parity test: eager HSTU vs scripted (sparse + dense) on a
synthetic batch.

Tolerances are deliberately loose because the scripted path replaces the
Triton fused kernels with PyTorch fallbacks and skips ``torch.autocast`` in
the user-forward block; both can perturb low-order bits in bf16.
"""

import unittest
from typing import Dict, Tuple

import torch
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import get_random_data
from generative_recommenders.dlrm_v3.inference.dense_predict_module import (
    HSTUDenseScriptModule,
)
from generative_recommenders.dlrm_v3.inference.sparse_predict_module import (
    HSTUSparseScriptModule,
)


_DATASET = "kuairand-1k"


def _move_dense_inputs(
    seq_emb_values: Dict[str, torch.Tensor],
    seq_emb_lengths: Dict[str, torch.Tensor],
    payload_features: Dict[str, torch.Tensor],
    uih_seq_lengths: torch.Tensor,
    num_candidates: torch.Tensor,
    device: torch.device,
) -> Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """C++-side ``move_sparse_output_to_device`` analog for the test."""
    return (
        {k: v.to(device).to(torch.bfloat16) for k, v in seq_emb_values.items()},
        {k: v.to(device) for k, v in seq_emb_lengths.items()},
        {k: v.to(device) for k, v in payload_features.items()},
        uih_seq_lengths.to(device),
        num_candidates.to(device),
    )


class HSTUScriptedParityTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_scripted_matches_eager(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        hstu_config = get_hstu_configs(_DATASET)
        table_config = get_embedding_table_config(_DATASET)

        uih_kjt, candidates_kjt = get_random_data(
            contexual_features=list(
                hstu_config.contextual_feature_to_max_length.keys()
            ),
            hstu_uih_keys=hstu_config.hstu_uih_feature_names,
            hstu_candidates_keys=hstu_config.hstu_candidate_feature_names,
            uih_max_seq_len=128,
            max_num_candidates=hstu_config.max_num_candidates_inference,
        )

        sparse_module = HSTUSparseScriptModule(
            table_config=table_config,
            hstu_config=hstu_config,
            use_no_copy_embedding_collection=True,
        ).eval()
        dense_module = (
            HSTUDenseScriptModule(
                hstu_config=hstu_config,
                table_config=table_config,
            )
            .to(torch.bfloat16)
            .to(device)
            .eval()
        )

        # === Eager path ===
        with torch.no_grad():
            sparse_out_e = sparse_module(
                uih_features=uih_kjt, candidates_features=candidates_kjt
            )
            dense_inputs_e = _move_dense_inputs(*sparse_out_e, device=device)
            preds_eager, _, _ = dense_module(*dense_inputs_e)

        # === Scripted path (separately scripted sparse + dense) ===
        scripted_sparse = torch.jit.script(sparse_module)
        scripted_dense = torch.jit.script(dense_module)

        with torch.no_grad():
            sparse_out_s = scripted_sparse(
                uih_features=uih_kjt, candidates_features=candidates_kjt
            )
            dense_inputs_s = _move_dense_inputs(*sparse_out_s, device=device)
            preds_scripted, _, _ = scripted_dense(*dense_inputs_s)

        torch.testing.assert_close(
            preds_eager.float(),
            preds_scripted.float(),
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
