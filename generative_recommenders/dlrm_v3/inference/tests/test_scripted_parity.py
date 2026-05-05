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
Numerical parity test: eager HSTU vs traced (sparse + dense) on a synthetic
batch.

The production deployment path (see ``end_to_end_test.py``) uses
``torch.jit.trace``, not ``torch.jit.script``, for the HSTU sparse/dense
wrappers. Tracing records the actual tensor ops executed during a forward
pass and ignores source-level dispatch logic (HammerKernel enum,
``is_fx_tracing()``, ``torch.autocast``, IntEnum branches) that scripting
cannot compile. This unit test mirrors that path.

Tolerances are deliberately loose because the traced path replaces the
Triton fused kernels with PyTorch fallbacks and skips ``torch.autocast`` in
the user-forward block; both can perturb low-order bits in bf16.
"""

import unittest
from typing import Dict, List, Tuple

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
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
from generative_recommenders.dlrm_v3.inference.ts_types import (
    SeqEmbLengths,
    SeqEmbValues,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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


class _SparseTraceShim(torch.nn.Module):
    """Adapter that takes raw tensors and rebuilds the KJTs inside forward.

    ``torch.jit.trace`` does not accept ``KeyedJaggedTensor`` (or any
    non-Tensor / non-collection-of-Tensor type) as a top-level forward
    input, so we make the traced boundary tensor-only and bake the
    ``List[str]`` of feature keys in as module attributes.
    """

    def __init__(
        self,
        sparse_module: HSTUSparseScriptModule,
        uih_keys: List[str],
        candidates_keys: List[str],
    ) -> None:
        super().__init__()
        self._sparse_module: HSTUSparseScriptModule = sparse_module
        self._uih_keys: List[str] = uih_keys
        self._candidates_keys: List[str] = candidates_keys

    def forward(
        self,
        uih_lengths: torch.Tensor,
        uih_values: torch.Tensor,
        candidates_lengths: torch.Tensor,
        candidates_values: torch.Tensor,
    ) -> Tuple[
        SeqEmbValues,
        SeqEmbLengths,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        uih_kjt = KeyedJaggedTensor(
            keys=self._uih_keys,
            lengths=uih_lengths,
            values=uih_values,
        )
        candidates_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=candidates_lengths,
            values=candidates_values,
        )
        return self._sparse_module(
            uih_features=uih_kjt, candidates_features=candidates_kjt
        )


class HSTUScriptedParityTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_scripted_matches_eager(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        hstu_config = get_hstu_configs(_DATASET)
        table_config = get_embedding_table_config(_DATASET)

        # Some embedding tables in kuairand-1k are tiny (e.g.
        # user_active_degree has num_embeddings=8). Clamp the random value
        # range so every index stays in range for every table; otherwise the
        # default value_bound=1000 triggers an out-of-range embedding lookup.
        min_rows = min(t.num_embeddings for t in table_config.values())
        value_bound = max(2, min_rows)

        uih_kjt, candidates_kjt = get_random_data(
            contexual_features=list(
                hstu_config.contextual_feature_to_max_length.keys()
            ),
            hstu_uih_keys=hstu_config.hstu_uih_feature_names,
            hstu_candidates_keys=hstu_config.hstu_candidate_feature_names,
            uih_max_seq_len=128,
            max_num_candidates=hstu_config.max_num_candidates_inference,
            value_bound=value_bound,
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

        # Pin the HammerKernel to PyTorch on both wrappers. The Triton
        # kernels use Python-level dispatch (autotune, constexpr arguments)
        # that interacts badly with torch.jit.trace's recording pass. The
        # eager reference run uses the same setting so the comparison is
        # apples-to-apples.
        sparse_module._sparse._hstu_model.set_hammer_kernel(HammerKernel.PYTORCH)
        dense_module._hstu_model.set_hammer_kernel(HammerKernel.PYTORCH)

        # === Eager reference path ===
        with torch.no_grad():
            sparse_out_e = sparse_module(
                uih_features=uih_kjt, candidates_features=candidates_kjt
            )
            dense_inputs_e = _move_dense_inputs(*sparse_out_e, device=device)
            preds_eager = dense_module(*dense_inputs_e)

        # === Traced path ===
        # Sparse is traced via a raw-tensor shim because KJT is not a valid
        # traced input. Dense is traced directly with the eager sparse
        # output as the example.
        sparse_shim = _SparseTraceShim(
            sparse_module=sparse_module,
            uih_keys=list(uih_kjt.keys()),
            candidates_keys=list(candidates_kjt.keys()),
        )
        traced_sparse = torch.jit.trace(
            sparse_shim,
            example_inputs=(
                uih_kjt.lengths(),
                uih_kjt.values(),
                candidates_kjt.lengths(),
                candidates_kjt.values(),
            ),
            strict=False,
            check_trace=False,
        )
        traced_dense = torch.jit.trace(
            dense_module,
            example_inputs=tuple(dense_inputs_e),
            strict=False,
            check_trace=False,
        )

        with torch.no_grad():
            sparse_out_t = traced_sparse(
                uih_kjt.lengths(),
                uih_kjt.values(),
                candidates_kjt.lengths(),
                candidates_kjt.values(),
            )
            dense_inputs_t = _move_dense_inputs(*sparse_out_t, device=device)
            preds_traced = traced_dense(*dense_inputs_t)

        torch.testing.assert_close(
            preds_eager.float(),
            preds_traced.float(),
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
