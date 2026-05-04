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
PredictFactory for the HSTU model family.

Builds a single ``torch.nn.Module`` that exposes the two scripted sub-modules
(``self.sparse``, ``self.dense``) the predictor can address independently.
The C++ glue calls them in sequence (sparse → device transfer → dense),
mirroring the eager ``HSTUModelFamily.predict`` flow.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
)
from generative_recommenders.dlrm_v3.inference.dense_predict_module import (
    HSTUDenseScriptModule,
)
from generative_recommenders.dlrm_v3.inference.sparse_predict_module import (
    HSTUSparseScriptModule,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.inference.modules import (
    BatchingMetadata,
    PredictFactory,
    QualNameMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingConfig


logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class HSTUModelConfig:
    """Inputs needed to build (and optionally checkpoint-load) the HSTU."""

    hstu_config: DlrmHSTUConfig
    table_config: Dict[str, EmbeddingConfig]
    checkpoint_path: Optional[str] = None


class _HSTUPredictModule(torch.nn.Module):
    """Composite container holding the two scripted sub-modules.

    The C++ side accesses ``self.sparse`` and ``self.dense`` as
    ``pkg.attr("sparse").toModule()`` / ``pkg.attr("dense").toModule()``.

    ``forward`` is a no-op placeholder so that ``torch.jit.script`` accepts
    the wrapper itself; the predictor and C++ glue invoke the sparse / dense
    sub-modules directly via attribute access.
    """

    def __init__(
        self,
        sparse: torch.jit.ScriptModule,
        dense: torch.jit.ScriptModule,
    ) -> None:
        super().__init__()
        self.sparse: torch.jit.ScriptModule = sparse
        self.dense: torch.jit.ScriptModule = dense

    def forward(self) -> torch.Tensor:
        return torch.zeros(1)


class HSTUPredictFactory(PredictFactory):
    """Builds the HSTU sparse + dense scripted modules for inference.

    Mirrors :class:`torchrec.inference.dlrm_predict.DLRMPredictFactory`.
    """

    def __init__(self, model_config: HSTUModelConfig) -> None:
        self.model_config = model_config

    # pyrefly: ignore[bad-override]
    def create_predict_module(
        self, world_size: int = 1, device: str = "cuda:0"
    ) -> torch.nn.Module:
        logging.basicConfig(level=logging.INFO)

        sparse_module = HSTUSparseScriptModule(
            table_config=self.model_config.table_config,
            hstu_config=self.model_config.hstu_config,
            use_no_copy_embedding_collection=True,
        )
        sparse_module.eval()
        if self.model_config.checkpoint_path is not None:
            load_sparse_checkpoint(
                model=sparse_module._sparse._hstu_model,
                path=self.model_config.checkpoint_path,
            )

        dense_module = HSTUDenseScriptModule(
            hstu_config=self.model_config.hstu_config,
            table_config=self.model_config.table_config,
        )
        dense_module = dense_module.to(torch.bfloat16).to(torch.device(device))
        dense_module._hstu_model.set_training_dtype(torch.bfloat16)
        dense_module.eval()
        if self.model_config.checkpoint_path is not None:
            load_nonsparse_checkpoint(
                model=dense_module._hstu_model,
                device=torch.device(device),
                optimizer=None,
                path=self.model_config.checkpoint_path,
            )

        scripted_sparse = torch.jit.script(sparse_module)
        scripted_dense = torch.jit.script(dense_module)

        return _HSTUPredictModule(sparse=scripted_sparse, dense=scripted_dense)

    # pyrefly: ignore[bad-override]
    def batching_metadata(self) -> Dict[str, BatchingMetadata]:
        # Sparse side consumes two KJTs; declared as sparse for the predictor.
        # device="cpu" because the sparse module runs on CPU; the dense-side
        # device move is performed by the C++ glue / predictor batcher.
        return {
            "uih_features": BatchingMetadata(
                type="sparse", device="cpu", pinned=["lengths"]
            ),
            "candidates_features": BatchingMetadata(
                type="sparse", device="cpu", pinned=["lengths"]
            ),
        }

    def model_inputs_data(self) -> Dict[str, Any]:
        return {}

    def qualname_metadata(self) -> Dict[str, QualNameMetadata]:
        return {
            "sparse.forward": QualNameMetadata(need_preproc=False),
            "dense.forward": QualNameMetadata(need_preproc=False),
        }

    def result_metadata(self) -> str:
        return "tuple_of_tensor"

    def run_weights_independent_tranformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        return predict_module

    def run_weights_dependent_transformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        return predict_module
