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
model_family for dlrm_v3.
"""

import os
import time
import uuid
from threading import Event
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
import torchrec
from generative_recommenders.dlrm_v3.checkpoint import (
    load_nonsparse_checkpoint,
    load_sparse_checkpoint,
)
from generative_recommenders.dlrm_v3.datasets.dataset import Samples
from generative_recommenders.dlrm_v3.inference.inference_modules import (
    get_hstu_model,
    HSTUSparseInferenceModule,
    move_sparse_output_to_device,
)
from generative_recommenders.dlrm_v3.utils import Profiler
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig, SequenceEmbedding
from torch import quantization as quant
from torchrec.distributed.quant_embedding import QuantEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig, QuantConfig
from torchrec.test_utils import get_free_port


class HSTUModelFamily:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
    ) -> None:
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.sparse: ModelFamilySparseDist = ModelFamilySparseDist(
            hstu_config=hstu_config,
            table_config=table_config,
        )

        assert torch.cuda.is_available(), "CUDA is required for this benchmark."
        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        print(f"Using {self.world_size} GPU(s)...")
        dense_model_family_clazz = (
            ModelFamilyDenseDist
            if self.world_size > 1
            else ModelFamilyDenseSingleWorker
        )
        self.dense: Union[ModelFamilyDenseDist, ModelFamilyDenseSingleWorker] = (
            dense_model_family_clazz(
                hstu_config=hstu_config,
                table_config=table_config,
                output_trace=output_trace,
            )
        )

    def version(self) -> str:
        return torch.__version__

    def name(self) -> str:
        return "model-family-hstu"

    def load(self, model_path: str) -> None:
        self.sparse.load(model_path=model_path)
        self.dense.load(model_path=model_path)

    def predict(
        self, samples: Optional[Samples]
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        if samples is None:
            return self.dense.predict(None, None, 0, None, 0, None)
        (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        ) = self.sparse.predict(samples)
        return self.dense.predict(
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )


class ModelFamilySparseDist:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
    ) -> None:
        super(ModelFamilySparseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.module: Optional[torch.nn.Module] = None

    def load(self, model_path: str) -> None:
        print(f"Loading sparse module from {model_path}")

        sparse_arch: HSTUSparseInferenceModule = HSTUSparseInferenceModule(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
        )
        load_sparse_checkpoint(model=sparse_arch._hstu_model, path=model_path)
        sparse_arch.eval()
        self.module = quant.quantize_dynamic(
            sparse_arch,
            qconfig_spec={
                torchrec.EmbeddingCollection: QuantConfig(
                    activation=quant.PlaceholderObserver.with_args(dtype=torch.float),
                    weight=quant.PlaceholderObserver.with_args(dtype=torch.int8),
                ),
            },
            mapping={
                torchrec.EmbeddingCollection: QuantEmbeddingCollection,
            },
            inplace=False,
        )
        print(f"sparse module is {self.module}")

    def predict(
        self, samples: Samples
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        with torch.profiler.record_function("sparse forward"):
            assert self.module is not None
            uih_features = samples.uih_features_kjt
            candidates_features = samples.candidates_features_kjt
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.module(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )
            return (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )


class ModelFamilyDenseDist:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
    ) -> None:
        super(ModelFamilyDenseDist, self).__init__()
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace

        ngpus = torch.cuda.device_count()
        self.world_size = int(os.environ.get("WORLD_SIZE", str(ngpus)))
        self.rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.dist_backend = "nccl"

        ctx = mp.get_context("spawn")
        self.samples_q: List[mp.Queue] = [ctx.Queue() for _ in range(self.world_size)]
        self.predictions_cache = [  # pyre-ignore[4]
            mp.Manager().dict() for _ in range(self.world_size)
        ]
        self.main_lock: Event = ctx.Event()

    def load(self, model_path: str) -> None:
        print(f"Loading dense module from {model_path}")

        ctx = mp.get_context("spawn")
        processes = []
        for rank in range(self.world_size):
            p = ctx.Process(
                target=self.distributed_setup,
                args=(
                    rank,
                    self.world_size,
                    model_path,
                ),
            )
            p.start()
            processes.append(p)
        self.main_lock.wait()

    def distributed_setup(self, rank: int, world_size: int, model_path: str) -> None:
        model = get_hstu_model(
            table_config=self.table_config,
            hstu_config=self.hstu_config,
            table_device="cpu",
            max_hash_size=100,
            is_dense=True,
        ).to(torch.bfloat16)
        load_nonsparse_checkpoint(model=model, optimizer=None, path=model_path)

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(f"cuda:{rank}")
        self.main_lock.set()
        model = model.to(device)
        model.eval()
        profiler = Profiler(rank) if self.output_trace else None

        while True:
            if self.samples_q[rank].empty():
                time.sleep(0.001)
                continue
            item = self.samples_q[rank].get()
            # If -1 is received terminate all subprocesses
            if item == -1:
                break
            (
                id,
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = item
            assert seq_embeddings is not None
            if self.output_trace:
                assert profiler is not None
                profiler.step()
            with torch.profiler.record_function("dense forward"):
                (
                    _,
                    _,
                    _,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.main_forward(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    max_uih_len=max_uih_len,
                    uih_seq_lengths=uih_seq_lengths,
                    max_num_candidates=max_num_candidates,
                    num_candidates=num_candidates,
                )
                assert mt_target_preds is not None
                mt_target_preds = mt_target_preds.detach().to(
                    device="cpu", non_blocking=True
                )
                if mt_target_labels is not None:
                    mt_target_labels = mt_target_labels.detach().to(
                        device="cpu", non_blocking=True
                    )
                if mt_target_weights is not None:
                    mt_target_weights = mt_target_weights.detach().to(
                        device="cpu", non_blocking=True
                    )
                self.predictions_cache[rank][id] = (
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                )

    def capture_output(
        self, id: uuid.UUID, rank: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        out = None
        while out is None:
            time.sleep(0.001)
            out = self.predictions_cache[rank].get(id, None)
        self.predictions_cache[rank].pop(id)
        return out

    def get_rank(self) -> int:
        rank = self.rank
        self.rank = (self.rank + 1) % self.world_size
        return rank

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        id = uuid.uuid4()
        # If none is received terminate all subprocesses
        if seq_embeddings is None:
            for rank in range(self.world_size):
                self.samples_q[rank].put(-1)
            return None
        rank = self.get_rank()
        device = torch.device(f"cuda:{rank}")
        assert (
            payload_features is not None
            and num_candidates is not None
            and uih_seq_lengths is not None
        )
        seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
            move_sparse_output_to_device(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                uih_seq_lengths=uih_seq_lengths,
                num_candidates=num_candidates,
                device=device,
            )
        )
        self.main_lock.wait()
        self.main_lock.clear()
        self.samples_q[rank].put(
            (
                id,
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            )
        )
        out = self.capture_output(id, rank)
        self.main_lock.set()
        return out


class ModelFamilyDenseSingleWorker:
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        table_config: Dict[str, EmbeddingConfig],
        output_trace: bool = False,
    ) -> None:
        self.model: Optional[torch.nn.Module] = None
        self.hstu_config = hstu_config
        self.table_config = table_config
        self.output_trace = output_trace
        self.device: torch.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.profiler: Optional[Profiler] = (
            Profiler(rank=0) if self.output_trace else None
        )

    def load(self, model_path: str) -> None:
        print(f"Loading dense module from {model_path}")
        self.model = (
            get_hstu_model(
                table_config=self.table_config,
                hstu_config=self.hstu_config,
                table_device="cpu",
                is_dense=True,
            )
            .to(self.device)
            .to(torch.bfloat16)
        )
        load_nonsparse_checkpoint(model=self.model, optimizer=None, path=model_path)
        assert self.model is not None
        self.model.eval()

    def predict(
        self,
        seq_embeddings: Optional[Dict[str, SequenceEmbedding]],
        payload_features: Optional[Dict[str, torch.Tensor]],
        max_uih_len: int,
        uih_seq_lengths: Optional[torch.Tensor],
        max_num_candidates: int,
        num_candidates: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.output_trace:
            assert self.profiler is not None
            self.profiler.step()
        assert (
            payload_features is not None
            and uih_seq_lengths is not None
            and num_candidates is not None
            and seq_embeddings is not None
        )
        with torch.profiler.record_function("dense forward"):
            seq_embeddings, payload_features, uih_seq_lengths, num_candidates = (
                move_sparse_output_to_device(
                    seq_embeddings=seq_embeddings,
                    payload_features=payload_features,
                    uih_seq_lengths=uih_seq_lengths,
                    num_candidates=num_candidates,
                    device=self.device,
                )
            )
            assert self.model is not None
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = self.model.main_forward(  # pyre-ignore [29]
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
            assert mt_target_preds is not None
            mt_target_preds = mt_target_preds.detach().to(
                device="cpu", non_blocking=True
            )
            if mt_target_labels is not None:
                mt_target_labels = mt_target_labels.detach().to(
                    device="cpu", non_blocking=True
                )
            if mt_target_weights is not None:
                mt_target_weights = mt_target_weights.detach().to(
                    device="cpu", non_blocking=True
                )
            return mt_target_preds, mt_target_labels, mt_target_weights
