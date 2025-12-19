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
Checkpoint utilities for saving and loading DLRMv3 model checkpoints.

This module provides functions for saving and loading distributed model checkpoints,
including both sparse (embedding) and dense (non-embedding) components.
"""

import gc
import os
from datetime import datetime
from typing import Any, Dict, Optional, Set

import gin

import torch
from generative_recommenders.dlrm_v3.utils import MetricsLogger
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.optimizer import Optimizer
from torchrec.distributed.types import ShardedTensor


class SparseState(Stateful):
    """
    Stateful wrapper for sparse (embedding) tensors in a model.

    This class implements the Stateful interface for distributed checkpointing,
    allowing sparse tensors to be saved and loaded separately from dense tensors.

    Args:
        model: The PyTorch model containing sparse tensors.
        sparse_tensor_keys: Set of keys identifying sparse tensors in the model's state dict.
    """

    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


def load_dense_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    own_state = model.state_dict()
    own_state_dense_keys = {k for k, v in own_state.items() if not is_sparse_key(k, v)}
    state_dict_dense_keys = {
        k for k, v in state_dict.items() if not is_sparse_key(k, v)
    }
    assert (
        own_state_dense_keys == state_dict_dense_keys
    ), f"expects {own_state_dense_keys} but gets {state_dict_dense_keys}"
    for name in state_dict_dense_keys:
        param = state_dict[name]
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


@gin.configurable
def save_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    rank: int,
    batch_idx: int,
    path: str = "",
) -> None:
    """
    Save a distributed model checkpoint including sparse and dense components.

    Saves the model's sparse tensors using distributed checkpointing and dense
    tensors, optimizer state, and metrics using standard PyTorch serialization.

    Args:
        model: The model to checkpoint.
        optimizer: The optimizer whose state should be saved.
        metric_logger: The metrics logger containing training/eval metrics.
        rank: The current process rank in distributed training.
        batch_idx: The current batch index (used for checkpoint naming).
        path: Base path for saving the checkpoint. If empty, no checkpoint is saved.
    """
    if path == "":
        return
    now = datetime.now()
    formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")
    path = f"{path}/{batch_idx}"
    if not os.path.exists(path) and rank == 0:
        os.makedirs(path)
    sparse_path = f"{path}/sparse/"
    if not os.path.exists(sparse_path) and rank == 0:
        os.makedirs(sparse_path)
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if isinstance(v, ShardedTensor)
    }
    if rank == 0:
        dense_state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if not isinstance(v, ShardedTensor)
        }
        class_metric_state_dict = {
            "train": [m.state_dict() for m in metric_logger.class_metrics["train"]],
            "eval": [m.state_dict() for m in metric_logger.class_metrics["eval"]],
        }
        regression_metric_state_dict = {
            "train": [
                m.state_dict() for m in metric_logger.regression_metrics["train"]
            ],
            "eval": [m.state_dict() for m in metric_logger.regression_metrics["eval"]],
        }
        torch.save(
            {
                "dense_dict": dense_state_dict,
                "optimizer_dict": optimizer.state_dict(),
                "class_metrics": class_metric_state_dict,
                "reg_metrics": regression_metric_state_dict,
                "global_step": metric_logger.global_step,
                "sparse_tensor_keys": sparse_tensor_keys,
            },
            non_sparse_ckpt,
        )
    torch.distributed.barrier()
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    torch.distributed.checkpoint.save(
        sparse_dict,
        storage_writer=torch.distributed.checkpoint.FileSystemWriter(sparse_path),
    )
    torch.distributed.barrier()
    print("checkpoint successfully saved")


@gin.configurable
def load_sparse_checkpoint(
    model: torch.nn.Module,
    path: str = "",
) -> None:
    if path == "":
        return
    sparse_path = f"{path}/sparse/"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if is_sparse_key(k, v)
    }
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    gc.collect()
    torch.distributed.checkpoint.load(
        sparse_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(sparse_path),
    )
    gc.collect()
    print("sparse checkpoint successfully loaded")


@gin.configurable
def load_nonsparse_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    metric_logger: Optional[MetricsLogger] = None,
    path: str = "",
) -> None:
    """
    Load non-sparse (dense) components from a checkpoint.

    Loads dense model parameters, and optionally optimizer state and metrics.

    Args:
        model: The model to load dense parameters into.
        device: The device to load tensors onto.
        optimizer: Optional optimizer to restore state for.
        metric_logger: Optional metrics logger to restore state for.
        path: Base path of the checkpoint. If empty, no loading is performed.
    """
    if path == "":
        return
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

    non_sparse_state_dict = torch.load(non_sparse_ckpt, map_location=device)
    load_dense_state_dict(model, non_sparse_state_dict["dense_dict"])
    print("dense checkpoint successfully loaded")
    if optimizer is not None:
        optimizer.load_state_dict(non_sparse_state_dict["optimizer_dict"])
        print("optimizer checkpoint successfully loaded")
    if metric_logger is not None:
        metric_logger.global_step = non_sparse_state_dict["global_step"]
        class_metric_state_dict = non_sparse_state_dict["class_metrics"]
        regression_metric_state_dict = non_sparse_state_dict["reg_metrics"]
        for i, m in enumerate(metric_logger.class_metrics["train"]):
            m.load_state_dict(class_metric_state_dict["train"][i])
        for i, m in enumerate(metric_logger.class_metrics["eval"]):
            m.load_state_dict(class_metric_state_dict["eval"][i])
        for i, m in enumerate(metric_logger.regression_metrics["train"]):
            m.load_state_dict(regression_metric_state_dict["train"][i])
        for i, m in enumerate(metric_logger.regression_metrics["eval"]):
            m.load_state_dict(regression_metric_state_dict["eval"][i])


@gin.configurable
def load_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    path: str = "",
) -> None:
    """
    Load a complete distributed model checkpoint (both sparse and dense components).

    This is a convenience function that calls both load_sparse_checkpoint and
    load_nonsparse_checkpoint.

    Args:
        model: The model to load the checkpoint into.
        optimizer: The optimizer to restore state for.
        metric_logger: The metrics logger to restore state for.
        device: The device to load tensors onto.
        path: Base path of the checkpoint. If empty, no loading is performed.
    """
    load_sparse_checkpoint(model=model, path=path)
    load_nonsparse_checkpoint(
        model=model,
        optimizer=optimizer,
        metric_logger=metric_logger,
        path=path,
        device=device,
    )
