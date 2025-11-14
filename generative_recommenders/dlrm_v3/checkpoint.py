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


class SparseStateSingleRank(Stateful):
    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str], rank: int) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys
        self.rank = rank

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Memory-optimized state_dict that extracts only sparse tensors.
        
        This method is called during checkpoint saving to get the ShardedTensors
        that need to be saved. Each rank will only save its local shards.
        """
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        
        # Extract only the sparse tensors (ShardedTensors)
        # Note: model.state_dict() returns the full metadata, but each rank
        # only has access to its local shards via .local_shards()
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor), \
                    f"Expected ShardedTensor for key '{k}', got {type(v)}"
                
                if isinstance(v, ShardedTensor):
                    local_shards = v.local_shards()[0].tensor
                    out_dict[k] = local_shards
        
        print(f"Rank {self.rank}: Extracted {len(out_dict)} sparse tensors for checkpoint")
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
def save_dmp_checkpoint_single_rank(
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    rank: int,
    batch_idx: int,
    path: str = "",
) -> None:
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
    
    # SEQUENTIAL CHECKPOINT SAVING: Save one rank at a time to reduce memory pressure
    # Instead of all ranks saving in parallel, we serialize the saves
    world_size = torch.distributed.get_world_size()
    
    print(f"Rank {rank}: Preparing sparse checkpoint (world_size={world_size})")
    
    # Each rank saves sequentially to minimize peak memory usage
    # Using regular torch.save instead of distributed checkpoint to reduce memory overhead
    for saving_rank in range(world_size):
        if rank == saving_rank:
            print(f"Rank {rank}: Extracting local sparse tensors")
            
            # Extract local shards directly without SparseState wrapper
            sparse_state = SparseStateSingleRank(model, sparse_tensor_keys, rank=rank)
            local_sparse_dict = sparse_state.state_dict()
            
            # Save to a per-rank file using regular torch.save
            rank_file = os.path.join(sparse_path, f"rank_{rank}.pt")
            print(f"Rank {rank}: Saving sparse checkpoint to {rank_file}")
            
            torch.save(
                {
                    "sparse_tensors": local_sparse_dict,
                    "rank": rank,
                },
                rank_file,
            )
            
            print(f"Rank {rank}: Checkpoint saved successfully")
            
            # Clean up immediately after saving
            del sparse_state
            del local_sparse_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # All ranks wait for current rank to finish before next rank starts
        torch.distributed.barrier()
        if rank == 0:
            print(f"Rank {saving_rank} checkpoint complete, proceeding to next rank...")
    
    if rank == 0:
        print("All ranks checkpoint successfully saved")
    torch.distributed.barrier()


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
    torch.distributed.checkpoint.load(
        sparse_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(sparse_path),
    )
    print("sparse checkpoint successfully loaded")

@gin.configurable
def load_sparse_checkpoint_single_rank(
    model: torch.nn.Module,
    path: str = "",
) -> None:
    if path == "":
        return
    sparse_path = f"{path}/sparse/"
    
    # Get all rank files
    import glob
    rank_files = sorted(glob.glob(f"{sparse_path}/rank_*.pt"))
    if not rank_files:
        print(f"No rank files found in {sparse_path}")
        return
    
    print(f"Found {len(rank_files)} rank files: {rank_files}")
    
    # Load sparse tensor keys from model
    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if is_sparse_key(k, v)
    }
    
    # Dictionary to hold concatenated tensors for each key
    concatenated_tensors = {}
    
    # Load each rank file and concatenate tensors
    for rank_idx, rank_file in enumerate(rank_files):
        print(f"Loading {rank_file}...")
        rank_data = torch.load(rank_file, map_location="cpu", mmap=True)
        
        # rank_data should have structure like {"sparse_tensors": {key: tensor}}
        if "sparse_tensors" in rank_data:
            sparse_tensors = rank_data["sparse_tensors"]
        else:
            sparse_tensors = rank_data
        
        # For each tensor in this rank, add to concatenation list
        for key in sparse_tensor_keys:
            if key in sparse_tensors:
                tensor = sparse_tensors[key]
                print(f"  Rank {rank_idx}: {key} shape = {tensor.shape}, device = {tensor.device}")
                
                if key not in concatenated_tensors:
                    concatenated_tensors[key] = []
                concatenated_tensors[key].append(tensor)
    
    # Concatenate all tensors along dimension 0
    final_state_dict = {}
    for key, tensor_list in concatenated_tensors.items():
        concatenated = torch.cat(tensor_list, dim=0)
        print(f"Concatenated {key}: {concatenated.shape}")
        final_state_dict[key] = concatenated
    
    # Load the concatenated state dict into the model
    incompatible_keys = model.load_state_dict(final_state_dict, strict=False)
    
    if incompatible_keys.unexpected_keys:
        print(f"Warning: unexpected keys: {incompatible_keys.unexpected_keys}")
    if incompatible_keys.missing_keys:
        # Filter out non-sparse keys (those are loaded separately)
        missing_sparse = [k for k in incompatible_keys.missing_keys if any(sk in k for sk in ["embedding"])]
        if missing_sparse:
            print(f"Warning: missing sparse keys: {missing_sparse}")
    
    print("sparse checkpoint successfully loaded")


@gin.configurable
def load_nonsparse_checkpoint(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[Optimizer] = None,
    metric_logger: Optional[MetricsLogger] = None,
    path: str = "",
) -> None:
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
    load_sparse_checkpoint(model=model, path=path)
    load_nonsparse_checkpoint(
        model=model,
        optimizer=optimizer,
        metric_logger=metric_logger,
        path=path,
        device=device,
    )