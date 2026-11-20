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
import logging
import os
from collections.abc import Iterator
from datetime import timedelta
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import gin
import torch
import torchrec
from generative_recommenders.dlrm_v3.checkpoint import save_dmp_checkpoint
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn, Dataset
from generative_recommenders.dlrm_v3.utils import get_dataset, MetricsLogger, Profiler
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import _T_co, DistributedSampler
from torchrec.distributed.model_parallel import (
    DataParallelWrapper,
    DistributedModelParallel,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import (
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

def _is_xccl_available() -> bool:
    try:
        return dist.is_xccl_available()
    except (AttributeError, RuntimeError):
        return False


def get_dist_backend(device: torch.device) -> str:
    """Return the best distributed backend for the given device."""
    if device.type == "xpu":
        if _is_xccl_available():
            return "xccl"
        logger.warning("XCCL not available, falling back to gloo for XPU")
    elif device.type == "cuda":
        return "nccl"
    return "gloo"

def setup(
    rank: int, world_size: int, master_port: int, device: torch.device
) -> dist.ProcessGroup:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    BACKEND = get_dist_backend(device)
    TIMEOUT = 1800

    # initialize the process group
    if not dist.is_initialized():
        dist.init_process_group(BACKEND, rank=rank, world_size=world_size)

    pg = dist.new_group(
        backend=BACKEND,
        timeout=timedelta(seconds=TIMEOUT),
    )
    assert isinstance(pg, dist.ProcessGroup)

    # set device
    if device.type == "xpu":
        torch.xpu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)

    return pg


def cleanup() -> None:
    dist.destroy_process_group()


class HammerToTorchDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        self.dataset: Dataset = dataset

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        self.dataset.load_query_samples([idx])
        sample = self.dataset.get_sample(idx)
        self.dataset.unload_query_samples([idx])
        return sample

    def __getitems__(
        self, indices: List[int]
    ) -> List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]:
        self.dataset.load_query_samples(indices)
        samples = [self.dataset.get_sample(i) for i in indices]
        self.dataset.unload_query_samples(indices)
        # pyrefly: ignore [bad-return]
        return samples


class ChunkDistributedSampler(DistributedSampler[_T_co]):
    """
    Each rank reads a contiguous chunk (trunk) of the input data
    """

    def __init__(
        self,
        dataset: TorchDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 1,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * 1001 + int(self.rank))
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))
        assert self.drop_last is True, (
            "drop_last must be True for ChunkDistributedSampler"
        )
        indices = [i + self.num_samples * self.rank for i in indices]

        assert len(indices) == self.num_samples
        # pyrefly: ignore [bad-return]
        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        logger.warning(f"Setting epoch to {epoch}")
        self.epoch = epoch


@gin.configurable
def make_model(
    dataset: str,
    hstu_ultra_stack_name: Optional[str] = None,
    hstu_fp8_addmm_fwd: bool = False,
    num_embeddings: Optional[int] = None,
) -> Tuple[torch.nn.Module, DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:
    hstu_config = get_hstu_configs(
        dataset,
        hstu_ultra_stack_name=hstu_ultra_stack_name,
        hstu_fp8_addmm_fwd=hstu_fp8_addmm_fwd,
    )
    table_config = get_embedding_table_config(dataset)
    # Optional row count for every table, for datasets whose ids are far below
    # the default hash size. It must exceed the largest id of every feature.
    if num_embeddings is not None:
        table_config = {
            name: dataclasses.replace(config, num_embeddings=num_embeddings)
            for name, config in table_config.items()
        }
    # The PYTORCH attention kernel forced below materialises a dense
    # [B, H, N, N] score tensor; the 16384 default needs ~60 GiB, so keep N small.
    if torch.xpu.is_available():
        hstu_config.max_seq_len = 256

    model = DlrmHSTU(
        hstu_configs=hstu_config,
        embedding_tables=table_config,
        is_inference=False,
    )

    return (
        model,
        hstu_config,
        table_config,
    )


@gin.configurable()
def dense_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    elif optimizer_name == "SGD":
        optimizer_cls = torch.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
        kwargs.update({"betas": betas, "eps": eps, "weight_decay": weight_decay})
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


@gin.configurable()
def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[
    Type[Optimizer], Dict[str, Any], Callable[[Iterable[torch.Tensor]], Optimizer]
]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "Adam":
        optimizer_cls = torch.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "SGD":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "RowWiseAdagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    optimizer_factory = lambda params: optimizer_cls(params, **kwargs)

    return optimizer_cls, kwargs, optimizer_factory


def _unsharded_tensors(module: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Parameters and buffers of `module` outside any torchrec ShardedModule."""
    if isinstance(module, ShardedModule):
        return
    yield from module.parameters(recurse=False)
    yield from module.buffers(recurse=False)
    for child in module.children():
        yield from _unsharded_tensors(child)


class _XpuGradAveragingDataParallelWrapper(DataParallelWrapper):
    """XPU replacement for torchrec's default DDP wrapper.

    On XPU, DDP's XCCL gradient allreduce running inside backward, concurrently
    with torchrec's embedding backward comms, segfaults in oneCCL (a freed SYCL
    dependency event) at >=4 ranks. This wrapper does what DDP does at
    construction (move the unsharded modules to the device and broadcast their
    parameters and buffers from rank 0); gradient averaging happens after
    backward, in the dense optimizer's step pre-hook (_average_grads_pre_step).
    """

    def wrap(
        self,
        dmp: DistributedModelParallel,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        module = dmp._dmp_wrapped_module.to(device)
        for t in _unsharded_tensors(module):
            dist.broadcast(t.data, src=0, group=env.process_group)


def _average_grads_pre_step(
    optimizer: Optimizer, args: Any, kwargs: Any
) -> None:
    """Average the optimizer's grads across ranks, like DDP, in one allreduce.

    A missing grad becomes zeros so every rank reduces the same buffer layout.
    """
    params = [p for group in optimizer.param_groups for p in group["params"]]
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
        p.grad = g


def make_optimizer_and_shard(
    model: torch.nn.Module,
    device: torch.device,
    world_size: int,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    dense_opt_cls, dense_opt_args, dense_opt_factory = (
        dense_optimizer_factory_and_class()
    )

    sparse_opt_cls, sparse_opt_args, sparse_opt_factory = (
        sparse_optimizer_factory_and_class()
    )
    # Fuse sparse optimizer to backward step
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
    sharders = get_default_sharders()
    # XPU cannot keep tables in host memory yet (no UVM allocation in fbgemm-xpu,
    # and TorchRec offers fused_uvm* only on CUDA), so plan against the real capacity.
    hbm_cap = (
        torch.xpu.get_device_properties(device).total_memory
        if device.type == "xpu"
        else 160 * 1024 * 1024 * 1024
    )
    
    constraints = None
    if device.type == "xpu":
        constraints = {}
        for module in model.modules():
            if type(module) not in TORCHREC_TYPES:
                continue
            configs = (
                module.embedding_configs()
                if isinstance(module, EmbeddingCollection)
                else module.embedding_bag_configs()
            )
            for config in configs:
                constraints[config.name] = ParameterConstraints(
                    sharding_types=[
                        ShardingType.TABLE_WISE.value,
                        ShardingType.ROW_WISE.value,
                    ]
                )
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=world_size,
            world_size=world_size,
            compute_device=device.type,
            hbm_cap=hbm_cap,
            ddr_cap=32 * 1024 * 1024 * 1024,
        ),
        constraints=constraints,
    )
    pg = dist.GroupMember.WORLD
    env = ShardingEnv.from_process_group(pg)  # pyre-ignore [6]
    pg = env.process_group

    plan = planner.collective_plan(model, sharders, pg)

    # XPU: no DDP (see _XpuGradAveragingDataParallelWrapper). DATA_PARALLEL
    # tables rely on DDP to sync their weights, so they are not supported here.
    data_parallel_wrapper = None
    if device.type == "xpu":
        assert not any(
            ps.sharding_type == ShardingType.DATA_PARALLEL.value
            for module_plan in plan.plan.values()
            for ps in module_plan.values()
        ), "DATA_PARALLEL embedding tables need DDP, which is disabled on XPU"
        data_parallel_wrapper = _XpuGradAveragingDataParallelWrapper()

    # Shard model
    model = DistributedModelParallel(
        module=model,
        device=device,
        plan=plan,
        sharders=sharders,
        data_parallel_wrapper=data_parallel_wrapper,
    )
    # Create keyed optimizer
    all_optimizers = []
    all_params = {}
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
            else:
                all_params[k] = v

    if non_fused_sparse_params:
        all_optimizers.append(
            (
                "sparse_non_fused",
                KeyedOptimizerWrapper(
                    params=non_fused_sparse_params, optim_factory=sparse_opt_factory
                ),
            )
        )

    if all_params:
        dense_factory = dense_opt_factory
        if device.type == "xpu":
            # Replaces DDP's in-backward allreduce: runs on every step(), i.e.
            # after backward has returned.
            def dense_factory(params: Iterable[torch.Tensor]) -> Optimizer:
                optimizer = dense_opt_factory(params)
                optimizer.register_step_pre_hook(_average_grads_pre_step)
                return optimizer

        all_optimizers.append(
            (
                "dense",
                KeyedOptimizerWrapper(
                    params=all_params,
                    optim_factory=dense_factory,
                ),
            )
        )
    # pyrefly: ignore [bad-argument-type]
    output_optimizer = CombinedOptimizer(all_optimizers)
    output_optimizer.init_state(set(model.sparse_grad_parameter_names()))
    return model, output_optimizer


@gin.configurable
def make_streaming_dataloader(
    dataset: HammerToTorchDataset,
    ts: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    dataset.dataset.set_ts(ts)  # pyre-ignore [16]
    total_items = dataset.dataset.get_item_count()
    subset = torch.utils.data.Subset(dataset, range(total_items))
    dataloader = DataLoader(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=DistributedSampler(subset, drop_last=True),
        pin_memory=True,
    )
    return dataloader


@gin.configurable
def make_train_test_dataloaders(
    batch_size: int,
    dataset_type: str,
    hstu_config: DlrmHSTUConfig,
    train_split_percentage: float,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    new_path_prefix: str = "",
    num_workers: int = 0,
    num_blocks: int = 1,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    dataset_class, kwargs = get_dataset(
        name=dataset_type, new_path_prefix=new_path_prefix
    )
    kwargs["embedding_config"] = embedding_table_configs

    # Create dataset
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    total_items = dataset.dataset.get_item_count()
    items_per_block = total_items // num_blocks
    train_size_per_block = round(train_split_percentage * items_per_block)
    train_inds, test_inds = [], []
    for i in range(num_blocks):
        train_inds.extend(
            range(i * items_per_block, i * items_per_block + train_size_per_block)
        )
        test_inds.extend(
            range(i * items_per_block + train_size_per_block, (i + 1) * items_per_block)
        )
    train_set = torch.utils.data.Subset(dataset, train_inds)
    test_set = torch.utils.data.Subset(dataset, test_inds)

    # Wrap dataset with dataloader
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=ChunkDistributedSampler(train_set, drop_last=True, shuffle=True),
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        sampler=ChunkDistributedSampler(test_set, drop_last=True, shuffle=True),
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


@gin.configurable
def train_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    start_batch_idx: int = 0,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.train()
    batch_idx: int = start_batch_idx
    profiler = Profiler(rank, active=10) if output_trace else None

    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        for sample in dataloader:
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if batch_idx % metric_log_frequency != 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if batch_idx % checkpoint_frequency == 0 and batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=batch_idx,
                )
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
        if num_batches is not None and batch_idx >= num_batches:
            break


@gin.configurable
def eval_loop(
    rank: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    metric_logger: MetricsLogger,
    device: torch.device,
    metric_log_frequency: int = 1,
    num_batches: Optional[int] = None,
    output_trace: bool = False,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    model.eval()
    batch_idx: int = 0
    profiler = Profiler(rank, active=10) if output_trace else None
    metric_logger.reset(mode="eval")
    with torch.no_grad():
        for sample in dataloader:
            sample.to(device)
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            metric_logger.update(
                mode="eval",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if batch_idx % metric_log_frequency != 0:
                metric_logger.compute_and_log(mode="eval")
            batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_batches is not None and batch_idx >= num_batches:
                break
    metric_logger.compute_and_log(mode="eval")
    for k, v in metric_logger.compute(mode="eval").items():
        print(f"{k}: {v}")


@gin.configurable
def train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_epochs: int,
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
    eval_frequency: int = 1,
    start_train_batch_idx: int = 0,
    start_eval_batch_idx: int = 0,
    # lr_scheduler: to-do: Add a scheduler
) -> None:
    train_batch_idx: int = start_train_batch_idx
    eval_batch_idx: int = start_eval_batch_idx
    profiler = Profiler(rank, active=10) if output_trace else None
    assert train_dataloader is not None and eval_dataloader is not None
    eval_data_iterator = iter(eval_dataloader)
    train_data_iterator = iter(train_dataloader)
    # metric_logger.reset(mode="train")
    # metric_logger.reset(mode="eval")

    for epoch in range(num_epochs):
        train_dataloader.sampler.set_epoch(epoch)  # pyre-ignore [16]
        while True:
            model.train()
            try:
                sample = next(train_data_iterator)
            except StopIteration:
                train_data_iterator = iter(train_dataloader)
                break
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if train_batch_idx % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            if train_batch_idx % checkpoint_frequency == 0 and train_batch_idx > 0:
                save_dmp_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    metric_logger=metric_logger,
                    rank=rank,
                    batch_idx=train_batch_idx,
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if train_batch_idx % eval_frequency == 0:
                model.eval()
                eval_batch_idx: int = 0
                with torch.no_grad():
                    while True:
                        try:
                            sample = next(eval_data_iterator)
                        except StopIteration:
                            eval_data_iterator = iter(eval_dataloader)
                            sample = next(eval_data_iterator)
                        sample.to(device)
                        (
                            _,
                            _,
                            _,
                            mt_target_preds,
                            mt_target_labels,
                            mt_target_weights,
                        ) = model.forward(
                            sample.uih_features_kjt,
                            sample.candidates_features_kjt,
                        )
                        metric_logger.update(
                            mode="eval",
                            predictions=mt_target_preds,
                            labels=mt_target_labels,
                            weights=mt_target_weights,
                            num_candidates=sample.candidates_features_kjt.lengths().view(
                                len(sample.candidates_features_kjt.keys()), -1
                            )[0],
                        )
                        eval_batch_idx += 1
                        if output_trace:
                            assert profiler is not None
                            profiler.step()
                        if eval_batch_idx % metric_log_frequency == 0:
                            metric_logger.compute_and_log(mode="eval")
                        if (
                            num_eval_batches is not None
                            and eval_batch_idx >= num_eval_batches
                        ):
                            break
                    for k, v in metric_logger.compute(mode="eval").items():
                        print(f"{k}: {v}")
                model.train()
            if num_train_batches is not None and train_batch_idx >= num_train_batches:
                break


@gin.configurable
def streaming_train_eval_loop(
    rank: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    metric_logger: MetricsLogger,
    device: torch.device,
    num_train_ts: int,
    hstu_config: DlrmHSTUConfig,
    embedding_table_configs: Dict[str, EmbeddingConfig],
    num_train_batches: Optional[int] = None,
    num_eval_batches: Optional[int] = None,
    output_trace: bool = False,
    metric_log_frequency: int = 1,
    checkpoint_frequency: int = 100,
) -> None:
    profiler = Profiler(rank, active=10) if output_trace else None
    dataset_class, kwargs = get_dataset()
    kwargs["embedding_config"] = embedding_table_configs
    dataset = HammerToTorchDataset(
        dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
    )
    for train_ts in range(num_train_ts):
        train_batch_idx: int = 0
        train_dataloader = make_streaming_dataloader(dataset=dataset, ts=train_ts)
        train_data_iterator = iter(train_dataloader)
        while True:
            model.train()
            try:
                sample = next(train_data_iterator)
            except StopIteration:
                break
            optimizer.zero_grad()
            sample.to(device)
            (
                _,
                _,
                aux_losses,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            # pyre-ignore
            sum(aux_losses.values()).backward()
            optimizer.step()
            metric_logger.update(
                mode="train",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            if train_batch_idx % metric_log_frequency == 0:
                metric_logger.compute_and_log(
                    mode="train",
                    additional_logs={
                        "losses": aux_losses,
                    },
                )
            train_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if num_train_batches is not None and train_batch_idx >= num_train_batches:
                break
        eval_ts = train_ts + 1
        dataset.dataset.is_eval = True  # pyre-ignore [16]
        model.eval()
        eval_batch_idx: int = 0
        eval_dataloader = make_streaming_dataloader(dataset=dataset, ts=eval_ts)
        eval_data_iterator = iter(eval_dataloader)
        with torch.no_grad():
            while True:
                try:
                    sample = next(eval_data_iterator)
                except StopIteration:
                    break
                sample.to(device)
                (
                    _,
                    _,
                    _,
                    mt_target_preds,
                    mt_target_labels,
                    mt_target_weights,
                ) = model.forward(
                    sample.uih_features_kjt,
                    sample.candidates_features_kjt,
                )
                metric_logger.update(
                    mode="eval",
                    predictions=mt_target_preds,
                    labels=mt_target_labels,
                    weights=mt_target_weights,
                    num_candidates=sample.candidates_features_kjt.lengths().view(
                        len(sample.candidates_features_kjt.keys()), -1
                    )[0],
                )
                eval_batch_idx += 1
                if output_trace:
                    assert profiler is not None
                    profiler.step()
                if eval_batch_idx % metric_log_frequency == 0:
                    metric_logger.compute_and_log(mode="eval")
                if num_eval_batches is not None and eval_batch_idx >= num_eval_batches:
                    break
            for k, v in metric_logger.compute(mode="eval").items():
                print(f"{k}: {v}")
        if (
            train_ts % checkpoint_frequency == 0 and train_ts > 0
        ) or train_ts == num_train_ts - 1:
            save_dmp_checkpoint(
                model=model,
                optimizer=optimizer,
                metric_logger=metric_logger,
                rank=rank,
                batch_idx=train_ts,
            )

    eval_ts = num_train_ts
    # pyrefly: ignore [missing-attribute]
    dataset.dataset.is_eval = True
    model.eval()
    eval_batch_idx: int = 0
    eval_dataloader = make_streaming_dataloader(dataset=dataset, ts=eval_ts)
    eval_data_iterator = iter(eval_dataloader)
    with torch.no_grad():
        while True:
            try:
                sample = next(eval_data_iterator)
            except StopIteration:
                break
            sample.to(device)
            (
                _,
                _,
                _,
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
            ) = model.forward(
                sample.uih_features_kjt,
                sample.candidates_features_kjt,
            )
            metric_logger.update(
                mode="eval",
                predictions=mt_target_preds,
                labels=mt_target_labels,
                weights=mt_target_weights,
                num_candidates=sample.candidates_features_kjt.lengths().view(
                    len(sample.candidates_features_kjt.keys()), -1
                )[0],
            )
            eval_batch_idx += 1
            if output_trace:
                assert profiler is not None
                profiler.step()
            if eval_batch_idx % metric_log_frequency == 0:
                metric_logger.compute_and_log(mode="eval")
            if num_eval_batches is not None and eval_batch_idx >= num_eval_batches:
                break
        for k, v in metric_logger.compute(mode="eval").items():
            print(f"{k}: {v}")
