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
import argparse
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys

import traceback

import gin
import torch
from generative_recommenders.dlrm_v3.checkpoint import load_dmp_checkpoint
from generative_recommenders.dlrm_v3.train.utils import (
    cleanup,
    eval_loop,
    make_model,
    make_optimizer_and_shard,
    make_train_test_dataloaders,
    setup,
    streaming_train_eval_loop,
    train_eval_loop,
    train_loop,
)
from generative_recommenders.dlrm_v3.utils import MetricsLogger
from torch import multiprocessing as mp
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger(__name__)


SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-1m": "movielens_1m.gin",
    "movielens-20m": "movielens_20m.gin",
    "movielens-13b": "movielens_13b.gin",
    "movielens-18b": "movielens_18b.gin",
    "streaming-400m": "streaming_400m.gin",
    "streaming-200b": "streaming_200b.gin",
    "streaming-100b": "streaming_100b.gin",
}


def _main_func(
    rank: int,
    world_size: int,
    master_port: int,
    gin_file: str,
    mode: str,
) -> None:
    device = torch.device(f"cuda:{rank}")
    logger.info(f"rank: {rank}, world_size: {world_size}, device: {device}")
    setup(
        rank=rank,
        world_size=world_size,
        master_port=master_port,
        device=device,
    )
    # parse all arguments
    gin.parse_config_file(gin_file)

    model, model_configs, embedding_table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(model=model, device=device, world_size=world_size)
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
    )
    metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=2500,
        device=device,
        rank=rank,
    )
    load_dmp_checkpoint(
        model=model, optimizer=optimizer, metric_logger=metrics, device=device
    )

    # train loop
    try:
        if mode == "train":
            train_loop(
                rank=rank,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "eval":
            # reinit metrics logger for eval
            metrics = MetricsLogger(
                multitask_configs=model_configs.multitask_configs,
                batch_size=train_dataloader.batch_size,
                window_size=1000,
                device=device,
                rank=rank,
            )
            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "train-eval":
            train_eval_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "streaming-train-eval":
            streaming_train_eval_loop(
                rank=rank,
                model=model,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
                hstu_config=model_configs,
                embedding_table_configs=embedding_table_configs,
            )
    except Exception as e:
        logger.info(traceback.format_exc())
        cleanup()
        raise Exception(e)


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="mode",
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    assert args.dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {args.dataset}"
    assert args.mode in [
        "train",
        "eval",
        "train-eval",
        "streaming-train-eval",
    ], f"Unsupported mode: {args.mode}"
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    MASTER_PORT = str(get_free_port())
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"

    mp.start_processes(
        _main_func,
        args=(WORLD_SIZE, MASTER_PORT, gin_path, args.mode),
        nprocs=WORLD_SIZE,
        join=True,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
