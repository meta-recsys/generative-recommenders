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
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
"""

import argparse
import json
import logging

import numpy as np
import torch
from generative_recommenders.dlrm_v3.configs import get_hstu_configs
from generative_recommenders.dlrm_v3.utils import MetricsLogger

logger: logging.Logger = logging.getLogger("main")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="path to mlperf_log_accuracy.json",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    logger.warning("Parsing loadgen accuracy log...")
    with open(args.path, "r") as f:
        results = json.load(f)
    hstu_config = get_hstu_configs(dataset="sampled-streaming-100b")
    metrics = MetricsLogger(
        multitask_configs=hstu_config.multitask_configs,
        batch_size=1,
        window_size=3000,
        device=torch.device("cpu"),
        rank=0,
    )
    logger.warning(f"results have {len(results)} entries")
    for result in results:
        data = np.frombuffer(bytes.fromhex(result["data"]), np.float32)
        num_candidates = data[-1].astype(int)
        assert len(data) == 1 + num_candidates * 3
        mt_target_preds = torch.from_numpy(data[0:num_candidates])
        mt_target_labels = torch.from_numpy(data[num_candidates : num_candidates * 2])
        mt_target_weights = torch.from_numpy(
            data[num_candidates * 2 : num_candidates * 3]
        )
        num_candidates = torch.tensor([num_candidates])
        metrics.update(
            predictions=mt_target_preds.view(1, -1),
            labels=mt_target_labels.view(1, -1),
            weights=mt_target_weights.view(1, -1),
            num_candidates=num_candidates,
        )
    for k, v in metrics.compute().items():
        logger.warning(f"{k}: {v}")


if __name__ == "__main__":
    main()
