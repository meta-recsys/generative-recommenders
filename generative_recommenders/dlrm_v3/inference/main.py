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
mlperf dlrm_v3 inference benchmarking tool.
"""

import argparse
import array
import json
import logging

logging.basicConfig(level=logging.INFO)
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

import gin

# pyre-ignore [21]
import mlperf_loadgen as lg  # @manual
import numpy as np
import torch
from generative_recommenders.common import set_dev_mode, set_verbose_level
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import Dataset
from generative_recommenders.dlrm_v3.inference.data_producer import (
    MultiThreadDataProducer,
    QueryItem,
    SingleThreadDataProducer,
)
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily
from generative_recommenders.dlrm_v3.utils import (
    get_dataset,
    MetricsLogger,
    profiler_or_nullcontext,
    SUPPORTED_DATASETS,
)


logger: logging.Logger = logging.getLogger("main")

torch.multiprocessing.set_start_method("spawn", force=True)

NANO_SEC = 1e9

MLPERF_CONF = f"{os.path.dirname(__file__)}/mlperf.conf"
USER_CONF = f"{os.path.dirname(__file__)}/user.conf"

SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-13b": "movielens_13b.gin",
}


SCENARIO_MAP = {  # pyre-ignore [5]
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_DATASETS, help="dataset"
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


class Runner:
    def __init__(
        self,
        model: HSTUModelFamily,
        ds: Dataset,
        data_producer_threads: int = 1,
        batchsize: int = 128,
        compute_eval: bool = False,
    ) -> None:
        self.model = model
        if data_producer_threads == 1:
            self.data_producer: Union[
                MultiThreadDataProducer, SingleThreadDataProducer
            ] = SingleThreadDataProducer(ds, self.run_one_item)
        else:
            self.data_producer = MultiThreadDataProducer(
                ds, data_producer_threads, self.run_one_item
            )
        self.batchsize = batchsize
        self.compute_eval = compute_eval
        self.result_timing: List[float] = []
        self.result_batches: List[int] = []
        self.current_query_ids: List[int] = []
        self.current_content_ids: List[int] = []
        self.metrics: Optional[MetricsLogger] = None
        if compute_eval:
            self.metrics = MetricsLogger(
                multitask_configs=model.hstu_config.multitask_configs,  # pyre-ignore [6]
                batch_size=batchsize,
                window_size=1000,
                device=torch.device("cpu"),
                rank=0,
            )

    def run_one_item(self, qitem: QueryItem) -> None:
        try:
            t0 = time.time()
            prediction_output = self.model.predict(qitem.samples)
            assert prediction_output is not None
            mt_target_preds, mt_target_labels, mt_target_weights = prediction_output
            if self.compute_eval:
                assert mt_target_labels is not None
                assert mt_target_weights is not None
                assert self.metrics is not None
                self.metrics.update(
                    predictions=mt_target_preds.t(),
                    labels=mt_target_labels.t(),
                    weights=mt_target_weights.t(),
                )
            self.result_timing.append(time.time() - t0)
            self.result_batches.append(len(qitem.query_ids))
        except Exception as ex:  # pylint: disable=broad-except
            logger.error("thread: failed, %s", ex)
        finally:
            response = []
            for query_id in qitem.query_ids:
                response_array = array.array(
                    "B", np.array([1]).astype(np.float32).tobytes()
                )
                bi = response_array.buffer_info()
                response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
            lg.QuerySamplesComplete(response)

    def enqueue(self, query_samples) -> None:  # pyre-ignore [2]
        self.current_query_ids.extend([q.id for q in query_samples])
        self.current_content_ids.extend([q.index for q in query_samples])
        if len(self.current_query_ids) >= self.batchsize:
            self.data_producer.enqueue(self.current_query_ids, self.current_content_ids)
            self.current_query_ids = []
            self.current_content_ids = []

    def finish(self) -> None:
        self.data_producer.finish()


def add_results(
    final_results: Dict[str, Any],
    result_timing: List[float],
    result_batches: List[int],
    metrics: Optional[MetricsLogger],
) -> None:
    percentiles = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    buckets = np.percentile(result_timing, percentiles).tolist()
    buckets_str = ",".join(
        ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)]
    )
    total_batches = sum(result_batches)

    final_results["good"] = len(result_timing)
    final_results["avg_time"] = np.mean(result_timing)
    final_results["percentiles"] = {str(k): v for k, v in zip(percentiles, buckets)}
    final_results["qps"] = total_batches / final_results["took"]
    final_results["count"] = total_batches

    if metrics is not None:
        for k, v in metrics.compute().items():
            print(f"{k}: {v}")

    print(
        "{} qps={:.2f}, avg_query_time={:.4f}, time={:.3f}, queries={}, tiles={}".format(
            final_results["scenario"],
            final_results["qps"],
            final_results["avg_time"],
            final_results["took"],
            len(result_timing),
            buckets_str,
        )
    )


@gin.configurable
def run(
    dataset: str = "debug",
    model_path: str = "",
    scenario_name: str = "Server",
    batchsize: int = 16,
    out_dir: str = "",
    output_trace: bool = False,
    data_producer_threads: int = 4,
    compute_eval: bool = False,
    find_peak_performance: bool = False,
    new_path_prefix: str = "",
    train_split_percentage: float = 0.75,
    # below will override mlperf rules compliant settings - don't use for official submission
    duration: Optional[int] = None,
    target_qps: Optional[int] = None,
    max_latency: Optional[float] = None,
    num_queries: Optional[int] = None,
    samples_per_query_multistream: int = 8,
    max_num_samples: int = 2048,
    numpy_rand_seed: int = 123,
    dev_mode: bool = False,
    dataset_percentage: float = 1.0,
) -> None:
    set_dev_mode(dev_mode)
    if scenario_name not in SCENARIO_MAP:
        raise NotImplementedError("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    np.random.seed(numpy_rand_seed)

    hstu_config = get_hstu_configs(dataset)
    table_config = get_embedding_table_config(dataset)
    set_is_inference(is_inference=not compute_eval)

    model_family = HSTUModelFamily(
        hstu_config=hstu_config,
        table_config=table_config,
        output_trace=output_trace,
    )
    dataset, kwargs = get_dataset(dataset, new_path_prefix)

    ds: Dataset = dataset(
        hstu_config=hstu_config,
        embedding_config=table_config,
        is_inference=not compute_eval,
        **kwargs,
    )
    model_family.load(model_path)

    mlperf_conf = os.path.abspath(MLPERF_CONF)
    if not os.path.exists(mlperf_conf):
        logger.error("{} not found".format(mlperf_conf))
        sys.exit(1)
    user_conf = os.path.abspath(USER_CONF)
    if not os.path.exists(user_conf):
        logger.error("{} not found".format(user_conf))
        sys.exit(1)

    if out_dir:
        output_dir = os.path.abspath(out_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    scenario = SCENARIO_MAP[scenario_name]
    if scenario != lg.TestScenario.Server or compute_eval:
        batchsize = 1

    # warmup
    warmup_ids = list(range(batchsize))
    ds.load_query_samples(warmup_ids)
    for _ in range(5 * int(os.environ.get("WORLD_SIZE", 1))):
        sample = ds.get_samples(warmup_ids)
        _ = model_family.predict(sample)
    ds.unload_query_samples(None)
    for h in logger.handlers:
        h.flush()
    logger.info("warmup done")

    count = int(ds.get_item_count() * dataset_percentage)
    train_size: int = round(train_split_percentage * count)

    settings = lg.TestSettings()
    settings.FromConfig(mlperf_conf, model_path, scenario_name)
    settings.FromConfig(user_conf, model_path, scenario_name)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly

    if compute_eval:
        settings.mode = lg.TestMode.AccuracyOnly
        count = count - train_size
        data_producer_threads = 1  # during eval, using multiple threads can have concurrency issue due to Triton autotune.

    runner: Runner = Runner(
        model_family,
        ds,
        data_producer_threads=data_producer_threads,
        batchsize=batchsize,
        compute_eval=compute_eval,
    )

    def issue_queries(query_samples) -> None:  # pyre-ignore [2]
        if compute_eval:
            for sample in query_samples:
                sample.index = sample.index + train_size
        runner.enqueue(query_samples)

    def load_query_samples(query_ids: List[int]) -> None:
        if compute_eval:
            query_ids = [q + train_size for q in query_ids]
        ds.load_query_samples(query_ids)

    def flush_queries() -> None:
        pass

    if find_peak_performance:
        settings.mode = lg.TestMode.FindPeakPerformance

    if duration:
        settings.min_duration_ms = duration
        settings.max_duration_ms = duration

    if target_qps:
        settings.server_target_qps = float(target_qps)
        settings.offline_expected_qps = float(target_qps)

    if num_queries:
        queries = math.ceil(num_queries / batchsize) * batchsize
        settings.min_query_count = queries
        settings.max_query_count = queries

    if samples_per_query_multistream:
        settings.multi_stream_samples_per_query = samples_per_query_multistream

    if max_latency:
        settings.server_target_latency_ns = int(max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(max_latency * NANO_SEC)

    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        count,
        min(count, max_num_samples),
        load_query_samples,
        ds.unload_query_samples,
    )

    final_results = {
        "runtime": model_family.name(),
        "version": model_family.version(),
        "time": int(time.time()),
        "scenario": str(scenario),
    }

    with profiler_or_nullcontext(enabled=output_trace, with_stack=False):
        logger.info(f"starting {scenario}")
        lg.StartTest(sut, qsl, settings)
        runner.finish()
        final_results["took"] = time.time() - ds.last_loaded
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)

    add_results(
        final_results,
        runner.result_timing,
        runner.result_batches,
        runner.metrics,
    )
    # If multiple subprocesses are running the model send a signal to stop them
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        model_family.predict(None)

    if out_dir:
        with open("results.json", "w") as f:
            json.dump(final_results, f, sort_keys=True, indent=4)


def main() -> None:
    set_verbose_level(1)
    args = get_args()
    logger.info(args)
    gin_path = f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[args.dataset]}"
    gin.parse_config_file(gin_path)
    run(dataset=args.dataset)


if __name__ == "__main__":
    main()
