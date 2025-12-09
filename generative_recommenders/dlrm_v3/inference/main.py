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
import random

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
from generative_recommenders.dlrm_v3.datasets.dataset import Dataset, Samples
from generative_recommenders.dlrm_v3.datasets.synthetic_streaming import (
    DLRMv3SyntheticStreamingDataset,
)
from generative_recommenders.dlrm_v3.inference.data_producer import (
    MultiThreadDataProducer,
    QueryItem,
    SingleThreadDataProducer,
)
from generative_recommenders.dlrm_v3.inference.inference_modules import set_is_inference
from generative_recommenders.dlrm_v3.inference.model_family import HSTUModelFamily
from generative_recommenders.dlrm_v3.utils import (
    get_dataset,
    profiler_or_nullcontext,
    SUPPORTED_DATASETS,
)


logger: logging.Logger = logging.getLogger("main")

torch.multiprocessing.set_start_method("spawn", force=True)

NANO_SEC = 1e9

USER_CONF = f"{os.path.dirname(__file__)}/user.conf"

SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-13b": "movielens_13b.gin",
    "streaming-400m": "streaming_400m.gin",
    "sampled-streaming-100b": "streaming_100b.gin",
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
        self.init_states()

    def init_states(self) -> None:
        self.result_timing: List[Dict[str, float]] = []
        self.result_batches: List[int] = []
        self.current_query_ids: List[int] = []
        self.current_content_ids: List[int] = []
        self.current_t0: List[float] = []

    def run_one_item(self, qitem: QueryItem) -> None:
        try:
            t0_prediction: float = time.time()
            prediction_output = self.model.predict(qitem.samples)
            dt_prediction: float = time.time() - t0_prediction
            assert prediction_output is not None
            (
                mt_target_preds,
                mt_target_labels,
                mt_target_weights,
                dt_sparse,
                dt_dense,
            ) = prediction_output
            if self.compute_eval:
                assert mt_target_labels is not None
                assert mt_target_weights is not None
            self.result_timing.append(
                {
                    "total": time.time() - qitem.start,
                    "prediction": dt_prediction,
                    "queue": qitem.dt_queue,
                    "batching": qitem.dt_batching,
                    "sparse": dt_sparse,
                    "dense": dt_dense,
                }
            )
            self.result_batches.append(len(qitem.query_ids))
        except Exception as ex:  # pylint: disable=broad-except
            logger.error("thread: failed, %s", ex)
        finally:
            candidate_size = mt_target_preds.size(1) // len(qitem.query_ids)
            if not self.compute_eval:
                for i, query_id in enumerate(qitem.query_ids):
                    query_mt_target_preds = (
                        mt_target_preds[  # pyre-ignore [61]
                            0,
                            candidate_size * i : candidate_size * (i + 1),
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    response_array = array.array("B", query_mt_target_preds.tobytes())
                    bi = response_array.buffer_info()
                    # since we send buffer to loadgen, needs `response_array` in memory during send
                    lg.QuerySamplesComplete(
                        [lg.QuerySampleResponse(query_id, bi[0], bi[1])]
                    )
            else:
                for i, query_id in enumerate(qitem.query_ids):
                    query_mt_target_preds = (
                        mt_target_preds[  # pyre-ignore [61]
                            0, candidate_size * i : candidate_size * (i + 1)
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    query_mt_target_labels = (
                        mt_target_labels[  # pyre-ignore [16,61]
                            0, candidate_size * i : candidate_size * (i + 1)
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    query_mt_target_weights = (
                        mt_target_weights[  # pyre-ignore [61]
                            0, candidate_size * i : candidate_size * (i + 1)
                        ]
                        .view(-1)
                        .float()
                        .numpy()
                    )
                    np_array = np.concatenate(
                        [
                            query_mt_target_preds,
                            query_mt_target_labels,
                            query_mt_target_weights,
                            np.array([candidate_size]).astype(np.float32),
                        ]
                    )
                    response_array = array.array("B", np_array.tobytes())
                    bi = response_array.buffer_info()
                    # since we send buffer to loadgen, needs `response_array` in memory during send
                    lg.QuerySamplesComplete(
                        [lg.QuerySampleResponse(query_id, bi[0], bi[1])]
                    )

    def enqueue(self, query_samples, t0: float) -> None:  # pyre-ignore [2]
        self.current_query_ids.extend([q.id for q in query_samples])
        self.current_content_ids.extend([q.index for q in query_samples])
        self.current_t0.append(t0)
        if len(self.current_query_ids) >= self.batchsize:
            self.data_producer.enqueue(
                query_ids=self.current_query_ids,
                content_ids=self.current_content_ids,
                t0=min(self.current_t0),
                dt_queue=max(self.current_t0) - min(self.current_t0),
            )
            self.current_query_ids = []
            self.current_content_ids = []
            self.current_t0 = []

    def finish(self) -> None:
        self.data_producer.finish()


def add_results(
    final_results: Dict[str, Any],
    result_timing: List[Dict[str, float]],
    result_batches: List[int],
) -> None:
    percentiles: list[float] = [50.0, 80.0, 90.0, 95.0, 99.0, 99.9]
    total_timing: list[float] = [result["total"] for result in result_timing]
    buckets = np.percentile(total_timing, percentiles).tolist()
    buckets_str: str = ",".join(
        ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)]
    )
    total_batches = sum(result_batches)

    final_results["good"] = len(total_timing)
    final_results["avg_time"] = np.mean(total_timing)
    final_results["percentiles"] = {str(k): v for k, v in zip(percentiles, buckets)}
    final_results["qps"] = total_batches / final_results["took"]
    final_results["count"] = total_batches

    for i, timing in enumerate(result_timing):
        logger.warning(f"timing of {i}: {timing}")

    logger.warning(
        "{} qps={:.2f}, avg_query_time={:.4f}, time={:.3f}, queries={}, tiles={}".format(
            final_results["scenario"],
            final_results["qps"],
            final_results["avg_time"],
            final_results["took"],
            len(result_timing),
            buckets_str,
        )
    )


def get_num_queries(input_size: Optional[int], one_pass_size: int) -> int:
    if input_size is None:
        return one_pass_size
    return math.ceil(input_size / one_pass_size) * one_pass_size


class StreamingQuerySampler:
    """
    Sampler for streaming dataset
    The execution order is determined by `StreamingQuerySampler.run_order`, not by the QSL or input query ID.
    This ensures that queries are executed according to their timestamp constraints.
    """

    def __init__(
        self,
        ds: DLRMv3SyntheticStreamingDataset,
        batchsize: int,
        dataset_percentage: float,
        input_queries: Optional[int] = None,
    ) -> None:
        self.ds: DLRMv3SyntheticStreamingDataset = ds
        self.ds.is_inference = True
        self.batchsize = batchsize
        self.inference_ts: int = self.ds.total_ts - self.ds.train_ts
        self.start_ts: int = self.ds.train_ts
        self.dataset_percentage: float = dataset_percentage
        self.num_requests: List[int] = self.get_num_requests(warmup_ratio=1.0)
        self.num_requests_cumsum: List[int] = np.cumsum(self.num_requests).tolist()
        self.total_requests: int = sum(self.num_requests)
        self.run_order: List[List[int]] = self.build_random_exec_order()
        self.ts: int = self.start_ts
        self.cnt: int = 0
        self.last_loaded: float = -1.0
        self.num_repeats: int = (
            get_num_queries(input_queries, self.total_requests) // self.total_requests
        )
        self.repeat: int = 0

    def get_num_requests(self, warmup_ratio: float) -> List[int]:
        return [
            int(
                (
                    self.ds.ts_to_users_cumsum[t][-1]
                    * self.dataset_percentage
                    * warmup_ratio
                )
                // self.batchsize
                * self.batchsize
            )
            for t in range(self.start_ts, self.start_ts + self.inference_ts)
        ]

    def build_random_exec_order(self) -> List[List[int]]:
        order = []
        for req_size in self.num_requests:
            within_ts_order = list(range(req_size))
            random.shuffle(within_ts_order)
            order.append(within_ts_order)
        return order

    def init_sut(self) -> None:
        self.ts = self.start_ts
        self.ds.set_ts(self.start_ts)
        self.cnt = 0

    def load_query_samples(self, query_ids: List[Optional[int]]) -> None:
        length = len(query_ids)
        ts_idx: int = 0
        while self.num_requests_cumsum[ts_idx] < length:
            ts_idx += 1
        for i in range(0, ts_idx):
            self.ds.set_ts(i + self.start_ts)
            self.ds.load_query_samples(self.run_order[i])
        self.ds.set_ts(ts_idx + self.start_ts)
        delta_length = (
            length if ts_idx == 0 else length - self.num_requests_cumsum[ts_idx - 1]
        )
        self.ds.load_query_samples(self.run_order[ts_idx][:delta_length])
        self.init_sut()
        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list: List[int]) -> None:
        self.ds.unload_query_samples(sample_list)

    def get_samples(self, id_list: List[int]) -> Samples:
        batch_size: int = len(id_list)
        ts_idx: int = 0
        while self.num_requests_cumsum[ts_idx] <= self.cnt:
            ts_idx += 1
        offset: int = 0 if ts_idx == 0 else self.num_requests_cumsum[ts_idx - 1]
        output: Samples = self.ds.get_samples_with_ts(
            self.run_order[ts_idx][self.cnt - offset : self.cnt + batch_size - offset],
            ts_idx + self.start_ts,
        )
        self.repeat += 1
        if self.repeat == self.num_repeats:
            self.repeat = 0
            self.cnt += batch_size
        return output

    def get_item_count(self) -> int:
        return self.total_requests


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
    warmup_ratio: float = 0.1,
    # below will override mlperf rules compliant settings - don't use for official submission
    duration: Optional[int] = None,
    target_qps: Optional[int] = None,
    max_latency: Optional[float] = None,
    num_queries: Optional[int] = None,
    samples_per_query_multistream: int = 8,
    max_num_samples: int = -1,
    numpy_rand_seed: int = 123,
    dev_mode: bool = False,
    sparse_quant: bool = False,
    dataset_percentage: float = 1.0,
) -> None:
    set_dev_mode(dev_mode)
    if scenario_name not in SCENARIO_MAP:
        raise NotImplementedError("valid scanarios:" + str(list(SCENARIO_MAP.keys())))
    scenario = SCENARIO_MAP[scenario_name]
    np.random.seed(numpy_rand_seed)

    hstu_config = get_hstu_configs(dataset)
    hstu_config.max_num_candidates = hstu_config.max_num_candidates_inference
    table_config = get_embedding_table_config(dataset)
    set_is_inference(is_inference=not compute_eval)

    model_family = HSTUModelFamily(
        hstu_config=hstu_config,
        table_config=table_config,
        sparse_quant=sparse_quant,
        output_trace=output_trace,
        compute_eval=compute_eval,
    )
    is_streaming: bool = "streaming" in dataset
    dataset, kwargs = get_dataset(dataset, new_path_prefix)

    ds: Dataset = dataset(
        hstu_config=hstu_config,
        embedding_config=table_config,
        is_inference=not compute_eval,
        **kwargs,
    )
    if is_streaming:
        ds = StreamingQuerySampler(  # pyre-ignore
            ds=ds,  # pyre-ignore [6]
            batchsize=batchsize,
            dataset_percentage=dataset_percentage,
            input_queries=num_queries,
        )
    model_family.load(model_path)

    user_conf = os.path.abspath(USER_CONF)
    if not os.path.exists(user_conf):
        logger.error("{} not found".format(user_conf))
        sys.exit(1)

    if out_dir:
        output_dir = os.path.abspath(out_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    # warmup
    warmup_ids = list(range(batchsize))
    ds.load_query_samples(warmup_ids)
    for _ in range(20 * int(os.environ.get("WORLD_SIZE", 1))):
        if is_streaming:
            ds.init_sut()  # pyre-ignore [16]
        sample = ds.get_samples(warmup_ids)
        _ = model_family.predict(sample)
    ds.unload_query_samples(None)
    for h in logger.handlers:
        h.flush()
    logger.info("Model forward warmup done")

    count = int(
        ds.get_item_count() * dataset_percentage
        if not is_streaming
        else ds.get_item_count()
    )
    train_size: int = round(train_split_percentage * count) if not is_streaming else 0

    settings = lg.TestSettings()
    settings.FromConfig(user_conf, model_path, scenario_name)
    settings.scenario = scenario
    settings.mode = lg.TestMode.PerformanceOnly

    if compute_eval:
        settings.mode = lg.TestMode.AccuracyOnly
        count = count - train_size

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
        runner.enqueue(query_samples, time.time())

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

    if samples_per_query_multistream:
        settings.multi_stream_samples_per_query = samples_per_query_multistream

    if max_latency:
        settings.server_target_latency_ns = int(max_latency * NANO_SEC)
        settings.multi_stream_expected_latency_ns = int(max_latency * NANO_SEC)

    # inference benchmark warmup
    if is_streaming:
        ds.init_sut()
        warmup_count: int = sum(
            ds.get_num_requests(warmup_ratio=warmup_ratio)  # pyre-ignore [16]
        )
    else:
        warmup_count: int = int(count * warmup_ratio)
    final_results = {
        "runtime": model_family.name(),
        "version": model_family.version(),
        "time": int(time.time()),
        "scenario": str(scenario),
    }
    settings.min_query_count = warmup_count
    settings.max_query_count = warmup_count
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        warmup_count,
        warmup_count,
        load_query_samples,
        ds.unload_query_samples,
    )
    with profiler_or_nullcontext(enabled=output_trace, with_stack=False):
        logger.info(f"starting warmup {scenario} with {warmup_count} queries")
        lg.StartTest(sut, qsl, settings)
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)

    # official run
    if is_streaming:
        ds.init_sut()
    runner.init_states()
    final_results = {
        "runtime": model_family.name(),
        "version": model_family.version(),
        "time": int(time.time()),
        "scenario": str(scenario),
    }
    query_size: int = get_num_queries(num_queries, count)
    settings.min_query_count = query_size
    settings.max_query_count = query_size
    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(
        count,
        min(count, max_num_samples) if max_num_samples > 0 else count,
        load_query_samples,
        ds.unload_query_samples,
    )
    with profiler_or_nullcontext(enabled=output_trace, with_stack=False):
        logger.info(
            f"starting {scenario} with {query_size} queries and {query_size // count} repeats"
        )
        lg.StartTest(sut, qsl, settings)
        runner.finish()
        final_results["took"] = time.time() - ds.last_loaded
        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)

    add_results(
        final_results,
        runner.result_timing,
        runner.result_batches,
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
