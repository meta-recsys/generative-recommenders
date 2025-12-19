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
Data producer module for DLRMv3 inference.

This module provides classes for producing and managing query data during inference,
supporting both single-threaded and multi-threaded data production modes.
"""

import logging
import threading
import time
from queue import Queue
from typing import List, Optional, Tuple, Union

import torch
from generative_recommenders.dlrm_v3.datasets.dataset import Dataset, Samples

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("data_producer")


class QueryItem:
    """
    Container for a query item to be processed by the inference thread pool.

    Attributes:
        query_ids: List of unique identifiers for the queries in this batch.
        samples: The sample data containing features for the queries.
        start: Time when the query was first received.
        dt_queue: Time spent in the queue before processing.
        dt_batching: Time spent on batching the data.
    """

    def __init__(
        self,
        query_ids: List[int],
        samples: Samples,
        start: float,
        dt_queue: float,
        dt_batching: float,
    ) -> None:
        self.query_ids = query_ids
        self.samples = samples
        self.start: float = start
        self.dt_queue: float = dt_queue
        self.dt_batching: float = dt_batching


class SingleThreadDataProducer:
    """
    Single-threaded data producer for synchronous query processing.

    This producer processes queries on the main thread without any parallelism,
    suitable for debugging or low-throughput scenarios.

    Args:
        ds: The dataset to fetch samples from.
        run_one_item: Callback function to process a single QueryItem.
    """

    def __init__(self, ds: Dataset, run_one_item) -> None:  # pyre-ignore [2]
        self.ds = ds
        self.run_one_item = run_one_item  # pyre-ignore [4]

    def enqueue(
        self, query_ids: List[int], content_ids: List[int], t0: float, dt_queue: float
    ) -> None:
        """
        Enqueue queries for immediate synchronous processing.

        Args:
            query_ids: List of unique query identifiers.
            content_ids: List of content/sample identifiers to fetch.
            t0: Timestamp when the query batch was created.
            dt_queue: Time spent waiting in the queue.
        """
        with torch.profiler.record_function("data batching"):
            t0_batching: float = time.time()
            samples: Union[Samples, List[Samples]] = self.ds.get_samples(content_ids)
            dt_batching: float = time.time() - t0_batching
            if isinstance(samples, Samples):
                query = QueryItem(
                    query_ids=query_ids,
                    samples=samples,
                    start=t0,
                    dt_queue=dt_queue,
                    dt_batching=dt_batching,
                )
                self.run_one_item(query)
            else:
                start_idx = 0
                for sample in samples:
                    batch_size: int = sample.batch_size()
                    query = QueryItem(
                        query_ids=query_ids[start_idx : start_idx + batch_size],
                        samples=sample,
                        start=t0,
                        dt_queue=dt_queue,
                        dt_batching=dt_batching,
                    )
                    start_idx += batch_size
                    self.run_one_item(query)

    def finish(self) -> None:
        """Finalize the producer. No-op for single-threaded mode."""
        pass


class MultiThreadDataProducer:
    """
    Multi-threaded data producer for parallel query processing.

    Uses a thread pool to fetch and batch data in parallel with model inference,
    improving throughput for high-load scenarios.

    Args:
        ds: The dataset to fetch samples from.
        threads: Number of worker threads to use.
        run_one_item: Callback function to process a single QueryItem.
    """

    def __init__(
        self,
        ds: Dataset,
        threads: int,
        run_one_item,  # pyre-ignore [2]
    ) -> None:
        queue_size_multiplier = 4
        self.ds = ds
        self.threads = threads
        self.run_one_item = run_one_item  # pyre-ignore [4]
        self.tasks: Queue[Optional[Tuple[List[int], List[int], float, float]]] = Queue(
            maxsize=threads * queue_size_multiplier
        )
        self.workers: List[threading.Thread] = []
        for _ in range(self.threads):
            worker = threading.Thread(target=self.handle_tasks, args=(self.tasks,))
            worker.daemon = True
            self.workers.append(worker)
            worker.start()

    def handle_tasks(
        self, tasks_queue: Queue[Optional[Tuple[List[int], List[int], float, float]]]
    ) -> None:
        """
        Worker thread main loop to process tasks from the queue.

        Each worker maintains its own CUDA stream for parallel execution.

        Args:
            tasks_queue: Queue containing task tuples or None for termination.
        """
        stream = torch.cuda.Stream()
        while True:
            query_and_content_ids = tasks_queue.get()
            if query_and_content_ids is None:
                tasks_queue.task_done()
                break
            query_ids, content_ids, t0, dt_queue = query_and_content_ids
            t0_batching: float = time.time()
            samples: Union[Samples, List[Samples]] = self.ds.get_samples(content_ids)
            dt_batching: float = time.time() - t0_batching
            if isinstance(samples, Samples):
                qitem = QueryItem(
                    query_ids=query_ids,
                    samples=samples,
                    start=t0,
                    dt_queue=dt_queue,
                    dt_batching=dt_batching,
                )
                with torch.inference_mode(), torch.cuda.stream(stream):
                    self.run_one_item(qitem)
            else:
                start_idx = 0
                for sample in samples:
                    batch_size: int = sample.batch_size()
                    qitem = QueryItem(
                        query_ids=query_ids[start_idx : start_idx + batch_size],
                        samples=sample,
                        start=t0,
                        dt_queue=dt_queue,
                        dt_batching=dt_batching,
                    )
                    start_idx += batch_size
                    with torch.inference_mode(), torch.cuda.stream(stream):
                        self.run_one_item(qitem)
            tasks_queue.task_done()

    def enqueue(
        self, query_ids: List[int], content_ids: List[int], t0: float, dt_queue: float
    ) -> None:
        """
        Enqueue queries for asynchronous processing by worker threads.

        Args:
            query_ids: List of unique query identifiers.
            content_ids: List of content/sample identifiers to fetch.
            t0: Timestamp when the query batch was created.
            dt_queue: Time spent waiting in the queue.
        """
        with torch.profiler.record_function("data batching"):
            self.tasks.put((query_ids, content_ids, t0, dt_queue))

    def finish(self) -> None:
        """
        Signal all worker threads to terminate and wait for completion.

        Sends None to each worker to trigger graceful shutdown.
        """
        for _ in self.workers:
            self.tasks.put(None)
        for worker in self.workers:
            worker.join()
