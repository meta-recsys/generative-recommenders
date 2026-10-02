# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch
from generative_recommenders.dlrm_v3.inference.data_producer import QueryItem
from generative_recommenders.dlrm_v3.inference.main import Runner


class _ConcurrentModel:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    def predict(self, _samples):  # pyrefly: ignore [missing-parameter-annotation]
        with self._lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        time.sleep(0.05)
        with self._lock:
            self.active_calls -= 1
        return torch.ones((1, 1)), None, None, 0.0, 0.0


class RunnerTest(unittest.TestCase):
    @patch("generative_recommenders.dlrm_v3.inference.main.lg.QuerySamplesComplete")
    def test_concurrent_requests_serialize_model_prediction(
        self, query_samples_complete: MagicMock
    ) -> None:
        model = _ConcurrentModel()
        runner = Runner(
            model=model,  # pyrefly: ignore [bad-argument-type]
            ds=MagicMock(),
            num_queries=2,
            data_producer_threads=1,
            batchsize=1,
        )
        requests = [
            QueryItem([query_id], MagicMock(), time.time(), 0.0, 0.0)
            for query_id in range(2)
        ]
        workers = [
            threading.Thread(target=runner.run_one_item, args=(request,))
            for request in requests
        ]

        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()

        self.assertEqual(model.max_active_calls, 1)
        self.assertEqual(len(runner.result_timing), 2)
        self.assertEqual(query_samples_complete.call_count, 2)
