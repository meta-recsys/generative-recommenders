# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generative_recommenders.dlrm_v3.inference.main import (
    get_args as get_inference_args,
    get_num_queries,
    write_results,
)
from generative_recommenders.dlrm_v3.train.train_ranker import (
    get_args as get_training_args,
)


class BenchmarkCLITest(unittest.TestCase):
    def test_explicit_query_count_overrides_offline_duration(self) -> None:
        self.assertEqual(
            get_num_queries(
                input_size=32,
                one_pass_size=10_000,
                scenario_name="Offline",
                offline_target_qps=2_000,
                target_duration=600_000,
            ),
            32,
        )

    def test_training_accepts_public_dataset_and_checkpoint_paths(self) -> None:
        argv = [
            "train_ranker",
            "--dataset",
            "kuairand-1k",
            "--hstu-ultra-fp8",
            "--dataset-path-prefix",
            "/datasets",
            "--checkpoint-path",
            "/checkpoints",
            "--checkpoint-frequency",
            "50",
        ]

        with patch.object(sys, "argv", argv):
            args = get_training_args()

        self.assertEqual(args.dataset, "kuairand-1k")
        self.assertTrue(args.hstu_ultra_fp8)
        self.assertEqual(args.dataset_path_prefix, "/datasets")
        self.assertEqual(args.checkpoint_path, "/checkpoints")
        self.assertEqual(args.checkpoint_frequency, 50)

    def test_inference_accepts_portable_benchmark_overrides(self) -> None:
        argv = [
            "inference",
            "--dataset",
            "kuairand-1k",
            "--hstu-ultra-fp8",
            "--model-path",
            "/checkpoints/50",
            "--dataset-path-prefix",
            "/datasets",
            "--scenario",
            "Offline",
            "--batch-size",
            "8",
            "--target-qps",
            "100",
            "--num-queries",
            "500",
            "--warmup-ratio",
            "0.01",
            "--data-producer-threads",
            "2",
            "--results-path",
            "/results/fp8.json",
        ]

        with patch.object(sys, "argv", argv):
            args = get_inference_args()

        self.assertEqual(args.model_path, "/checkpoints/50")
        self.assertEqual(args.dataset_path_prefix, "/datasets")
        self.assertEqual(args.scenario, "Offline")
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.target_qps, 100)
        self.assertEqual(args.num_queries, 500)
        self.assertEqual(args.warmup_ratio, 0.01)
        self.assertEqual(args.data_producer_threads, 2)
        self.assertEqual(args.results_path, "/results/fp8.json")

    def test_write_results_creates_machine_readable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "nested" / "results.json"

            write_results(
                {"qps": 17.5, "peak_gpu_memory_bytes": 1024}, str(results_path)
            )

            self.assertEqual(
                json.loads(results_path.read_text(encoding="utf-8")),
                {"peak_gpu_memory_bytes": 1024, "qps": 17.5},
            )
