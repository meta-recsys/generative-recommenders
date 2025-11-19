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
import csv
import logging
import sys
import time
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.utils import (
    json_loads,
    maybe_truncate_seq,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

csv.field_size_limit(sys.maxsize)
logger: logging.Logger = logging.getLogger(__name__)


class DLRMv3SyntheticStreamingDataset(DLRMv3RandomDataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        ratings_file_prefix: str,
        is_inference: bool,
        train_ts: int,
        total_ts: int,
        num_files: int,
        num_users: int,
        num_items: int,
        num_categories: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(hstu_config=hstu_config, is_inference=is_inference)
        self.ratings_file_prefix = ratings_file_prefix
        self.file_to_offsets: Dict[int, List[int]] = {}
        with open(f"{self.ratings_file_prefix}offset.csv", "r") as file:
            reader = csv.reader(file)
            for size in range(num_files):
                row = next(reader)
                assert len(row) == 1
                offset = json_loads(row[0])
                assert len(offset) == num_users // num_files
                self.file_to_offsets[size] = offset
        self.ts_requests_offsets: List[int] = []
        with open(f"{self.ratings_file_prefix}requests_per_ts_offset.csv", "r") as file:
            reader = csv.reader(file)
            row = next(reader)
            assert len(row) == 1
            self.ts_requests_offsets = json_loads(row[0])
            assert len(self.ts_requests_offsets) == total_ts
        self.requests: List[int] = []
        self.ts_to_users_cumsum: Dict[int, List[int]] = {}
        with open(
            f"{self.ratings_file_prefix}users_cumsum_per_ts.csv", "r"
        ) as cumsum_file:
            reader = csv.reader(cumsum_file)
            ts = 0
            for row in reader:
                assert len(row) == 1
                cumsum = json_loads(row[0])
                self.ts_to_users_cumsum[ts] = cumsum
                ts += 1
        self.train_ts = train_ts
        self.total_ts = total_ts
        self.num_files = num_files
        self.ts: int = -1
        self.is_inference: bool = False
        self.is_eval: bool = False
        self.users_per_file: int = num_users // num_files
        self.cached_files: Set[str] = set()
        self.items_per_category: int = num_items // num_categories
        assert hstu_config.action_weights is not None
        self.action_weights: List[int] = hstu_config.action_weights
        self.items_in_memory: Dict[
            int, Dict[int, Tuple[KeyedJaggedTensor, KeyedJaggedTensor]]
        ] = {}

    def get_item_count(self) -> int:
        return len(self.requests)

    def load_query_samples(self, sample_list: List[int]) -> None:
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        for idx in sample_list:
            data = self.iloc(idx)
            sample = self.load_item(data, max_num_candidates)
            if self.ts not in self.items_in_memory:
                self.items_in_memory[self.ts] = {}
            self.items_in_memory[self.ts][idx] = sample

        self.last_loaded = time.time()

    def unload_query_samples(self, sample_list: List[int]) -> None:
        self.items_in_memory = {}

    def get_sample(self, id: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        return self.items_in_memory[self.ts][id]

    def _process_line(self, line: str, user_id: int) -> pd.Series:
        reader = csv.reader([line])
        parsed_line = next(reader)
        # total ts + one more eval ts + one base ts so that uih won't be zero
        # for each ts, ordered as candidate_ids, candidate_ratings, uih_ids, uih_ratings
        assert len(parsed_line) == 4 * (self.total_ts + 2)
        uih_item_ids_list = []
        uih_ratings_list = []
        candidate_item_ids = ""
        candidate_ratings = ""
        if (not self.is_eval) and (not self.is_inference):
            assert self.ts < self.train_ts
            for i in range(self.ts + 1):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            candidate_item_ids = parsed_line[4 * (self.ts + 1)]
            candidate_ratings = parsed_line[1 + 4 * (self.ts + 1)]
        elif self.is_eval:
            for i in range(self.ts + 1):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            candidate_item_ids = parsed_line[4 * (self.ts + 1)]
            candidate_ratings = parsed_line[1 + 4 * (self.ts + 1)]
        else:
            assert self.is_inference is True
            assert self.ts >= self.train_ts
            for i in range(self.train_ts + 1):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            for i in range(self.train_ts + 2, self.ts + 2):
                if parsed_line[4 * i]:
                    uih_item_ids_list.append(parsed_line[2 + 4 * i])
                    uih_ratings_list.append(parsed_line[3 + 4 * i])
            candidate_item_ids = parsed_line[4 * (self.ts + 2)]
            candidate_ratings = parsed_line[1 + 4 * (self.ts + 2)]
        uih_item_ids = ",".join(uih_item_ids_list)
        uih_ratings = ",".join(uih_ratings_list)
        assert candidate_item_ids != "" and candidate_ratings != ""
        return pd.Series(
            data={
                "user_id": user_id,
                "uih_item_ids": uih_item_ids,
                "uih_ratings": uih_ratings,
                "candidate_item_ids": candidate_item_ids,
                "candidate_ratings": candidate_ratings,
            }
        )

    def iloc(self, idx: int) -> pd.Series:
        cumsum: List[int] = self.ts_to_users_cumsum[self.ts]
        assert cumsum != []
        assert idx < cumsum[-1]
        file_idx: int = 0
        while cumsum[file_idx] <= idx:
            file_idx += 1
        user_idx = self.requests[idx]
        filename = f"{self.ratings_file_prefix}{file_idx}.csv"
        with open(filename, "r") as file:
            idx = user_idx % self.users_per_file
            file.seek(self.file_to_offsets[file_idx][idx])
            line = file.readline()
        data = self._process_line(line=line, user_id=user_idx)
        return data

    def get_timestamp_uih(
        self, data: pd.Series, max_num_candidates: int, size: int
    ) -> List[int]:
        return [1] * size

    def set_ts(self, ts: int) -> None:
        logger.warning(f"Streaming dataset ts set to {ts}")
        if ts == self.ts:
            return
        self.ts = ts
        with open(
            f"{self.ratings_file_prefix}requests_per_ts.csv", "r"
        ) as request_file:
            request_file.seek(self.ts_requests_offsets[self.ts])
            line = request_file.readline()
            reader = csv.reader([line])
            row = next(reader)
            assert len(row) == 1
            requests = json_loads(row[0])
            self.requests = requests
            logger.warning(f"DLRMv3SyntheticStreamingDataset: ts={ts} requests loaded")
        assert self.ts_to_users_cumsum[self.ts][-1] == len(self.requests)
        logger.warning(
            f"DLRMv3SyntheticStreamingDataset: ts={ts} users_cumsum={self.ts_to_users_cumsum[self.ts]}"
        )

    def load_item(
        self, data: pd.Series, max_num_candidates: int
    ) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        ids_uih = json_loads(data.uih_item_ids)
        ids_candidates = json_loads(data.candidate_item_ids)
        ratings_uih = json_loads(data.uih_ratings)
        ratings_candidates = json_loads(data.candidate_ratings)
        timestamps_uih = self.get_timestamp_uih(
            data=data,
            max_num_candidates=max_num_candidates,
            size=len(ids_uih),
        )
        assert len(ids_uih) == len(
            timestamps_uih
        ), "history len differs from timestamp len."
        assert len(ids_uih) == len(
            ratings_uih
        ), f"history len {len(ids_uih)} differs from ratings len {len(ratings_uih)}."
        assert (
            len(ids_candidates) == len(ratings_candidates)
        ), f"candidates len {len(ids_candidates)} differs from ratings len {len(ratings_candidates)}."

        ids_uih = maybe_truncate_seq(ids_uih, self._max_uih_len)
        ratings_uih = maybe_truncate_seq(ratings_uih, self._max_uih_len)
        timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)
        ids_candidates = maybe_truncate_seq(ids_candidates, max_num_candidates)
        num_candidates = len(ids_candidates)
        ratings_candidates = maybe_truncate_seq(ratings_candidates, max_num_candidates)
        action_weights_uih = [
            self.action_weights[int(rating) - 1] for rating in ratings_uih
        ]
        action_weights_candidates = [
            int(rating >= 3.5) for rating in ratings_candidates
        ]

        uih_kjt_values: List[int] = []
        uih_kjt_lengths: List[int] = []
        for name, length in self._contextual_feature_to_max_length.items():
            uih_kjt_values.append(data[name])
            uih_kjt_lengths.append(length)

        uih_seq_len = len(ids_uih)
        dummy_watch_times_uih = [0 for _ in range(uih_seq_len)]
        item_category_ids = [id // self.items_per_category for id in ids_uih]
        extend_uih_kjt_values: List[int] = (
            ids_uih
            + ratings_uih
            + timestamps_uih
            + action_weights_uih
            + dummy_watch_times_uih
            + item_category_ids
        )
        uih_kjt_values.extend(extend_uih_kjt_values)
        uih_kjt_lengths.extend(
            [
                uih_seq_len
                for _ in range(
                    len(self._uih_keys) - len(self._contextual_feature_to_max_length)
                )
            ]
        )

        dummy_query_time = 0 if timestamps_uih == [] else max(timestamps_uih)
        uih_kjt_values.append(dummy_query_time)
        uih_kjt_lengths.append(1)
        uih_features_kjt: KeyedJaggedTensor = KeyedJaggedTensor(
            keys=self._uih_keys + ["dummy_query_time"],
            lengths=torch.tensor(uih_kjt_lengths).long(),
            values=torch.tensor(uih_kjt_values).long(),
        )

        candidates_kjt_lengths = num_candidates * torch.ones(len(self._candidates_keys))
        item_candidate_category_ids = [
            id // self.items_per_category for id in ids_candidates
        ]
        candidates_kjt_values = (
            ids_candidates
            + ratings_candidates
            + [dummy_query_time] * num_candidates  # item_query_time
            + action_weights_candidates
            + [1] * num_candidates  # item_dummy_watchtime
            + item_candidate_category_ids
        )
        candidates_features_kjt: KeyedJaggedTensor = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=candidates_kjt_lengths.detach().clone().long(),
            values=torch.tensor(candidates_kjt_values).long(),
        )
        return uih_features_kjt, candidates_features_kjt
