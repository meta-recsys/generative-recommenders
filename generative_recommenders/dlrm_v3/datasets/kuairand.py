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

# pyre-unsafe
import json
import time
from functools import partial
from typing import Any, Dict, List

import pandas as pd
import torch
from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.utils import (
    maybe_truncate_seq,
    separate_uih_candidates,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def process_and_hash_x(x: Any, hash_size: int) -> Any:
    if isinstance(x, str):
        x = json.loads(x)
    if isinstance(x, list):
        return [x_i % hash_size for x_i in x]
    else:
        return x % hash_size


class DLRMv3KuaiRandDataset(DLRMv3RandomDataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        embedding_config: Dict[str, Any],
        seq_logs_file: str,
        is_inference: bool,
        **kwargs,
    ) -> None:
        super().__init__(hstu_config=hstu_config, is_inference=is_inference)
        self.seq_logs_frame: pd.DataFrame = pd.read_csv(seq_logs_file, delimiter=",")
        # apply hashing from embedding table config
        for key, table in embedding_config.items():
            assert key in self.seq_logs_frame.columns, (
                "Rename key in embedding table configs!"
            )
            hash_size = table.num_embeddings
            self.seq_logs_frame[key] = self.seq_logs_frame[key].apply(
                partial(process_and_hash_x, hash_size=hash_size)
            )

    def get_item_count(self):
        return len(self.seq_logs_frame)

    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}

    def load_query_samples(self, sample_list):
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for idx in sample_list:
            data = self.seq_logs_frame.iloc[idx]
            if len(data.video_id) <= max_num_candidates:
                continue
            sample = self.load_item(data, max_num_candidates)
            self.items_in_memory[idx] = sample

        self.last_loaded = time.time()

    def load_item(self, data, max_num_candidates):
        with torch.profiler.record_function("load_item"):
            video_history_uih, video_history_candidates = separate_uih_candidates(
                data.video_id,
                candidates_max_seq_len=max_num_candidates,
            )
            action_weights_uih, action_weights_candidates = separate_uih_candidates(
                data.action_weights,
                candidates_max_seq_len=max_num_candidates,
            )
            timestamps_uih, _ = separate_uih_candidates(
                data.time_ms,
                candidates_max_seq_len=max_num_candidates,
            )
            watch_time_uih, watch_time_candidates = separate_uih_candidates(
                data.play_time_ms,
                candidates_max_seq_len=max_num_candidates,
            )

            video_history_uih = maybe_truncate_seq(video_history_uih, self._max_uih_len)
            action_weights_uih = maybe_truncate_seq(
                action_weights_uih, self._max_uih_len
            )
            timestamps_uih = maybe_truncate_seq(timestamps_uih, self._max_uih_len)
            watch_time_uih = maybe_truncate_seq(watch_time_uih, self._max_uih_len)

            uih_seq_len = len(video_history_uih)
            assert uih_seq_len == len(timestamps_uih), (
                "history len differs from timestamp len."
            )
            assert uih_seq_len == len(action_weights_uih), (
                "history len differs from weights len."
            )
            assert uih_seq_len == len(watch_time_uih), (
                "history len differs from watch time len."
            )

            uih_kjt_values: List[torch.Tensor] = []
            uih_kjt_lengths: List[torch.Tensor] = []
            for name, length in self._contextual_feature_to_max_length.items():
                uih_kjt_values.append(data[name])
                uih_kjt_lengths.append(length)

            uih_kjt_values.extend(
                video_history_uih + timestamps_uih + action_weights_uih + watch_time_uih
            )

            uih_kjt_lengths.extend(
                [
                    uih_seq_len
                    for _ in range(
                        len(self._uih_keys)
                        - len(self._contextual_feature_to_max_length)
                    )
                ]
            )

            dummy_query_time = max(timestamps_uih)
            uih_features_kjt = KeyedJaggedTensor(
                keys=self._uih_keys,
                lengths=torch.tensor(uih_kjt_lengths).long(),
                values=torch.tensor(uih_kjt_values).long(),
            )

            candidates_kjt_lengths = max_num_candidates * torch.ones(
                len(self._candidates_keys)
            )
            candidates_kjt_values = (
                video_history_candidates
                + action_weights_candidates
                + watch_time_candidates
                + [dummy_query_time] * max_num_candidates
            )
            candidates_features_kjt = KeyedJaggedTensor(
                keys=self._candidates_keys,
                lengths=torch.tensor(candidates_kjt_lengths).long(),
                values=torch.tensor(candidates_kjt_values).long(),
            )

        return uih_features_kjt, candidates_features_kjt
