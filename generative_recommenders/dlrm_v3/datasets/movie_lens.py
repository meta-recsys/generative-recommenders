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
import time
from typing import List, Optional

import pandas as pd
import torch

from generative_recommenders.dlrm_v3.datasets.dataset import DLRMv3RandomDataset
from generative_recommenders.dlrm_v3.datasets.utils import (
    maybe_truncate_seq,
    separate_uih_candidates,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class DLRMv3MovieLensDataset(DLRMv3RandomDataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        ratings_file: str,
        is_inference: bool,
        *args,
        **kwargs,
    ):
        super().__init__(hstu_config=hstu_config, is_inference=is_inference)
        self.ratings_frame: Optional[pd.DataFrame] = None
        if ratings_file != "":
            self.ratings_frame = pd.read_csv(
                ratings_file,
                delimiter=",",
            )

    def get_item_count(self):
        assert self.ratings_frame is not None
        return len(self.ratings_frame)

    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}

    def iloc(self, idx):
        assert self.ratings_frame is not None
        return self.ratings_frame.iloc[idx]

    def load_query_samples(self, sample_list):
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for idx in sample_list:
            data = self.iloc(idx)
            if len(data.sequence_item_ids) <= max_num_candidates:
                continue
            sample = self.load_item(data, max_num_candidates)
            self.items_in_memory[idx] = sample

        self.last_loaded = time.time()

    def get_timestamp_uih(self, data, max_num_candidates, size):
        movie_timestamps_uih, _ = separate_uih_candidates(
            data.sequence_timestamps,
            candidates_max_seq_len=max_num_candidates,
        )
        return movie_timestamps_uih

    def load_item(self, data, max_num_candidates):
        movie_history_uih, movie_history_candidates = separate_uih_candidates(
            data.sequence_item_ids,
            candidates_max_seq_len=max_num_candidates,
        )
        movie_history_ratings_uih, _ = separate_uih_candidates(
            data.sequence_ratings,
            candidates_max_seq_len=max_num_candidates,
        )
        movie_timestamps_uih = self.get_timestamp_uih(
            data=data,
            max_num_candidates=max_num_candidates,
            size=len(movie_history_uih),
        )

        assert len(movie_history_uih) == len(
            movie_timestamps_uih
        ), "history len differs from timestamp len."
        assert len(movie_history_uih) == len(
            movie_history_ratings_uih
        ), "history len differs from ratings len."

        movie_history_uih = maybe_truncate_seq(movie_history_uih, self._max_uih_len)
        movie_history_ratings_uih = maybe_truncate_seq(
            movie_history_ratings_uih, self._max_uih_len
        )
        movie_timestamps_uih = maybe_truncate_seq(
            movie_timestamps_uih, self._max_uih_len
        )

        uih_kjt_values: List[torch.Tensor] = []
        uih_kjt_lengths: List[torch.Tensor] = []
        for name, length in self._contextual_feature_to_max_length.items():
            uih_kjt_values.append(data[name])
            uih_kjt_lengths.append(length)

        uih_seq_len = len(movie_history_uih)
        movie_dummy_weights_uih = [0.0 for _ in range(uih_seq_len)]
        movie_dummy_watch_times_uih = [0.0 for _ in range(uih_seq_len)]
        uih_kjt_values.extend(
            movie_history_uih
            + movie_history_ratings_uih
            + movie_timestamps_uih
            + movie_dummy_weights_uih
            + movie_dummy_watch_times_uih
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

        dummy_query_time = max(movie_timestamps_uih)
        uih_kjt_values.append(dummy_query_time)
        uih_kjt_lengths.append(1)
        uih_features_kjt = KeyedJaggedTensor(
            keys=self._uih_keys + ["dummy_query_time"],
            lengths=torch.tensor(uih_kjt_lengths).long(),
            values=torch.tensor(uih_kjt_values).long(),
        )

        candidates_kjt_lengths = max_num_candidates * torch.ones(
            len(self._candidates_keys)
        )
        candidates_kjt_values = (
            movie_history_candidates
            + [dummy_query_time] * max_num_candidates  # item_query_time
            + [1] * max_num_candidates  # item_dummy_weights
            + [1] * max_num_candidates  # item_dummy_watchtime
        )
        candidates_features_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=torch.tensor(candidates_kjt_lengths).long(),
            values=torch.tensor(candidates_kjt_values).long(),
        )

        return (
            uih_features_kjt,
            candidates_features_kjt,
        )
