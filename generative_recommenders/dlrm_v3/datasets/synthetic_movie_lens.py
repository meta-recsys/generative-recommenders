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
import csv
import linecache
import logging
import sys
from typing import List

import numpy as np
import pandas as pd

from generative_recommenders.dlrm_v3.datasets.movie_lens import DLRMv3MovieLensDataset
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig

csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)


class DLRMv3SyntheticMovieLensDataset(DLRMv3MovieLensDataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        ratings_file_prefix: str,
        is_inference: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            hstu_config=hstu_config, is_inference=is_inference, ratings_file=""
        )
        self.ratings_file_prefix = ratings_file_prefix
        with open(f"{self.ratings_file_prefix}_users.csv", "r") as file:
            reader = csv.reader(file)
            self.users_cumsum: List[int] = np.cumsum(
                [int(row[1]) for row in reader]
            ).tolist()

    def get_item_count(self):
        return self.users_cumsum[-1]

    def _process_line(self, line: str) -> pd.Series:
        reader = csv.reader([line])
        parsed_line = next(reader)
        user_id = int(parsed_line[0])
        sequence_item_ids = parsed_line[1]
        sequence_ratings = parsed_line[2]
        return pd.Series(
            data={
                "user_id": user_id,
                "sequence_item_ids": sequence_item_ids,
                "sequence_ratings": sequence_ratings,
            }
        )

    def iloc(self, idx) -> pd.Series:
        assert idx < self.users_cumsum[-1]
        file_idx: int = 0
        while self.users_cumsum[file_idx] <= idx:
            file_idx += 1
        if file_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.users_cumsum[file_idx - 1]
        line = linecache.getline(
            f"{self.ratings_file_prefix}_{file_idx}.csv", local_idx + 1
        )
        data = self._process_line(line)
        return data

    def get_timestamp_uih(self, data, max_num_candidates, size):
        return [1] * size
