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
import argparse
import logging
import os
import tarfile
from typing import Dict, List
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

"""
Usage: mkdir -p data/ && python3 preprocess_public_data.py --dataset kuairand-1k
"""

SUPPORTED_DATASETS = ["kuairand-1k", "kuairand-27k"]


def get_feature_merge_weights(dataset: str = "debug") -> Dict[str, int]:
    if "kuairand" in dataset:
        return {
            "is_click": 1,
            "is_like": 2,
            "is_follow": 4,
            "is_comment": 8,
            "is_forward": 16,
            "is_hate": 32,
            "long_view": 64,
            "is_profile_enter": 128,
        }
    else:
        return {"dummy": 1}


class DataProcessor:
    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        self._download_url = download_url
        self._data_path = data_path
        self._file_name = file_name
        self._prefix = prefix

    def download(self) -> None:
        return

    def preprocess(self) -> None:
        return

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class DLRMKuaiRandProcessor(DataProcessor):
    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        super().__init__(download_url, data_path, file_name, prefix)
        if prefix == "KuaiRand-1K":
            self._log_files: List[str] = [
                f"{data_path}{prefix}/data/log_standard_4_08_to_4_21_1k.csv",
                f"{data_path}{prefix}/data/log_standard_4_22_to_5_08_1k.csv",
            ]
            self._user_features_file: str = (
                f"{data_path}{prefix}/data/user_features_1k.csv"
            )
        elif prefix == "KuaiRand-27K":
            self._log_files: List[str] = [
                f"{data_path}{prefix}/data/log_standard_4_08_to_4_21_27k_part1.csv",
                f"{data_path}{prefix}/data/log_standard_4_08_to_4_21_27k_part2.csv",
                f"{data_path}{prefix}/data/log_standard_4_22_to_5_08_27k_part1.csv",
                f"{data_path}{prefix}/data/log_standard_4_22_to_5_08_27k_part2.csv",
            ]
            self._user_features_file: str = (
                f"{data_path}{prefix}/data/user_features_27k.csv"
            )
        self._output_file: str = f"{data_path}{prefix}/data/processed_seqs.csv"
        self._event_merge_weight: Dict[str, int] = get_feature_merge_weights(
            prefix.lower()
        )

    def download(self) -> None:
        file_path = f"{self._data_path}{self._file_name}"
        if not self.file_exists(file_path):
            log.info(f"Downloading {self._download_url}")
            urlretrieve(self._download_url, file_path)
            log.info(f"Downloaded to {file_path}")
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(path=self._data_path)
                log.info("Data files extracted")
            os.remove(file_path)
            log.info("Tar file removed")

    def preprocess(self) -> None:
        self.download()
        log.info("Preprocessing data...")
        seq_cols = [
            "video_id",
            "time_ms",
            "action_weights",
            "play_time_ms",
            "duration_ms",
        ]
        df = None
        for idx, log_file in enumerate(self._log_files):
            log.info(f"Processing {log_file}...")
            log_df = pd.read_csv(
                log_file,
                delimiter=",",
            )
            df_grouped_by_user = log_df.groupby("user_id").agg(list).reset_index()

            for event, weight in self._event_merge_weight.items():
                df_grouped_by_user[event] = df_grouped_by_user[event].apply(
                    lambda seq: np.where(np.array(seq) == 0, 0, weight)
                )

            events = list(self._event_merge_weight.keys())
            df_grouped_by_user["action_weights"] = df_grouped_by_user.apply(
                lambda row: [int(sum(x)) for x in zip(*[row[col] for col in events])],
                axis=1,
            )
            df_grouped_by_user = df_grouped_by_user[["user_id"] + seq_cols]

            if idx == 0:
                df = df_grouped_by_user
            else:
                # pyrefly: ignore [missing-attribute]
                df = df.merge(df_grouped_by_user, on="user_id", suffixes=("_x", "_y"))
                for col in seq_cols:
                    df[col] = df.apply(
                        lambda row: row[col + "_x"] + row[col + "_y"], axis=1
                    )
                    df = df.drop(columns=[col + "_x", col + "_y"])

        # pyrefly: ignore [unsupported-operation]
        max_seq_len = df["video_id"].apply(len).max()
        # pyrefly: ignore [unsupported-operation]
        min_seq_len = df["video_id"].apply(len).min()
        # pyrefly: ignore [unsupported-operation]
        average_seq_len = df["video_id"].apply(len).mean()
        log.info(f"{max_seq_len=}, {min_seq_len=}, {average_seq_len=}")

        log.info("Merging user features...")
        user_features_df = pd.read_csv(self._user_features_file, delimiter=",")

        def _one_hot_encode(row):
            mapping = {category: i + 1 for i, category in enumerate(row.unique())}
            row = row.map(mapping)
            return row

        for col in [
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]:
            user_features_df[col] = _one_hot_encode(user_features_df[col])

        # pyrefly: ignore [bad-argument-type]
        final_df = pd.merge(df, user_features_df, on="user_id")
        final_df.to_csv(self._output_file, index=False, sep=",")
        log.info(f"Processed file saved to {self._output_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, help="dataset")
    args = parser.parse_args()
    if args.dataset == "kuairand-1k":
        kuairand_processor = DLRMKuaiRandProcessor(
            download_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
            data_path="data/",
            file_name="KuaiRand-1K.tar.gz",
            prefix="KuaiRand-1K",
        )
        kuairand_processor.preprocess()
    elif args.dataset == "kuairand-27k":
        kuairand_processor = DLRMKuaiRandProcessor(
            download_url="https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
            data_path="data/",
            file_name="KuaiRand-27K.tar.gz",
            prefix="KuaiRand-27K",
        )
        kuairand_processor.preprocess()


if __name__ == "__main__":
    main()
