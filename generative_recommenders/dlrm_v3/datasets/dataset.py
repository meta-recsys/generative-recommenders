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
"""
implementation of the dataset for dlrm_v3.
Implementation borrowed from dlrm_v2 benchmark (https://github.com/mlcommons/inference/blob/master/recommendation/dlrm_v2/pytorch/python/dataset.py).
TODO: use this as a template and implements the expanded MovieLens dataset
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger("dlrmv3_dataset")


@dataclass
class Samples:
    uih_features_kjt: KeyedJaggedTensor
    candidates_features_kjt: KeyedJaggedTensor

    def to(self, device: torch.device) -> None:
        for attr in vars(self):
            setattr(self, attr, getattr(self, attr).to(device=device))


def collate_fn(
    samples: List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]],
) -> Samples:
    (
        uih_features_kjt_list,
        candidates_features_kjt_list,
    ) = list(zip(*samples))

    return Samples(
        uih_features_kjt=kjt_batch_func(uih_features_kjt_list),
        candidates_features_kjt=kjt_batch_func(candidates_features_kjt_list),
    )


class Dataset:
    def __init__(self, hstu_config: DlrmHSTUConfig, **args):
        self.arrival = None
        self.image_list = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.last_loaded = -1.0

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def load_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:load_query_samples")

    def unload_query_samples(self, sample_list):
        raise NotImplementedError("Dataset:unload_query_samples")

    def get_sample(self, id: int):
        raise NotImplementedError("Dataset:get_sample")

    def get_samples(self, id_list: List[int]) -> Samples:
        # Collate multiple examples same for all classes
        list_samples = [self.get_sample(ix) for ix in id_list]
        return collate_fn(list_samples)


@torch.jit.script
def kjt_batch_func(
    kjt_list: List[KeyedJaggedTensor],
) -> KeyedJaggedTensor:
    bs_list = [kjt.stride() for kjt in kjt_list]
    bs = sum(bs_list)
    batched_length = torch.cat([kjt.lengths() for kjt in kjt_list], dim=0)
    batched_indices = torch.cat([kjt.values() for kjt in kjt_list], dim=0)
    bs_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(bs_list)
    ).int()
    batched_offset = torch.ops.fbgemm.asynchronous_complete_cumsum(batched_length)
    reorder_length = torch.ops.fbgemm.reorder_batched_ad_lengths(
        batched_length, bs_offset, bs
    )
    reorder_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(reorder_length)
    reorder_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
        batched_offset, batched_indices, reorder_offsets, bs_offset, bs
    )
    out = KeyedJaggedTensor(
        keys=kjt_list[0].keys(),
        lengths=reorder_length.long(),
        values=reorder_indices.long(),
    )
    return out


def get_random_data(
    contexual_features: List[str],
    hstu_uih_keys: List[str],
    hstu_candidates_keys: List[str],
    uih_max_seq_len: int,
    max_num_candidates: int,
    value_bound: int = 1000,
):
    uih_non_seq_feature_keys = contexual_features
    uih_seq_feature_keys = [
        k for k in hstu_uih_keys if k not in uih_non_seq_feature_keys
    ]
    uih_seq_len = torch.randint(
        int(uih_max_seq_len * 0.8),
        uih_max_seq_len + 1,
        (1,),
    ).item()
    uih_lengths = torch.tensor(
        [1 for _ in uih_non_seq_feature_keys]
        + [uih_seq_len for _ in uih_seq_feature_keys]
    )
    # logging.info(f"uih_lengths: {uih_lengths}")
    uih_values = torch.randint(
        1,
        value_bound,
        # pyre-ignore[6]
        (uih_seq_len * len(uih_seq_feature_keys) + len(uih_non_seq_feature_keys),),
    )
    uih_features_kjt = KeyedJaggedTensor(
        keys=uih_non_seq_feature_keys + uih_seq_feature_keys,
        lengths=uih_lengths.long(),
        values=uih_values.long(),
    )
    num_candidates = torch.randint(
        1,
        max_num_candidates + 1,
        (1,),
    ).item()
    candidates_lengths = num_candidates * torch.ones(len(hstu_candidates_keys))
    candidates_values = torch.randint(
        1,
        value_bound,
        (num_candidates * len(hstu_candidates_keys),),  # pyre-ignore[6]
    )
    candidates_features_kjt = KeyedJaggedTensor(
        keys=hstu_candidates_keys,
        lengths=candidates_lengths.long(),
        values=candidates_values.long(),
    )
    return uih_features_kjt, candidates_features_kjt


class DLRMv3RandomDataset(Dataset):
    def __init__(
        self,
        hstu_config: DlrmHSTUConfig,
        num_aggregated_samples: int = 10000,
        is_inference: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            hstu_config=hstu_config,
        )
        self.hstu_config: DlrmHSTUConfig = hstu_config
        self._max_num_candidates: int = hstu_config.max_num_candidates
        self._max_num_candidates_inference: int = (
            hstu_config.max_num_candidates_inference
        )
        self._max_seq_len: int = hstu_config.max_seq_len
        self._uih_keys: List[str] = hstu_config.hstu_uih_feature_names
        self._candidates_keys: List[str] = hstu_config.hstu_candidate_feature_names
        self._contextual_feature_to_max_length: Dict[str, int] = (
            hstu_config.contextual_feature_to_max_length
        )
        self._max_uih_len: int = (
            self._max_seq_len
            - self._max_num_candidates
            - (
                len(self._contextual_feature_to_max_length)
                if self._contextual_feature_to_max_length
                else 0
            )
        )
        self._is_inference = is_inference

        self.contexual_features = []
        if hstu_config.contextual_feature_to_max_length is not None:
            self.contexual_features = [
                p[0] for p in hstu_config.contextual_feature_to_max_length
            ]

        self.num_aggregated_samples = num_aggregated_samples
        self.items_in_memory = {}

    def get_sample(self, id: int) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
        return self.items_in_memory[id]

    def get_item_count(self):
        # get number of items in the dataset
        return self.num_aggregated_samples

    def unload_query_samples(self, sample_list):
        self.items_in_memory = {}

    def load_query_samples(self, sample_list):
        max_num_candidates = (
            self._max_num_candidates_inference
            if self._is_inference
            else self._max_num_candidates
        )
        self.items_in_memory = {}
        for sample in sample_list:
            self.items_in_memory[sample] = get_random_data(
                contexual_features=self.contexual_features,
                hstu_uih_keys=self.hstu_config.hstu_uih_feature_names,
                hstu_candidates_keys=self.hstu_config.hstu_candidate_feature_names,
                uih_max_seq_len=self._max_uih_len,
                max_num_candidates=max_num_candidates,
            )
        self.last_loaded = time.time()
