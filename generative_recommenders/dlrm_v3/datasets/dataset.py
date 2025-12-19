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
Dataset implementations for DLRMv3.

This module provides dataset classes for loading and processing recommendation
data, including sample containers, collation functions, and random data generation.
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
    """
    Container for batched samples with user interaction history and candidate features.

    Attributes:
        uih_features_kjt: User interaction history features as KeyedJaggedTensor.
        candidates_features_kjt: Candidate item features as KeyedJaggedTensor.
    """

    uih_features_kjt: KeyedJaggedTensor
    candidates_features_kjt: KeyedJaggedTensor

    def to(self, device: torch.device) -> None:
        """
        Move all tensors to the specified device.

        Args:
            device: Target device to move tensors to.
        """
        for attr in vars(self):
            setattr(self, attr, getattr(self, attr).to(device=device))

    def batch_size(self) -> int:
        """
        Get the batch size of the samples.

        Returns:
            Number of samples in the batch.
        """
        return self.uih_features_kjt.stride()


def collate_fn(
    samples: List[Tuple[KeyedJaggedTensor, KeyedJaggedTensor]],
) -> Samples:
    """
    Collate multiple samples into a batched Samples object.

    Args:
        samples: List of (uih_features, candidates_features) tuples.

    Returns:
        Batched Samples object with concatenated features.
    """
    (
        uih_features_kjt_list,
        candidates_features_kjt_list,
    ) = list(zip(*samples))

    return Samples(
        uih_features_kjt=kjt_batch_func(uih_features_kjt_list),
        candidates_features_kjt=kjt_batch_func(candidates_features_kjt_list),
    )


class Dataset:
    """
    Base dataset class for DLRMv3.

    Provides the interface for loading, accessing, and managing samples
    for recommendation model training and inference.

    Args:
        hstu_config: HSTU model configuration.
        **args: Additional arguments (unused in base class).
    """

    def __init__(self, hstu_config: DlrmHSTUConfig, **args):
        self.arrival = None
        self.image_list = []
        self.label_list = []
        self.image_list_inmemory = {}
        self.last_loaded = -1.0

    def preprocess(self, use_cache=True):
        """
        Preprocess the dataset.

        Args:
            use_cache: Whether to use cached preprocessed data.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        """
        Get the total number of items in the dataset.

        Returns:
            Number of items.
        """
        return len(self.image_list)

    def load_query_samples(self, sample_list):
        """
        Load specified samples into memory.

        Args:
            sample_list: List of sample indices to load.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:load_query_samples")

    def unload_query_samples(self, sample_list):
        """
        Unload specified samples from memory.

        Args:
            sample_list: List of sample indices to unload.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:unload_query_samples")

    def get_sample(self, id: int):
        """
        Get a single sample by ID.

        Args:
            id: Sample identifier.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Dataset:get_sample")

    def get_samples(self, id_list: List[int]) -> Samples:
        """
        Get multiple samples and collate them into a batch.

        Args:
            id_list: List of sample identifiers.

        Returns:
            Collated Samples object containing the batch.
        """
        list_samples = [self.get_sample(ix) for ix in id_list]
        return collate_fn(list_samples)


@torch.jit.script
def kjt_batch_func(
    kjt_list: List[KeyedJaggedTensor],
) -> KeyedJaggedTensor:
    """
    Batch multiple KeyedJaggedTensors into a single tensor.

    Uses FBGEMM operations for efficient batching and reordering of
    jagged tensor data.

    Args:
        kjt_list: List of KeyedJaggedTensors to batch.

    Returns:
        Batched KeyedJaggedTensor with reordered indices and lengths.
    """
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
    """
    Generate random sample data for testing and debugging.

    Creates synthetic user interaction history and candidate features
    with random values.

    Args:
        contexual_features: List of contextual feature names.
        hstu_uih_keys: List of UIH feature keys.
        hstu_candidates_keys: List of candidate feature keys.
        uih_max_seq_len: Maximum sequence length for UIH.
        max_num_candidates: Maximum number of candidates.
        value_bound: Upper bound for random values.

    Returns:
        Tuple of (uih_features_kjt, candidates_features_kjt).
    """
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
    """
    Dataset that generates random synthetic data for DLRMv3.

    Useful for testing and benchmarking without real data dependencies.

    Args:
        hstu_config: HSTU model configuration.
        num_aggregated_samples: Total number of samples to generate.
        is_inference: Whether the dataset is used for inference mode.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """

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
        """
        Get a sample by ID from in-memory storage.

        Args:
            id: Sample identifier.

        Returns:
            Tuple of (uih_features_kjt, candidates_features_kjt).
        """
        return self.items_in_memory[id]

    def get_item_count(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            Number of aggregated samples.
        """
        return self.num_aggregated_samples

    def unload_query_samples(self, sample_list):
        """
        Clear all samples from memory.

        Args:
            sample_list: Ignored; clears all samples.
        """
        self.items_in_memory = {}

    def load_query_samples(self, sample_list):
        """
        Generate and load random samples into memory.

        Args:
            sample_list: List of sample IDs to generate.
        """
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
