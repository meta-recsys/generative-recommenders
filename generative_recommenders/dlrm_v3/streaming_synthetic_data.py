# pyre-strict
"""
Streaming synthetic data generator for DLRMv3.

This module generates synthetic streaming recommendation data for benchmarking
and testing purposes. It creates user-item interaction histories with timestamps,
ratings, and category-based item distributions.
"""

import csv
import logging
import math
import multiprocessing
import os
import random
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class StreamingSyntheticDataGenerator:
    """
    Generator for streaming synthetic recommendation data.

    Creates realistic user-item interaction data with temporal dynamics,
    category preferences, and rating distributions for benchmarking
    recommendation systems.

    Args:
        num_categories: Number of item categories.
        categories_per_user: Number of categories each user is interested in.
        num_users: Total number of users to generate.
        num_items: Total number of items in the catalog.
        num_timestamps: Number of time periods in the streaming data.
        avg_samples_per_item: Average number of interactions per item.
        train_ratio: Fraction of timestamps used for training.
        user_sampling_ratio: Probability of sampling a user at each timestamp.
        num_eval_candidates: Number of candidates for evaluation.
        num_inference_candidates: Number of candidates for inference.
        debug: If True, use deterministic ratings for debugging.
        rank: Process rank for distributed generation.
    """

    def __init__(
        self,
        num_categories: int,
        categories_per_user: int,
        num_users: int,
        num_items: int,
        num_timestamps: int,
        avg_samples_per_item: int,
        train_ratio: float,
        user_sampling_ratio: float,
        num_eval_candidates: int,
        num_inference_candidates: int,
        debug: bool = False,
        rank: int = 0,
    ) -> None:
        self.num_categories = num_categories
        self.categories_per_user = categories_per_user
        self.num_users = num_users
        self.num_items = num_items
        self.num_timestamps = num_timestamps
        self.avg_samples_per_item = avg_samples_per_item
        self.avg_seq_len_per_timestamp = int(
            num_items * avg_samples_per_item / num_users / num_timestamps
        )
        self.items_per_category: int = num_items // num_categories
        self.category_to_start_end_item_idx: Dict[int, Tuple[int, int]] = {}
        for i in range(num_categories):
            start_idx = i * self.items_per_category
            end_idx = (i + 1) * self.items_per_category
            self.category_to_start_end_item_idx[i] = (start_idx, end_idx)
        self.alpha_range = (1, 500)
        self.min_seq_len: int = num_eval_candidates + 1
        self.train_ratio = train_ratio
        self.num_eval_candidates = num_eval_candidates
        self.num_inference_candidates = num_inference_candidates
        self.debug = debug
        self.total_cnt = 0
        self.rank = rank
        logger.warning(f"rank {self.rank}: start generating item rating")
        np.random.seed(1001)
        self.item_rating = np.random.choice(  # pyre-ignore [4]
            [5.0, 4.0, 3.0, 2.0, 1.0], size=num_items, p=[0.2, 0.25, 0.25, 0.2, 0.1]
        )
        logger.warning(f"rank {self.rank}: finish generating item rating")
        self.user_sampling_ratio = user_sampling_ratio

    def generate_one_timestamp(
        self,
        category_to_cnt: Dict[int, int],
        categories: List[int],
        t: int,
        id: int,
        output_folder: str,
        uih_seq_len: int,
        eval: bool,
        inference: bool,
        file_idx: int,
        ts_buffers: Dict[int, List[int]],
    ) -> Tuple[List[int], List[float], List[int], List[float], Dict[int, int]]:
        """
        Generate interaction data for a single user at one timestamp.

        Args:
            category_to_cnt: Running count of interactions per category.
            categories: Categories this user is interested in.
            t: Current timestamp index.
            id: User ID.
            output_folder: Output directory for files.
            uih_seq_len: Length of user interaction history to generate.
            eval: Whether this is for evaluation.
            inference: Whether this is for inference.
            file_idx: File index for output.
            ts_buffers: Buffer for timestamp data.

        Returns:
            Tuple of (uih_item_ids, uih_ratings, candidate_ids, candidate_ratings,
            updated_category_counts).
        """
        if t >= 0 and (not eval):
            if t not in ts_buffers:
                ts_buffers[t] = []
            ts_buffers[t].append(id)
        seq_len: int = self.num_inference_candidates if inference else uih_seq_len
        self.total_cnt += seq_len
        alpha = random.randint(self.alpha_range[0], self.alpha_range[1])
        total_cnt = sum(category_to_cnt.values())
        p = np.array(
            [
                (alpha / len(categories) + category_to_cnt[c]) / (alpha + total_cnt)
                for c in categories
            ]
        )
        item_categories = np.random.choice(categories, size=seq_len, p=p)
        unique, counts = np.unique(item_categories, return_counts=True)
        for cat, cnt in zip(unique, counts):
            category_to_cnt[cat] += int(cnt)
        sample_end_idx = int(
            self.items_per_category * max((t + 1), 1) / self.num_timestamps
        )
        sample_inds = np.random.randint(0, sample_end_idx, size=seq_len)
        offsets = np.array(
            [self.category_to_start_end_item_idx[cat][0] for cat in item_categories]
        )
        sample_inds = sample_inds + offsets
        num_categories = len(categories)
        quarter = num_categories // 4
        half = num_categories // 2
        three_quarter = num_categories // 4 * 3
        category_to_ratings = {}
        cos1 = math.cos(t * math.pi / 4)
        cos2 = math.cos((t + 2) * math.pi / 4)
        cos3 = math.cos((t + 4) * math.pi / 4)
        for i, cat in enumerate(categories):
            if i < quarter:
                if self.debug:
                    ratings = np.full(seq_len, 5.0)
                else:
                    ratings = np.random.choice(
                        [4.5 + 0.5 * cos1, 4.0 + 0.5 * cos2],
                        size=seq_len,
                        p=[0.8, 0.2],
                    )
            elif i < half:
                if self.debug:
                    ratings = np.full(seq_len, 4.0)
                else:
                    ratings = np.random.choice(
                        [4.5 + 0.5 * cos1, 4.0 + 0.5 * cos2, 3.5 + 0.5 * cos3],
                        size=seq_len,
                        p=[0.1, 0.8, 0.1],
                    )
            elif i < three_quarter:
                if self.debug:
                    ratings = np.full(seq_len, 3.0)
                else:
                    ratings = np.random.choice(
                        [3.5 + 0.5 * cos1, 3.0 + 0.5 * cos2, 2.5 + 0.5 * cos3],
                        size=seq_len,
                        p=[0.1, 0.8, 0.1],
                    )
            else:
                if self.debug:
                    ratings = np.full(seq_len, 2.0)
                else:
                    ratings = np.random.choice(
                        [2.5 + 0.5 * cos1, 2.0 + 0.5 * cos2, 1.5 + 0.5 * cos3],
                        size=seq_len,
                        p=[0.1, 0.8, 0.1],
                    )
            category_to_ratings[cat] = ratings
        sample_inds = sample_inds.tolist()
        sample_ratings = [
            (
                category_to_ratings[item_categories[i]][i]
                + self.item_rating[sample_inds[i]]
            )
            / 2
            for i in range(seq_len)
        ]
        if not inference:
            sub_indices = random.sample(range(seq_len), self.num_eval_candidates)
            sample_candidate_inds = [sample_inds[i] for i in sub_indices]
            sample_candidate_ratings = [sample_ratings[i] for i in sub_indices]
            sample_uih_inds = sample_inds
            sample_uih_ratings = sample_ratings
        else:
            sub_indices = random.sample(range(seq_len), uih_seq_len)
            sample_uih_inds = [sample_inds[i] for i in sub_indices]
            sample_uih_ratings = [sample_ratings[i] for i in sub_indices]
            sample_candidate_inds = sample_inds
            sample_candidate_ratings = sample_ratings
        return (
            sample_uih_inds,
            sample_uih_ratings,
            sample_candidate_inds,
            sample_candidate_ratings,
            category_to_cnt,
        )

    def gen_rand_seq_len(self) -> int:
        """
        Generate a random sequence length from a Gaussian distribution.

        Returns:
            Sequence length, guaranteed to be at least min_seq_len.
        """
        seq_len = round(
            random.gauss(
                self.avg_seq_len_per_timestamp, self.avg_seq_len_per_timestamp // 4
            )
        )
        seq_len = self.min_seq_len if seq_len < self.min_seq_len else seq_len
        return seq_len

    def get_timestamp_sample(self, t: int) -> int:
        """
        Determine if a user should be sampled at this timestamp.

        Args:
            t: Timestamp index. Base timestamp (-1) is always sampled.

        Returns:
            1 if the user should be sampled, 0 otherwise.
        """
        if t == -1:
            sample = 1
        else:
            sample = np.random.choice(
                [1, 0],
                size=1,
                p=[self.user_sampling_ratio, 1 - self.user_sampling_ratio],
            )[0]
        return sample

    def generate_one_user(
        self,
        id: int,
        output_folder: str,
        file_idx: int,
        ts_buffers: Dict[int, List[int]],
    ) -> List[str]:
        """
        Generate complete interaction history for one user.

        Creates training, evaluation, and inference data for a single user
        across all timestamps.

        Args:
            id: User ID.
            output_folder: Output directory.
            file_idx: File index for output.
            ts_buffers: Buffer for timestamp metadata.

        Returns:
            List of CSV row values for this user's data.
        """
        categories = random.sample(range(self.num_categories), self.categories_per_user)
        category_to_cnt = {c: 0 for c in categories}
        out_list: List[str] = []
        # t = -1 as base UIH
        (
            sample_inds,
            sample_ratings,
            sample_candidate_inds,
            sample_candidate_ratings,
            category_to_cnt,
        ) = self.generate_one_timestamp(
            category_to_cnt=category_to_cnt,
            categories=categories,
            t=-1,
            id=id,
            output_folder=output_folder,
            uih_seq_len=self.gen_rand_seq_len(),
            eval=False,
            inference=False,
            file_idx=file_idx,
            ts_buffers=ts_buffers,
        )
        out_list.append(",".join([str(ind) for ind in sample_candidate_inds]))
        out_list.append(",".join([str(rat) for rat in sample_candidate_ratings]))
        out_list.append(",".join([str(ind) for ind in sample_inds]))
        out_list.append(",".join([str(rat) for rat in sample_ratings]))
        # train
        for t in range(int(self.num_timestamps * self.train_ratio)):
            if self.get_timestamp_sample(t):
                (
                    sample_inds,
                    sample_ratings,
                    sample_candidate_inds,
                    sample_candidate_ratings,
                    category_to_cnt,
                ) = self.generate_one_timestamp(
                    category_to_cnt=category_to_cnt,
                    categories=categories,
                    t=t,
                    id=id,
                    output_folder=output_folder,
                    uih_seq_len=self.gen_rand_seq_len(),
                    eval=False,
                    inference=False,
                    file_idx=file_idx,
                    ts_buffers=ts_buffers,
                )
                out_list.append(",".join([str(ind) for ind in sample_candidate_inds]))
                out_list.append(
                    ",".join([str(rat) for rat in sample_candidate_ratings])
                )
                out_list.append(",".join([str(ind) for ind in sample_inds]))
                out_list.append(",".join([str(rat) for rat in sample_ratings]))
            else:
                out_list += ["", "", "", ""]
        # eval
        (
            sample_inds,
            sample_ratings,
            sample_candidate_inds,
            sample_candidate_ratings,
            category_to_cnt,
        ) = self.generate_one_timestamp(
            category_to_cnt=category_to_cnt,
            categories=categories,
            t=int(self.num_timestamps * self.train_ratio),
            id=id,
            output_folder=output_folder,
            uih_seq_len=self.num_eval_candidates,
            eval=True,
            inference=False,
            file_idx=file_idx,
            ts_buffers=ts_buffers,
        )
        out_list.append(",".join([str(ind) for ind in sample_candidate_inds]))
        out_list.append(",".join([str(rat) for rat in sample_candidate_ratings]))
        out_list.append(",".join([str(ind) for ind in sample_inds]))
        out_list.append(",".join([str(rat) for rat in sample_ratings]))
        # inference
        for t in range(
            int(self.num_timestamps * self.train_ratio), self.num_timestamps
        ):
            if self.get_timestamp_sample(t):
                (
                    sample_inds,
                    sample_ratings,
                    sample_candidate_inds,
                    sample_candidate_ratings,
                    category_to_cnt,
                ) = self.generate_one_timestamp(
                    category_to_cnt=category_to_cnt,
                    categories=categories,
                    t=t,
                    id=id,
                    output_folder=output_folder,
                    uih_seq_len=self.gen_rand_seq_len(),
                    eval=False,
                    inference=True,
                    file_idx=file_idx,
                    ts_buffers=ts_buffers,
                )
                out_list.append(",".join([str(ind) for ind in sample_candidate_inds]))
                out_list.append(
                    ",".join([str(rat) for rat in sample_candidate_ratings])
                )
                out_list.append(",".join([str(ind) for ind in sample_inds]))
                out_list.append(",".join([str(rat) for rat in sample_ratings]))
            else:
                out_list += ["", "", "", ""]
        return out_list

    def write_dataset(
        self, output_folder: str, num_files: int, file_idx: int, seed: int
    ) -> None:
        """
        Write dataset for a single file partition.

        Args:
            output_folder: Output directory path.
            num_files: Total number of files in the dataset.
            file_idx: Index of this file partition.
            seed: Random seed for reproducibility.
        """
        t0 = time.time()
        num_users_per_file = self.num_users // num_files
        user_id: int = num_users_per_file * file_idx
        random.seed(seed + file_idx)
        np.random.seed(seed + file_idx)
        # Buffer timestamp data in memory to avoid excessive file I/O
        ts_buffers: Dict[int, List[int]] = {}
        output_file = output_folder + f"{file_idx}.csv"
        with open(output_file, "w") as file:
            writer = csv.writer(file)
            for i in range(num_users_per_file):
                out_list = self.generate_one_user(
                    id=user_id,
                    output_folder=output_folder,
                    file_idx=file_idx,
                    ts_buffers=ts_buffers,
                )
                user_id += 1
                writer.writerow(out_list)
                if i % 10000 == 0:
                    logger.warning(
                        f"rank {self.rank}: Done with users {i} for file {file_idx + 1} / {num_files}, total_cnt = {self.total_cnt}, spends {time.time() - t0} seconds."
                    )
        # Write buffered timestamp data after all users are processed
        for ts, user_ids in ts_buffers.items():
            ts_file = output_folder + f"ts_{file_idx}_{ts}.csv"
            with open(ts_file, "w") as f:
                writer = csv.writer(f)
                for uid in user_ids:
                    writer.writerow([uid])
        logger.warning(
            f"rank {self.rank}: Wrote {len(ts_buffers)} timestamp files for file {file_idx}"
        )


def worker(
    rank: int,
    world_size: int,
    num_files: int,
    num_users: int,
    num_items: int,
    num_categories: int,
    categories_per_user: int,
    num_timestamps: int,
    avg_samples_per_item: int,
    num_eval_candidates: int,
    num_inference_candidates: int,
    train_ratio: float,
    user_sampling_ratio: float,
    output_folder: str,
) -> None:
    """
    Worker function for parallel data generation.

    Each worker generates a subset of the dataset files.

    Args:
        rank: Worker rank.
        world_size: Total number of workers.
        num_files: Total files to generate.
        num_users: Total users in dataset.
        num_items: Total items in catalog.
        num_categories: Number of item categories.
        categories_per_user: Categories per user.
        num_timestamps: Number of time periods.
        avg_samples_per_item: Average interactions per item.
        num_eval_candidates: Eval candidates count.
        num_inference_candidates: Inference candidates count.
        train_ratio: Training data fraction.
        user_sampling_ratio: User sampling probability.
        output_folder: Output directory.
    """
    generator = StreamingSyntheticDataGenerator(
        num_categories=num_categories,
        categories_per_user=categories_per_user,
        num_users=num_users,
        num_items=num_items,
        num_timestamps=num_timestamps,
        avg_samples_per_item=avg_samples_per_item,
        train_ratio=train_ratio,
        user_sampling_ratio=user_sampling_ratio,
        num_eval_candidates=num_eval_candidates,
        num_inference_candidates=num_inference_candidates,
        debug=False,
        rank=rank,
    )
    num_files_per_rank = num_files // world_size
    file_indices = [i + rank * num_files_per_rank for i in range(num_files_per_rank)]
    for file_idx in file_indices:
        logger.warning(f"rank {rank}: start generating file {file_idx}")
        generator.write_dataset(
            output_folder=output_folder,
            num_files=num_files,
            file_idx=file_idx,
            seed=1001,
        )
        logger.warning(f"rank {rank}: finish generating file {file_idx}")


def write_offset(output_folder: str, num_files: int, num_users: int) -> None:
    """
    Write file byte offsets for random access to user data.

    Creates an offset.csv file containing byte positions for each user
    within their respective data files.

    Args:
        output_folder: Directory containing data files.
        num_files: Number of data files.
        num_users: Total number of users.
    """
    with open(output_folder + "offset.csv", "a") as output_file:
        writer = csv.writer(output_file)
        for i in range(num_files):
            input_file = output_folder + f"{i}.csv"
            offsets = []
            with open(input_file, "r") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    offsets.append(offset)
            assert len(offsets) == num_users // num_files, (
                f"num_users {num_users // num_files} != {len(offsets)}"
            )
            logger.warning(f"offsets for file {i} finished")
            writer.writerow([",".join([str(offset) for offset in offsets])])


def write_ts_metadata(output_folder: str, total_ts: int, num_files: int) -> None:
    """
    Write timestamp metadata for streaming simulation.

    Creates files tracking which users are active at each timestamp
    and cumulative counts for efficient streaming access.

    Args:
        output_folder: Output directory path.
        total_ts: Total number of timestamps.
        num_files: Number of data files.
    """
    with open(output_folder + "requests_per_ts.csv", "w") as file_requests:
        with open(output_folder + "users_cumsum_per_ts.csv", "w") as file_cumsum:
            requests_writer = csv.writer(file_requests)
            cumsum_writer = csv.writer(file_cumsum)
            for ts in range(total_ts):
                requests = []
                num_users_per_file = []
                for file in range(num_files):
                    with open(f"{output_folder}ts_{file}_{ts}.csv", "r") as file:
                        reader = csv.reader(file)
                        size = 0
                        for row in reader:
                            requests.append(int(row[0]))
                            size += 1
                        num_users_per_file.append(size)
                cumsum = np.cumsum(num_users_per_file).tolist()
                assert cumsum[-1] == len(requests)
                requests_writer.writerow([",".join([str(r) for r in requests])])
                cumsum_writer.writerow([",".join([str(s) for s in cumsum])])
                logger.warning(f"ts {ts} finished")
    with open(
        output_folder + "requests_per_ts_offset.csv", "w"
    ) as file_requests_offset:
        writer = csv.writer(file_requests_offset)
        input_file = output_folder + "requests_per_ts.csv"
        offsets = []
        with open(input_file, "r") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(offset)
        assert len(offsets) == total_ts, f"total_ts {total_ts} != {len(offsets)}"
        logger.warning("offsets for file requests_per_ts.csv finished")
        writer.writerow([",".join([str(offset) for offset in offsets])])


def copy_sub_dataset(src_folder: str) -> None:
    """
    Copy a subset of dataset files for quick testing.

    Creates a sampled_data subdirectory with essential files.

    Args:
        src_folder: Source folder containing full dataset.
    """
    dst_folder = src_folder + "sampled_data/"
    files_to_copy = [
        "0.csv",
        "offset.csv",
        "requests_per_ts.csv",
        "requests_per_ts_offset.csv",
        "users_cumsum_per_ts.csv",
    ]
    os.makedirs(dst_folder, exist_ok=True)
    for filename in files_to_copy:
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.copy2(src_path, dst_path)
    logger.warning("Files copied successfully.")


def main() -> None:
    """
    Main entry point for synthetic data generation.

    Configures and launches parallel workers to generate a complete
    streaming recommendation dataset.
    """
    processes = []
    num_files = 100
    num_users = 5_000_000
    num_items = 1_000_000_000
    num_categories = 128
    categories_per_user = 4
    num_timestamps = 100
    avg_samples_per_item = 50
    num_eval_candidates = 32
    num_inference_candidates = 2048
    train_ratio = 0.9
    user_sampling_ratio = 0.7
    world_size = 5
    username = os.getlogin()
    output_folder = f"/home/{username}/data/streaming-100b/"
    for i in range(world_size):
        p = multiprocessing.Process(
            target=worker,
            args=(
                i,
                world_size,
                num_files,
                num_users,
                num_items,
                num_categories,
                categories_per_user,
                num_timestamps,
                avg_samples_per_item,
                num_eval_candidates,
                num_inference_candidates,
                train_ratio,
                user_sampling_ratio,
                output_folder,
            ),
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    write_offset(output_folder, num_files, num_users)
    write_ts_metadata(output_folder, num_timestamps, num_files)
    copy_sub_dataset(src_folder=output_folder)


if __name__ == "__main__":
    main()
