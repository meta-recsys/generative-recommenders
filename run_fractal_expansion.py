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
Run fractal expansion introduced in https://arxiv.org/abs/1901.08910.
Implementation adapted from the scripts used to generate MovieLens-1B
(https://grouplens.org/datasets/movielens/movielens-1b/).
"""

# Generate a 3B dataset (takes around 50 minutes):
# python run_fractal_expansion.py --input-csv-file ~/data/ml-20m/ratings.csv --write-dataset True --output-prefix ~/data/ml-3b/
# Generate a 13B dataset with 440M item size:
# python run_fractal_expansion.py --input-csv-file ~/data/ml-20m/ratings.csv --write-dataset True --output-prefix ~/data/ml-13b/ --num-row-multiplier 16 --num-col-multiplier 16384 --element-sample-rate 0.2 --block-sample-rate 0.05
# Generate a 18B dataset with 1B item size:
# python run_fractal_expansion.py --input-csv-file ~/data/ml-20m/ratings.csv --write-dataset True --output-prefix ~/data/ml-18b/ --num-row-multiplier 20 --num-col-multiplier 36864 --element-sample-rate 0.08 --block-sample-rate 0.05

import csv
import linecache
import logging
import os
import pickle

from dataclasses import dataclass

import click
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.linalg
import skimage.transform as transform
from scipy import sparse
from scipy.sparse import linalg
from sklearn.utils import shuffle
from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SparseMatrixMetadata:
    num_interactions: int = 0
    num_rows: int = 0
    num_cols: int = 0


def _dropout_sparse_coo_matrix(
    sparse_matrix, rate, min_dropout_rate=0.005, max_dropout_rate=0.999
):
    assert min_dropout_rate <= max_dropout_rate
    sampling_rate = 1.0 - rate

    sampled_fraction = min(
        max(sampling_rate, 1.0 - max_dropout_rate), 1.0 - min_dropout_rate
    )
    if sampled_fraction != sampling_rate:
        logger.warning(
            f"Desired sampling rate {sampling_rate} clipped to {sampled_fraction}."
        )
    num_sampled = min(
        max(int(sparse_matrix.nnz * sampled_fraction), 1), sparse_matrix.nnz
    )
    sampled_indices = np.random.choice(
        sparse_matrix.nnz, size=num_sampled, replace=False
    )
    return sparse.coo_matrix(
        (
            sparse_matrix.data[sampled_indices],
            (sparse_matrix.row[sampled_indices], sparse_matrix.col[sampled_indices]),
        ),
        shape=sparse_matrix.shape,
    )


def shuffle_sparse_matrix(
    sparse_matrix, dropout_rate=0.0, min_dropout_rate=0.005, max_dropout_rate=0.999
):
    """
    Shuffle sparse matrix encoded as a SciPy csr matrix.
    """

    assert dropout_rate >= 0.0 and dropout_rate <= 1.0
    (num_rows, num_cols) = sparse_matrix.shape
    shuffled_rows = shuffle(np.arange(num_rows))
    shuffled_cols = shuffle(np.arange(num_cols))
    sparse_matrix = _dropout_sparse_coo_matrix(
        sparse_matrix, dropout_rate, min_dropout_rate, max_dropout_rate
    )
    new_row = np.take(shuffled_rows, sparse_matrix.row)
    new_col = np.take(shuffled_cols, sparse_matrix.col)
    return sparse.csr_matrix(
        (sparse_matrix.data, (new_row, new_col)), shape=(num_rows, num_cols)
    )


def graph_reduce(usv, num_rows, num_cols):
    """Apply algorithm 2 in https://arxiv.org/pdf/1901.08910.pdf."""

    def _closest_column_orthogonal_matrix(matrix):
        return np.matmul(
            matrix, np.linalg.inv(scipy.linalg.sqrtm(np.matmul(matrix.T, matrix)))
        )

    u, s, v = usv
    k = min(num_rows, num_cols)
    u_random_proj = transform.resize(u[:, :k], (num_rows, k))
    v_random_proj = transform.resize(v[:k, :], (k, num_cols))
    u_random_proj_orth = _closest_column_orthogonal_matrix(u_random_proj)
    v_random_proj_orth = _closest_column_orthogonal_matrix(v_random_proj.T).T
    return np.matmul(u_random_proj_orth, np.matmul(np.diag(s[:k]), v_random_proj_orth))


def rescale(matrix, rescale_w_abs=False, element_sample_rate=1.0):
    """Rescale all values of the matrix into [0, 1]."""
    if rescale_w_abs:
        abs_matrix = np.abs(matrix.copy())
        out = abs_matrix / abs_matrix.max()
    else:
        out = (matrix - matrix.min()) / (matrix.max() - matrix.min())
        assert out.min() >= 0 and out.max() <= 1
    return out * element_sample_rate


def _compute_row_block(
    i, left_matrix, right_matrix, block_sample_rate, indices_out_path, remove_empty_rows
):
    """Compute row block of expansion for row i of the left_matrix."""

    kron_blocks = []
    num_rows = 0
    num_removed_rows = 0
    num_interactions = 0

    for j in range(left_matrix.shape[1]):
        if np.random.random() <= block_sample_rate:
            dropout_rate = 1.0 - left_matrix[i, j]
            kron_block = shuffle_sparse_matrix(right_matrix, dropout_rate).tocsr()
            num_interactions += kron_block.nnz
            kron_blocks.append(kron_block)
            logger.info(f"Kronecker block ({i}, {j}) processed.")
        else:
            kron_blocks.append(sparse.csr_matrix(right_matrix.shape))
            logger.info(f"Kronecker block ({i}, {j}) skipped.")

    rows_to_write = sparse.hstack(kron_blocks).tocsr()
    logger.info("Writing dataset row by row.")

    # Write Kronecker product line per line.
    filepath = f"{indices_out_path}_{i}.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        for k in range(right_matrix.shape[0]):
            items_to_write = rows_to_write.getrow(k).indices
            ratings_to_write = rows_to_write.getrow(k).data
            num = items_to_write.shape[0]
            if remove_empty_rows and (not num):
                logger.info(f"Removed empty output row {i * left_matrix.shape[0] + k}.")
                num_removed_rows += 1
                continue
            num_rows += 1
            writer.writerow(
                [
                    i * right_matrix.shape[0] + k,
                    ",".join([str(x) for x in items_to_write]),
                    ",".join([str(x) for x in ratings_to_write]),
                ]
            )
            if k % 100000 == 0:
                logger.info(f"Done producing data set row {k}.")

    num_cols = rows_to_write.shape[1]
    metadata = SparseMatrixMetadata(
        num_interactions=num_interactions, num_rows=num_rows, num_cols=num_cols
    )
    logger.info(
        f"Done with left matrix row {i}, {num_interactions} interactions written in shard, {num_removed_rows} rows removed in shard."
    )
    return (num_removed_rows, metadata)


def visualize_samples(
    right_matrix,
    visualize_num_samples,
    expanded_file_name,
    output_prefix,
):
    # Note: only the rows of the first Kronecker block are visualized.
    logger.info("visualize dataset row by row.")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].set_title("Original data Histogram")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Frequency")
    axs[1].set_title("Expended Row Histogram")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Frequency")
    for k in range(visualize_num_samples):
        original_row = right_matrix.getrow(k).data
        line = linecache.getline(expanded_file_name, k + 1)
        reader = csv.reader([line])
        parsed_line = next(reader)
        expended_row = eval(parsed_line[2])
        original_hist_counts, original_bin_edges = np.histogram(original_row, bins=9)
        expended_hist_counts, expended_bin_edges = np.histogram(expended_row, bins=9)
        axs[0].plot(original_bin_edges[:-1], original_hist_counts, alpha=0.2)
        axs[1].plot(expended_bin_edges[:-1], expended_hist_counts, alpha=0.2)
        axs[0].fill_between(original_bin_edges[:-1], original_hist_counts, alpha=0.2)
        axs[1].fill_between(expended_bin_edges[:-1], expended_hist_counts, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_sample_distribution.png")
    logger.info("Sample visualization finished.")


def build_randomized_kronecker(
    left_matrix,
    right_matrix,
    block_sample_rate,
    indices_out_path,
    metadata_out_path=None,
    remove_empty_rows=True,
):
    """Compute randomized Kronecker product and dump it on the fly based on https://arxiv.org/pdf/1901.08910.pdf."""
    logger.info(f"Writing item sequences to pickle files {metadata_out_path}.")

    num_rows = 0
    num_removed_rows = 0
    num_cols = left_matrix.shape[1] * right_matrix.shape[1]
    num_interactions = 0

    filepath = f"{indices_out_path}_users.csv"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        for i in tqdm(range(left_matrix.shape[0])):
            (shard_num_removed_rows, shard_metadata) = _compute_row_block(
                i,
                left_matrix,
                right_matrix,
                block_sample_rate,
                indices_out_path,
                remove_empty_rows,
            )
            writer.writerow([i, shard_metadata.num_rows])
            file.flush()
            num_rows += shard_metadata.num_rows
            num_removed_rows += shard_num_removed_rows
            num_interactions += shard_metadata.num_interactions

    logger.info(f"{num_interactions / num_rows} average sequence length")
    logger.info(f"{num_interactions} total interactions written.")
    logger.info(f"{num_removed_rows} total rows removed.")

    metadata = SparseMatrixMetadata(
        num_interactions=num_interactions, num_rows=num_rows, num_cols=num_cols
    )
    if metadata_out_path is not None:
        logger.info(f"Writing metadata file to {metadata_out_path}")
        with open(metadata_out_path, "wb") as output_file:
            pickle.dump(metadata, output_file)
    return metadata


def _preprocess_movie_lens(ratings_df, binary=False):
    """
    Filters out users with less than three distinct timestamps.
    """

    def _create_index(df, colname):
        value_set = sorted(set(df[colname].values))
        num_unique = len(value_set)
        return dict(zip(value_set, range(num_unique)))

    if not binary:
        ratings_df["data"] = ratings_df["rating"]
    else:
        ratings_df["data"] = 1.0
    ratings_df["binary_data"] = 1.0
    num_timestamps = ratings_df[["userId", "timestamp"]].groupby("userId").nunique()
    ratings_df["numberOfTimestamps"] = ratings_df["userId"].apply(
        lambda x: num_timestamps["timestamp"][x]
    )
    ratings_df = ratings_df[ratings_df["numberOfTimestamps"] > 2]
    user_id_to_user_idx = _create_index(ratings_df, "userId")
    item_id_to_item_idx = _create_index(ratings_df, "movieId")
    ratings_df["row"] = ratings_df["userId"].apply(lambda x: user_id_to_user_idx[x])
    ratings_df["col"] = ratings_df["movieId"].apply(lambda x: item_id_to_item_idx[x])
    return ratings_df


def normalize(matrix):
    norm_matrix = matrix.copy()
    if isinstance(norm_matrix, np.ndarray):
        norm_matrix -= norm_matrix.mean()
    else:
        norm_matrix.data -= norm_matrix.mean()
    max_val = norm_matrix.max()
    min_val = norm_matrix.min()
    if isinstance(norm_matrix, np.ndarray):
        norm_matrix /= max(abs(max_val), abs(min_val))
    else:
        norm_matrix.data /= max(abs(max_val), abs(min_val))
    return norm_matrix


def plot_distribution(user_wise_sum, item_wise_sum, s, title_prefix, normalized=False):
    y_label = "rating sums" if normalized else "number of ratings"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.loglog(
        np.arange(len(user_wise_sum)) + 1,
        np.sort(user_wise_sum)[::-1],
        linestyle="-",
        color="blue",
        marker="",
    )
    ax1.set_title(f"{title_prefix} matrix user-wise rating sums")
    ax1.set_xlabel("User rank")
    ax1.set_ylabel(y_label)
    ax1.grid(True)
    ax2.loglog(
        np.arange(len(item_wise_sum)) + 1,
        np.sort(item_wise_sum)[::-1],
        linestyle="-",
        color="green",
        marker="",
    )
    ax2.set_title(f"{title_prefix} matrix item-wise rating sums")
    ax2.set_xlabel("Item rank")
    ax2.set_ylabel(y_label)
    ax2.grid(True)
    ax3.loglog(
        np.arange(len(s)) + 1, np.sort(s)[::-1], linestyle="-", color="red", marker=""
    )
    ax3.set_title(f"{title_prefix} matrix singular values")
    ax3.set_xlabel("Singular value Rank")
    ax3.set_ylabel("Magnitude")
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title_prefix}_distribution.png")


def visualize_distribution(mat, reduced_mat, s, reduced_s, normalized=False, title=""):
    user_wise_sum = np.asarray(mat.sum(axis=1)).flatten()
    item_wise_sum = np.asarray(mat.sum(axis=0)).flatten()
    assert len(user_wise_sum) == mat.shape[0]
    assert len(item_wise_sum) == mat.shape[1]
    plot_distribution(
        user_wise_sum,
        item_wise_sum,
        s,
        title_prefix=f"{title}_Original",
        normalized=normalized,
    )

    reduced_user_wise_sum = np.asarray(reduced_mat.sum(axis=1)).flatten()
    reduced_item_wise_sum = np.asarray(reduced_mat.sum(axis=0)).flatten()
    assert len(reduced_user_wise_sum) == reduced_mat.shape[0]
    assert len(reduced_item_wise_sum) == reduced_mat.shape[1]
    plot_distribution(
        reduced_user_wise_sum,
        reduced_item_wise_sum,
        reduced_s,
        title_prefix=f"{title}_Reduced",
        normalized=normalized,
    )

    expanded_s = np.einsum("i,j->ij", reduced_s, s).flatten()
    expanded_user_wise_sum = np.einsum("ij,k->ik", reduced_mat, user_wise_sum).flatten()
    expanded_item_wise_sum = np.einsum("ij,k->jk", reduced_mat, item_wise_sum).flatten()
    assert len(expanded_user_wise_sum) == reduced_mat.shape[0] * mat.shape[0]
    assert len(expanded_item_wise_sum) == reduced_mat.shape[1] * mat.shape[1]
    plot_distribution(
        expanded_user_wise_sum,
        expanded_item_wise_sum,
        expanded_s,
        title_prefix=f"{title}_Expanded",
        normalized=normalized,
    )


def expand_dataset(
    ratings_matrix,
    binary_ratings_matrix,
    num_users,
    num_items,
    reduced_num_rows,
    reduced_num_cols,
    rescale_w_abs,
    element_sample_rate,
    block_sample_rate,
    visualize,
    write_dataset,
    output_prefix,
):
    k = min(reduced_num_rows, reduced_num_cols)
    norm_rating_matrix = normalize(ratings_matrix)
    (u, s, v) = linalg.svds(
        norm_rating_matrix, k=k, maxiter=None, return_singular_vectors=True
    )

    logger.info(
        f"Creating reduced rating matrix (size {reduced_num_rows}, {reduced_num_cols})"
    )
    reduced_matrix = graph_reduce((u, s, v), reduced_num_rows, reduced_num_cols)
    norm_reduced_matrix = normalize(reduced_matrix)
    (_, s_reduce, _) = linalg.svds(
        norm_reduced_matrix, k=k - 1, maxiter=None, return_singular_vectors=True
    )
    reduced_matrix = rescale(
        reduced_matrix,
        rescale_w_abs=rescale_w_abs,
        element_sample_rate=element_sample_rate,
    )
    logger.info(f"largest singular value of the reduced matrix is {s_reduce[-1]}")
    logger.info(
        f"Sampling rate mean is {reduced_matrix.mean()}, var is {reduced_matrix.var()}, min is {reduced_matrix.min()}, max is {reduced_matrix.max()}"
    )
    samples = reduced_matrix.sum() * ratings_matrix.nnz * block_sample_rate
    logger.info(
        f"Expected number of synthetic samples: {samples}, sparsity is {samples / (num_users * num_items * reduced_num_rows * reduced_num_cols)}, average seqlen is {samples / (num_users * reduced_num_rows)}"
    )

    if visualize:
        s = linalg.svds(
            norm_rating_matrix, k=20 * k, maxiter=None, return_singular_vectors=False
        )
        visualize_distribution(
            norm_rating_matrix,
            norm_reduced_matrix,
            s,
            s_reduce,
            normalized=True,
            title="Normalized",
        )
        visualize_distribution(
            binary_ratings_matrix,
            reduced_matrix,
            s,
            s_reduce,
            normalized=False,
            title="Binary",
        )
    if write_dataset:
        output_file = (
            output_prefix + str(reduced_num_rows) + "x" + str(reduced_num_cols)
        )
        output_file_metadata = None

        logger.info(f"Creating synthetic dataset and dumping to {output_file}.")
        build_randomized_kronecker(
            left_matrix=reduced_matrix,
            right_matrix=ratings_matrix.tocoo(),
            block_sample_rate=block_sample_rate,
            indices_out_path=output_file,
            metadata_out_path=output_file_metadata,
        )


@click.command()
@click.option(
    "--random-seed",
    type=int,
    default=0,
)
@click.option(
    "--input-csv-file",
    type=str,
    default="ratings.csv",
)
@click.option(
    "--output-prefix",
    type=str,
    default="",
)
@click.option(
    "--num-row-multiplier",
    type=int,
    default=16,
)
@click.option(
    "--num-col-multiplier",
    type=int,
    default=32,
)
@click.option(
    "--element-sample-rate",
    type=float,
    default=1.0,
)
@click.option(
    "--block-sample-rate",
    type=float,
    default=1.0,
)
@click.option(
    "--visualize",
    type=bool,
    default=False,
)
@click.option(
    "--write-dataset",
    type=bool,
    default=False,
)
@click.option(
    "--visualize-num-samples",
    type=int,
    default=0,
)
def main(
    random_seed: int,
    input_csv_file: str,
    output_prefix: str,
    num_row_multiplier: int,
    num_col_multiplier: int,
    element_sample_rate: float,
    block_sample_rate: float,
    visualize: bool,
    write_dataset: bool,
    visualize_num_samples: int,
):
    np.random.seed(random_seed)

    logger.info(f"Loading and preprocessing MovieLens-20m from {input_csv_file}")
    with open(input_csv_file, "r") as infile:
        ratings_df = pd.read_csv(infile, sep=",", header=0)
    ratings_df = _preprocess_movie_lens(ratings_df, binary=False)
    num_ratings = len(ratings_df)
    num_users = len(set(ratings_df["row"].values))
    num_items = len(set(ratings_df["col"].values))
    logger.info(
        f"number of ratings of input dataset is {num_ratings}, number of users is {num_users}, number of items is {num_items}, sparsity is {num_ratings / (num_users * num_items)}, average seqlen is {num_ratings / num_users}"
    )

    ratings_matrix = sparse.csr_matrix(
        (
            ratings_df["data"].values,
            (ratings_df["row"].values, ratings_df["col"].values),
        ),
        shape=(num_users, num_items),
    )
    binary_ratings_matrix = sparse.csr_matrix(
        (
            ratings_df["binary_data"].values,
            (ratings_df["row"].values, ratings_df["col"].values),
        ),
        shape=(num_users, num_items),
    )
    if write_dataset or visualize:
        expand_dataset(
            ratings_matrix=ratings_matrix,
            binary_ratings_matrix=binary_ratings_matrix,
            num_users=num_users,
            num_items=num_items,
            reduced_num_rows=num_row_multiplier,
            reduced_num_cols=num_col_multiplier,
            rescale_w_abs=False,
            element_sample_rate=element_sample_rate,
            block_sample_rate=block_sample_rate,
            visualize=visualize,
            write_dataset=write_dataset,
            output_prefix=output_prefix,
        )
    if visualize_num_samples > 0:
        logger.info(f"Visualizing {visualize_num_samples} samples.")
        visualize_samples(
            right_matrix=ratings_matrix.tocoo(),
            visualize_num_samples=visualize_num_samples,
            expanded_file_name=f"{output_prefix}{num_row_multiplier}x{num_col_multiplier}_0.csv",
            output_prefix="Sample_Histogram",
        )


if __name__ == "__main__":
    main()
