#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from generative_recommenders.common import gpu_unavailable
from hammer.ops.jagged import replace_last_n_with_jagged

# buck2 test @mode/opt -c fbcode.nvcc_arch=h100 fbcode//generative_recommenders/ops/cpp/tests:replace_last_n_with_jagged_test

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class OpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_replace_last_n_with_jagged(self) -> None:
        torch.manual_seed(42)
        batch_size = 8
        embedding_dim = 64
        max_seq_len_left = 25
        max_seq_len_right = 10

        lengths_left = torch.randint(
            max_seq_len_right, max_seq_len_left + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            1, max_seq_len_right + 1, (batch_size,), dtype=torch.int32
        )

        lengths_right = torch.min(lengths_right, lengths_left)

        total_left = int(lengths_left.sum().item())
        total_right = int(lengths_right.sum().item())

        values_left = torch.randn(total_left, embedding_dim, dtype=torch.float32)
        values_right = torch.randn(total_right, embedding_dim, dtype=torch.float32)

        custom_cpu_result = torch.ops.hstu.replace_last_n_with_jagged(
            lengths_left=lengths_left,
            values_left=values_left,
            lengths_right=lengths_right,
            values_right=values_right,
        )

        custom_cuda_result = torch.ops.hstu.replace_last_n_with_jagged(
            lengths_left=lengths_left.cuda(),
            values_left=values_left.cuda(),
            lengths_right=lengths_right.cuda(),
            values_right=values_right.cuda(),
        )

        torch.testing.assert_close(custom_cuda_result.cpu(), custom_cpu_result)

    @unittest.skipIf(*gpu_unavailable)
    def test_replace_last_n_with_jagged_vs_hammer(self) -> None:
        torch.manual_seed(42)
        batch_size = 8
        embedding_dim = 32
        max_seq_len_left = 20
        max_seq_len_right = 8

        lengths_left = torch.randint(
            max_seq_len_right, max_seq_len_left + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            1, max_seq_len_right + 1, (batch_size,), dtype=torch.int32
        )

        lengths_right = torch.min(lengths_right, lengths_left)

        total_left = int(lengths_left.sum().item())
        total_right = int(lengths_right.sum().item())

        values_left = torch.randn(total_left, embedding_dim, dtype=torch.float32)
        values_right = torch.randn(total_right, embedding_dim, dtype=torch.float32)

        offsets_left = torch.zeros(
            (batch_size + 1,), dtype=lengths_left.dtype, device=lengths_left.device
        )
        offsets_left[1:] = torch.cumsum(lengths_left.view(-1), dim=0)
        offsets_right = torch.zeros(
            (batch_size + 1,), dtype=lengths_right.dtype, device=lengths_right.device
        )
        offsets_right[1:] = torch.cumsum(lengths_right.view(-1), dim=0)

        result_ref = replace_last_n_with_jagged(
            max_seq_len_left=max_seq_len_left,
            offsets_left=offsets_left,
            values_left=values_left,
            offsets_right=offsets_right,
            values_right=values_right,
        )

        custom_cuda_result = torch.ops.hstu.replace_last_n_with_jagged(
            lengths_left=lengths_left.cuda(),
            values_left=values_left.cuda(),
            lengths_right=lengths_right.cuda(),
            values_right=values_right.cuda(),
        )

        torch.testing.assert_close(custom_cuda_result.cpu(), result_ref)
