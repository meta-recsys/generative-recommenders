#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from generative_recommenders.common import gpu_unavailable
from hammer.ops.jagged import split_1D_jagged_jagged

# buck2 test @mode/opt -c fbcode.nvcc_arch=h100 fbcode//generative_recommenders/ops/cpp/tests:split_1d_jagged_jagged_test

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class OpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_split_1d_jagged_jagged(self) -> None:
        torch.manual_seed(42)
        batch_size = 8
        max_seq_len_left = 25
        max_seq_len_right = 20

        lengths_left = torch.randint(
            0, max_seq_len_left + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            0, max_seq_len_right + 1, (batch_size,), dtype=torch.int32
        )

        combined_lengths = lengths_left + lengths_right
        combined_offsets = torch.zeros(
            (batch_size + 1,), dtype=lengths_left.dtype, device=lengths_left.device
        )
        combined_offsets[1:] = torch.cumsum(combined_lengths.view(-1), dim=0)

        combined_values = torch.randn(
            int(combined_offsets[-1].item()), dtype=torch.float32
        )

        custom_cpu_left, custom_cpu_right = torch.ops.hstu.split_1d_jagged_jagged(
            lengths_left=lengths_left,
            lengths_right=lengths_right,
            combined_values=combined_values,
        )

        custom_cuda_left, custom_cuda_right = torch.ops.hstu.split_1d_jagged_jagged(
            lengths_left=lengths_left.cuda(),
            lengths_right=lengths_right.cuda(),
            combined_values=combined_values.cuda(),
        )

        torch.testing.assert_close(custom_cuda_left.cpu(), custom_cpu_left)
        torch.testing.assert_close(custom_cuda_right.cpu(), custom_cpu_right)

    @unittest.skipIf(*gpu_unavailable)
    def test_split_1d_jagged_jagged_vs_hammer(self) -> None:
        torch.manual_seed(42)
        batch_size = 8
        max_seq_len_left = 25
        max_seq_len_right = 20

        lengths_left = torch.randint(
            0, max_seq_len_left + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            0, max_seq_len_right + 1, (batch_size,), dtype=torch.int32
        )

        offsets_left = torch.zeros(
            (batch_size + 1,), dtype=lengths_left.dtype, device=lengths_left.device
        )
        offsets_left[1:] = torch.cumsum(lengths_left.view(-1), dim=0)
        offsets_right = torch.zeros(
            (batch_size + 1,), dtype=lengths_right.dtype, device=lengths_right.device
        )
        offsets_right[1:] = torch.cumsum(lengths_right.view(-1), dim=0)

        combined_offsets = offsets_left + offsets_right
        combined_values = torch.randn(
            int(combined_offsets[-1].item()), dtype=torch.float32
        )

        left_ref, right_ref = split_1D_jagged_jagged(
            max_seq_len=max_seq_len_left + max_seq_len_right,
            values=combined_values,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )

        custom_cuda_left, custom_cuda_right = torch.ops.hstu.split_1d_jagged_jagged(
            lengths_left=lengths_left.cuda(),
            lengths_right=lengths_right.cuda(),
            combined_values=combined_values.cuda(),
        )

        torch.testing.assert_close(custom_cuda_left.cpu(), left_ref)
        torch.testing.assert_close(custom_cuda_right.cpu(), right_ref)
