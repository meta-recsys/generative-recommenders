#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from generative_recommenders.common import gpu_unavailable
from hammer.ops.jagged import concat_1D_jagged_jagged
from hypothesis import given, settings, strategies as st, Verbosity

# buck2 test @mode/opt -c fbcode.nvcc_arch=h100 fbcode//generative_recommenders/ops/cpp/tests:concat_1d_jagged_jagged_test

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class OpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(10, 500),
        max_seq_len_left=st.integers(10, 1000),
        max_seq_len_right=st.integers(10, 1000),
        val_dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=100,
        deadline=None,
    )
    def test_concat_1d_jagged_jagged(
        self,
        batch_size: int,
        max_seq_len_left: int,
        max_seq_len_right: int,
        val_dtype: torch.dtype,
    ) -> None:
        batch_size = 3
        max_seq_len_left = 4
        max_seq_len_right = 2
        lengths_left = torch.randint(
            0, max_seq_len_left + 1, (batch_size,), device="cpu"
        )
        values_left = torch.rand(
            (int(torch.sum(lengths_left).cpu().item()),), dtype=val_dtype, device="cpu"
        )
        offsets_left = torch.zeros(
            (batch_size + 1,),
            dtype=lengths_left.dtype,
            device=lengths_left.device,
        )
        offsets_left[1:] = torch.cumsum(lengths_left.view(-1), dim=0)
        lengths_right = torch.randint(
            0, max_seq_len_right + 1, (batch_size,), device="cpu"
        )
        values_right = torch.rand(
            (int(torch.sum(lengths_right).cpu().item()),), dtype=val_dtype, device="cpu"
        )
        offsets_right = torch.zeros(
            (batch_size + 1,),
            dtype=lengths_right.dtype,
            device=lengths_right.device,
        )
        offsets_right[1:] = torch.cumsum(lengths_right.view(-1), dim=0)
        custom_cpu_result = torch.ops.hstu.concat_1d_jagged_jagged(
            lengths_left=lengths_left,
            values_left=values_left,
            lengths_right=lengths_right,
            values_right=values_right,
        )

        custom_cuda_result = torch.ops.hstu.concat_1d_jagged_jagged(
            lengths_left=lengths_left.cuda(),
            values_left=values_left.cuda(),
            lengths_right=lengths_right.cuda(),
            values_right=values_right.cuda(),
        )
        torch.testing.assert_close(custom_cuda_result.cpu(), custom_cpu_result)

    @unittest.skipIf(*gpu_unavailable)
    def test_concat_1d_jagged_jagged_vs_hammer(self) -> None:
        torch.manual_seed(42)
        batch_size = 8
        max_seq_len_left = 50
        max_seq_len_right = 30

        lengths_left = torch.randint(
            0, max_seq_len_left + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            0, max_seq_len_right + 1, (batch_size,), dtype=torch.int32
        )

        total_left = int(lengths_left.sum().item())
        total_right = int(lengths_right.sum().item())

        values_left = (
            torch.randn(total_left, dtype=torch.float32)
            if total_left > 0
            else torch.empty(0, dtype=torch.float32)
        )
        values_right = (
            torch.randn(total_right, dtype=torch.float32)
            if total_right > 0
            else torch.empty(0, dtype=torch.float32)
        )

        offsets_left = torch.zeros(
            (batch_size + 1,), dtype=lengths_left.dtype, device=lengths_left.device
        )
        offsets_left[1:] = torch.cumsum(lengths_left.view(-1), dim=0)
        offsets_right = torch.zeros(
            (batch_size + 1,), dtype=lengths_right.dtype, device=lengths_right.device
        )
        offsets_right[1:] = torch.cumsum(lengths_right.view(-1), dim=0)

        combined_values_ref = concat_1D_jagged_jagged(
            max_seq_len_left=max_seq_len_left,
            offsets_left=offsets_left,
            values_left=values_left,
            max_seq_len_right=max_seq_len_right,
            offsets_right=offsets_right,
            values_right=values_right,
        )

        custom_cuda_result = torch.ops.hstu.concat_1d_jagged_jagged(
            lengths_left=lengths_left.cuda(),
            values_left=values_left.cuda(),
            lengths_right=lengths_right.cuda(),
            values_right=values_right.cuda(),
        )

        torch.testing.assert_close(custom_cuda_result.cpu(), combined_values_ref)
