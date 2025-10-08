#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from generative_recommenders.common import gpu_unavailable
from hammer.ops.jagged import jagged_transpose_1D

from hypothesis import given, settings, strategies as st, Verbosity

# buck2 test @mode/opt -c fbcode.nvcc_arch=h100 fbcode//generative_recommenders/ops/cpp/tests:jagged_transpose_1d_test

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class OpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        size1=st.integers(2, 10),
        size2=st.integers(2, 10),
        max_len=st.integers(5, 50),
        val_dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=100,
        deadline=None,
    )
    def test_jagged_transpose_1d(
        self,
        size1: int,
        size2: int,
        max_len: int,
        val_dtype: torch.dtype,
    ) -> None:
        lengths = torch.randint(
            0, max_len + 1, (size1 * size2,), dtype=torch.int32, device="cpu"
        )
        offsets = torch.zeros(
            (size1 * size2 + 1,), dtype=lengths.dtype, device=lengths.device
        )
        offsets[1:] = torch.cumsum(lengths.view(-1), dim=0)

        values = torch.randn(int(offsets[-1].item()), dtype=val_dtype, device="cpu")

        (
            custom_cpu_values,
            custom_cpu_offsets,
            custom_cpu_lengths,
        ) = torch.ops.hstu.jagged_transpose_1d(
            values=values,
            offsets=offsets,
            lengths=lengths,
            max_len=max_len,
            size1=size1,
            size2=size2,
        )

        (
            custom_cuda_values,
            custom_cuda_offsets,
            custom_cuda_lengths,
        ) = torch.ops.hstu.jagged_transpose_1d(
            values=values.cuda(),
            offsets=offsets.cuda(),
            lengths=lengths.cuda(),
            max_len=max_len,
            size1=size1,
            size2=size2,
        )

        torch.testing.assert_close(custom_cuda_values.cpu(), custom_cpu_values)
        torch.testing.assert_close(custom_cuda_offsets.cpu(), custom_cpu_offsets)
        torch.testing.assert_close(custom_cuda_lengths.cpu(), custom_cpu_lengths)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        size1=st.integers(2, 10),
        size2=st.integers(2, 10),
        max_len=st.integers(5, 50),
        val_dtype=st.sampled_from([torch.float32, torch.float16, torch.bfloat16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=100,
        deadline=None,
    )
    def test_jagged_transpose_1d_vs_hammer(
        self,
        size1: int,
        size2: int,
        max_len: int,
        val_dtype: torch.dtype,
    ) -> None:
        lengths = torch.randint(0, max_len + 1, (size1 * size2,), dtype=torch.int32)
        offsets = torch.zeros(
            (size1 * size2 + 1,), dtype=lengths.dtype, device=lengths.device
        )
        offsets[1:] = torch.cumsum(lengths.view(-1), dim=0)

        values = torch.randn(int(offsets[-1].item()), dtype=val_dtype)

        values_ref, offsets_ref, lengths_ref = jagged_transpose_1D(
            values=values,
            offsets=offsets,
            lengths=lengths,
            max_len=max_len,
            size1=size1,
            size2=size2,
        )

        (
            custom_cuda_values,
            custom_cuda_offsets,
            custom_cuda_lengths,
        ) = torch.ops.hstu.jagged_transpose_1d(
            values=values.cuda(),
            offsets=offsets.cuda(),
            lengths=lengths.cuda(),
            max_len=max_len,
            size1=size1,
            size2=size2,
        )

        torch.testing.assert_close(custom_cuda_values.cpu(), values_ref)
        torch.testing.assert_close(custom_cuda_offsets.cpu(), offsets_ref)
        torch.testing.assert_close(custom_cuda_lengths.cpu(), lengths_ref)
