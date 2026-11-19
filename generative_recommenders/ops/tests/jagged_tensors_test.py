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

#!/usr/bin/env python3

# pyre-strict

import itertools
import unittest
from typing import Any, Dict, List, Optional

import torch
from generative_recommenders.common import (
    generate_sparse_seq_len,
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from generative_recommenders.ops.jagged_tensors import (
    concat_2D_jagged,
    concat_2D_jagged_multirow,
    split_2D_jagged,
    split_2D_jagged_multirow,
)
from generative_recommenders.ops.triton.triton_jagged import (
    set_split_concat_2d_jagged_multirow_kernel,
    triton_split_2D_jagged,
)
from generative_recommenders.ops.utils import is_sm100_plus
from hypothesis import given, settings, strategies as st, Verbosity


# Kernels jagged_dense_bmm_broadcast_add tests run against the PyTorch reference.
# CuTeDSL (tcgen05, cta_group=2) is Blackwell-only.
_JDBBA_KERNELS: List[HammerKernel] = [HammerKernel.TRITON] + (
    [HammerKernel.CUTEDSL] if is_sm100_plus() else []
)


def _jdbba_inputs(kernel: HammerKernel) -> st.SearchStrategy[Dict[str, Any]]:
    """Inputs each kernel computes itself. CuTeDSL takes bf16 with feature dims that are
    multiples of 8 (TMA's 16-byte rows); anything else falls back to Triton."""
    if kernel == HammerKernel.CUTEDSL:
        return st.fixed_dictionaries(
            {
                "real_kernel": st.just(kernel),
                "D": st.sampled_from([64, 136, 256, 520]),
                "K": st.sampled_from([64, 200, 256, 832]),
                "dtype": st.just(torch.bfloat16),
                "contiguous": st.booleans(),
            }
        )
    return st.fixed_dictionaries(
        {
            "real_kernel": st.just(kernel),
            "D": st.integers(20, 200),
            "K": st.integers(30, 200),
            "dtype": st.sampled_from(
                [torch.float32, torch.bfloat16, torch.float16]
                if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
                else [torch.float32]
            ),
            "contiguous": st.booleans(),
        }
    )


class JaggedTensorsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_len_a=st.integers(20, 100),
        max_len_b=st.integers(20, 100),
        D=st.integers(10, 30),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_split_2D_jagged_triton(self, *args, **kwargs) -> None:
        self._test_split_2D_jagged(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_split_2D_jagged(
        self,
        batch_size: int,
        max_len_a: int,
        max_len_b: int,
        D: int,
        is_dense_a: bool,
        is_dense_b: bool,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        dtype: torch.dtype = torch.float32,
        skip_comparisons: bool = False,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import split_2D_jagged

        max_seq_len = max_len_a + max_len_b
        if not is_dense_a:
            lengths_a = torch.randint(
                1, max_len_a + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_a = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
            total_len_a = int(offsets_a[-1].item())
        else:
            offsets_a = None
            total_len_a = batch_size * max_len_a
            is_dense_b = False
        if not is_dense_b:
            lengths_b = torch.randint(
                1, max_len_b + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_b = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
            total_len_b = int(offsets_b[-1].item())
        else:
            offsets_b = None
            total_len_b = batch_size * max_len_b
        values = (
            torch.empty(
                (total_len_a + total_len_b, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        ref_values_a, ref_values_b = split_2D_jagged(
            max_seq_len=max_len_a + max_len_b,
            values=values,
            max_len_left=max_len_a if is_dense_a else None,
            max_len_right=max_len_b if is_dense_b else None,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=ref_kernel,
        )
        d_values_a = torch.randn_like(ref_values_a)
        d_values_b = torch.randn_like(ref_values_b)
        ref_values_a.backward(d_values_a, retain_graph=True)
        ref_values_b.backward(d_values_b)
        if skip_comparisons:
            return

        assert values.grad is not None
        ref_d_values, values.grad = values.grad.clone(), None

        values = values.detach().clone().requires_grad_()
        real_values_a, real_values_b = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            max_len_left=max_len_a if is_dense_a else None,
            max_len_right=max_len_b if is_dense_b else None,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_values_a, real_values_a)
        torch.testing.assert_close(ref_values_b, real_values_b)

        if test_backward:
            d_values_a = d_values_a.detach().clone()
            d_values_b = d_values_b.detach().clone()
            real_values_a.backward(d_values_a, retain_graph=True)
            real_values_b.backward(d_values_b)
            # pyrefly: ignore [missing-attribute]
            real_d_values = values.grad.clone()
            torch.testing.assert_close(ref_d_values, real_d_values)

    @unittest.skipIf(*gpu_unavailable)
    def test_split_2D_jagged_backward_no_host_sync(self) -> None:
        # The lengths arrive as device tensors, as they do from callers that
        # pass `lengths.sum()`. The backward must size its outputs without
        # reading them back to the host.
        device = torch.device("cuda")
        batch_size, prefix_len, max_len_b, D = 16, 8, 64, 32
        lengths_a = torch.full(
            (batch_size,), prefix_len, dtype=torch.int64, device=device
        )
        lengths_b = torch.randint(
            1, max_len_b + 1, (batch_size,), dtype=torch.int64, device=device
        )
        offsets_a = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        offsets_b = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
        offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
        total_len_a = int(offsets_a[-1].item())
        total_len_b = int(offsets_b[-1].item())
        # skip_left with the single-row kernel takes the backward's zero-fill
        # path; the multirow kernel skips it.
        self.addCleanup(set_split_concat_2d_jagged_multirow_kernel, None)
        # A jagged right part gets its lengths as device tensors, the left one
        # directly or through total_seq_len. A dense right part takes both
        # lengths from shapes, so it only checks the zero-fill sizing.
        cases = [(False, False), (False, True), (True, False)]
        for case, use_multirow, skip_left in itertools.product(
            cases, [False, True], [False, True]
        ):
            is_dense_b, use_total_seq_len = case
            set_split_concat_2d_jagged_multirow_kernel(use_multirow)
            num_rows_b = batch_size * max_len_b if is_dense_b else total_len_b
            values = torch.randn(
                (total_len_a + num_rows_b, D), device=device, requires_grad=True
            )
            _, values_b = triton_split_2D_jagged(
                values,
                prefix_len + max_len_b,
                offsets_a,
                None if is_dense_b else offsets_b,
                dense_size=max_len_b if is_dense_b else 0,
                seq_len_a=None if use_total_seq_len else lengths_a.sum(),
                seq_len_b=None if is_dense_b else lengths_b.sum(),
                total_seq_len=total_len_a + num_rows_b if use_total_seq_len else None,
                skip_left=skip_left,
            )
            d_values_b = torch.randn_like(values_b)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                torch.autograd.backward([values_b], [d_values_b])
            finally:
                torch.cuda.set_sync_debug_mode("default")
            ref_values = values.detach().clone().requires_grad_()
            _, ref_values_b = split_2D_jagged(
                max_seq_len=prefix_len + max_len_b,
                values=ref_values,
                max_len_left=None,
                max_len_right=max_len_b if is_dense_b else None,
                offsets_left=offsets_a,
                offsets_right=None if is_dense_b else offsets_b,
                kernel=HammerKernel.PYTORCH,
            )
            # The PyTorch reference returns a dense right part flat.
            ref_values_b.backward(d_values_b.reshape(ref_values_b.shape))
            torch.testing.assert_close(values.grad, ref_values.grad)

    @unittest.skipIf(*gpu_unavailable)
    def test_split_2D_jagged_rejects_uncovered_rows(self) -> None:
        device = torch.device("cuda")
        offsets_a = torch.tensor([0, 2, 4], device=device)
        offsets_b = torch.tensor([0, 3, 5], device=device)
        values = torch.randn((10, 8), device=device, requires_grad=True)
        with self.assertRaisesRegex(AssertionError, "values has 10 rows"):
            triton_split_2D_jagged(
                values, 8, offsets_a, offsets_b, seq_len_a=4, seq_len_b=5
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_split_2D_jagged_backward_compiled_caller(self) -> None:
        # Called from compiled code, the forward keeps the lengths as device
        # tensors (it reads the last offsets with index_select); the backward
        # runs eagerly and must still give the reference gradient without a
        # host sync.
        device = torch.device("cuda")
        batch_size, prefix_len, max_len_b, D = 16, 8, 64, 32
        max_seq_len = prefix_len + max_len_b
        lengths_a = torch.full(
            (batch_size,), prefix_len, dtype=torch.int64, device=device
        )
        lengths_b = torch.randint(
            1, max_len_b + 1, (batch_size,), dtype=torch.int64, device=device
        )
        offsets_a = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        offsets_b = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
        offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
        total_len = int(offsets_a[-1].item()) + int(offsets_b[-1].item())

        @torch.compile
        def right_part(v: torch.Tensor) -> torch.Tensor:
            _, values_b = triton_split_2D_jagged(v, max_seq_len, offsets_a, offsets_b)
            return values_b * 2.0

        # The first backward also compiles, so only the second one is checked.
        for check_sync in [False, True]:
            values = torch.randn((total_len, D), device=device, requires_grad=True)
            out = right_part(values)
            d_out = torch.randn_like(out)
            torch.cuda.synchronize()
            if check_sync:
                torch.cuda.set_sync_debug_mode("error")
            try:
                out.backward(d_out)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            ref_values = values.detach().clone().requires_grad_()
            _, ref_values_b = split_2D_jagged(
                max_seq_len=max_seq_len,
                values=ref_values,
                max_len_left=None,
                max_len_right=None,
                offsets_left=offsets_a,
                offsets_right=offsets_b,
                kernel=HammerKernel.PYTORCH,
            )
            (ref_values_b * 2.0).backward(d_out)
            torch.testing.assert_close(values.grad, ref_values.grad)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_len_a=st.integers(20, 100),
        max_len_b=st.integers(20, 100),
        D=st.integers(10, 30),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_concat_2D_jagged_triton(self, *args, **kwargs) -> None:
        self._test_concat_2D_jagged(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.sampled_from([130]),
        max_len_a=st.sampled_from([32768]),
        max_len_b=st.sampled_from([10]),
        D=st.sampled_from([512]),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_concat_2D_jagged_large_tensor(self, *args, **kwargs) -> None:
        self._test_concat_2D_jagged(
            *args,
            **kwargs,
            test_backward=True,
            skip_comparisons=True,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_concat_2D_jagged(
        self,
        batch_size: int,
        max_len_a: int,
        max_len_b: int,
        D: int,
        is_dense_a: bool,
        is_dense_b: bool,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        dtype: torch.dtype = torch.float32,
        skip_comparisons: bool = False,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import concat_2D_jagged

        if not is_dense_a:
            lengths_a = torch.randint(
                1, max_len_a + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_a = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
            total_len_a = int(offsets_a[-1].item())
        else:
            offsets_a = None
            total_len_a = batch_size * max_len_a
        values_a = (
            torch.empty(
                (total_len_a, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if not is_dense_b:
            lengths_b = torch.randint(
                1, max_len_b + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_b = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
            total_len_b = int(offsets_b[-1].item())
        else:
            offsets_b = None
            total_len_b = batch_size * max_len_b
        values_b = (
            torch.empty(
                (total_len_b, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        ref_values = concat_2D_jagged(
            max_seq_len=max_len_a + max_len_b,
            values_left=values_a,
            values_right=values_b,
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_values)
        ref_values.backward(dout)
        if skip_comparisons:
            return

        assert values_a.grad is not None
        ref_d_a, values_a.grad = values_a.grad.clone(), None
        assert values_b.grad is not None
        ref_d_b, values_b.grad = values_b.grad.clone(), None

        values_a = values_a.detach().clone().requires_grad_()
        values_b = values_b.detach().clone().requires_grad_()
        dout = dout.detach().clone()
        real_values = concat_2D_jagged(
            max_seq_len=max_len_a + max_len_b,
            values_left=values_a,
            values_right=values_b,
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_values, real_values)

        if test_backward:
            real_values.backward(dout)
            # pyrefly: ignore [missing-attribute]
            real_d_a = values_a.grad.clone()
            # pyrefly: ignore [missing-attribute]
            real_d_b = values_b.grad.clone()
            torch.testing.assert_close(ref_d_a, real_d_a)
            torch.testing.assert_close(ref_d_b, real_d_b)

    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_uih_len=st.integers(20, 100),
        max_l2_len=st.integers(10, 30),
        contextual_seq_len=st.sampled_from([0, 10]),
        max_targets=st.sampled_from([10, 20]),
        D=st.integers(10, 30),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_hstu_split_l2_embeddings(
        self,
        batch_size: int,
        max_uih_len: int,
        max_l2_len: int,
        contextual_seq_len: int,
        max_targets: int,
        D: int,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import hstu_split_l2_embeddings

        max_seq_len = max_uih_len + max_targets + contextual_seq_len
        num_targets = torch.randint(
            1, max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        x_lengths = torch.randint(
            0,
            max_uih_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        x_lengths = num_targets + x_lengths + contextual_seq_len
        x_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        x_offsets[1:] = torch.cumsum(x_lengths, dim=0)
        total_len = int(x_offsets[-1].item())
        x = (
            torch.empty(
                (total_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        prefix_lengths = x_lengths - max_l2_len - num_targets - contextual_seq_len
        prefix_lengths = torch.clamp(prefix_lengths, min=0)
        prefix_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(prefix_lengths)
        l2_offsets = x_offsets - prefix_offsets
        ref_prefix_x, ref_l2_x = hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            prefix_offsets=prefix_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.PYTORCH,
        )
        d_prefix_x = torch.randn_like(ref_prefix_x)
        d_l2_x = torch.randn_like(ref_l2_x)
        ref_prefix_x.backward(d_prefix_x, retain_graph=True)
        ref_l2_x.backward(d_l2_x)
        assert x.grad is not None
        ref_d_x, x.grad = x.grad.clone(), None
        x = x.detach().clone().requires_grad_()
        real_prefix_x, real_l2_x = hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            prefix_offsets=prefix_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.TRITON,
        )
        print(ref_prefix_x.shape, real_prefix_x.shape)
        torch.testing.assert_close(ref_prefix_x, real_prefix_x)
        torch.testing.assert_close(ref_l2_x, real_l2_x)
        d_prefix_x = d_prefix_x.detach().clone()
        d_l2_x = d_l2_x.detach().clone()
        real_prefix_x.backward(d_prefix_x, retain_graph=True)
        real_l2_x.backward(d_l2_x)
        # pyrefly: ignore [missing-attribute]
        real_d_x = x.grad.clone()
        torch.testing.assert_close(ref_d_x, real_d_x)

    # pyre-ignore
    @given(
        batch_size=st.integers(1, 1),
        max_prefix_len=st.integers(10, 10),
        max_l2_len=st.integers(5, 5),
        contextual_seq_len=st.sampled_from([3]),
        max_targets=st.sampled_from([2]),
        D=st.integers(10, 10),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_hstu_concat_l2_embeddings(
        self,
        batch_size: int,
        max_prefix_len: int,
        max_l2_len: int,
        contextual_seq_len: int,
        max_targets: int,
        D: int,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import hstu_concat_l2_embeddings

        num_targets = torch.randint(
            1, max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        l2_lengths = torch.randint(
            0,
            max_l2_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        l2_lengths = num_targets + l2_lengths + contextual_seq_len
        max_l2_len = max_l2_len + contextual_seq_len + max_targets
        l2_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        l2_offsets[1:] = torch.cumsum(l2_lengths, dim=0)
        total_l2_len = int(l2_offsets[-1].item())
        l2_x = (
            torch.empty(
                (total_l2_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        prefix_lengths = torch.randint(
            0,
            max_prefix_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        prefix_lengths = torch.randint(
            0,
            max_prefix_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        prefix_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        prefix_offsets[1:] = torch.cumsum(prefix_lengths, dim=0)
        total_prefix_len = int(prefix_offsets[-1].item())
        prefix_x = (
            torch.empty(
                (total_prefix_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        ref_x = hstu_concat_l2_embeddings(
            max_prefix_len=max_prefix_len,
            prefix_x=prefix_x,
            prefix_offsets=prefix_offsets,
            max_l2_len=max_l2_len,
            l2_x=l2_x,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.PYTORCH,
        )
        dout = torch.randn_like(ref_x)
        ref_x.backward(dout)

        assert prefix_x.grad is not None
        ref_d_prefix_x, prefix_x.grad = prefix_x.grad.clone(), None
        assert l2_x.grad is not None
        ref_d_l2_x, l2_x.grad = l2_x.grad.clone(), None

        prefix_x = prefix_x.detach().clone().requires_grad_()
        l2_x = l2_x.detach().clone().requires_grad_()
        real_x = hstu_concat_l2_embeddings(
            max_prefix_len=max_prefix_len,
            prefix_x=prefix_x,
            prefix_offsets=prefix_offsets,
            max_l2_len=max_l2_len,
            l2_x=l2_x,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.TRITON,
        )
        torch.testing.assert_close(ref_x, real_x)
        dout = dout.detach().clone()
        real_x.backward(dout)
        # pyrefly: ignore [missing-attribute]
        real_d_prefix_x = prefix_x.grad.clone()
        # pyrefly: ignore [missing-attribute]
        real_d_l2_x = l2_x.grad.clone()
        torch.testing.assert_close(ref_d_prefix_x, real_d_prefix_x)
        torch.testing.assert_close(ref_d_l2_x, real_d_l2_x)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        max_seq_len=st.integers(50, 500),
        case=st.sampled_from(_JDBBA_KERNELS).flatmap(_jdbba_inputs),
    )
    @settings(
        verbosity=Verbosity.verbose,
        # About 20 per kernel.
        max_examples=20 * len(_JDBBA_KERNELS),
        deadline=None,
    )
    def test_jagged_dense_bmm_broadcast_add(
        self, batch_size: int, max_seq_len: int, case: Dict[str, Any]
    ) -> None:
        self._test_jagged_dense_bmm_broadcast_add(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            **case,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=HammerKernel.PYTORCH,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.sampled_from([130]),
        max_seq_len=st.sampled_from([32768]),
        D=st.sampled_from([512]),
        K=st.sampled_from([512]),
        dtype=st.sampled_from([torch.float32, torch.bfloat16]),
        contiguous=st.booleans(),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=1,
        deadline=None,
    )
    def test_jagged_dense_bmm_broadcast_add_triton_large_tensor(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        **kwargs,  # pyre-ignore[2]
    ) -> None:
        self._test_jagged_dense_bmm_broadcast_add(
            *args,
            **kwargs,
            test_backward=True,
            atol=None,
            rtol=None,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_jagged_dense_bmm_broadcast_add(
        self,
        batch_size: int,
        max_seq_len: int,
        D: int,
        K: int,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        test_backward: bool,
        contiguous: bool = True,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        sparsity: float = -1,
        lengths: Optional[torch.Tensor] = None,
    ) -> None:
        set_dev_mode(True)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        from generative_recommenders.ops.jagged_tensors import (
            jagged_dense_bmm_broadcast_add,
        )

        if lengths is not None:
            # Caller-supplied, for deterministic length patterns the random draw below
            # cannot reach -- see test_jagged_dense_bmm_broadcast_add_cutedsl_shapes.
            lengths = lengths.to(device=torch.device("cuda"), dtype=torch.int64)
        elif sparsity > 0.0:
            lengths = generate_sparse_seq_len(
                size=batch_size,
                max_seq_len=max_seq_len,
                sparsity=sparsity,
                device=torch.device("cuda"),
            ).to(torch.int64)
        else:
            lengths = torch.randint(
                max_seq_len + 1, size=(batch_size,), device=torch.device("cuda")
            )
        # Test the edge case with an empty row
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        jagged_size = int(seq_offsets[-1].item())
        jagged = (
            torch.empty((jagged_size, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        dense = (
            torch.empty((batch_size, D, K), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias = (
            torch.empty((batch_size, K), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if not contiguous:
            dense = (
                dense.transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
                .detach()
                .clone()
                .requires_grad_()
            )

        ref_out = jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
            kernel=ref_kernel,
        ).to(jagged.dtype)
        if test_backward:
            dout = torch.randn_like(ref_out) * 0.01
            ref_out.backward(dout)
            # pyre-ignore
            ref_d_jagged, jagged.grad = jagged.grad.clone(), None
            # pyrefly: ignore [missing-attribute]
            ref_d_dense, dense.grad = dense.grad.clone(), None
            # pyrefly: ignore [missing-attribute]
            ref_d_bias, bias.grad = bias.grad.clone(), None

        jagged = jagged.detach().clone().requires_grad_()
        dense = dense.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()
        real_out = jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
            kernel=real_kernel,
        )
        torch.testing.assert_close(
            ref_out,
            real_out,
            atol=atol,
            rtol=rtol,
        )
        if test_backward:
            real_out.backward(dout)  # pyre-ignore
            # pyrefly: ignore [missing-attribute]
            real_d_jagged = jagged.grad.clone()
            # pyrefly: ignore [missing-attribute]
            real_d_dense = dense.grad.clone()
            # pyrefly: ignore [missing-attribute]
            real_d_bias = bias.grad.clone()
            torch.testing.assert_close(
                ref_d_jagged,  # pyre-ignore
                real_d_jagged,
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(
                ref_d_dense,  # pyre-ignore
                real_d_dense,
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(
                ref_d_bias,  # pyre-ignore
                real_d_bias,
                atol=atol,
                rtol=rtol,
            )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_len_a=st.integers(20, 100),
        max_len_b=st.integers(20, 100),
        D=st.integers(10, 30),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_concat_2D_jagged_multirow_triton(self, *args, **kwargs) -> None:
        self._test_concat_2D_jagged_multirow(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_concat_2D_jagged_multirow(
        self,
        batch_size: int,
        max_len_a: int,
        max_len_b: int,
        D: int,
        is_dense_a: bool,
        is_dense_b: bool,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        set_dev_mode(True)

        if not is_dense_a:
            lengths_a = torch.randint(
                1, max_len_a + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_a = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
            total_len_a = int(offsets_a[-1].item())
        else:
            offsets_a = None
            total_len_a = batch_size * max_len_a
        values_a = (
            torch.empty(
                (total_len_a, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if not is_dense_b:
            lengths_b = torch.randint(
                1, max_len_b + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_b = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
            total_len_b = int(offsets_b[-1].item())
        else:
            offsets_b = None
            total_len_b = batch_size * max_len_b
        values_b = (
            torch.empty(
                (total_len_b, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        ref_values = concat_2D_jagged(
            max_seq_len=max_len_a + max_len_b,
            values_left=values_a,
            values_right=values_b,
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_values) * 0.1
        ref_values.backward(dout)
        assert values_a.grad is not None
        ref_d_a, values_a.grad = values_a.grad.clone(), None
        assert values_b.grad is not None
        ref_d_b, values_b.grad = values_b.grad.clone(), None

        values_a = values_a.detach().clone().requires_grad_()
        values_b = values_b.detach().clone().requires_grad_()
        dout = dout.detach().clone()

        real_values = concat_2D_jagged_multirow(
            max_seq_len=max_len_a + max_len_b,
            values_left=values_a,
            values_right=values_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_values, real_values)
        if test_backward:
            real_values.backward(dout)
            # pyrefly: ignore [missing-attribute]
            real_d_a = values_a.grad.clone()
            # pyrefly: ignore [missing-attribute]
            real_d_b = values_b.grad.clone()
            torch.testing.assert_close(ref_d_a, real_d_a)
            torch.testing.assert_close(ref_d_b, real_d_b)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_len_a=st.integers(20, 100),
        max_len_b=st.integers(20, 100),
        D=st.integers(10, 30),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_split_2D_jagged_multirow_triton(self, *args, **kwargs) -> None:
        self._test_split_2D_jagged_multirow(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_split_2D_jagged_multirow(
        self,
        batch_size: int,
        max_len_a: int,
        max_len_b: int,
        D: int,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        set_dev_mode(True)

        lengths_a = torch.randint(
            1, max_len_a + 1, size=(batch_size,), device=torch.device("cuda")
        )
        offsets_a = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        offsets_a[1:] = torch.cumsum(lengths_a, dim=0)

        lengths_b = torch.randint(
            1, max_len_b + 1, size=(batch_size,), device=torch.device("cuda")
        )
        offsets_b = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        offsets_b[1:] = torch.cumsum(lengths_b, dim=0)

        total_len = int(offsets_a[-1].item()) + int(offsets_b[-1].item())
        values = (
            torch.empty(
                (total_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        ref_values_a, ref_values_b = split_2D_jagged(
            max_seq_len=max_len_a + max_len_b,
            values=values,
            total_len_left=int(offsets_a[-1].item()),
            total_len_right=int(offsets_b[-1].item()),
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=ref_kernel,
        )
        d_values_a = torch.randn_like(ref_values_a) * 0.1
        d_values_b = torch.randn_like(ref_values_b) * 0.1
        ref_values_a.backward(d_values_a, retain_graph=True)
        ref_values_b.backward(d_values_b)
        assert values.grad is not None
        ref_d_values, values.grad = values.grad.clone(), None

        values = values.detach().clone().requires_grad_()
        d_values_a = d_values_a.detach().clone()
        d_values_b = d_values_b.detach().clone()

        max_len_a_actual = int((offsets_a[1:] - offsets_a[:-1]).max().item())
        max_len_b_actual = int((offsets_b[1:] - offsets_b[:-1]).max().item())

        real_values_a, real_values_b = split_2D_jagged_multirow(
            max_seq_len=max_len_a + max_len_b,
            values=values,
            total_len_left=int(offsets_a[-1].item()),
            total_len_right=int(offsets_b[-1].item()),
            max_len_left=max_len_a_actual,
            max_len_right=max_len_b_actual,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_values_a, real_values_a)
        torch.testing.assert_close(ref_values_b, real_values_b)
        if test_backward:
            real_values_a.backward(d_values_a, retain_graph=True)
            real_values_b.backward(d_values_b)
            # pyrefly: ignore [missing-attribute]
            real_d_values = values.grad.clone()
            torch.testing.assert_close(ref_d_values, real_d_values)
