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

# pyre-strict

import unittest
from typing import List, Tuple

import fbgemm_gpu.sparse_ops  # noqa: F401 -- fake (abstract) impls of fbgemm jagged ops
import torch
from generative_recommenders.ops.pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
    pytorch_split_2D_jagged,
)
from torch.fx.experimental.proxy_tensor import make_fx


def _offsets(lengths: List[int]) -> torch.Tensor:
    return torch.tensor([0] + lengths).cumsum(0)


def _ref_concat(
    left: torch.Tensor, ll: List[int], right: torch.Tensor, lr: List[int]
) -> torch.Tensor:
    parts, a, b = [], 0, 0
    for nl, nr in zip(ll, lr):
        parts += [left[a : a + nl], right[b : b + nr]]
        a, b = a + nl, b + nr
    return torch.cat(parts)


def _ref_split(
    values: torch.Tensor, ll: List[int], lr: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    left, right, pos = [], [], 0
    for nl, nr in zip(ll, lr):
        left.append(values[pos : pos + nl])
        right.append(values[pos + nl : pos + nl + nr])
        pos += nl + nr
    return torch.cat(left), torch.cat(right)


# Includes empty rows on either side and a left side longer than the right.
LEFT = [3, 0, 5, 2]
RIGHT = [2, 4, 0, 1]
D = 3


class PtJaggedTensorsStaticTest(unittest.TestCase):
    def test_concat_matches_reference(self) -> None:
        left = torch.randn(sum(LEFT), D, requires_grad=True)
        right = torch.randn(sum(RIGHT), D, requires_grad=True)
        out = pytorch_concat_2D_jagged(
            values_left=left,
            values_right=right,
            max_len_left=max(LEFT),
            max_len_right=max(RIGHT),
            offsets_left=_offsets(LEFT),
            offsets_right=_offsets(RIGHT),
        )
        ref = _ref_concat(left, LEFT, right, RIGHT)
        torch.testing.assert_close(out, ref)
        grad = torch.randn_like(out)
        got = torch.autograd.grad(out, (left, right), grad)
        want = torch.autograd.grad(ref, (left, right), grad)
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w)

    def test_split_matches_reference(self) -> None:
        values = torch.randn(sum(LEFT) + sum(RIGHT), D, requires_grad=True)
        max_seq_len = max(a + b for a, b in zip(LEFT, RIGHT))
        for totals in [
            (sum(LEFT), sum(RIGHT)),
            (sum(LEFT), None),
            (None, sum(RIGHT)),
            (None, None),
        ]:
            out = pytorch_split_2D_jagged(
                max_seq_len=max_seq_len,
                values=values,
                max_len_left=None,
                max_len_right=None,
                offsets_left=_offsets(LEFT),
                offsets_right=_offsets(RIGHT),
                total_len_left=totals[0],
                total_len_right=totals[1],
            )
            ref = _ref_split(values, LEFT, RIGHT)
            for o, r in zip(out, ref):
                torch.testing.assert_close(o, r)
            grads = [torch.randn_like(o) for o in out]
            got = torch.autograd.grad(out, values, grads)[0]
            want = torch.autograd.grad(ref, values, grads)[0]
            torch.testing.assert_close(got, want)

    def test_split_dense_side(self) -> None:
        # offsets_right=None: the right side is dense with max_len_right rows.
        n_right = 2
        lr = [n_right] * len(LEFT)
        values = torch.randn(sum(LEFT) + sum(lr), D)
        left, right = pytorch_split_2D_jagged(
            max_seq_len=max(LEFT) + n_right,
            values=values,
            max_len_left=None,
            max_len_right=n_right,
            offsets_left=_offsets(LEFT),
            offsets_right=None,
            total_len_right=sum(lr),
        )
        ref_left, ref_right = _ref_split(values, LEFT, lr)
        torch.testing.assert_close(left, ref_left)
        torch.testing.assert_close(right, ref_right)

    def test_fake_trace_has_static_shapes(self) -> None:
        """With totals given, forward + backward trace on fake tensors (any
        data-dependent size would raise)."""
        max_seq_len = max(a + b for a, b in zip(LEFT, RIGHT))
        total_left, total_right = sum(LEFT), sum(RIGHT)

        def step(
            left: torch.Tensor,
            right: torch.Tensor,
            offsets_left: torch.Tensor,
            offsets_right: torch.Tensor,
        ) -> Tuple[torch.Tensor, ...]:
            left = left.detach().requires_grad_()
            right = right.detach().requires_grad_()
            merged = pytorch_concat_2D_jagged(
                values_left=left,
                values_right=right,
                max_len_left=max(LEFT),
                max_len_right=max(RIGHT),
                offsets_left=offsets_left,
                offsets_right=offsets_right,
            )
            out_left, out_right = pytorch_split_2D_jagged(
                max_seq_len=max_seq_len,
                values=merged * 2.0,
                max_len_left=None,
                max_len_right=None,
                offsets_left=offsets_left,
                offsets_right=offsets_right,
                total_len_left=total_left,
                total_len_right=total_right,
            )
            loss = out_left.sum() + out_right.square().sum()
            return (out_left, out_right) + torch.autograd.grad(loss, (left, right))

        args = (
            torch.randn(total_left, D),
            torch.randn(total_right, D),
            _offsets(LEFT),
            _offsets(RIGHT),
        )
        gm = make_fx(step, tracing_mode="fake")(*args)
        out_left, out_right, g_left, g_right = gm(*args)
        self.assertEqual(tuple(out_left.shape), (total_left, D))
        self.assertEqual(tuple(out_right.shape), (total_right, D))
        torch.testing.assert_close(g_left, torch.full_like(g_left, 2.0))
        torch.testing.assert_close(g_right, 8.0 * args[1])
