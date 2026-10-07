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

"""Triton kernels trace with fake tensors (make_fx), forward and backward.

Each case traces ``fn`` with ``tracing_mode="fake"`` (a raw kernel launch on a
fake tensor would fail), then runs the traced graph on real inputs and checks
it against eager.
"""

import unittest
from typing import Callable, Sequence, Tuple

import fbgemm_gpu.sparse_ops  # noqa: F401 -- fake (abstract) impls of fbgemm ops
import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from generative_recommenders.ops.layer_norm import TraceableSwishLayerNorm
from generative_recommenders.ops.position import add_timestamp_positional_embeddings
from torch.fx.experimental.proxy_tensor import make_fx

TRITON = HammerKernel.TRITON


def _offsets(lengths: Sequence[int]) -> torch.Tensor:
    return torch.tensor([0] + list(lengths), device="cuda").cumsum(0)


def _with_grads(
    fn: Callable[..., torch.Tensor], num_diff: int
) -> Callable[..., Tuple[torch.Tensor, ...]]:
    """Forward plus gradients of a fixed loss w.r.t. the first ``num_diff``
    inputs."""

    def step(*args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        diff = [a.detach().requires_grad_() for a in args[:num_diff]]
        out = fn(*diff, *args[num_diff:])
        loss = (
            out.float() * torch.arange(out.numel(), device=out.device).view_as(out)
        ).sum()
        return (out,) + torch.autograd.grad(loss, diff)

    return step


class TritonFakeTraceTest(unittest.TestCase):
    def _check(
        self, step: Callable[..., Tuple[torch.Tensor, ...]], *args: torch.Tensor
    ) -> None:
        eager = step(*args)
        gm = make_fx(step, tracing_mode="fake")(*args)
        traced = gm(*args)
        for t, e in zip(traced, eager):
            torch.testing.assert_close(t, e)

    @unittest.skipIf(*gpu_unavailable)
    def test_concat_split_2D_jagged(self) -> None:
        left, right, D = [3, 0, 5, 2], [2, 4, 1, 1], 16
        offsets_left, offsets_right = _offsets(left), _offsets(right)
        max_seq_len = max(a + b for a, b in zip(left, right))

        def fn(
            values_left: torch.Tensor,
            values_right: torch.Tensor,
            offsets_left: torch.Tensor,
            offsets_right: torch.Tensor,
        ) -> torch.Tensor:
            merged = concat_2D_jagged(
                max_seq_len=max_seq_len,
                values_left=values_left,
                values_right=values_right,
                max_len_left=max(left),
                max_len_right=max(right),
                offsets_left=offsets_left,
                offsets_right=offsets_right,
                kernel=TRITON,
            )
            out_left, out_right = split_2D_jagged(
                max_seq_len=max_seq_len,
                values=merged * 2.0,
                total_len_left=sum(left),
                total_len_right=sum(right),
                max_len_left=max(left),
                max_len_right=max(right),
                offsets_left=offsets_left,
                offsets_right=offsets_right,
                kernel=TRITON,
            )
            return torch.cat([out_left, out_right * 3.0])

        self._check(
            _with_grads(fn, 2),
            torch.randn(sum(left), D, device="cuda"),
            torch.randn(sum(right), D, device="cuda"),
            offsets_left,
            offsets_right,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_add_timestamp_positional_embeddings(self) -> None:
        lengths, D, max_pos, num_buckets = [5, 3, 7], 32, 16, 64
        num_targets = torch.tensor([2, 1, 2], device="cuda")
        offsets = _offsets(lengths)
        seq_lengths = torch.tensor(lengths, device="cuda")
        N = sum(lengths)
        timestamps = torch.randint(0, 10_000, (N,), device="cuda")

        def fn(
            seq_embeddings: torch.Tensor,
            pos_weight: torch.Tensor,
            ts_weight: torch.Tensor,
            offsets: torch.Tensor,
            seq_lengths: torch.Tensor,
            timestamps: torch.Tensor,
            num_targets: torch.Tensor,
        ) -> torch.Tensor:
            return add_timestamp_positional_embeddings(
                alpha=1.0,
                max_seq_len=max(lengths),
                max_contextual_seq_len=0,
                position_embeddings_weight=pos_weight,
                timestamp_embeddings_weight=ts_weight,
                seq_offsets=offsets,
                seq_lengths=seq_lengths,
                seq_embeddings=seq_embeddings,
                timestamps=timestamps,
                num_targets=num_targets,
                interleave_targets=False,
                kernel=TRITON,
            )

        self._check(
            _with_grads(fn, 3),
            torch.randn(N, D, device="cuda"),
            torch.randn(max_pos, D, device="cuda"),
            torch.randn(num_buckets + 1, D, device="cuda"),
            offsets,
            seq_lengths,
            timestamps,
            num_targets,
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_traceable_swish_layer_norm(self) -> None:
        N, D = 37, 48
        module = TraceableSwishLayerNorm(D).cuda()
        module.set_hammer_kernel(TRITON)

        def fn(
            x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            with torch.nn.utils.stateless._reparametrize_module(
                module, {"weight": weight, "bias": bias}
            ):
                return module(x)

        self._check(
            _with_grads(fn, 3),
            torch.randn(N, D, device="cuda"),
            torch.randn(D, device="cuda"),
            torch.randn(D, device="cuda"),
        )
