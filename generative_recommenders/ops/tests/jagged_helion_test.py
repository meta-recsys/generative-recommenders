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

import unittest
from typing import List, Tuple

import torch
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.ops.triton.triton_jagged import (
    helion_split_2D_jagged,
    set_split_concat_2d_jagged_multirow_kernel,
    triton_split_2D_jagged,
)

# (B, D, dtype, max_len_a, max_len_b)
_CONFIGS: List[Tuple[int, int, torch.dtype, int, int]] = [
    (1024, 128, torch.bfloat16, 3070, 384),
    (96, 768, torch.bfloat16, 1360, 206),
    (64, 96, torch.bfloat16, 1024, 3),
    (128, 512, torch.float32, 1024, 128),
    (37, 129, torch.float32, 300, 37),
    (1, 1, torch.bfloat16, 40, 40),
    (8, 64, torch.float32, 16, 16),
]


def _rand_offsets(
    B: int, max_len: int, device: torch.device, seed: int
) -> Tuple[torch.Tensor, int]:
    g = torch.Generator(device=device).manual_seed(seed)
    lengths = torch.randint(0, max_len + 1, (B,), generator=g, device=device)
    offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(lengths, 0)
    return offsets, int(lengths.max())


def _rand(
    rows: int, D: int, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(rows, D, dtype=dtype, device=device, generator=g)


def _ref_concat(
    va: torch.Tensor, vb: torch.Tensor, oa: torch.Tensor, ob: torch.Tensor
) -> torch.Tensor:
    oa_l, ob_l = oa.tolist(), ob.tolist()
    outs: List[torch.Tensor] = []
    for i in range(len(oa_l) - 1):
        outs.append(va[oa_l[i] : oa_l[i + 1]])
        outs.append(vb[ob_l[i] : ob_l[i + 1]])
    return torch.cat(outs, 0)


def _poison_allocator(
    rows: int, D: int, dtype: torch.dtype, device: torch.device
) -> None:
    """Leave a NaN-filled block of the right size in the caching allocator.

    The backward writes its input gradient into a torch.empty buffer and relies
    on the concat kernel to zero the left region itself. Fresh CUDA pages are
    usually zero, and the block freed by the previous run already holds the
    right answer, so without poisoning a kernel that stopped writing there would
    still pass.
    """
    block = torch.full((rows, D), float("nan"), dtype=dtype, device=device)
    del block


class JaggedHelionSkipLeftTest(unittest.TestCase):
    """SKIP_LEFT on the Helion split path.

    The Helion split kernel owns the forward; its backward is the Triton concat,
    which already took SKIP_LEFT. Both halves have to honour the same contract
    as the Triton split: the left output is dropped, the right output is
    untouched, and the left region of the input gradient is exactly zero.
    """

    def tearDown(self) -> None:
        set_split_concat_2d_jagged_multirow_kernel(None)

    @unittest.skipIf(*gpu_unavailable)
    def test_split_skip_left(self) -> None:
        for config in _CONFIGS:
            with self.subTest(config=config):
                self._test_split_skip_left(*config)

    def _test_split_skip_left(
        self,
        B: int,
        D: int,
        dtype: torch.dtype,
        max_len_a: int,
        max_len_b: int,
    ) -> None:
        """skip_left drops the left output without disturbing anything else.

        Runs the full (multirow, skip_left) matrix. multirow only reaches the
        backward -- the Helion forward has a single kernel and no fallback --
        but that is exactly the half that is easy to leave untested, since the
        default on the H100 this runs on is always multirow.

        Compares against a PyTorch reference rather than only cross-checking the
        runs, so a backward that is consistently wrong still fails. atol=0 is
        what pins the left region of the gradient to exactly zero.
        """
        dev = torch.device("cuda")
        oa, mla = _rand_offsets(B, max_len_a, dev, seed=B + D)
        ob, mlb = _rand_offsets(B, max_len_b, dev, seed=B + D + 100)
        cmax = mla + mlb
        va = _rand(int(oa[-1]), D, dtype, dev, seed=B + D + 200)
        vb = _rand(int(ob[-1]), D, dtype, dev, seed=B + D + 300)
        values = _ref_concat(va, vb, oa, ob)
        gb = _rand(int(ob[-1]), D, dtype, dev, seed=B + D + 400)
        ref_grad = _ref_concat(torch.zeros_like(va), gb, oa, ob)

        rights: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []
        for multirow in (False, True):
            set_split_concat_2d_jagged_multirow_kernel(multirow)
            for skip_left in (False, True):
                v = values.detach().clone().requires_grad_(True)
                sa, sb = helion_split_2D_jagged(v, cmax, oa, ob, skip_left=skip_left)
                if skip_left:
                    self.assertEqual(tuple(sa.shape), (0, D))
                else:
                    torch.testing.assert_close(sa, va, rtol=0, atol=0)
                rights.append(sb.detach())
                _poison_allocator(values.shape[0], D, dtype, dev)
                torch.autograd.backward([sa, sb], [torch.zeros_like(sa), gb])
                assert v.grad is not None
                grads.append(v.grad)

        torch.testing.assert_close(rights[0], vb, rtol=0, atol=0)
        torch.testing.assert_close(grads[0], ref_grad, rtol=0, atol=0)
        for i in range(1, len(rights)):
            torch.testing.assert_close(rights[i], rights[0], rtol=0, atol=0)
            torch.testing.assert_close(grads[i], grads[0], rtol=0, atol=0)

    @unittest.skipIf(*gpu_unavailable)
    def test_skip_left_matches_triton_multirow(self) -> None:
        """Helion and Triton multirow must be interchangeable under SKIP_LEFT.

        The two are selected by config, not by shape, so a model can switch
        between them; anything less than bit-exact agreement would show up as an
        unexplained numerics change when it does. Only the forward carries
        signal here -- both backwards call the same concat kernel -- so the
        gradient is checked in test_split_skip_left instead.
        """
        for config in _CONFIGS:
            with self.subTest(config=config):
                B, D, dtype, max_len_a, max_len_b = config
                dev = torch.device("cuda")
                oa, mla = _rand_offsets(B, max_len_a, dev, seed=B + D + 7)
                ob, mlb = _rand_offsets(B, max_len_b, dev, seed=B + D + 107)
                cmax = mla + mlb
                va = _rand(int(oa[-1]), D, dtype, dev, seed=B + D + 207)
                vb = _rand(int(ob[-1]), D, dtype, dev, seed=B + D + 307)
                values = _ref_concat(va, vb, oa, ob)

                set_split_concat_2d_jagged_multirow_kernel(True)
                outs: List[torch.Tensor] = []
                for split_fn in (helion_split_2D_jagged, triton_split_2D_jagged):
                    with torch.no_grad():
                        sa, sb = split_fn(values, cmax, oa, ob, skip_left=True)
                    self.assertEqual(tuple(sa.shape), (0, D))
                    outs.append(sb)

                torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)
                torch.testing.assert_close(outs[0], vb, rtol=0, atol=0)

    @unittest.skipIf(*gpu_unavailable)
    def test_skip_left_grad_materialized_by_autograd(self) -> None:
        """Production never hands the left grad in explicitly.

        It backwards through the right output alone and lets autograd
        materialize the (0, D) left grad. Equivalent to the explicit form only
        because materialize_grads defaults on, which is worth pinning.

        `.sum()` also makes this the only test where the incoming gradient is a
        stride-0 expanded view rather than a contiguous tensor -- the concat
        kernels index rows by stride(-2), so without the contiguity fix in
        backward this reads out of bounds.
        """
        dev = torch.device("cuda")
        B, D = 16, 64
        oa, mla = _rand_offsets(B, 96, dev, seed=21)
        ob, mlb = _rand_offsets(B, 12, dev, seed=121)
        cmax = mla + mlb
        va = _rand(int(oa[-1]), D, torch.float32, dev, seed=221)
        vb = _rand(int(ob[-1]), D, torch.float32, dev, seed=321)
        values = _ref_concat(va, vb, oa, ob)
        expected = _ref_concat(torch.zeros_like(va), torch.ones_like(vb), oa, ob)

        for skip_left in (False, True):
            with self.subTest(skip_left=skip_left):
                v = values.detach().clone().requires_grad_(True)
                _, sb = helion_split_2D_jagged(v, cmax, oa, ob, skip_left=skip_left)
                _poison_allocator(values.shape[0], D, torch.float32, dev)
                sb.sum().backward()

                assert v.grad is not None
                torch.testing.assert_close(v.grad, expected, rtol=0, atol=0)

    @unittest.skipIf(*gpu_unavailable)
    def test_skip_left_rejects_dense_left(self) -> None:
        dev = torch.device("cuda")
        B, D, dense_size = 8, 16, 4
        oa, _ = _rand_offsets(B, 32, dev, seed=5)
        ob, mlb = _rand_offsets(B, 32, dev, seed=55)
        values = _rand(int(oa[-1]) + int(ob[-1]), D, torch.float32, dev, seed=555)

        with self.assertRaisesRegex(AssertionError, "skip_left needs a jagged"):
            helion_split_2D_jagged(
                values, mlb + dense_size, None, ob, dense_size, skip_left=True
            )
        with self.assertRaisesRegex(AssertionError, "skip_left needs a jagged"):
            helion_split_2D_jagged(
                values, mlb + dense_size, oa, ob, dense_size, skip_left=True
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_skip_left_when_left_is_entirely_empty(self) -> None:
        """A batch whose left side is empty everywhere still has to work.

        len_a == 0 makes every row a right-hand row, so neither the SKIP_LEFT
        guard nor the tightened has_work may be what produces them. Guards
        against wrapping the guard around too much of the kernel body, and
        against an off-by-one in the has_work bound.
        """
        dev = torch.device("cuda")
        B, D = 16, 64
        oa = torch.zeros(B + 1, dtype=torch.long, device=dev)
        ob, mlb = _rand_offsets(B, 48, dev, seed=3)
        vb = _rand(int(ob[-1]), D, torch.float32, dev, seed=33)

        outs: List[torch.Tensor] = []
        for skip_left in (False, True):
            with torch.no_grad():
                sa, sb = helion_split_2D_jagged(vb, mlb, oa, ob, skip_left=skip_left)
            self.assertEqual(tuple(sa.shape), (0, D))
            outs.append(sb)

        torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)
        torch.testing.assert_close(outs[1], vb, rtol=0, atol=0)
