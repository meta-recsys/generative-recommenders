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

import unittest
from typing import List, Tuple

import torch
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.ops.triton.triton_jagged import (
    _should_use_multirow,
    set_split_concat_2d_jagged_multirow_kernel,
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from generative_recommenders.ops.utils import is_sm90_plus
from hypothesis import example, given, settings, strategies as st, Verbosity

# (B, D, dtype, max_len_a, max_len_b). Shapes are memory-paired (large B with
# small D and vice versa) so any single config fits comfortably on one GPU.
_CONFIGS: List[Tuple[int, int, torch.dtype, int, int]] = [
    (1024, 128, torch.bfloat16, 3070, 384),  # prod shape (bf16, D=128)
    (512, 128, torch.bfloat16, 4096, 512),
    (256, 256, torch.bfloat16, 2048, 256),
    (128, 512, torch.float32, 1024, 128),  # large D, fp32
    (64, 64, torch.bfloat16, 777, 97),
    (37, 129, torch.float32, 300, 37),  # non-pow2 D, odd B, fp32
]


def _rand_offsets(
    B: int, max_len: int, device: torch.device, seed: int
) -> Tuple[torch.Tensor, int]:
    g = torch.Generator(device=device).manual_seed(seed)
    lengths = torch.randint(0, max_len + 1, (B,), generator=g, device=device)
    offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(lengths, 0)
    return offsets, int(lengths.max())


def _ref_concat(
    va: torch.Tensor, vb: torch.Tensor, oa: torch.Tensor, ob: torch.Tensor
) -> torch.Tensor:
    # Per batch row, values_a's slice followed by values_b's slice.
    oa_l, ob_l = oa.tolist(), ob.tolist()
    outs: List[torch.Tensor] = []
    for i in range(len(oa_l) - 1):
        outs.append(va[oa_l[i] : oa_l[i + 1]])
        outs.append(vb[ob_l[i] : ob_l[i + 1]])
    return torch.cat(outs, 0)


class JaggedMultirowParityTest(unittest.TestCase):
    """Bit-exact parity for the multirow concat/split 2D jagged kernels.

    The multirow flip only changes the launch/tiling of pure gather-scatter
    kernels, so the multirow path must be bit-identical to the single-row path
    and to a PyTorch reference, for both forward and backward.
    """

    def tearDown(self) -> None:
        set_split_concat_2d_jagged_multirow_kernel(None)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(config=st.sampled_from(_CONFIGS))
    @settings(verbosity=Verbosity.verbose, max_examples=len(_CONFIGS), deadline=None)
    @example(config=_CONFIGS[0])  # always exercise prod shape
    @example(config=_CONFIGS[-1])  # always exercise the non-pow2 D / odd B edge
    def test_concat_parity(
        self, config: Tuple[int, int, torch.dtype, int, int]
    ) -> None:
        self._test_concat(*config)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(config=st.sampled_from(_CONFIGS))
    @settings(verbosity=Verbosity.verbose, max_examples=len(_CONFIGS), deadline=None)
    @example(config=_CONFIGS[0])
    @example(config=_CONFIGS[-1])
    def test_split_parity(self, config: Tuple[int, int, torch.dtype, int, int]) -> None:
        self._test_split(*config)

    def _test_concat(
        self,
        B: int,
        D: int,
        dtype: torch.dtype,
        max_len_a: int,
        max_len_b: int,
        test_backward: bool = True,
    ) -> None:
        dev = torch.device("cuda")
        oa, mla = _rand_offsets(B, max_len_a, dev, seed=B + D)
        ob, mlb = _rand_offsets(B, max_len_b, dev, seed=B + D + 100)
        cmax = mla + mlb
        va = torch.randn(int(oa[-1]), D, dtype=dtype, device=dev)
        vb = torch.randn(int(ob[-1]), D, dtype=dtype, device=dev)
        ref = _ref_concat(va, vb, oa, ob)
        dout = torch.randn_like(ref) if test_backward else None

        outs: List[torch.Tensor] = []
        grads: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for multirow in (False, True):
            set_split_concat_2d_jagged_multirow_kernel(multirow)
            a = va.detach().clone().requires_grad_(test_backward)
            b = vb.detach().clone().requires_grad_(test_backward)
            out = triton_concat_2D_jagged(cmax, a, b, oa, ob)
            outs.append(out.detach())
            if test_backward:
                out.backward(dout)
                assert a.grad is not None and b.grad is not None
                grads.append((a.grad, b.grad))

        torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)
        torch.testing.assert_close(outs[0], ref, rtol=0, atol=0)
        if test_backward:
            torch.testing.assert_close(grads[1][0], grads[0][0], rtol=0, atol=0)
            torch.testing.assert_close(grads[1][1], grads[0][1], rtol=0, atol=0)

    def _test_split(
        self,
        B: int,
        D: int,
        dtype: torch.dtype,
        max_len_a: int,
        max_len_b: int,
        test_backward: bool = True,
    ) -> None:
        dev = torch.device("cuda")
        oa, mla = _rand_offsets(B, max_len_a, dev, seed=B + D)
        ob, mlb = _rand_offsets(B, max_len_b, dev, seed=B + D + 100)
        cmax = mla + mlb
        va = torch.randn(int(oa[-1]), D, dtype=dtype, device=dev)
        vb = torch.randn(int(ob[-1]), D, dtype=dtype, device=dev)
        values = _ref_concat(va, vb, oa, ob)
        ga = torch.randn_like(va) if test_backward else None
        gb = torch.randn_like(vb) if test_backward else None

        outs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        grads: List[torch.Tensor] = []
        for multirow in (False, True):
            set_split_concat_2d_jagged_multirow_kernel(multirow)
            v = values.detach().clone().requires_grad_(test_backward)
            sa, sb = triton_split_2D_jagged(v, cmax, oa, ob)
            outs.append((sa.detach(), sb.detach()))
            if test_backward:
                torch.autograd.backward([sa, sb], [ga, gb])
                assert v.grad is not None
                grads.append(v.grad)

        torch.testing.assert_close(outs[1][0], outs[0][0], rtol=0, atol=0)
        torch.testing.assert_close(outs[1][1], outs[0][1], rtol=0, atol=0)
        torch.testing.assert_close(outs[0][0], va, rtol=0, atol=0)
        torch.testing.assert_close(outs[0][1], vb, rtol=0, atol=0)
        if test_backward:
            torch.testing.assert_close(grads[1], grads[0], rtol=0, atol=0)

    @unittest.skipIf(*gpu_unavailable)
    def test_default_selection_matches_singlerow(self) -> None:
        # With no override, the default path must select multirow on sm90+ and
        # stay bit-exact vs the single-row kernel at the prod shape.
        dev = torch.device("cuda")
        B, D, dtype, max_len_a, max_len_b = _CONFIGS[0]
        oa, _ = _rand_offsets(B, max_len_a, dev, seed=11)
        ob, _ = _rand_offsets(B, max_len_b, dev, seed=111)
        cmax = int((oa[1:] - oa[:-1]).max()) + int((ob[1:] - ob[:-1]).max())
        va = torch.randn(int(oa[-1]), D, dtype=dtype, device=dev)
        vb = torch.randn(int(ob[-1]), D, dtype=dtype, device=dev)

        set_split_concat_2d_jagged_multirow_kernel(False)
        with torch.no_grad():
            single = triton_concat_2D_jagged(cmax, va, vb, oa, ob)

        set_split_concat_2d_jagged_multirow_kernel(None)  # restore default
        if is_sm90_plus():
            self.assertTrue(_should_use_multirow())
        with torch.no_grad():
            default_out = triton_concat_2D_jagged(cmax, va, vb, oa, ob)
        torch.testing.assert_close(single, default_out, rtol=0, atol=0)
