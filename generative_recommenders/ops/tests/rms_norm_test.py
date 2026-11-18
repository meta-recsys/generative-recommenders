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

import copy
import unittest
from unittest import mock

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel, set_dev_mode
from generative_recommenders.ops.layer_norm import rms_norm, RMSNorm
from generative_recommenders.ops.triton import triton_layer_norm
from generative_recommenders.ops.utils import is_sm100_plus
from hammer.ops.triton.cc.utils import set_triton_cc_version
from hypothesis import given, settings, strategies as st, Verbosity


class LayerNormTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.sampled_from([2000000]),
        D=st.sampled_from([512]),
        dtype=st.sampled_from(
            [torch.bfloat16]
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else [torch.float32]
        ),
        silu=st.booleans(),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=1,
    )
    # pyre-ignore[2]
    def test_large_tensors(self, *args, **kwargs) -> None:
        self._test_rms_norm(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
            skip_comparisons=True,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=0, max_value=10000),
        D=st.integers(min_value=32, max_value=512),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else [torch.float32]
        ),
        silu=st.booleans(),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=50,
    )
    # pyre-ignore[2]
    def test_rms_norm(self, *args, **kwargs) -> None:
        self._test_rms_norm(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    @unittest.skipUnless(is_sm100_plus(), "Wide RMS autotuning regression for SM100")
    def test_rms_norm_wide(self) -> None:
        for N, D, dtype in [
            (13, 2048, torch.float32),
            (13, 2049, torch.float32),
            (13, 4097, torch.float32),
            (13, 9843, torch.float32),
            (13, 15859, torch.float32),
            (4096, 16384, torch.bfloat16),
        ]:
            for silu in [False, True]:
                with self.subTest(N=N, D=D, dtype=dtype, silu=silu):
                    self._test_rms_norm(
                        N=N,
                        D=D,
                        dtype=dtype,
                        silu=silu,
                        ref_kernel=HammerKernel.PYTORCH,
                        real_kernel=HammerKernel.TRITON,
                    )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=4, max_value=10000),
        D=st.sampled_from([256, 512]),
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
        triton_cc_version=st.sampled_from(["", "repkg"]),
        silu=st.just(False),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=10,
    )
    # pyre-ignore[2]
    def test_rms_norm_triton_cc(self, triton_cc_version: str, *args, **kwargs) -> None:
        set_triton_cc_version(triton_cc_version)
        self._test_rms_norm(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON_CC,
            test_backward=False,
        )

    def _test_rms_norm(
        self,
        N: int,
        D: int,
        dtype: torch.dtype,
        silu: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        skip_comparisons: bool = False,
        test_backward: bool = True,
    ) -> None:
        # enable auto-tuning to verify correctness of multi-row kernel
        set_dev_mode(False)
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .normal_(0.0, 1.0)
            .requires_grad_()
        )
        weight = (
            torch.empty((D,), device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        # ref
        ref_out = rms_norm(x, weight, eps=1e-6, silu=silu, kernel=ref_kernel)
        opt_x = x.detach().clone().requires_grad_()
        opt_weight = weight.detach().clone().requires_grad_()
        opt_out = rms_norm(opt_x, opt_weight, eps=1e-6, silu=silu, kernel=real_kernel)
        torch.testing.assert_close(ref_out, opt_out)

        if not test_backward:
            return

        dout = torch.randn_like(ref_out) * 0.05
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        # pyrefly: ignore [missing-attribute]
        ref_dw, weight.grad = weight.grad.detach().clone(), None
        # opt
        dout = dout.detach().clone()
        opt_out.backward(dout)
        # pyrefly: ignore [missing-attribute]
        opt_dx, x.grad = opt_x.grad.detach().clone(), None
        # pyrefly: ignore [missing-attribute]
        opt_dw, weight.grad = opt_weight.grad.detach().clone(), None
        torch.testing.assert_close(ref_dx, opt_dx)
        torch.testing.assert_close(ref_dw, opt_dw)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=32, max_value=10000),
        D=st.integers(min_value=32, max_value=512),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=50,
    )
    # pyre-ignore[2]
    def test_modules(self, *args, **kwargs) -> None:
        self._test_rms_norm_module(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_rms_norm_module(
        self,
        N: int,
        D: int,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        skip_comparisons: bool = False,
    ) -> None:
        set_dev_mode(True)
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .normal_(0.0, 1.0)
            .requires_grad_()
        )
        # ref
        ref_layer = RMSNorm(
            dim=D,
            eps=1e-6,
        ).to(device="cuda")
        ref_layer._hammer_kernel = ref_kernel
        opt_layer = copy.deepcopy(ref_layer)
        opt_layer._hammer_kernel = real_kernel

        ref_out = ref_layer(x)
        dout = torch.randn_like(ref_out) * 0.05
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_dw = ref_layer.weight.grad.detach().clone()
        # opt
        x = x.detach().clone().requires_grad_()
        opt_out = opt_layer(x)
        dout = dout.detach().clone()
        opt_out.backward(dout)
        # pyrefly: ignore [missing-attribute]
        opt_dx, x.grad = x.grad.detach().clone(), None
        opt_dw = opt_layer.weight.grad.detach().clone()
        torch.testing.assert_close(ref_out.to(dtype), opt_out)
        torch.testing.assert_close(
            ref_dx,
            opt_dx,
        )
        torch.testing.assert_close(
            ref_dw,
            opt_dw,
        )


class StridedGradTest(unittest.TestCase):
    """The backward kernels read dy rows with a unit column stride."""

    def _grads(
        self,
        name: str,
        x0: torch.Tensor,
        w0: torch.Tensor,
        b0: torch.Tensor,
        dy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        if name == "rms":
            y = triton_layer_norm.triton_rms_norm(x, w, 1e-6)
        elif name == "layer":
            y = triton_layer_norm.triton_layer_norm(x, w, b0, 1e-6)
        else:
            y = triton_layer_norm.triton_swish_layer_norm(x, [x.shape[-1]], w, b0, 1e-6)
        y.backward(dy)
        return x.grad, w.grad

    @unittest.skipIf(*gpu_unavailable)
    def test_strided_dy(self) -> None:
        torch.manual_seed(0)
        n, d = 1000, 256
        for dtype in (torch.float32, torch.bfloat16):
            x0 = torch.randn(n, d, device="cuda", dtype=dtype)
            w0 = torch.rand(d, device="cuda", dtype=dtype) + 0.5
            b0 = torch.randn(d, device="cuda", dtype=dtype)
            dy0 = torch.randn(n, d, device="cuda", dtype=dtype)
            strided = {
                "transposed": dy0.t().contiguous().t(),
                "col_stride_2": torch.stack([dy0, torch.full_like(dy0, 9.0)], -1)[
                    ..., 0
                ],
                "expanded": dy0[:1, :1].expand(n, d),
                "row_broadcast": dy0[:1].expand(n, d),
            }
            for name in ("rms", "layer", "swish_layer"):
                for layout, dy in strided.items():
                    with self.subTest(name=name, dtype=dtype, layout=layout):
                        ref = self._grads(name, x0, w0, b0, dy.contiguous())
                        got = self._grads(name, x0, w0, b0, dy)
                        torch.testing.assert_close(got[0], ref[0])
                        torch.testing.assert_close(got[1], ref[1])

    @unittest.skipIf(*gpu_unavailable)
    def test_layer_norm_transposed_x(self) -> None:
        torch.manual_seed(0)
        n, d = 1000, 256
        x0 = torch.randn(n, d, device="cuda")
        w0 = torch.rand(d, device="cuda") + 0.5
        b0 = torch.randn(d, device="cuda")
        dy = torch.randn(n, d, device="cuda")
        ref = self._grads("layer", x0, w0, b0, dy)
        got = self._grads("layer", x0.t().contiguous().t(), w0, b0, dy)
        torch.testing.assert_close(got[0], ref[0])
        torch.testing.assert_close(got[1], ref[1])


class RMSNormAutotuneTest(unittest.TestCase):
    def test_narrow_configs_unchanged(self) -> None:
        with mock.patch.object(triton_layer_norm, "is_sm100_plus", return_value=True):
            original = triton_layer_norm._get_layer_norm_fwd_configs()
            backward = triton_layer_norm._get_rms_norm_bwd_configs()
            forward = triton_layer_norm._get_rms_norm_fwd_configs()
            for block_d in [256, 512, 2048]:
                self.assertEqual(
                    [str(c) for c in original],
                    [
                        str(c)
                        for c in triton_layer_norm._prune_rms_norm_bwd_configs(
                            backward, {}, BLOCK_D=block_d
                        )
                    ],
                )
                self.assertIs(
                    forward,
                    triton_layer_norm._prune_rms_norm_configs(
                        forward, {}, BLOCK_D=block_d
                    ),
                )

    def test_wide_configs(self) -> None:
        with mock.patch.object(triton_layer_norm, "is_sm100_plus", return_value=True):
            configs = triton_layer_norm._get_rms_norm_bwd_configs()
            for block_d in [4096, 8192, 16384, 32768]:
                with self.subTest(block_d=block_d):
                    retained = triton_layer_norm._prune_rms_norm_bwd_configs(
                        configs, {}, BLOCK_D=block_d
                    )
                    self.assertEqual(
                        retained,
                        triton_layer_norm._prune_rms_norm_bwd_configs(
                            configs, {"BLOCK_D": block_d}
                        ),
                    )
                    self.assertGreater(len(retained), 1)
                    self.assertIn(
                        (1, 8), [(c.kwargs["BLOCK_N"], c.num_warps) for c in retained]
                    )
                    self.assertNotIn(
                        (16, 1), [(c.kwargs["BLOCK_N"], c.num_warps) for c in retained]
                    )
                    self.assertTrue(
                        triton_layer_norm._prune_rms_norm_configs(
                            triton_layer_norm._get_rms_norm_fwd_configs(),
                            {},
                            BLOCK_D=block_d,
                        )
                    )
            retained = triton_layer_norm._prune_rms_norm_bwd_configs(
                configs, {}, BLOCK_D=16384
            )
            choices = [(c.kwargs["BLOCK_N"], c.num_warps) for c in retained]
            self.assertIn((4, 8), choices)
            self.assertNotIn((4, 1), choices)
            self.assertNotEqual(
                [str(c) for c in configs],
                [str(c) for c in triton_layer_norm._get_layer_norm_fwd_configs()],
            )

    def test_other_architectures_unchanged(self) -> None:
        with mock.patch.object(triton_layer_norm, "is_sm100_plus", return_value=False):
            original = triton_layer_norm._get_layer_norm_fwd_configs()
            configs = triton_layer_norm._get_rms_norm_bwd_configs()
            self.assertEqual([str(c) for c in original], [str(c) for c in configs])
            for prune in [
                triton_layer_norm._prune_rms_norm_configs,
                triton_layer_norm._prune_rms_norm_bwd_configs,
            ]:
                self.assertIs(configs, prune(configs, {}, BLOCK_D=16384))
