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

import random
import unittest

from typing import Optional

import torch

from generative_recommenders.common import (
    generate_sparse_seq_len,
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from hypothesis import given, settings, strategies as st, Verbosity


class HSTUComputeTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=1000, max_value=1000),
        D=st.integers(min_value=128, max_value=128),
        L=st.integers(min_value=512, max_value=512),
        concat_ux=st.booleans(),
        group_norm=st.booleans(),
        num_heads=st.sampled_from([4]),
        training=st.just(False),
        recompute_y_in_backward=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_compute_output(self, *args, **kwargs) -> None:
        self._test_compute_output(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            opt_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.just(1500000),
        D=st.just(512),
        L=st.just(512),
        concat_ux=st.sampled_from([True]),
        group_norm=st.sampled_from([False]),
        num_heads=st.sampled_from([4]),
        training=st.just(False),
        recompute_y_in_backward=st.sampled_from([False]),
        dtype=st.just(torch.bfloat16),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=1,
    )
    # pyre-ignore[2]
    def test_long_sequences_compute_output(self, *args, **kwargs) -> None:
        self._test_compute_output(
            *args,
            **kwargs,
            test_backward=False,
            ref_kernel=HammerKernel.TRITON,
            opt_kernel=HammerKernel.TRITON,
            skip_comparisons=True,
        )

    def _test_compute_output(
        self,
        N: int,
        D: int,
        L: int,
        concat_ux: bool,
        group_norm: bool,
        num_heads: int,
        training: bool,
        recompute_y_in_backward: bool,
        dtype: torch.dtype,
        test_backward: bool,
        ref_kernel: HammerKernel,
        opt_kernel: HammerKernel,
        skip_comparisons: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        from generative_recommenders.ops.hstu_compute import hstu_compute_output

        torch.manual_seed(0)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        dropout_ratio = 0.3 if training else 0.0
        attn = (
            torch.empty((N, L), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        u = (
            torch.empty((N, L), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_weight = (
            torch.empty(
                (L if not group_norm else num_heads,),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_bias = (
            torch.empty(
                (L if not group_norm else num_heads,),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_eps = 1e-6
        output_weight = (
            torch.empty(
                (L * 3 if concat_ux else L, D), dtype=dtype, device=torch.device("cuda")
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        # ref
        ref_out = hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            dropout_ratio=dropout_ratio,
            output_weight=output_weight,
            concat_ux=concat_ux,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=L // num_heads,
            training=training,
            recompute_y_in_backward=recompute_y_in_backward,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_out) * 0.1
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dattn, attn.grad = attn.grad.detach().clone(), None
        ref_du, u.grad = u.grad.detach().clone(), None
        ref_d_norm_w, norm_weight.grad = norm_weight.grad.detach().clone(), None
        ref_d_norm_b, norm_bias.grad = norm_bias.grad.detach().clone(), None
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_d_output_w, output_weight.grad = output_weight.grad.detach().clone(), None

        # opt
        attn = attn.detach().clone().requires_grad_()
        u = u.detach().clone().requires_grad_()
        norm_weight = norm_weight.detach().clone().requires_grad_()
        norm_bias = norm_bias.detach().clone().requires_grad_()
        output_weight = output_weight.detach().clone().requires_grad_()
        x = x.detach().clone().requires_grad_()
        opt_out = hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            dropout_ratio=dropout_ratio,
            output_weight=output_weight,
            concat_ux=concat_ux,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=L // num_heads,
            training=training,
            recompute_y_in_backward=recompute_y_in_backward,
            kernel=opt_kernel,
        )
        torch.testing.assert_close(
            ref_out,
            opt_out,
            atol=atol,
            rtol=rtol,
        )

        if test_backward:
            dout = dout.detach().clone()
            opt_out.backward(dout)
            opt_dattn, attn.grad = attn.grad.detach().clone(), None
            opt_du, u.grad = u.grad.detach().clone(), None
            opt_d_norm_w, norm_weight.grad = norm_weight.grad.detach().clone(), None
            opt_d_norm_b, norm_bias.grad = norm_bias.grad.detach().clone(), None
            opt_dx, x.grad = x.grad.detach().clone(), None
            opt_d_output_w, output_weight.grad = (
                output_weight.grad.detach().clone(),
                None,
            )
            torch.testing.assert_close(ref_du, opt_du)
            torch.testing.assert_close(ref_dattn, opt_dattn)
            torch.testing.assert_close(ref_d_norm_w, opt_d_norm_w)
            torch.testing.assert_close(ref_d_norm_b, opt_d_norm_b)
            torch.testing.assert_close(ref_dx, opt_dx)
            torch.testing.assert_close(ref_d_output_w, opt_d_output_w)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([100, 128, 256, 1300]),
        max_targets=st.sampled_from([20, 512]),
        embedding_dim=st.sampled_from([16, 32, 64]),
        attn_dim=st.sampled_from([16, 32, 64, 128]),
        hidden_dim=st.sampled_from([16, 32, 64, 128]),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        contextual_seq_len=st.sampled_from([0]),
        has_max_attn_len=st.sampled_from([False, True]),
        sort_by_length=st.sampled_from([True, False]),
        recompute_uvqk_in_backward=st.sampled_from([True, False]),
        recompute_normed_x_in_backward=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=150,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_preprocess_and_attention(self, *args, **kwargs) -> None:
        self._test_hstu_preprocess_and_attention(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_hstu_preprocess_and_attention(
        self,
        batch_size: int,
        heads: int,
        max_uih_len: int,
        max_targets: int,
        embedding_dim: int,
        attn_dim: int,
        hidden_dim: int,
        causal: bool,
        has_multiple_targets: bool,
        has_max_attn_len: bool,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        test_backward: bool,
        contextual_seq_len: int,
        sort_by_length: bool,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        sparsity: float = -1.0,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        set_dev_mode(True)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

        from generative_recommenders.ops.hstu_compute import (
            hstu_preprocess_and_attention,
        )

        alpha = 1.0 / (attn_dim**0.5)
        if sparsity > 0.0:
            lengths = generate_sparse_seq_len(
                size=batch_size,
                max_seq_len=max_uih_len,
                sparsity=sparsity,
                device=torch.device("cuda"),
            )
        else:
            lengths = torch.randint(
                max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
            )

        num_targets = torch.randint(
            max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        lengths = lengths + num_targets
        max_seq_len = max_uih_len + max_targets
        if has_max_attn_len:
            max_attn_len = random.randint(1, max_uih_len // 5)
        else:
            max_attn_len = 0
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)

        L = int(seq_offsets[-1].item())

        x = (
            torch.empty((L, embedding_dim), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_weight = (
            torch.empty((embedding_dim,), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_bias = (
            torch.empty(
                (embedding_dim,),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        norm_eps = 1e-6
        uvqk_weight = (
            torch.empty(
                (
                    embedding_dim,
                    (hidden_dim * 2 + attn_dim * 2) * heads,
                ),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        uvqk_bias = (
            torch.empty(
                (hidden_dim * 2 + attn_dim * 2) * heads,
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        # ref implementation
        ref_u, ref_attn_output, _, _ = hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            attn_alpha=alpha,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            sort_by_length=sort_by_length,
            kernel=ref_kernel,
        )
        ref_out = ref_u + ref_attn_output
        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)

        # pyre-ignore
        ref_dx, x.grad = x.grad.clone(), None
        ref_d_norm_weight, norm_weight.grad = norm_weight.grad.clone(), None
        ref_d_norm_bias, norm_bias.grad = norm_bias.grad.clone(), None
        ref_d_uvqk_weight, uvqk_weight.grad = uvqk_weight.grad.clone(), None
        ref_d_uvqk_bias, uvqk_bias.grad = uvqk_bias.grad.clone(), None

        # real implementation
        x = x.detach().clone().requires_grad_()
        norm_weight = norm_weight.detach().clone().requires_grad_()
        norm_bias = norm_bias.detach().clone().requires_grad_()
        uvqk_weight = uvqk_weight.detach().clone().requires_grad_()
        uvqk_bias = uvqk_bias.detach().clone().requires_grad_()
        real_u, real_attn_output, _, _ = hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            attn_alpha=alpha,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            sort_by_length=sort_by_length,
            kernel=real_kernel,
        )
        real_out = real_u + real_attn_output
        torch.testing.assert_close(
            ref_u,
            real_u,
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            ref_attn_output,
            real_attn_output,
            atol=atol,
            rtol=rtol,
        )
        if test_backward:
            # real implementation
            dout = dout.detach().clone()
            real_out.backward(dout)
            (
                real_dx,
                real_d_norm_weight,
                real_d_norm_bias,
                real_d_uvqk_weight,
                real_d_uvqk_bias,
            ) = (
                x.grad.clone(),
                norm_weight.grad.clone(),
                norm_bias.grad.clone(),
                uvqk_weight.grad.clone(),
                uvqk_bias.grad.clone(),
            )
            torch.testing.assert_close(ref_dx, real_dx, atol=atol, rtol=rtol)
            torch.testing.assert_close(
                ref_d_norm_weight, real_d_norm_weight, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                ref_d_norm_bias, real_d_norm_bias, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                ref_d_uvqk_weight, real_d_uvqk_weight, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                ref_d_uvqk_bias, real_d_uvqk_bias, atol=atol, rtol=rtol
            )
