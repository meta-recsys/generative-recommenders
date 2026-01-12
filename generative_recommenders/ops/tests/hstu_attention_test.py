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
from generative_recommenders.ops.jagged_tensors import split_2D_jagged
from hypothesis import given, settings, strategies as st, Verbosity


def test_attn(
    batch_size: int,
    heads: int,
    max_uih_len: int,
    max_targets: int,
    attn_dim: int,
    hidden_dim: int,
    causal: bool,
    has_multiple_targets: bool,
    has_max_attn_len: bool,
    dtype: torch.dtype,
    test_backward: bool,
    ref_kernel: HammerKernel,
    real_kernel: HammerKernel,
    skip_comparisons: bool = False,
    sparsity: float = -1.0,
    contextual_seq_len: int = 0,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    enable_tma: bool = False,
) -> None:
    set_dev_mode(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from generative_recommenders.ops.hstu_attention import hstu_mha

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
        1, max_targets + 1, size=(batch_size,), device=torch.device("cuda")
    )
    lengths = lengths + num_targets + contextual_seq_len
    max_seq_len = max_uih_len + max_targets + contextual_seq_len
    if has_max_attn_len:
        max_attn_len = random.randint(1, max_uih_len // 5)
    else:
        max_attn_len = 0
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)

    L = int(seq_offsets[-1].item())
    q = (
        torch.empty((L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    k = (
        torch.empty((L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    v = (
        torch.empty((L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )

    # ref implementation
    ref_out = hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        dropout_pr=0.0,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=ref_kernel,
        enable_tma=enable_tma,
    )
    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)

    if skip_comparisons:
        return

    # pyre-ignore
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    q = q.detach().clone().requires_grad_()
    k = k.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    dout = dout.detach().clone()
    real_out = hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        dropout_pr=0.0,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=real_kernel,
        enable_tma=enable_tma,
    )

    torch.testing.assert_close(
        ref_out,
        real_out,
        atol=atol,
        rtol=rtol,
    )
    if test_backward:
        real_out.backward(dout)
        real_dq, real_dk, real_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        torch.testing.assert_close(ref_dv, real_dv, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_dk, real_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_dq, real_dq, atol=atol, rtol=rtol)


def test_delta_attn(
    batch_size: int,
    heads: int,
    max_uih_len: int,
    max_targets: int,
    delta_size: int,
    attn_dim: int,
    hidden_dim: int,
    has_multiple_targets: bool,
    has_max_attn_len: bool,
    dtype: torch.dtype,
    ref_kernel: HammerKernel,
    real_kernel: HammerKernel,
    contextual_seq_len: int = 0,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    enable_tma: bool = False,
) -> None:
    set_dev_mode(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from generative_recommenders.ops.hstu_attention import delta_hstu_mha

    alpha = 1.0 / (attn_dim**0.5)
    lengths = torch.randint(
        max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
    )
    num_targets = torch.randint(
        1, delta_size + 1, size=(batch_size,), device=torch.device("cuda")
    )
    lengths = lengths + delta_size + contextual_seq_len
    max_seq_len = max_uih_len + delta_size + contextual_seq_len
    if has_max_attn_len:
        max_attn_len = random.randint(1, max_uih_len // 5)
    else:
        max_attn_len = 0
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)

    L = int(seq_offsets[-1].item())
    delta_q = torch.empty(
        (batch_size * delta_size, heads, attn_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)
    k = torch.empty(
        (L, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(-0.1, 0.1)
    v = torch.empty(
        (L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(-0.1, 0.1)

    # ref implementation
    ref_out = delta_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets if has_multiple_targets else None,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=ref_kernel,
        enable_tma=enable_tma,
    )

    # real implementation
    real_out = delta_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets if has_multiple_targets else None,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=real_kernel,
        enable_tma=enable_tma,
    )
    torch.testing.assert_close(
        ref_out,
        real_out,
        atol=atol,
        rtol=rtol,
    )


class HSTUAttentionTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([20, 100, 128, 256]),
        max_targets=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([16, 32, 64, 128]),
        hidden_dim=st.sampled_from([16, 32, 64, 128]),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        has_max_attn_len=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=200,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_triton(self, *args, **kwargs) -> None:
        test_attn(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.just(64),
        heads=st.just(4),
        max_uih_len=st.sampled_from([32768]),
        max_targets=st.sampled_from([32]),
        attn_dim=st.just(128),
        hidden_dim=st.just(128),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from([torch.bfloat16]),
        has_max_attn_len=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=5,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_triton_long_seqs(self, *args, **kwargs) -> None:
        test_attn(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
            skip_comparisons=True,
            sparsity=1.0,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([100, 128, 256]),
        max_targets=st.sampled_from([20, 512]),
        delta_size=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([16, 32, 64, 128]),
        hidden_dim=st.sampled_from([16, 32, 64, 128]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        has_max_attn_len=st.sampled_from([False, True]),
        contextual_seq_len=st.sampled_from([0, 10]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=200,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_delta_attn_triton(self, *args, **kwargs) -> None:
        test_delta_attn(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([20, 100, 128]),
        max_targets=st.sampled_from([20, 512]),
        delta_size=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([16, 32, 64]),
        hidden_dim=st.sampled_from([16, 32, 64]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
        has_max_attn_len=st.sampled_from([False, True]),
        contextual_seq_len=st.sampled_from([0, 10]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=200,
        deadline=None,
    )
    def test_cache(
        self,
        batch_size: int,
        heads: int,
        max_uih_len: int,
        max_targets: int,
        delta_size: int,
        attn_dim: int,
        hidden_dim: int,
        has_multiple_targets: bool,
        dtype: torch.dtype,
        has_max_attn_len: bool,
        contextual_seq_len: int,
    ) -> None:
        set_dev_mode(True)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        from generative_recommenders.ops.hstu_attention import delta_hstu_mha, hstu_mha

        alpha = 1.0 / (attn_dim**0.5)
        lengths = torch.randint(
            max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
        )
        num_targets = torch.randint(
            1, delta_size + 1, size=(batch_size,), device=torch.device("cuda")
        )
        lengths = lengths + delta_size + contextual_seq_len
        max_seq_len = max_uih_len + delta_size + contextual_seq_len
        if has_max_attn_len:
            max_attn_len = random.randint(1, max_uih_len // 5)
        else:
            max_attn_len = 0
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)

        L = int(seq_offsets[-1].item())
        q = torch.empty(
            (L, heads, attn_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.1, 0.1)
        _, delta_q = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=q.view(-1, heads * attn_dim),
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                lengths - delta_size
            ),
            offsets_right=None,
            kernel=HammerKernel.TRITON,
        )
        delta_q = delta_q.view(-1, heads, attn_dim)
        k = torch.empty(
            (L, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
        ).uniform_(-0.1, 0.1)
        v = torch.empty(
            (L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda")
        ).uniform_(-0.1, 0.1)
        prime_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            lengths - delta_size
        )

        # ref implementation
        ref_out = hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            num_targets=num_targets if has_multiple_targets else None,
            dropout_pr=0.0,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.TRITON,
        )
        _, delta_out = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=ref_out.view(-1, heads * hidden_dim),
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=HammerKernel.TRITON,
        )
        delta_out = delta_out.view(-1, heads, hidden_dim)

        # real implementation
        real_delta_out = delta_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets if has_multiple_targets else None,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
        torch.testing.assert_close(
            delta_out,
            real_delta_out,
        )
