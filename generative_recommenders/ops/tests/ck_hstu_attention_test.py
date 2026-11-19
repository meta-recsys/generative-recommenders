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
from typing import Optional, Tuple

import torch
from generative_recommenders.common import generate_sparse_seq_len, set_dev_mode
from generative_recommenders.ops.ck.ck_hstu_attention import (
    ck_hstu_mha,
    is_ck_hstu_available,
)
from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha
from hypothesis import given, settings, strategies as st, Verbosity

ck_hstu_unavailable: Tuple[bool, str] = (
    not is_ck_hstu_available(),
    "Requires an AMD GPU and the CK HSTU attention extension",
)


def _test_attn(
    batch_size: int,
    heads: int,
    max_uih_len: int,
    attn_scale: float,
    max_targets: int,
    attn_dim: int,
    causal: bool,
    alpha: float,
    has_multiple_targets: bool,
    dtype: torch.dtype,
    test_backward: bool,
    sparsity_w_dense: Tuple[float, bool] = (1.0, False),
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    max_attn_len: int = 0,
    tensor_range: Tuple[float, float] = (-0.1, 0.1),
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    set_dev_mode(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    hidden_dim = attn_dim

    sparsity, dense = sparsity_w_dense
    lengths = generate_sparse_seq_len(
        size=batch_size,
        max_seq_len=max_uih_len,
        sparsity=sparsity,
        device=device,
    )
    num_targets = torch.randint(
        0 if not dense else max_targets,
        max_targets + 1,
        size=(batch_size,),
        device=device,
    ).to(torch.int32)
    lengths = lengths + contextual_seq_len
    if has_multiple_targets:
        # The kernel needs at least one UIH token per sequence.
        lengths = torch.where(lengths > 0, lengths, 1)
    lengths = lengths + num_targets
    max_seq_len = max_uih_len + max_targets + contextual_seq_len
    if min_full_attn_seq_len == 0 and max_attn_len > 0:
        return
    seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)

    L = int(seq_offsets[-1].item())

    def rand(*shape: int) -> torch.Tensor:
        return (
            torch.empty(shape, dtype=dtype, device=device)
            .uniform_(tensor_range[0], tensor_range[1])
            .requires_grad_()
        )

    q = rand(L, heads, attn_dim)
    k = rand(L, heads, attn_dim)
    v = rand(L, heads, hidden_dim)
    attn_scale_tensor = torch.tensor(attn_scale, dtype=torch.float32)

    ref_out = pytorch_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q.to(torch.float32),
        k=k.to(torch.float32),
        v=v.to(torch.float32),
        seq_offsets=seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        attn_scale=attn_scale_tensor,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    ).to(dtype)
    dout = torch.empty((L, heads, hidden_dim), dtype=dtype, device=device).uniform_(
        tensor_range[0], tensor_range[1]
    )
    ref_out.backward(dout)
    # pyrefly: ignore [missing-attribute]
    ref_dv, v.grad = v.grad.clone(), None
    # pyrefly: ignore [missing-attribute]
    ref_dk, k.grad = k.grad.clone(), None
    # pyrefly: ignore [missing-attribute]
    ref_dq, q.grad = q.grad.clone(), None

    q = q.detach().clone().requires_grad_()
    k = k.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    dout = dout.detach().clone()
    ck_seq_offsets: Optional[torch.Tensor] = seq_offsets.to(torch.int32)
    if dense:
        assert sparsity == 1.0
        q = q.reshape(batch_size, max_seq_len, heads, attn_dim)
        q.retain_grad()
        k = k.reshape(batch_size, max_seq_len, heads, attn_dim)
        k.retain_grad()
        v = v.reshape(batch_size, max_seq_len, heads, hidden_dim)
        v.retain_grad()
        dout = dout.reshape(batch_size, max_seq_len, heads, hidden_dim)
        ck_seq_offsets = None
    min_seq_len = int(lengths.min().item())
    real_out = ck_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets_q=ck_seq_offsets,
        seq_offsets_kv=ck_seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        attn_scale=attn_scale_tensor,
        max_attn_len=max_attn_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
        contextual_seq_len=contextual_seq_len,
        use_softmax=False,
        is_cross_attention=False,
        # Self-attention: KV length == Q length.
        max_seq_len_kv=max_seq_len,
        min_seq_len_q=min_seq_len,
        min_seq_len_kv=min_seq_len,
    )
    torch.testing.assert_close(
        real_out, ref_out.reshape(real_out.shape), atol=atol, rtol=rtol
    )
    if test_backward:
        real_out.backward(dout)
        # pyrefly: ignore [missing-attribute]
        real_dq, real_dk, real_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        torch.testing.assert_close(
            ref_dq.reshape(q.shape), real_dq, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            ref_dk.reshape(k.shape), real_dk, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            ref_dv.reshape(v.shape), real_dv, atol=atol, rtol=rtol
        )


class CKHSTUAttentionTest(unittest.TestCase):
    @unittest.skipIf(*ck_hstu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.sampled_from([8, 16]),
        heads=st.sampled_from([4]),
        alpha=st.sampled_from([0.1, 0.3, 0.8]),
        attn_scale=st.sampled_from([0.0001, 0.0003]),
        max_uih_len=st.integers(min_value=1, max_value=1000),
        max_targets=st.integers(min_value=1, max_value=1000),
        attn_dim=st.sampled_from([8, 64, 128, 256]),
        causal=st.sampled_from([True]),
        sparsity_w_dense=st.sampled_from([(1.0, False), (0.5, False)]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from([torch.bfloat16]),
        contextual_seq_len=st.sampled_from([0]),
        max_attn_len=st.integers(min_value=0, max_value=300),
        min_full_attn_seq_len=st.integers(min_value=0, max_value=300),
        tensor_range=st.sampled_from([(-0.1, 0.1), (0.1, 0.1), (0.0, 1.0), (1.0, 1.0)]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=500,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_ck(self, *args, **kwargs) -> None:
        _test_attn(*args, **kwargs, test_backward=True)

    @unittest.skipIf(*ck_hstu_unavailable)
    def test_attn_ck_dense(self) -> None:
        torch.manual_seed(0)
        _test_attn(
            batch_size=4,
            heads=2,
            max_uih_len=200,
            attn_scale=0.0003,
            max_targets=20,
            attn_dim=128,
            causal=True,
            alpha=0.3,
            has_multiple_targets=True,
            dtype=torch.bfloat16,
            test_backward=True,
            sparsity_w_dense=(1.0, True),
        )
