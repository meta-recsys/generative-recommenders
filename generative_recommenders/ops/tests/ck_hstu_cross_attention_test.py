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
from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_cross_mha
from hypothesis import given, settings, strategies as st, Verbosity

ck_hstu_unavailable: Tuple[bool, str] = (
    not is_ck_hstu_available(),
    "Requires an AMD GPU and the CK HSTU attention extension",
)


def _test_attn(
    batch_size: int,
    heads: int,
    max_uih_len: int,
    max_targets: int,
    attn_dim: int,
    causal: bool,
    alpha: float,
    has_multiple_targets: bool,
    dtype: torch.dtype,
    test_backward: bool,
    sparsity: float = -1.0,
    sparsity_w_dense: Tuple[float, bool] = (1.0, False),
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    max_attn_len: int = 0,
    tensor_range: Tuple[float, float] = (-0.1, 0.1),
    sort_by_length: bool = False,
    use_softmax: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    attn_scale_dtype: torch.dtype = torch.float32,
) -> None:
    set_dev_mode(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    hidden_dim = attn_dim

    sparsity, dense = sparsity_w_dense
    if sparsity > 0.0:
        q_lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
        kv_lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
    else:
        assert sparsity == 1.0
        q_lengths = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int64)
            * max_uih_len
        )
        kv_lengths = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int64)
            * max_uih_len
        )

    num_targets = torch.randint(
        0 if not dense else max_targets,
        max_targets + 1,
        size=(batch_size,),
        device=torch.device("cuda"),
    ).to(torch.int32)
    # Ensure num_targets does not exceed the sequence lengths (minus 1 to leave at least 1 for uih)
    # This prevents integer underflow in the kernel when computing seqlen - num_target
    # For cross-attention, num_targets must be bounded by BOTH q_lengths and kv_lengths
    min_seq_len = torch.minimum(q_lengths, kv_lengths)
    max_allowed_targets = (min_seq_len - 1).clamp(min=0).to(torch.int32)
    num_targets = torch.minimum(num_targets, max_allowed_targets)

    q_lengths = q_lengths + contextual_seq_len
    if has_multiple_targets:
        # current implementation needs uih to be at least 1
        q_lengths = torch.where(q_lengths > 0, q_lengths, 1)

    kv_lengths = kv_lengths + contextual_seq_len
    if has_multiple_targets:
        # current implementation needs uih to be at least 1
        kv_lengths = torch.where(kv_lengths > 0, kv_lengths, 1)

    q_lengths = q_lengths + num_targets
    kv_lengths = kv_lengths + num_targets

    max_seq_len = max_uih_len + max_targets + contextual_seq_len
    if min_full_attn_seq_len == 0 and max_attn_len > 0:
        return

    q_seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    q_seq_offsets[1:] = torch.cumsum(q_lengths, dim=0)

    kv_seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    kv_seq_offsets[1:] = torch.cumsum(kv_lengths, dim=0)

    q_L = int(q_seq_offsets[-1].item())
    kv_L = int(kv_seq_offsets[-1].item())
    q = (
        torch.empty((q_L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(tensor_range[0], tensor_range[1])
        .requires_grad_()
    )
    k = (
        torch.empty((kv_L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(tensor_range[0], tensor_range[1])
        .requires_grad_()
    )
    v = (
        torch.empty((kv_L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(tensor_range[0], tensor_range[1])
        .requires_grad_()
    )

    # A device tensor reaches the kernel through attn_scale_ptr; lowering can
    # hand the op a bf16 one, which the op converts to fp32.
    attn_scale = torch.tensor(0.5, dtype=attn_scale_dtype, device=torch.device("cuda"))

    # ref implementation
    pt_out = pytorch_hstu_cross_mha(
        max_seq_len_q=max_seq_len,
        max_seq_len_kv=max_seq_len,
        alpha=alpha,
        q=q.to(torch.float32),
        k=k.to(torch.float32),
        v=v.to(torch.float32),
        seq_offsets_q=q_seq_offsets,
        seq_offsets_kv=kv_seq_offsets,
        num_softmax_heads=heads if use_softmax else 0,
        num_targets=num_targets if has_multiple_targets else None,
        targets_in_kv=False,
        causal=causal,
        attn_scale=attn_scale,
    ).to(dtype)
    dout = torch.empty_like(pt_out).uniform_(-0.1, 0.1)
    pt_out.backward(dout)
    # pyre-ignore
    pt_dq, q.grad = q.grad.clone(), None
    # pyrefly: ignore [missing-attribute]
    pt_dk, k.grad = k.grad.clone(), None
    # pyrefly: ignore [missing-attribute]
    pt_dv, v.grad = v.grad.clone(), None

    torch.cuda.synchronize()

    # cuda implementation
    q = q.detach().clone().requires_grad_()
    k = k.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    dout = dout.detach().clone()
    if dense:
        assert sparsity == 1.0
        q = q.reshape(batch_size, max_seq_len, heads, attn_dim)
        k = k.reshape(batch_size, max_seq_len, heads, attn_dim)
        k.retain_grad()
        v = v.reshape(batch_size, max_seq_len, heads, hidden_dim)
        v.retain_grad()
        q.retain_grad()
        dout = dout.reshape(batch_size, max_seq_len, heads, hidden_dim)
        q_seq_offsets = None
        kv_seq_offsets = None
    else:
        q_seq_offsets = q_seq_offsets.to(torch.int32)
        kv_seq_offsets = kv_seq_offsets.to(torch.int32)

    max_seq_len_kv = int(kv_lengths.max().item())
    min_seq_len_q = int(q_lengths.min().item())
    min_seq_len_kv = int(kv_lengths.min().item())
    ck_out = ck_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets_q=q_seq_offsets,
        seq_offsets_kv=kv_seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        max_attn_len=0,
        min_full_attn_seq_len=0,
        contextual_seq_len=0,
        use_softmax=use_softmax,
        is_cross_attention=True,
        attn_scale=attn_scale,
        max_seq_len_kv=max_seq_len_kv,
        min_seq_len_q=min_seq_len_q,
        min_seq_len_kv=min_seq_len_kv,
    )

    torch.cuda.synchronize()
    pt_out = pt_out.reshape(ck_out.shape)
    torch.testing.assert_close(
        ck_out,
        pt_out,
        atol=atol,
        rtol=rtol,
    )
    if test_backward:
        pt_dq = pt_dq.reshape(q.shape)
        pt_dk = pt_dk.reshape(k.shape)
        pt_dv = pt_dv.reshape(v.shape)
        ck_out.backward(dout)
        # pyrefly: ignore [missing-attribute]
        ck_dq, ck_dk, ck_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        # Long BF16 rows accumulate in a different order in CK and PyTorch. With
        # constant inputs P is uniform, so its BF16 rounding does not cancel and
        # dq/dk (exactly 0) pick up ~alpha * rowsum(dO) * 2^-9.
        grad_atol_floor = max(
            1e-3 if atol is None else atol,
            alpha * float(dout.sum(-1).abs().max()) * 2**-8,
        )
        for ref_grad, ck_grad in (
            (pt_dq, ck_dq),
            (pt_dk, ck_dk),
            (pt_dv, ck_dv),
        ):
            backward_atol = max(
                grad_atol_floor,
                5e-2 * float(ref_grad.abs().max()),
            )
            torch.testing.assert_close(ref_grad, ck_grad, atol=backward_atol, rtol=rtol)


def _test_attn_gqa(
    batch_size: int,
    heads_q: int,
    heads_kv: int,
    max_uih_len: int,
    max_targets: int,
    attn_dim: int,
    alpha: float,
    has_multiple_targets: bool,
    dtype: torch.dtype,
    sparsity: float = 0.5,
    tensor_range: Tuple[float, float] = (-0.1, 0.1),
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    """Test GQA cross-attention where Q has more heads than K/V."""
    set_dev_mode(True)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    hidden_dim = attn_dim

    if sparsity > 0.0 and sparsity < 1.0:
        q_lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
        kv_lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
    else:
        q_lengths = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int64)
            * max_uih_len
        )
        kv_lengths = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int64)
            * max_uih_len
        )

    num_targets = torch.randint(
        0 if not has_multiple_targets else max_targets,
        max_targets + 1,
        size=(batch_size,),
        device=torch.device("cuda"),
    ).to(torch.int32)
    min_seq_len = torch.minimum(q_lengths, kv_lengths)
    max_allowed_targets = (min_seq_len - 1).clamp(min=0).to(torch.int32)
    num_targets = torch.minimum(num_targets, max_allowed_targets)

    if has_multiple_targets:
        q_lengths = torch.where(q_lengths > 0, q_lengths, 1)
        kv_lengths = torch.where(kv_lengths > 0, kv_lengths, 1)

    q_lengths = q_lengths + num_targets
    kv_lengths = kv_lengths + num_targets

    max_seq_len = max_uih_len + max_targets

    q_seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    q_seq_offsets[1:] = torch.cumsum(q_lengths, dim=0)

    kv_seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    kv_seq_offsets[1:] = torch.cumsum(kv_lengths, dim=0)

    q_L = int(q_seq_offsets[-1].item())
    kv_L = int(kv_seq_offsets[-1].item())

    # Q has heads_q heads, K/V has heads_kv heads (GQA)
    q = torch.empty(
        (q_L, heads_q, attn_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(tensor_range[0], tensor_range[1])
    k = torch.empty(
        (kv_L, heads_kv, attn_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(tensor_range[0], tensor_range[1])
    v = torch.empty(
        (kv_L, heads_kv, hidden_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(tensor_range[0], tensor_range[1])

    attn_scale = torch.tensor(0.5, dtype=torch.float32, device=torch.device("cuda"))

    # Reference: PyTorch (supports GQA via repeat_interleave in _pad_cross_qkv)
    pt_out = pytorch_hstu_cross_mha(
        max_seq_len_q=max_seq_len,
        max_seq_len_kv=max_seq_len,
        alpha=alpha,
        q=q.to(torch.float32),
        k=k.to(torch.float32),
        v=v.to(torch.float32),
        seq_offsets_q=q_seq_offsets,
        seq_offsets_kv=kv_seq_offsets,
        num_softmax_heads=0,
        num_targets=num_targets if has_multiple_targets else None,
        targets_in_kv=False,
        causal=False,
        attn_scale=attn_scale,
    ).to(dtype)

    torch.cuda.synchronize()

    # CK implementation with GQA (stride-based broadcasting)
    q_seq_offsets_i32 = q_seq_offsets.to(torch.int32)
    kv_seq_offsets_i32 = kv_seq_offsets.to(torch.int32)

    max_seq_len_kv = int(kv_lengths.max().item())
    min_seq_len_q = int(q_lengths.min().item())
    min_seq_len_kv = int(kv_lengths.min().item())
    ck_out = ck_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets_q=q_seq_offsets_i32,
        seq_offsets_kv=kv_seq_offsets_i32,
        causal=False,
        num_targets=num_targets if has_multiple_targets else None,
        max_attn_len=0,
        min_full_attn_seq_len=0,
        contextual_seq_len=0,
        use_softmax=False,
        is_cross_attention=True,
        attn_scale=attn_scale,
        max_seq_len_kv=max_seq_len_kv,
        min_seq_len_q=min_seq_len_q,
        min_seq_len_kv=min_seq_len_kv,
    )

    torch.cuda.synchronize()
    pt_out = pt_out.reshape(ck_out.shape)
    torch.testing.assert_close(
        ck_out,
        pt_out,
        atol=atol,
        rtol=rtol,
    )


class CKHSTUCrossAttentionTest(unittest.TestCase):
    @unittest.skipIf(*ck_hstu_unavailable)
    @given(
        batch_size=st.sampled_from([8, 16]),
        heads=st.sampled_from([4]),
        alpha=st.sampled_from([0.1, 0.3, 0.8]),
        max_uih_len=st.integers(min_value=128, max_value=1000),
        max_targets=st.integers(min_value=1, max_value=1000),
        attn_dim=st.sampled_from([64, 128, 256]),
        causal=st.sampled_from([False]),
        sparsity_w_dense=st.sampled_from([(1.0, False), (0.5, False)]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from([torch.bfloat16]),
        contextual_seq_len=st.sampled_from([0, 5]),
        max_attn_len=st.integers(min_value=0, max_value=300),
        min_full_attn_seq_len=st.integers(min_value=0, max_value=300),
        sort_by_length=st.sampled_from([False]),
        use_softmax=st.sampled_from([True, False]),
        tensor_range=st.sampled_from([(-0.1, 0.1), (0.1, 0.1), (0.0, 1.0), (1.0, 1.0)]),
        attn_scale_dtype=st.sampled_from([torch.float32, torch.bfloat16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=500,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_ck_cross(self, *args, **kwargs) -> None:
        _test_attn(
            *args,
            **kwargs,
            test_backward=True,
            atol=1e-3,
            rtol=0.05,
        )

    @unittest.skipIf(*ck_hstu_unavailable)
    @given(
        batch_size=st.sampled_from([8, 16]),
        alpha=st.sampled_from([0.1, 0.3, 0.8]),
        max_uih_len=st.integers(min_value=128, max_value=500),
        max_targets=st.integers(min_value=1, max_value=100),
        attn_dim=st.sampled_from([64, 128]),
        sparsity=st.sampled_from([1.0, 0.5]),
        has_multiple_targets=st.sampled_from([True]),
        dtype=st.sampled_from([torch.bfloat16]),
        tensor_range=st.sampled_from([(-0.1, 0.1), (0.1, 0.1)]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=100,
        deadline=None,
    )
    def test_attn_ck_cross_gqa(
        self,
        batch_size: int,
        alpha: float,
        max_uih_len: int,
        max_targets: int,
        attn_dim: int,
        sparsity: float,
        has_multiple_targets: bool,
        dtype: torch.dtype,
        tensor_range: Tuple[float, float],
    ) -> None:
        _test_attn_gqa(
            batch_size=batch_size,
            heads_q=2,
            heads_kv=1,
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            attn_dim=attn_dim,
            alpha=alpha,
            has_multiple_targets=has_multiple_targets,
            dtype=dtype,
            sparsity=sparsity,
            tensor_range=tensor_range,
            atol=1e-3,
            rtol=0.05,
        )

    @unittest.skipIf(*ck_hstu_unavailable)
    @given(
        batch_size=st.sampled_from([4, 8]),
        heads=st.sampled_from([1, 4]),
        max_q_len=st.integers(min_value=1, max_value=500),
        attn_dim=st.sampled_from([64, 128]),
        use_softmax=st.sampled_from([True, False]),
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=50,
        deadline=None,
    )
    def test_attn_ck_cross_empty_kv(
        self,
        batch_size: int,
        heads: int,
        max_q_len: int,
        attn_dim: int,
        use_softmax: bool,
        dtype: torch.dtype,
    ) -> None:
        # With no keys/values in the whole batch, every query row attends to
        # nothing, so the output is zeros and q gets no gradient.
        set_dev_mode(True)
        device = torch.device("cuda")
        q_lengths = torch.randint(1, max_q_len + 1, (batch_size,), device=device)
        max_seq_len = int(q_lengths.max().item())
        q_seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
        q_seq_offsets[1:] = torch.cumsum(q_lengths, dim=0)
        kv_seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )
        q = (
            torch.empty(
                (int(q_seq_offsets[-1].item()), heads, attn_dim),
                dtype=dtype,
                device=device,
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        k = torch.empty(
            (0, heads, attn_dim), dtype=dtype, device=device
        ).requires_grad_()
        v = torch.empty(
            (0, heads, attn_dim), dtype=dtype, device=device
        ).requires_grad_()
        # Leave NaN-filled blocks of the output's size in the caching allocator,
        # so an output the op never writes reads back as NaN instead of passing
        # as zeros by chance.
        poison = [torch.full_like(q, float("nan")) for _ in range(4)]
        del poison

        out = ck_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=1.0 / attn_dim**0.5,
            q=q,
            k=k,
            v=v,
            seq_offsets_q=q_seq_offsets,
            seq_offsets_kv=kv_seq_offsets,
            use_softmax=use_softmax,
            is_cross_attention=True,
            max_seq_len_kv=max_seq_len,
        )
        torch.testing.assert_close(out, torch.zeros_like(q))
        out.backward(torch.empty_like(out).uniform_(-0.1, 0.1))
        dq = q.grad
        assert dq is not None
        torch.testing.assert_close(dq, torch.zeros_like(q))

    @unittest.skipIf(*ck_hstu_unavailable)
    def test_ck_train_ops_opcheck(self) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")
        heads, attn_dim, alpha = 4, 128, 0.3
        q_lengths = torch.tensor([3, 17, 64, 1], dtype=torch.int32, device=device)
        kv_lengths = torch.tensor([40, 9, 128, 5], dtype=torch.int32, device=device)
        q_offsets = torch.zeros(5, dtype=torch.int32, device=device)
        q_offsets[1:] = torch.cumsum(q_lengths, dim=0)
        kv_offsets = torch.zeros(5, dtype=torch.int32, device=device)
        kv_offsets[1:] = torch.cumsum(kv_lengths, dim=0)
        max_q, max_kv, min_q, min_kv = 64, 128, 1, 5

        def rand(n: int) -> torch.Tensor:
            return torch.empty(
                (n, heads, attn_dim), dtype=torch.bfloat16, device=device
            ).uniform_(-0.1, 0.1)

        q = rand(int(q_lengths.sum().item()))
        k = rand(int(kv_lengths.sum().item()))
        v = rand(int(kv_lengths.sum().item()))
        attn_scale = torch.tensor(0.5, dtype=torch.float32, device=device)

        for use_softmax in (False, True):
            with self.subTest(use_softmax=use_softmax):
                shared = (q_offsets, kv_offsets, False, None, 0, 0, 0, attn_scale)
                tail = (use_softmax, True, max_kv, min_q, min_kv)
                fwd_args = (max_q, alpha, q, k, v) + shared + tail
                torch.library.opcheck(
                    torch.ops.ck_hstu_experimental.ck_hstu_mha_fwd.default,
                    fwd_args,
                )
                out, lse = torch.ops.ck_hstu_experimental.ck_hstu_mha_fwd(*fwd_args)
                dout = torch.empty_like(out).uniform_(-0.1, 0.1)
                bwd_args = (
                    (max_q, alpha, dout, q, k, v, out, lse if use_softmax else None)
                    + shared
                    + tail
                )
                torch.library.opcheck(
                    torch.ops.ck_hstu_experimental.ck_hstu_mha_bwd.default,
                    bwd_args,
                )

    @unittest.skipIf(*ck_hstu_unavailable)
    @given(
        batch_size=st.sampled_from([1, 5, 13]),
        heads=st.sampled_from([1, 4]),
        max_q_len=st.integers(min_value=1, max_value=600),
        max_kv_len=st.integers(min_value=0, max_value=3000),
        kv_truncate_len=st.integers(min_value=1, max_value=2000),
        kv_keep_last=st.booleans(),
        attn_dim=st.sampled_from([64, 128]),
        use_softmax=st.booleans(),
        dtype=st.sampled_from([torch.bfloat16, torch.float16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=100,
        deadline=None,
    )
    def test_attn_ck_cross_kv_truncate(
        self,
        batch_size: int,
        heads: int,
        max_q_len: int,
        max_kv_len: int,
        kv_truncate_len: int,
        kv_keep_last: bool,
        attn_dim: int,
        use_softmax: bool,
        dtype: torch.dtype,
    ) -> None:
        # Truncating in the kernel must match the untruncated kernel run on K/V
        # gathered down to the kept window of each sequence. The two calls may
        # still dispatch differently (the strategy reads the total KV length,
        # which the truncated call can only bound), so compare with errors
        # relative to the output scale rather than bitwise.
        device = torch.device("cuda")
        q_lengths = torch.randint(1, max_q_len + 1, (batch_size,), device=device)
        kv_lengths = torch.randint(0, max_kv_len + 1, (batch_size,), device=device)
        kept = torch.clamp(kv_lengths, max=kv_truncate_len)

        def offsets(lengths: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)
            out[1:] = torch.cumsum(lengths, dim=0)
            return out

        q_offsets, kv_offsets, kept_offsets = (
            offsets(q_lengths),
            offsets(kv_lengths),
            offsets(kept),
        )
        total_q, total_kv = int(q_offsets[-1].item()), int(kv_offsets[-1].item())
        q = torch.empty((total_q, heads, attn_dim), dtype=dtype, device=device)
        k = torch.empty((total_kv, heads, attn_dim), dtype=dtype, device=device)
        v = torch.empty((total_kv, heads, attn_dim), dtype=dtype, device=device)
        for t in (q, k, v):
            t.uniform_(-0.5, 0.5)

        def window_index(keep_last: bool) -> torch.Tensor:
            parts = []
            for b in range(batch_size):
                begin, end = int(kv_offsets[b].item()), int(kv_offsets[b + 1].item())
                n = int(kept[b].item())
                parts.append(
                    torch.arange(end - n, end, device=device)
                    if keep_last
                    else torch.arange(begin, begin + n, device=device)
                )
            return torch.cat(parts)

        common = {
            "max_seq_len": int(q_lengths.max().item()),
            "alpha": 1.0 / attn_dim**0.5,
            "q": q,
            "seq_offsets_q": q_offsets,
            "attn_scale": torch.tensor(1.0 / max_q_len, device=device),
            "use_softmax": use_softmax,
            "is_cross_attention": True,
            # Same host KV extent for both calls, so they size split-kv alike.
            "max_seq_len_kv": kv_truncate_len,
        }

        def gathered(keep_last: bool) -> torch.Tensor:
            index = window_index(keep_last)
            return ck_hstu_mha(
                k=k[index].contiguous(),
                v=v[index].contiguous(),
                seq_offsets_kv=kept_offsets,
                **common,
            ).float()

        truncated = ck_hstu_mha(
            k=k,
            v=v,
            seq_offsets_kv=kv_offsets,
            kv_truncate_len=kv_truncate_len,
            kv_keep_last=kv_keep_last,
            **common,
        ).float()

        def errors(out: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
            diff = out - ref
            ref_norm = ref.norm().item()
            ref_max = ref.abs().max().item()
            if ref_norm == 0.0:
                return diff.norm().item(), diff.abs().max().item()
            return diff.norm().item() / ref_norm, diff.abs().max().item() / ref_max

        rel_fro, rel_max = errors(truncated, gathered(kv_keep_last))
        self.assertLess(rel_fro, 5e-3)
        self.assertLess(rel_max, 2e-2)
        # Negative control: on sequences at least twice kv_truncate_len long the
        # first and last windows are disjoint, so the opposite window must give
        # a clearly different result there. This shows the check above is
        # sensitive to which keys the kernel kept.
        disjoint = [
            b
            for b in range(batch_size)
            if int(kv_lengths[b].item()) >= 2 * kv_truncate_len
        ]
        if disjoint:
            rows = torch.cat(
                [
                    torch.arange(
                        int(q_offsets[b].item()),
                        int(q_offsets[b + 1].item()),
                        device=device,
                    )
                    for b in disjoint
                ]
            )
            other = gathered(not kv_keep_last)
            other_fro, _ = errors(truncated[rows], other[rows])
            self.assertGreater(other_fro, 5e-2)

    @unittest.skipIf(*ck_hstu_unavailable)
    def test_attn_ck_cross_kv_truncate_rejects_training(self) -> None:
        device = torch.device("cuda")
        offsets = torch.tensor([0, 8], dtype=torch.int32, device=device)
        q = torch.randn(
            (8, 1, 64), dtype=torch.bfloat16, device=device, requires_grad=True
        )
        kv = torch.randn((8, 1, 64), dtype=torch.bfloat16, device=device)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            ck_hstu_mha(
                max_seq_len=8,
                alpha=0.125,
                q=q,
                k=kv,
                v=kv,
                seq_offsets_q=offsets,
                seq_offsets_kv=offsets,
                use_softmax=True,
                is_cross_attention=True,
                max_seq_len_kv=4,
                kv_truncate_len=4,
            )
