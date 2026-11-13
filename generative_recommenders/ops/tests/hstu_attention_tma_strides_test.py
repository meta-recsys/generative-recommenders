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

"""Regression for full-attention TMA descriptors on fused UVQK views (#583)."""

import itertools
import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] == 9,
    "Hopper GPU required for the TMA stride regression",
)
class HSTUAttentionTmaStridesTest(unittest.TestCase):
    def test_packed_and_interleaved_rows(self) -> None:
        from generative_recommenders.ops.triton.triton_hstu_attention import (
            tensor_descriptor_tma,
        )

        if not tensor_descriptor_tma:
            self.skipTest("Triton host tensor descriptors are unavailable")

        # V may have a different head dimension from Q/K. Vary each layout
        # independently so an input-stride failure cannot mask a store failure.
        for (
            attn_dim,
            interleaved_inputs,
            interleaved_grads,
            strided_dout,
        ) in itertools.product((128, 64), (True, False), (True, False), (True, False)):
            with self.subTest(
                attn_dim=attn_dim,
                interleaved_inputs=interleaved_inputs,
                interleaved_grads=interleaved_grads,
                strided_dout=strided_dout,
            ):
                self._check_rows(
                    attn_dim, interleaved_inputs, interleaved_grads, strided_dout
                )

    def _check_rows(
        self,
        attn_dim: int,
        interleaved_inputs: bool,
        interleaved_grads: bool,
        strided_dout: bool,
    ) -> None:
        import fbgemm_gpu.sparse_ops  # noqa: F401 -- register jagged operators
        from generative_recommenders.ops.pytorch.pt_hstu_attention import (
            pytorch_hstu_mha,
        )
        from generative_recommenders.ops.triton.triton_hstu_attention import (
            triton_hstu_attention_bwd,
            triton_hstu_attention_fwd,
        )

        torch.manual_seed(1234)
        heads, hidden_dim, max_seq_len = 4, 128, 128
        # Unequal lengths exercise sequence offsets, partial tiles and sorting.
        lengths = [93, 117, 128]
        total = sum(lengths)
        offsets = torch.tensor([0, 93, 210, total], device="cuda", dtype=torch.int64)
        targets = torch.tensor([7, 13, 16], device="cuda", dtype=torch.int64)
        sort_indices = torch.argsort(
            torch.tensor(lengths, device="cuda"), descending=True
        )
        split_sizes = [
            heads * hidden_dim,
            heads * hidden_dim,
            heads * attn_dim,
            heads * attn_dim,
        ]
        fused = torch.empty(
            (total, sum(split_sizes)), device="cuda", dtype=torch.bfloat16
        ).uniform_(-0.1, 0.1)
        _, v, q, k = [
            part.view(total, heads, -1) for part in fused.split(split_sizes, dim=1)
        ]
        if not interleaved_inputs:
            q, k, v = [x.contiguous() for x in (q, k, v)]
        rq, rk, rv = [
            x.detach().clone().contiguous().requires_grad_() for x in (q, k, v)
        ]
        reference = pytorch_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=attn_dim**-0.5,
            q=rq,
            k=rk,
            v=rv,
            seq_offsets=offsets,
            num_targets=targets,
        )
        # Give DOut its own row pitch, distinct from both QKV and dQKV.
        dout_storage = torch.randn(
            (total, 2 * heads * hidden_dim), device="cuda", dtype=torch.bfloat16
        )
        dout = dout_storage[:, : heads * hidden_dim].view(total, heads, hidden_dim)
        if not strided_dout:
            dout = dout.contiguous()
        reference.backward(dout.contiguous())
        actual = triton_hstu_attention_fwd(
            N=max_seq_len,
            alpha=attn_dim**-0.5,
            q=q,
            k=k,
            v=v,
            seq_offsets=offsets,
            num_targets=targets,
            max_attn_len=0,
            contextual_seq_len=0,
            sort_by_length_indices=sort_indices,
            enable_tma=True,
            num_softmax_heads=0,
        )
        # Poison every gradient destination so missing writes cannot pass by
        # chance. Attention must also leave the adjacent U slice untouched.
        grads = torch.full_like(fused, 123.0)
        du, dv, dq, dk = [
            part.view(total, heads, -1) for part in grads.split(split_sizes, dim=1)
        ]
        if not interleaved_grads:
            dq, dk, dv = [x.contiguous() for x in (dq, dk, dv)]
        triton_hstu_attention_bwd(
            dout=dout,
            dq=dq,
            dk=dk,
            dv=dv,
            N=max_seq_len,
            alpha=attn_dim**-0.5,
            q=q,
            k=k,
            v=v,
            seq_offsets=offsets,
            num_targets=targets,
            max_attn_len=0,
            contextual_seq_len=0,
            sort_by_length_indices=sort_indices,
            enable_tma=True,
            num_softmax_heads=0,
        )
        for name, result, expected in (
            ("forward", actual, reference),
            ("dq", dq, rq.grad),
            ("dk", dk, rk.grad),
            ("dv", dv, rv.grad),
        ):
            with self.subTest(result=name):
                # Standard BF16 tolerances allow different accumulation orders.
                torch.testing.assert_close(result, expected, rtol=1.6e-2, atol=1e-5)
        torch.testing.assert_close(du, torch.full_like(du, 123.0), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
