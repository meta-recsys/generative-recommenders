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
from typing import List

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel, set_dev_mode
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.ops.jagged_tensors import split_2D_jagged
from hypothesis import given, settings, strategies as st, Verbosity


def _inplace_swap(
    batch_size: int,
    x: torch.Tensor,
    swap_from: torch.Tensor,
    swap_to: torch.Tensor,
) -> torch.Tensor:
    for i in range(batch_size):
        tmp = x[i, swap_from[i], :].detach().clone()
        x[i, swap_from[i], :] = x[i, swap_to[i], :]
        x[i, swap_to[i], :] = tmp
    return x


class StuTest(unittest.TestCase):
    # pyre-ignore
    @given(
        causal=st.sampled_from([True]),
        num_layers=st.sampled_from([2]),
        num_heads=st.sampled_from([1, 2]),
        max_uih_len=st.sampled_from([20, 64]),
        batch_size=st.sampled_from([8]),
        embedding_dim=st.sampled_from([16]),
        attention_dim=st.sampled_from([32]),
        linear_hidden_dim=st.sampled_from([64]),
        has_multiple_targets=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
        use_group_norm=st.sampled_from([True, False]),
        recompute_uvqk_in_backward=st.sampled_from([True, False]),
        recompute_normed_x_in_backward=st.sampled_from([True, False]),
        recompute_y_in_backward=st.sampled_from([True, False]),
        empty_inputs=st.sampled_from([False]),
        dtype=st.sampled_from(
            [torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=100, deadline=None)
    def test_triton(
        self,
        causal: bool,
        num_layers: int,
        num_heads: int,
        max_uih_len: int,
        batch_size: int,
        embedding_dim: int,
        attention_dim: int,
        linear_hidden_dim: int,
        has_multiple_targets: bool,
        contextual_seq_len: int,
        use_group_norm: bool,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        recompute_y_in_backward: bool,
        empty_inputs: bool,  # test the case where all the seqlen in the batch are 0
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")

        stu_layers: List[STU] = [
            STULayer(
                config=STULayerConfig(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    output_dropout_ratio=0.0,
                    causal=causal,
                    target_aware=has_multiple_targets,
                    max_attn_len=None,
                    attn_alpha=None,
                    use_group_norm=use_group_norm,
                    recompute_normed_x=recompute_normed_x_in_backward,
                    recompute_uvqk=recompute_uvqk_in_backward,
                    recompute_y=recompute_y_in_backward,
                    sort_by_length=True,
                    contextual_seq_len=contextual_seq_len,
                ),
                is_inference=False,
            )
            for _ in range(num_layers)
        ]
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=False,
        ).to(device)
        stu.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
        stu_triton = copy.deepcopy(stu)
        stu_triton.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)

        if empty_inputs:
            x_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
            num_targets = torch.zeros(batch_size, dtype=torch.int32, device=device)
            contextual_seq_len = 0
            max_seq_len = 16
        else:
            x_lengths = torch.randint(max_uih_len + 1, (batch_size,), device=device)
            x_lengths = x_lengths + contextual_seq_len
            max_seq_len = max_uih_len + contextual_seq_len
            max_targets = 20
            num_targets = torch.randint(
                1, max_targets, size=(batch_size,), device=device
            )
            if has_multiple_targets:
                x_lengths = x_lengths + num_targets
                max_seq_len = max_seq_len + max_targets
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total_seq_len = int(x_offsets[-1].cpu().item())
        x = torch.randn(
            int(total_seq_len),
            embedding_dim,
            device=device,
            dtype=dtype,
        ).requires_grad_(True)
        x_triton = x.clone().detach().requires_grad_()
        stu_output = stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        stu_triton_output = stu_triton(
            x=x_triton,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        atol = 5e-3 if dtype == torch.bfloat16 else None
        rtol = 1e-2 if dtype == torch.bfloat16 else None
        torch.testing.assert_close(stu_triton_output, stu_output, atol=atol, rtol=rtol)
        dout = torch.randn_like(stu_output)
        stu_output.backward(dout)
        dout = dout.detach().clone()
        stu_triton_output.backward(dout)
        torch.testing.assert_close(x.grad, x_triton.grad, atol=atol, rtol=rtol)

    # pyre-ignore
    @given(
        dtype=st.sampled_from(
            [torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_target_invariance(
        self,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")
        num_layers = 2
        num_heads = 2
        max_seq_len = 32
        batch_size = 8
        embedding_dim = 16
        attention_dim = 32
        linear_hidden_dim = 32
        causal = True
        use_group_norm = False
        recompute_normed_x_in_backward = False
        recompute_uvqk_in_backward = False
        recompute_y_in_backward = False
        max_attn_len = None
        stu_layers: List[STU] = [
            STULayer(
                config=STULayerConfig(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    output_dropout_ratio=0.0,
                    causal=causal,
                    target_aware=True,
                    max_attn_len=max_attn_len,
                    attn_alpha=None,
                    use_group_norm=use_group_norm,
                    recompute_normed_x=recompute_normed_x_in_backward,
                    recompute_uvqk=recompute_uvqk_in_backward,
                    recompute_y=recompute_y_in_backward,
                    sort_by_length=True,
                    contextual_seq_len=0,
                ),
                is_inference=False,
            )
            for _ in range(num_layers)
        ]
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=False,
        ).to(device)

        x_lengths = torch.randint(
            low=2, high=max_seq_len + 1, size=(batch_size,), device=device
        )
        num_targets = torch.randint(low=2, high=10, size=(batch_size,), device=device)
        x_lengths = x_lengths + num_targets
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total_seq_len = int(x_offsets[-1].cpu())

        swap_from = torch.remainder(
            torch.randint(20, (batch_size,), device=device), num_targets
        )
        swap_to = torch.remainder(
            torch.randint(20, (batch_size,), device=device), num_targets
        )
        swap_from = x_lengths - 1 - swap_from
        swap_to = x_lengths - 1 - swap_to
        max_seq_len = int(x_lengths.max().item())

        # forward()
        x = torch.randn(
            int(total_seq_len),
            embedding_dim,
            device=device,
            dtype=dtype,
        ).requires_grad_(True)
        stu_output = stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        stu_output_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            values=stu_output,
            offsets=[x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )

        # swapped forward().
        dense_x = torch.ops.fbgemm.jagged_to_padded_dense(
            x.detach(),
            [x_offsets],
            [max_seq_len],
        )
        swapped_dense_x = _inplace_swap(batch_size, dense_x, swap_from, swap_to)
        swapped_x = torch.ops.fbgemm.dense_to_jagged(
            swapped_dense_x,
            [x_offsets],
        )[0].requires_grad_(True)
        swapped_stu_output = stu(
            x=swapped_x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        swapped_stu_output_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            values=swapped_stu_output,
            offsets=[x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )

        # backward
        dout = torch.randn_like(stu_output_dense)
        stu_output_dense.backward(dout)
        dout = dout.detach().clone()
        swapped_stu_output_dense.backward(
            _inplace_swap(batch_size, dout, swap_from, swap_to)
        )

        swapped_swapped_stu_output_dense = _inplace_swap(
            batch_size, swapped_stu_output_dense, swap_from, swap_to
        )
        torch.testing.assert_close(stu_output_dense, swapped_swapped_stu_output_dense)

        # backward
        torch.testing.assert_close(
            torch.ops.fbgemm.jagged_to_padded_dense(
                swapped_x.grad,
                [x_offsets],
                [max_seq_len],
            ),
            _inplace_swap(
                batch_size,
                torch.ops.fbgemm.jagged_to_padded_dense(
                    x.grad,
                    [x_offsets],
                    [max_seq_len],
                ),
                swap_from,
                swap_to,
            ),
        )

    # pyre-ignore[56]
    @given(
        num_layers=st.sampled_from([1, 2, 4]),
        num_heads=st.sampled_from([1, 4]),
        max_uih_len=st.sampled_from([20, 128]),
        batch_size=st.sampled_from([4, 8]),
        embedding_dim=st.sampled_from([32]),
        attention_dim=st.sampled_from([16]),
        linear_hidden_dim=st.sampled_from([64]),
        contextual_seq_len=st.sampled_from([0, 10]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    @torch.inference_mode()
    def test_cached_forward(
        self,
        num_layers: int,
        num_heads: int,
        max_uih_len: int,
        batch_size: int,
        embedding_dim: int,
        attention_dim: int,
        linear_hidden_dim: int,
        contextual_seq_len: int,
    ) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")

        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        use_group_norm = False
        recompute_normed_x_in_backward = False
        recompute_uvqk_in_backward = False
        recompute_y_in_backward = False
        max_attn_len = None
        stu_layers: List[STU] = [
            STULayer(
                config=STULayerConfig(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    output_dropout_ratio=0.0,
                    causal=True,
                    target_aware=True,
                    max_attn_len=max_attn_len,
                    attn_alpha=None,
                    use_group_norm=use_group_norm,
                    recompute_normed_x=recompute_normed_x_in_backward,
                    recompute_uvqk=recompute_uvqk_in_backward,
                    recompute_y=recompute_y_in_backward,
                    sort_by_length=True,
                    contextual_seq_len=contextual_seq_len,
                ),
                is_inference=True,
            )
            for _ in range(num_layers)
        ]
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=True,
        ).to(device)
        stu.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)
        stu.eval()

        x_lengths = torch.randint(
            max_uih_len, max_uih_len + 1, (batch_size,), device=device
        )
        x_lengths = x_lengths + contextual_seq_len
        max_seq_len = max_uih_len + contextual_seq_len
        delta_size = 20
        max_targets = delta_size * 2
        num_targets = torch.randint(
            delta_size, max_targets + 1, size=(batch_size,), device=device
        )
        x_lengths = x_lengths + num_targets + contextual_seq_len
        max_seq_len = max_seq_len + max_targets + contextual_seq_len
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total_seq_len = int(x_offsets[-1].cpu().item())
        x = torch.randn(
            int(total_seq_len),
            embedding_dim,
            device=device,
        ).requires_grad_(True)

        # default forward().
        ref_y = stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        prime_lengths = x_lengths - delta_size
        prime_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(prime_lengths)
        _, ref_delta_y = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=ref_y,
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=HammerKernel.TRITON,
        )

        # cached forward().
        prime_x, delta_x = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=x,
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=HammerKernel.TRITON,
        )
        _ = stu(
            x=prime_x,
            x_lengths=prime_lengths,
            x_offsets=prime_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets - delta_size,
            max_kv_caching_len=max_seq_len - delta_size,
            kv_caching_lengths=x_lengths - delta_size,
        )
        delta_y = stu.cached_forward(
            delta_x=delta_x,
            num_targets=num_targets,
        )

        torch.testing.assert_close(ref_delta_y, delta_y)
