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
from generative_recommenders.modules.dynamic_stu import L2STU, SDSTU
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from hypothesis import given, settings, strategies as st, Verbosity


class DynamicStuTest(unittest.TestCase):
    # pyre-ignore
    @given(
        causal=st.sampled_from([True]),
        num_layers=st.sampled_from([2]),
        num_heads=st.sampled_from([2]),
        max_uih_len=st.sampled_from([300]),
        batch_size=st.sampled_from([8]),
        embedding_dim=st.sampled_from([16]),
        attention_dim=st.sampled_from([32]),
        linear_hidden_dim=st.sampled_from([64]),
        has_multiple_targets=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_l2_stu(
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
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        device = torch.device("cuda")
        l3_stu_layers: List[STU] = [
            STULayer(
                config=STULayerConfig(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    output_dropout_ratio=0.0,
                    causal=causal,
                    target_aware=has_multiple_targets,
                    contextual_seq_len=contextual_seq_len,
                ),
                is_inference=False,
            )
            for _ in range(num_layers)
        ]
        l3_stu: List[STU] = [
            L2STU(
                stu=STUStack(
                    stu_list=l3_stu_layers,
                    is_inference=False,
                ),
                max_l2_len=100,
                contextual_seq_len=contextual_seq_len,
                is_inference=False,
            )
        ]
        l2_stu_layers: List[STU] = [
            STULayer(
                config=STULayerConfig(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=linear_hidden_dim,
                    attention_dim=attention_dim,
                    output_dropout_ratio=0.0,
                    causal=causal,
                    target_aware=has_multiple_targets,
                    contextual_seq_len=contextual_seq_len,
                ),
                is_inference=False,
            )
            for _ in range(num_layers)
        ] + l3_stu
        l2_stu: List[STU] = [
            L2STU(
                stu=STUStack(
                    stu_list=l2_stu_layers,
                    is_inference=False,
                ),
                max_l2_len=200,
                contextual_seq_len=contextual_seq_len,
                is_inference=False,
            )
        ]
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
                    contextual_seq_len=contextual_seq_len,
                ),
                is_inference=False,
            )
            for _ in range(num_layers)
        ] + l2_stu
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=False,
        ).to(device)
        stu.recursive_setattr("_hammer_kernel", HammerKernel.TRITON)

        x_lengths = torch.randint(max_uih_len + 1, (batch_size,), device=device)
        x_lengths = x_lengths + contextual_seq_len
        max_seq_len = max_uih_len + contextual_seq_len
        max_targets = 20
        num_targets = torch.randint(1, max_targets, size=(batch_size,), device=device)
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
        stu_output = stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        dout = torch.randn_like(stu_output)
        stu_output.backward(dout)
        self.assertTrue(stu_output.shape == x.shape)

    # pyre-ignore
    @given(
        causal=st.sampled_from([True]),
        num_layers=st.sampled_from([2]),
        num_heads=st.sampled_from([2]),
        max_uih_len=st.sampled_from([300]),
        batch_size=st.sampled_from([8]),
        embedding_dim=st.sampled_from([16]),
        attention_dim=st.sampled_from([32]),
        linear_hidden_dim=st.sampled_from([64]),
        has_multiple_targets=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
        dropout_ratio=st.sampled_from([0.0, 0.3, 1.0]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_sd_stu(
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
        dropout_ratio: float,
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
        sd_stu = SDSTU(
            stu=copy.deepcopy(stu),
            dropout_ratio=dropout_ratio,
            is_inference=False,
        ).to(device)
        stu.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
        sd_stu.recursive_setattr("_hammer_kernel", HammerKernel.PYTORCH)
        x_lengths = torch.randint(max_uih_len + 1, (batch_size,), device=device)
        x_lengths = x_lengths + contextual_seq_len
        max_seq_len = max_uih_len + contextual_seq_len
        max_targets = 20
        num_targets = torch.randint(1, max_targets, size=(batch_size,), device=device)
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
        stu_output = stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        dout = torch.randn_like(stu_output)
        stu_output.backward(dout)
        assert x.grad is not None
        d_x, x.grad = x.grad.detach().clone(), None
        x = x.detach().clone().requires_grad_(True)
        sd_stu_output = sd_stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        dout = dout.detach().clone()
        sd_stu_output.backward(dout)
        d_sd_x = x.grad.detach().clone()

        self.assertTrue(sd_stu_output.shape == x.shape)
        if dropout_ratio == 0.0:
            torch.testing.assert_close(stu_output, sd_stu_output)
            torch.testing.assert_close(d_x, d_sd_x)
        if dropout_ratio == 1.0:
            torch.testing.assert_close(x, sd_stu_output)
