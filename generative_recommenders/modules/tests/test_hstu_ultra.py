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

from __future__ import annotations

import copy
import unittest

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel
from generative_recommenders.modules.hstu_ultra import (
    hstu_ultra_stack_configs,
    HSTUUltraStack,
    HSTUUltraStackConfig,
)
from generative_recommenders.ops.hstu_ultra import HSTUUltraAttentionConfig


def _small_stack_config(full_attn_size: int) -> HSTUUltraStackConfig:
    return HSTUUltraStackConfig(
        name="test_hstu_ultra",
        description="Small HSTU Ultra stack for tests",
        attention=HSTUUltraAttentionConfig(
            name="test_hstu_ultra_attention",
            description="Small semi-local attention configuration",
            heads=1,
            attention_dim=8,
            value_dim=8,
            max_uih_length=6,
            max_targets=0,
            max_attn_len=1,
            full_attn_size=full_attn_size,
            default_sequence_lengths=(6,),
        ),
        embedding_dim=8,
        num_layers=1,
    )


class HSTUUltraStackTest(unittest.TestCase):
    def test_model_derived_stack_configs(self) -> None:
        configs = hstu_ultra_stack_configs(fp8_addmm_fwd=True)

        semi_local = configs["hstu_ultra_semi_local"]
        self.assertEqual((semi_local.embedding_dim, semi_local.num_layers), (512, 8))
        self.assertEqual(
            (
                semi_local.attention.max_attn_len,
                semi_local.attention.full_attn_size,
            ),
            (256, 256),
        )
        self.assertTrue(semi_local.fp8_addmm_fwd)

        ultra = configs["hstu_ultra"]
        self.assertEqual((ultra.embedding_dim, ultra.num_layers), (512, 12))
        self.assertEqual(ultra.attention.max_targets, 512)
        self.assertEqual(
            (ultra.attention.max_attn_len, ultra.attention.full_attn_size),
            (0, 0),
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_full_attention_tail_changes_layer_output(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        full_tail = HSTUUltraStack(
            config=_small_stack_config(full_attn_size=2),
            kernel=HammerKernel.PYTORCH,
        ).to(device)
        local_only = HSTUUltraStack(
            config=_small_stack_config(full_attn_size=0),
            kernel=HammerKernel.PYTORCH,
        ).to(device)
        local_only.load_state_dict(full_tail.state_dict())
        full_tail.eval()
        local_only.eval()

        x = torch.arange(48, dtype=torch.float32, device=device).view(6, 8) / 10.0
        lengths = torch.tensor([6], dtype=torch.int32, device=device)
        offsets = torch.tensor([0, 6], dtype=torch.int32, device=device)
        num_targets = torch.zeros(1, dtype=torch.int32, device=device)

        full_tail_output = full_tail(x, lengths, offsets, 6, num_targets)
        local_only_output = local_only(x, lengths, offsets, 6, num_targets)

        self.assertFalse(torch.allclose(full_tail_output, local_only_output))

    @unittest.skipIf(*gpu_unavailable)
    def test_triton_matches_pytorch_for_complete_layer(self) -> None:
        torch.manual_seed(1)
        device = torch.device("cuda")
        pytorch_stack = HSTUUltraStack(
            config=_small_stack_config(full_attn_size=2),
            kernel=HammerKernel.PYTORCH,
        ).to(device)
        triton_stack = copy.deepcopy(pytorch_stack)
        triton_stack.set_hammer_kernel(HammerKernel.TRITON)
        pytorch_stack.eval()
        triton_stack.eval()

        lengths = torch.tensor([6, 4], dtype=torch.int32, device=device)
        offsets = torch.tensor([0, 6, 10], dtype=torch.int32, device=device)
        num_targets = torch.zeros(2, dtype=torch.int32, device=device)
        x = torch.randn(10, 8, device=device, dtype=torch.float32).requires_grad_()
        triton_x = x.detach().clone().requires_grad_()

        expected = pytorch_stack(x, lengths, offsets, 6, num_targets)
        actual = triton_stack(triton_x, lengths, offsets, 6, num_targets)
        torch.testing.assert_close(actual, expected, atol=5e-3, rtol=1e-2)

        output_gradient = torch.randn_like(expected)
        expected.backward(output_gradient)
        actual.backward(output_gradient)
        torch.testing.assert_close(triton_x.grad, x.grad, atol=5e-3, rtol=1e-2)

    @unittest.skipIf(*gpu_unavailable)
    def test_full_attention_tail_with_fp8_projections(self) -> None:
        torch.manual_seed(2)
        device = torch.device("cuda")
        attention = HSTUUltraAttentionConfig(
            name="test_hstu_ultra_fp8_attention",
            description="FP8-compatible semi-local attention configuration",
            heads=2,
            attention_dim=16,
            value_dim=16,
            max_uih_length=8,
            max_targets=0,
            max_attn_len=2,
            full_attn_size=2,
            default_sequence_lengths=(8,),
        )
        reference = HSTUUltraStack(
            config=HSTUUltraStackConfig(
                name="test_hstu_ultra_bf16",
                description="BF16 reference stack",
                attention=attention,
                embedding_dim=64,
                num_layers=1,
            ),
            kernel=HammerKernel.TRITON,
        ).to(device=device, dtype=torch.bfloat16)
        fp8 = HSTUUltraStack(
            config=HSTUUltraStackConfig(
                name="test_hstu_ultra_fp8",
                description="FP8 projection stack",
                attention=attention,
                embedding_dim=64,
                num_layers=1,
                fp8_addmm_fwd=True,
            ),
            kernel=HammerKernel.TRITON,
        ).to(device=device, dtype=torch.bfloat16)
        fp8.load_state_dict(reference.state_dict())
        reference.eval()
        fp8.eval()
        fp8.prepare_fp8_weights()
        for layer in fp8._stu_layers:
            self.assertIsNotNone(layer._uvqk_weight_fp8)
            self.assertIsNotNone(layer._output_weight_fp8)

        lengths = torch.full((4,), 8, dtype=torch.int32, device=device)
        offsets = torch.arange(5, dtype=torch.int32, device=device) * 8
        num_targets = torch.zeros(4, dtype=torch.int32, device=device)
        x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)

        expected = reference(x, lengths, offsets, 8, num_targets)
        actual = fp8(x, lengths, offsets, 8, num_targets)

        torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.15)

    def test_cached_forward_rejects_full_attention_tail(self) -> None:
        stack = HSTUUltraStack(
            config=_small_stack_config(full_attn_size=2),
            kernel=HammerKernel.PYTORCH,
        )

        with self.assertRaisesRegex(NotImplementedError, "cached HSTU attention"):
            stack.cached_forward(
                delta_x=torch.zeros(1, 8),
                num_targets=torch.zeros(1, dtype=torch.int32),
            )
