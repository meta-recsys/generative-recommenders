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

import unittest

import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.benchmarks.hstu_ultra_bench import (
    build_hstu_ultra_inputs,
)
from generative_recommenders.ops.hstu_ultra import (
    get_hstu_ultra_valid_attn_mask,
    hstu_ultra_attention_configs,
    hstu_ultra_mha,
    HSTUUltraAttentionConfig,
    pytorch_hstu_ultra_mha,
)
from generative_recommenders.ops.pytorch.pt_hstu_attention import pytorch_hstu_mha
from parameterized import parameterized


def _test_config(
    max_targets: int,
    max_attn_len: int,
    full_attn_size: int,
) -> HSTUUltraAttentionConfig:
    return HSTUUltraAttentionConfig(
        name="test",
        description="test configuration",
        heads=2,
        attention_dim=4,
        value_dim=6,
        max_uih_length=8,
        max_targets=max_targets,
        max_attn_len=max_attn_len,
        full_attn_size=full_attn_size,
        default_sequence_lengths=(5,),
    )


class HSTUUltraTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("semi_local", 0, 2, 2),
            ("post_cross", 1, 0, 0),
        ]
    )
    def test_matches_pytorch_hstu_attention(
        self,
        _name: str,
        max_targets: int,
        max_attn_len: int,
        full_attn_size: int,
    ) -> None:
        torch.manual_seed(0)
        config = _test_config(max_targets, max_attn_len, full_attn_size)
        seq_offsets = torch.tensor([0, 5, 9], dtype=torch.int32)
        num_targets = (
            torch.tensor([1, 1], dtype=torch.int32) if max_targets > 0 else None
        )
        q = torch.randn(9, 2, 4)
        k = torch.randn(9, 2, 4)
        v = torch.randn(9, 2, 6)

        actual = pytorch_hstu_ultra_mha(
            config=config,
            max_seq_len=5,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
        )
        expected = pytorch_hstu_mha(
            max_seq_len=5,
            alpha=0.25,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            training=False,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            min_full_attn_seq_len=full_attn_size,
        )

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    @parameterized.expand(
        [
            ("semi_local", 0, 3, 3),
            ("post_cross", 2, 0, 0),
        ]
    )
    def test_triton_matches_pytorch(
        self,
        _name: str,
        max_targets: int,
        max_attn_len: int,
        full_attn_size: int,
    ) -> None:
        torch.manual_seed(0)
        config = HSTUUltraAttentionConfig(
            name="test",
            description="test configuration",
            heads=2,
            attention_dim=16,
            value_dim=16,
            max_uih_length=9,
            max_targets=max_targets,
            max_attn_len=max_attn_len,
            full_attn_size=full_attn_size,
            default_sequence_lengths=(9,),
        )
        device = torch.device("cuda")
        seq_offsets = torch.tensor([0, 9, 16], dtype=torch.int32, device=device)
        num_targets = None
        if max_targets > 0:
            num_targets = torch.tensor([2, 1], dtype=torch.int32, device=device)
        q = torch.randn(16, 2, 16, dtype=torch.bfloat16, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        reference_inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
        triton_inputs = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
        reference = hstu_ultra_mha(
            config=config,
            max_seq_len=9,
            q=reference_inputs[0],
            k=reference_inputs[1],
            v=reference_inputs[2],
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            kernel=HammerKernel.PYTORCH,
        )
        actual = hstu_ultra_mha(
            config=config,
            max_seq_len=9,
            q=triton_inputs[0],
            k=triton_inputs[1],
            v=triton_inputs[2],
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            kernel=HammerKernel.TRITON,
        )
        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)

        output_gradient = torch.randn_like(reference)
        reference.backward(output_gradient)
        actual.backward(output_gradient)
        for actual_input, reference_input in zip(triton_inputs, reference_inputs):
            torch.testing.assert_close(
                actual_input.grad,
                reference_input.grad,
                atol=0.03,
                rtol=0.03,
            )

    def test_builds_regular_jagged_inputs(self) -> None:
        config = _test_config(max_targets=1, max_attn_len=0, full_attn_size=0)

        inputs = build_hstu_ultra_inputs(
            config=config,
            batch_size=2,
            sequence_length=5,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertEqual(inputs.q.shape, (10, 2, 4))
        self.assertEqual(inputs.k.shape, (10, 2, 4))
        self.assertEqual(inputs.v.shape, (10, 2, 6))
        torch.testing.assert_close(
            inputs.seq_offsets, torch.tensor([0, 5, 10], dtype=torch.int32)
        )
        self.assertIsNotNone(inputs.num_targets)
        assert inputs.num_targets is not None
        torch.testing.assert_close(
            inputs.num_targets, torch.tensor([1, 1], dtype=torch.int32)
        )
        self.assertEqual(inputs.tokens, 10)

    def test_requires_target_counts_for_post_cross(self) -> None:
        config = _test_config(max_targets=1, max_attn_len=0, full_attn_size=0)
        q = torch.randn(5, 2, 4)
        v = torch.randn(5, 2, 6)

        with self.assertRaisesRegex(ValueError, "requires one target count"):
            pytorch_hstu_ultra_mha(
                config=config,
                max_seq_len=5,
                q=q,
                k=q,
                v=v,
                seq_offsets=torch.tensor([0, 5], dtype=torch.int32),
            )

    def test_model_derived_attention_settings(self) -> None:
        configs = hstu_ultra_attention_configs()

        l1 = configs["hstu_ultra_l1"]
        self.assertEqual((l1.heads, l1.attention_dim), (4, 128))
        self.assertEqual((l1.max_attn_len, l1.full_attn_size), (256, 256))

        post_cross = configs["hstu_ultra_post_cross"]
        self.assertEqual(post_cross.max_targets, 512)
        self.assertEqual(post_cross.max_sequence_length, 1536)
        self.assertEqual((post_cross.max_attn_len, post_cross.full_attn_size), (0, 0))

    def test_semi_local_attention_mask(self) -> None:
        actual = get_hstu_ultra_valid_attn_mask(
            device=torch.device("cpu"),
            max_seq_len=6,
            seq_lengths=torch.tensor([6, 4]),
            max_attn_len=2,
            full_attn_size=2,
        )
        expected = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(actual, expected)
