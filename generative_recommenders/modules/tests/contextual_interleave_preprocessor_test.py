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

import torch
from generative_recommenders.common import (
    accelerator_supports_bf16,
    accelerator_unavailable,
    get_accelerator_device,
)
from generative_recommenders.modules.action_encoder import ActionEncoder
from generative_recommenders.modules.content_encoder import ContentEncoder
from generative_recommenders.modules.contextual_interleave_preprocessor import (
    ContextualInterleavePreprocessor,
)
from generative_recommenders.modules.contextualize_mlps import (
    ParameterizedContextualizedMLP,
    SimpleContextualizedMLP,
)
from hypothesis import given, settings, strategies as st, Verbosity


class ContextualInterleavePreprocessorTest(unittest.TestCase):
    # pyre-ignore
    @given(
        enable_interleaving=st.sampled_from([True, False]),
        enable_pmlp=st.sampled_from([True, False]),
        is_train=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.float32, torch.bfloat16]
            if accelerator_supports_bf16()
            else [torch.float32]
        ),
    )
    @unittest.skipIf(*accelerator_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=50, deadline=None)
    def test_forward(
        self,
        enable_interleaving: bool,
        enable_pmlp: bool,
        is_train: bool,
        dtype: torch.dtype,
    ) -> None:
        device = get_accelerator_device()

        input_embedding_dim = 64
        output_embedding_dim = 32
        action_embedding_dim = 16
        action_encoder_hidden_dim = 256
        content_encoder_hidden_dim = 128
        contextual_len = 3

        content_encoder = ContentEncoder(
            input_embedding_dim=input_embedding_dim,
            additional_content_features={
                "a0": input_embedding_dim,
                "a1": input_embedding_dim,
            },
            target_enrich_features={
                "t0": input_embedding_dim,
                "t1": input_embedding_dim,
            },
            is_inference=False,
        ).to(device)
        action_embedding_dim = 32
        action_weights = [1, 2, 4, 8, 16]
        watchtime_to_action_thresholds_and_weights = [
            (30, 32),
            (60, 64),
            (100, 128),
        ]
        action_encoder = ActionEncoder(
            watchtime_feature_name="watchtimes",
            action_feature_name="actions",
            action_weights=action_weights,
            watchtime_to_action_thresholds_and_weights=watchtime_to_action_thresholds_and_weights,
            action_embedding_dim=action_embedding_dim,
            is_inference=False,
        ).to(device)

        preprocessor = ContextualInterleavePreprocessor(
            input_embedding_dim=input_embedding_dim,
            output_embedding_dim=output_embedding_dim,
            contextual_feature_to_max_length={"c_0": 1, "c_1": 2},
            contextual_feature_to_min_uih_length={"c_1": 4},
            pmlp_contextual_dropout_ratio=0.2,
            content_encoder=content_encoder,
            content_contextualize_mlp_fn=lambda in_dim,
            out_dim,
            contextual_dim,
            is_inference: ParameterizedContextualizedMLP(
                contextual_embedding_dim=contextual_dim,
                sequential_input_dim=in_dim,
                sequential_output_dim=out_dim,
                hidden_dim=content_encoder_hidden_dim,
                is_inference=is_inference,
            )
            if enable_pmlp
            else SimpleContextualizedMLP(
                sequential_input_dim=in_dim,
                sequential_output_dim=out_dim,
                hidden_dim=content_encoder_hidden_dim,
                is_inference=is_inference,
            ),
            action_encoder=action_encoder,
            action_contextualize_mlp_fn=lambda in_dim,
            out_dim,
            contextual_dim,
            is_inference: ParameterizedContextualizedMLP(
                contextual_embedding_dim=contextual_dim,
                sequential_input_dim=in_dim,
                sequential_output_dim=out_dim,
                hidden_dim=action_encoder_hidden_dim,
                is_inference=is_inference,
            )
            if enable_pmlp
            else SimpleContextualizedMLP(
                sequential_input_dim=in_dim,
                sequential_output_dim=out_dim,
                hidden_dim=action_encoder_hidden_dim,
                is_inference=is_inference,
            ),
            enable_interleaving=enable_interleaving,
            is_inference=False,
        ).to(device)
        preprocessor.set_training_dtype(dtype)
        if not is_train:
            preprocessor.eval()

        # inputs
        seq_lengths = [6, 3]
        num_targets = [2, 1]
        seq_embeddings = torch.rand(
            (sum(seq_lengths), input_embedding_dim),
            device=device,
            dtype=dtype,
        )
        seq_timestamps = torch.tensor(
            [1, 2, 3, 4, 5, 6, 10, 20, 30],
            device=device,
        )
        watchtimes = [40, 20, 110, 31, 26, 55]
        actions = [1, 3, 26, 30, 6, 4]
        (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            _,
        ) = preprocessor(
            max_uih_len=4,
            max_targets=2,
            total_uih_len=sum(seq_lengths) - sum(num_targets),
            total_targets=sum(num_targets),
            seq_lengths=torch.tensor(seq_lengths, device=device),
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            seq_payloads={
                # contextual
                "c_0": torch.rand((2, input_embedding_dim), device=device, dtype=dtype),
                "c_0_offsets": torch.tensor([0, 1, 1], device=device),
                "c_1": torch.rand((4, input_embedding_dim), device=device, dtype=dtype),
                "c_1_offsets": torch.tensor([0, 2, 3], device=device),
                # action
                "watchtimes": torch.tensor(watchtimes, device=device),
                "actions": torch.tensor(actions, device=device),
                # content
                "a0": torch.rand_like(seq_embeddings).requires_grad_(True),
                "a1": torch.rand_like(seq_embeddings).requires_grad_(True),
                "t0": torch.rand(
                    sum(num_targets), input_embedding_dim, device=device, dtype=dtype
                ).requires_grad_(True),
                "t1": torch.rand(
                    sum(num_targets), input_embedding_dim, device=device, dtype=dtype
                ).requires_grad_(True),
            },
            num_targets=torch.tensor(num_targets, device=device),
        )
        if enable_interleaving:
            if is_train:
                expected_output_seq_lengths = [
                    2 * s + contextual_len for s in seq_lengths
                ]
                expected_max_seq_len = max(expected_output_seq_lengths)
                expected_output_num_targets = [2 * s for s in num_targets]
                expected_seq_embedding_size = (
                    sum(expected_output_seq_lengths),
                    output_embedding_dim,
                )
                expected_seq_timestamps_size = (sum(expected_output_seq_lengths),)
                expected_output_seq_timestamps = [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    0,
                    0,
                    0,
                    10,
                    10,
                    20,
                    20,
                    30,
                    30,
                ]
            else:
                expected_output_seq_lengths = [
                    2 * s - n + contextual_len for s, n in zip(seq_lengths, num_targets)
                ]
                expected_max_seq_len = max(expected_output_seq_lengths)
                expected_output_num_targets = num_targets
                expected_seq_embedding_size = (
                    sum(expected_output_seq_lengths),
                    output_embedding_dim,
                )
                expected_seq_timestamps_size = (sum(expected_output_seq_lengths),)
                expected_output_seq_timestamps = [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    6,
                    0,
                    0,
                    0,
                    10,
                    10,
                    20,
                    20,
                    30,
                ]
        else:
            expected_output_seq_lengths = [s + contextual_len for s in seq_lengths]
            expected_max_seq_len = max(expected_output_seq_lengths)
            expected_output_num_targets = num_targets
            expected_seq_embedding_size = (
                sum(expected_output_seq_lengths),
                output_embedding_dim,
            )
            expected_seq_timestamps_size = (sum(expected_output_seq_lengths),)
            expected_output_seq_timestamps = [
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                0,
                0,
                10,
                20,
                30,
            ]

        self.assertEqual(output_max_seq_len, expected_max_seq_len)
        self.assertEqual(output_seq_lengths.tolist(), expected_output_seq_lengths)
        torch.testing.assert_allclose(
            torch.ops.fbgemm.asynchronous_complete_cumsum(output_seq_lengths),
            output_seq_offsets,
        )
        self.assertEqual(output_num_targets.tolist(), expected_output_num_targets)
        self.assertEqual(
            output_seq_embeddings.size(),
            expected_seq_embedding_size,
        )
        self.assertEqual(
            output_seq_timestamps.size(),
            expected_seq_timestamps_size,
        )
        self.assertEqual(
            output_seq_timestamps.tolist(),
            expected_output_seq_timestamps,
        )

        # test combine embeddings
        batch_size = 10
        max_uih_len = 100
        max_targets = 20
        max_seq_len = max_uih_len + max_targets
        seq_lengths = torch.randint(0, max_uih_len, (batch_size,), device=device)
        total_uih_len = int(seq_lengths.sum().item())
        num_targets = torch.randint(1, max_targets, (batch_size,), device=device)
        total_targets = int(num_targets.sum().item())
        seq_lengths = seq_lengths + num_targets
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=device
        )
        seq_offsets[1:] = torch.cumsum(seq_lengths, dim=0)
        total_seq_len = int(torch.sum(seq_lengths).item())
        seq_timestamps = torch.randint(0, 1000000, (total_seq_len,), device=device)
        content_embeddings = torch.rand(
            (total_seq_len, output_embedding_dim),
            device=device,
        ).requires_grad_(True)
        action_embeddings = torch.rand(
            (total_seq_len, output_embedding_dim),
            device=device,
        ).requires_grad_(True)
        contextual_embeddings = torch.rand(
            (total_seq_len, 3 * output_embedding_dim), device=device
        ).requires_grad_(True)
        (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        ) = preprocessor.combine_embeddings(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            content_embeddings=content_embeddings,
            action_embeddings=action_embeddings,
            contextual_embeddings=contextual_embeddings,
            num_targets=num_targets,
        )
        seq_embeddings = action_embeddings + content_embeddings
        if enable_interleaving:
            if is_train:
                self.assertEqual(
                    output_max_seq_len,
                    max_seq_len * 2 + contextual_len,
                )
                self.assertEqual(
                    output_total_uih_len,
                    total_uih_len * 2 + contextual_len * batch_size,
                )
                self.assertEqual(
                    output_total_targets,
                    total_targets * 2,
                )
                torch.testing.assert_allclose(
                    output_seq_lengths, seq_lengths * 2 + contextual_len
                )
                torch.testing.assert_allclose(output_num_targets, num_targets * 2)
            else:
                self.assertEqual(
                    output_max_seq_len,
                    max_uih_len * 2 + max_targets + contextual_len,
                )
                self.assertEqual(
                    output_total_uih_len,
                    total_uih_len * 2 + contextual_len * batch_size,
                )
                self.assertEqual(
                    output_total_targets,
                    total_targets,
                )
                torch.testing.assert_allclose(
                    output_seq_lengths, seq_lengths * 2 - num_targets + contextual_len
                )
                torch.testing.assert_allclose(output_num_targets, num_targets)
        else:
            self.assertEqual(
                output_max_seq_len,
                max_seq_len + contextual_len,
            )
            self.assertEqual(
                output_total_uih_len,
                total_uih_len + contextual_len * batch_size,
            )
            self.assertEqual(
                output_total_targets,
                total_targets,
            )
            torch.testing.assert_allclose(
                output_seq_lengths, seq_lengths + contextual_len
            )
            torch.testing.assert_allclose(output_num_targets, num_targets)
        for b in range(batch_size):
            input_start = int(seq_offsets[b].item())
            input_end = int(seq_offsets[b + 1].item())
            output_start = int(output_seq_offsets[b].item())
            output_end = int(output_seq_offsets[b + 1].item())
            input_targets = int(num_targets[b].item())
            output_targets = int(output_num_targets[b].item())
            torch.testing.assert_allclose(
                output_seq_timestamps[output_start : output_start + contextual_len],
                torch.zeros(3, device=device),
            )
            torch.testing.assert_allclose(
                output_seq_embeddings[
                    output_start : output_start + contextual_len
                ].view(-1),
                contextual_embeddings[b],
            )
            if enable_interleaving:
                torch.testing.assert_allclose(
                    output_seq_timestamps[
                        output_start + contextual_len : output_end - output_targets
                    ].view(-1, 2)[:, 0],
                    seq_timestamps[input_start : input_end - input_targets],
                )
                torch.testing.assert_allclose(
                    output_seq_timestamps[
                        output_start + contextual_len : output_end - output_targets
                    ].view(-1, 2)[:, 1],
                    seq_timestamps[input_start : input_end - input_targets],
                )
                torch.testing.assert_allclose(
                    output_seq_embeddings[
                        output_start + contextual_len : output_end - output_targets
                    ].view(-1, 2, output_embedding_dim)[:, 0, :],
                    content_embeddings[input_start : input_end - input_targets],
                )
                torch.testing.assert_allclose(
                    output_seq_embeddings[
                        output_start + contextual_len : output_end - output_targets
                    ].view(-1, 2, output_embedding_dim)[:, 1, :],
                    action_embeddings[input_start : input_end - input_targets],
                )
                if is_train:
                    torch.testing.assert_allclose(
                        output_seq_timestamps[
                            output_end - output_targets : output_end
                        ].view(-1, 2)[:, 0],
                        seq_timestamps[input_end - input_targets : input_end],
                    )
                    torch.testing.assert_allclose(
                        output_seq_timestamps[
                            output_end - output_targets : output_end
                        ].view(-1, 2)[:, 1],
                        seq_timestamps[input_end - input_targets : input_end],
                    )
                    torch.testing.assert_allclose(
                        output_seq_embeddings[
                            output_end - output_targets : output_end
                        ].view(-1, 2, output_embedding_dim)[:, 0, :],
                        content_embeddings[input_end - input_targets : input_end],
                    )
                    torch.testing.assert_allclose(
                        output_seq_embeddings[
                            output_end - output_targets : output_end
                        ].view(-1, 2, output_embedding_dim)[:, 1, :],
                        action_embeddings[input_end - input_targets : input_end],
                    )
                else:
                    torch.testing.assert_allclose(
                        output_seq_timestamps[output_end - output_targets : output_end],
                        seq_timestamps[input_end - input_targets : input_end],
                    )
                    torch.testing.assert_allclose(
                        output_seq_embeddings[output_end - output_targets : output_end],
                        content_embeddings[input_end - input_targets : input_end],
                    )
            else:
                torch.testing.assert_allclose(
                    output_seq_timestamps[output_start + contextual_len : output_end],
                    seq_timestamps[input_start:input_end],
                )
                torch.testing.assert_allclose(
                    output_seq_embeddings[output_start + contextual_len : output_end],
                    seq_embeddings[input_start:input_end],
                )
