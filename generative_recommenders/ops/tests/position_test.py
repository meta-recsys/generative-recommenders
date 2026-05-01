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
    generate_sparse_seq_len,
    gpu_unavailable,
    HammerKernel,
    set_dev_mode,
)
from hypothesis import given, settings, strategies as st, Verbosity


class PositionEmbeddingsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        alpha=st.sampled_from([0.5]),
        max_uih_len=st.integers(50, 500),
        max_contextual_seq_len=st.sampled_from([10]),
        interleave_targets=st.sampled_from([True, False]),
        batch_size=st.integers(16, 32),
        D=st.integers(20, 200),
        max_targets=st.sampled_from([10, 20]),
        time_bucket_fn=st.sampled_from(["log"]),
        dtype=st.sampled_from([torch.float32, torch.bfloat16, torch.float16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_add_timestamp_positional_embeddings_triton(self, *args, **kwargs) -> None:
        self._test_add_timestamp_positional_embeddings(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        alpha=st.sampled_from([0.5]),
        max_uih_len=st.sampled_from([32768]),
        max_contextual_seq_len=st.sampled_from([10]),
        interleave_targets=st.sampled_from([False]),
        batch_size=st.sampled_from([130]),
        D=st.sampled_from([512]),
        max_targets=st.sampled_from([10]),
        time_bucket_fn=st.sampled_from(["log"]),
        dtype=st.sampled_from([torch.bfloat16]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=1,
        deadline=None,
    )
    def test_add_timestamp_positional_embeddings_triton_large_tensor(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-ignore[2]
        **kwargs,
    ) -> None:
        self._test_add_timestamp_positional_embeddings(
            *args,
            **kwargs,
            test_backward=False,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
            sparsity=1.0,
        )

    def _test_add_timestamp_positional_embeddings(
        self,
        alpha: float,
        max_uih_len: int,
        max_contextual_seq_len: int,
        interleave_targets: bool,
        batch_size: int,
        D: int,
        max_targets: int,
        time_bucket_fn: str,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        test_backward: bool,
        sparsity: float = -1,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.position import (
            add_timestamp_positional_embeddings,
        )

        num_targets = torch.randint(
            max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        if sparsity > 0.0:
            lengths = generate_sparse_seq_len(
                size=batch_size,
                max_seq_len=max_uih_len,
                sparsity=sparsity,
                device=torch.device("cuda"),
            ).to(torch.int64)
        else:
            lengths = torch.randint(
                max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
            )
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        max_seq_len = max_uih_len + max_targets

        position_embeddings_weight = (
            torch.empty(
                (max_seq_len, D), dtype=torch.float32, device=torch.device("cuda")
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        num_time_buckets = 1000
        timestamp_embeddings_weight = (
            torch.empty(
                (num_time_buckets, D), dtype=torch.float32, device=torch.device("cuda")
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        seq_embeddings = (
            torch.empty(
                (int(seq_offsets[-1].item()), D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        timestamp_deltas: torch.Tensor = torch.randint(
            86400,
            size=(batch_size, max_seq_len),
            device="cuda",
        )
        timestamps = timestamp_deltas.cumsum(dim=1)
        mask = torch.arange(max_seq_len, device=timestamps.device) < lengths.unsqueeze(
            1
        )
        timestamps = timestamps[mask.view(batch_size, -1)]

        ref_out = add_timestamp_positional_embeddings(
            alpha=alpha,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            position_embeddings_weight=position_embeddings_weight,
            timestamp_embeddings_weight=timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=lengths,
            seq_embeddings=seq_embeddings,
            timestamps=timestamps,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)
        # pyre-ignore
        ref_d_seq_embeddings, seq_embeddings.grad = seq_embeddings.grad.clone(), None
        ref_d_position_embeddings_weight, position_embeddings_weight.grad = (
            # pyrefly: ignore [missing-attribute]
            position_embeddings_weight.grad.clone(),
            None,
        )
        ref_d_timestamp_embeddings_weight, timestamp_embeddings_weight.grad = (
            # pyrefly: ignore [missing-attribute]
            timestamp_embeddings_weight.grad.clone(),
            None,
        )

        real_out = add_timestamp_positional_embeddings(
            alpha=alpha,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            position_embeddings_weight=position_embeddings_weight,
            timestamp_embeddings_weight=timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=lengths,
            seq_embeddings=seq_embeddings,
            timestamps=timestamps,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
            kernel=real_kernel,
        )

        torch.testing.assert_close(ref_out, real_out)
        if test_backward:
            real_out.backward(dout)
            # pyrefly: ignore [missing-attribute]
            real_d_seq_embeddings = seq_embeddings.grad.clone()
            # pyrefly: ignore [missing-attribute]
            real_d_position_embeddings_weight = position_embeddings_weight.grad.clone()
            real_d_timestamp_embeddings_weight = (
                # pyrefly: ignore [missing-attribute]
                timestamp_embeddings_weight.grad.clone()
            )
            torch.testing.assert_close(ref_d_seq_embeddings, real_d_seq_embeddings)
            torch.testing.assert_close(
                ref_d_position_embeddings_weight,
                real_d_position_embeddings_weight,
                atol=5e-2 if dtype != torch.float32 else None,
                rtol=2e-2 if dtype != torch.float32 else None,
            )
            torch.testing.assert_close(
                ref_d_timestamp_embeddings_weight,
                real_d_timestamp_embeddings_weight,
                atol=5e-2 if dtype != torch.float32 else None,
                rtol=2e-2 if dtype != torch.float32 else None,
            )
