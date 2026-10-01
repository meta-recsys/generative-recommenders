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

import itertools
import unittest
from typing import Optional

import torch
from generative_recommenders.ops.pytorch.pt_position import _get_col_indices
from torch.fx.experimental.proxy_tensor import make_fx


def _reference_col_indices(
    max_seq_len: int,
    max_contextual_seq_len: int,
    max_pos_ind: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    """The previous implementation, which overwrote the contextual slice in place."""
    B = seq_lengths.size(0)
    col_indices = torch.arange(max_seq_len).expand(B, max_seq_len)
    if num_targets is not None:
        multiplier = 2 if interleave_targets else 1
        high_inds = seq_lengths - num_targets * multiplier
        col_indices = torch.clamp(col_indices, max=high_inds.view(-1, 1))
        col_indices = high_inds.view(-1, 1) - col_indices
    else:
        col_indices = seq_lengths.view(-1, 1) - col_indices
    col_indices = col_indices + max_contextual_seq_len
    col_indices = torch.clamp(col_indices, max=max_pos_ind - 1)
    if max_contextual_seq_len > 0:
        col_indices[:, :max_contextual_seq_len] = torch.arange(
            0, max_contextual_seq_len, dtype=col_indices.dtype
        ).view(1, -1)
    return col_indices


class GetColIndicesTest(unittest.TestCase):
    def test_matches_the_in_place_reference(self) -> None:
        seq_lengths = torch.tensor([12, 7, 1, 15])
        num_targets = torch.tensor([3, 2, 0, 4])
        for max_ctx, with_targets, interleave, max_pos_ind in itertools.product(
            (0, 3), (False, True), (False, True), (6, 64)
        ):
            kwargs = {
                "max_seq_len": 16,
                "max_contextual_seq_len": max_ctx,
                "max_pos_ind": max_pos_ind,
                "seq_lengths": seq_lengths,
                "num_targets": num_targets if with_targets else None,
                "interleave_targets": interleave,
            }
            with self.subTest(
                **{k: v for k, v in kwargs.items() if isinstance(v, (int, bool))}
            ):
                self.assertTrue(
                    torch.equal(
                        _get_col_indices(**kwargs), _reference_col_indices(**kwargs)
                    )
                )

    def test_traces_without_an_in_place_view_write(self) -> None:
        # Tracers that lower make_fx graphs without functionalizing them (e.g.
        # recir/torchax) drop a copy_ into a slice.
        def fn(seq_lengths: torch.Tensor, num_targets: torch.Tensor) -> torch.Tensor:
            return _get_col_indices(
                max_seq_len=16,
                max_contextual_seq_len=3,
                max_pos_ind=64,
                seq_lengths=seq_lengths,
                num_targets=num_targets,
                interleave_targets=False,
            )

        gm = make_fx(fn, tracing_mode="fake")(
            torch.tensor([12, 7, 1, 15]), torch.tensor([3, 2, 0, 4])
        )
        targets = {n.target for n in gm.graph.nodes if n.op == "call_function"}
        self.assertNotIn(torch.ops.aten.copy_.default, targets)
