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

import os
import unittest
from unittest.mock import patch

import torch
from generative_recommenders.common import switch_to_contiguous_if_needed


class SwitchToContiguousIfNeededTest(unittest.TestCase):
    def test_empty_input_rejected_under_compile_by_default(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("torch.compiler.is_compiling", return_value=True),
        ):
            with self.assertRaises(RuntimeError):
                switch_to_contiguous_if_needed(torch.empty(0, 4))

    def test_empty_input_allowed_under_compile_when_opted_in(self) -> None:
        x = torch.empty(0, 4)
        with (
            patch.dict(os.environ, {"TRITON_JAGGED_ALLOW_EMPTY_UNDER_COMPILE": "1"}),
            patch("torch.compiler.is_compiling", return_value=True),
        ):
            self.assertIs(switch_to_contiguous_if_needed(x), x)

    def test_non_contiguous_input_is_made_contiguous(self) -> None:
        x = torch.arange(12.0).view(3, 4).t()
        out = switch_to_contiguous_if_needed(x)
        self.assertEqual(out.stride(-1), 1)
        torch.testing.assert_close(out, x)
