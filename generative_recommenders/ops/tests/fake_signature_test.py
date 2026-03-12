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

"""
Tests to ensure fake and real implementations of triton functions
have the same function signatures. This is critical for PT2 compile compatibility.
"""

import inspect
import unittest
from typing import Any, Callable, List


def get_custom_op_params(func: Callable[..., object]) -> List[str]:
    """
    Get parameter names from a function, handling custom_op decorated functions.

    For maybe_register_custom_op decorated functions, inspect.signature may return
    *args, **kwargs instead of the actual parameters. In this case, we need to
    access the underlying schema to get the real parameter names.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    if params == ["args", "kwargs"]:
        func_any: Any = func
        if hasattr(func_any, "_opoverload"):
            schema = func_any._opoverload._schema
            return [arg.name for arg in schema.arguments]

    return params


class FakeSignatureTest(unittest.TestCase):
    """Test to ensure fake and real implementations have the same function signatures."""

    def test_triton_addmm_fwd_and_fake_have_same_signature(self) -> None:
        """Verify triton_addmm_fwd and triton_addmm_fwd_fake have the same arguments."""
        from generative_recommenders.ops.triton.triton_addmm import (
            triton_addmm_fwd,
            triton_addmm_fwd_fake,
        )

        real_params = get_custom_op_params(triton_addmm_fwd)
        fake_params = get_custom_op_params(triton_addmm_fwd_fake)

        self.assertEqual(
            real_params,
            fake_params,
            f"triton_addmm_fwd and triton_addmm_fwd_fake have different arguments.\n"
            f"Real: {real_params}\n"
            f"Fake: {fake_params}",
        )

    def test_maybe_triton_addmm_fwd_and_fake_have_same_signature(self) -> None:
        """Verify maybe_triton_addmm_fwd and maybe_triton_addmm_fwd_fake have the same arguments."""
        from generative_recommenders.ops.triton.triton_addmm import (
            maybe_triton_addmm_fwd,
            maybe_triton_addmm_fwd_fake,
        )

        real_params = get_custom_op_params(maybe_triton_addmm_fwd)
        fake_params = get_custom_op_params(maybe_triton_addmm_fwd_fake)

        self.assertEqual(
            real_params,
            fake_params,
            f"maybe_triton_addmm_fwd and maybe_triton_addmm_fwd_fake have different arguments.\n"
            f"Real: {real_params}\n"
            f"Fake: {fake_params}",
        )

    def test_triton_hstu_attention_fwd_and_fake_have_same_signature(self) -> None:
        """Verify triton_hstu_attention_fwd and _triton_hstu_attention_fwd_fake have the same arguments."""
        from generative_recommenders.ops.triton.triton_hstu_attention import (
            _triton_hstu_attention_fwd_fake,
            triton_hstu_attention_fwd,
        )

        real_params = get_custom_op_params(triton_hstu_attention_fwd)
        fake_params = get_custom_op_params(_triton_hstu_attention_fwd_fake)

        self.assertEqual(
            real_params,
            fake_params,
            f"triton_hstu_attention_fwd and _triton_hstu_attention_fwd_fake have different arguments.\n"
            f"Real: {real_params}\n"
            f"Fake: {fake_params}",
        )

    def test_triton_hstu_attention_bwd_and_fake_have_same_signature(self) -> None:
        """Verify triton_hstu_attention_bwd and _triton_hstu_attention_bwd_fake have the same arguments."""
        from generative_recommenders.ops.triton.triton_hstu_attention import (
            _triton_hstu_attention_bwd_fake,
            triton_hstu_attention_bwd,
        )

        real_params = get_custom_op_params(triton_hstu_attention_bwd)
        fake_params = get_custom_op_params(_triton_hstu_attention_bwd_fake)

        self.assertEqual(
            real_params,
            fake_params,
            f"triton_hstu_attention_bwd and _triton_hstu_attention_bwd_fake have different arguments.\n"
            f"Real: {real_params}\n"
            f"Fake: {fake_params}",
        )

    def test_triton_layer_norm_mul_dropout_fwd_impl_and_fake_have_same_signature(
        self,
    ) -> None:
        """Verify _triton_layer_norm_mul_dropout_fwd_impl and its fake have the same arguments."""
        from generative_recommenders.ops.triton.triton_hstu_linear import (
            _triton_layer_norm_mul_dropout_fwd_impl,
            _triton_layer_norm_mul_dropout_fwd_impl_fake,
        )

        real_params = get_custom_op_params(_triton_layer_norm_mul_dropout_fwd_impl)
        fake_params = get_custom_op_params(_triton_layer_norm_mul_dropout_fwd_impl_fake)

        self.assertEqual(
            real_params,
            fake_params,
            f"_triton_layer_norm_mul_dropout_fwd_impl and _triton_layer_norm_mul_dropout_fwd_impl_fake have different arguments.\n"
            f"Real: {real_params}\n"
            f"Fake: {fake_params}",
        )

    def test_triton_layer_norm_mul_dropout_bwd_impl_and_fake_have_same_signature(
        self,
    ) -> None:
        """Verify _triton_layer_norm_mul_dropout_bwd_impl and its fake have the same arguments."""
        from generative_recommenders.ops.triton.triton_hstu_linear import (
            _triton_layer_norm_mul_dropout_bwd_impl,
            _triton_layer_norm_mul_dropout_bwd_impl_fake,
        )

        real_params = get_custom_op_params(_triton_layer_norm_mul_dropout_bwd_impl)
        fake_params = get_custom_op_params(_triton_layer_norm_mul_dropout_bwd_impl_fake)

        self.assertEqual(
            real_params,
            fake_params,
            f"_triton_layer_norm_mul_dropout_bwd_impl and _triton_layer_norm_mul_dropout_bwd_impl_fake have different arguments.\n"
            f"Real: {real_params}\n"
            f"Fake: {fake_params}",
        )
