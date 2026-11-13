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
from typing import Callable, Sequence, Tuple

import fbgemm_gpu  # noqa: F401 - register jagged operators before importing GR
import torch
import torch.nn.functional as F
from generative_recommenders.common import gpu_unavailable, is_dev_mode, set_dev_mode
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)
from generative_recommenders.ops.pytorch.pt_hstu_linear import pytorch_norm_mul_dropout
from generative_recommenders.ops.pytorch.pt_layer_norm import (
    pytorch_layer_norm,
    pytorch_rms_norm,
    pytorch_swish_layer_norm,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_cached_hstu_mha,
    triton_hstu_mha,
)
from generative_recommenders.ops.triton.triton_hstu_linear import (
    triton_layer_norm_mul_dropout_bwd,
    triton_layer_norm_mul_dropout_fwd,
)
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_layer_norm,
    triton_rms_norm,
    triton_swish_layer_norm,
)

DTYPE = torch.bfloat16


def compare(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float = 1e-2,
) -> None:
    actual = actual.detach()
    reference = reference.detach()
    assert torch.isfinite(actual).all(), name + ": non-finite actual values"
    assert torch.isfinite(reference).all(), name + ": non-finite reference values"
    torch.testing.assert_close(actual, reference, rtol=1.6e-2, atol=atol, msg=name)


def outputs_and_grads(
    fn: Callable[..., torch.Tensor],
    inputs: Sequence[torch.Tensor],
    dout: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    inputs = [x.detach().clone().requires_grad_() for x in inputs]
    output = fn(*inputs)
    grads = torch.autograd.grad(output, inputs, dout)
    return output, grads


def check_unweighted_norm(rows: int, dim: int) -> None:
    torch.manual_seed(1234)
    x = torch.randn(rows, dim, device="cuda", dtype=DTYPE)
    dout = torch.randn_like(x)
    actual, grads = outputs_and_grads(
        lambda x: triton_layer_norm(x, None, None, 1e-6), [x], dout
    )
    reference, ref_grads = outputs_and_grads(
        lambda x: F.layer_norm(x.float(), [dim], eps=1e-6).to(x.dtype), [x], dout
    )
    compare("unweighted_norm/fwd", actual, reference)
    compare("unweighted_norm/dx", grads[0], ref_grads[0])


def check_weighted_norm(rows: int, dim: int, kind: str) -> None:
    torch.manual_seed(1234)
    x = torch.randn(rows, dim, device="cuda", dtype=DTYPE)
    w = torch.randn(dim, device="cuda", dtype=DTYPE)
    b = torch.randn(dim, device="cuda", dtype=DTYPE)
    dout = torch.randn_like(x)
    cases = {
        "layer_norm": (
            lambda x, w, b: triton_layer_norm(x, w, b, 1e-6),
            lambda x, w, b: pytorch_layer_norm(x, [dim], w, b, 1e-6),
            (x, w, b),
        ),
        "swish_layer_norm": (
            lambda x, w, b: triton_swish_layer_norm(x, [dim], w, b, 1e-6),
            lambda x, w, b: pytorch_swish_layer_norm(x, [dim], w, b, 1e-6),
            (x, w, b),
        ),
        "rms_norm": (
            lambda x, w: triton_rms_norm(x, w, 1e-6),
            lambda x, w: pytorch_rms_norm(x, [dim], w, 1e-6),
            (x, w),
        ),
        "rms_norm_silu": (
            lambda x, w: triton_rms_norm(x, w, 1e-6, True),
            lambda x, w: pytorch_rms_norm(x, [dim], w, 1e-6, True),
            (x, w),
        ),
    }
    actual_fn, reference_fn, inputs = cases[kind]
    actual, grads = outputs_and_grads(actual_fn, inputs, dout)
    reference, ref_grads = outputs_and_grads(reference_fn, inputs, dout)
    compare(kind + "/fwd", actual, reference)
    compare(kind + "/dx", grads[0], ref_grads[0])
    for index, (grad, ref_grad) in enumerate(
        zip(grads[1:], ref_grads[1:], strict=True), start=1
    ):
        # Parameter gradients reduce over all rows; compare relative to their scale.
        scale = max(ref_grad.float().abs().max().item(), 1.0)
        compare(
            kind + "/dparam" + str(index) + "_scaled",
            grad.float() / scale,
            ref_grad.float() / scale,
        )


def check_fused_gradients(
    rows: int, dim: int, concat_u: bool, concat_x: bool, silu_u: bool
) -> None:
    torch.manual_seed(1234)
    x = torch.randn(rows, dim, device="cuda", dtype=DTYPE)
    u = torch.randn_like(x)
    weight = torch.ones(dim, device="cuda", dtype=DTYPE)
    bias = torch.zeros_like(weight)
    options = {
        "eps": 1e-6,
        "dropout_ratio": 0.0,
        "training": True,
        "silu_u": silu_u,
        "concat_u": concat_u,
        "concat_x": concat_x,
    }
    dout = torch.randn(
        rows, dim * (1 + concat_u + concat_x), device="cuda", dtype=DTYPE
    )
    # The v1 autograd wrapper collapses concat flags and disables the training
    # kernel at dropout=0. Call the low-level API to exercise the ported kernel.
    actual, mean, rstd, block_d, num_warps, seed, random_mask = (
        triton_layer_norm_mul_dropout_fwd(
            x=x, u=u, weight=weight, bias=bias, seed=1234, **options
        )
    )
    dx, du, dweight, dbias, _ = triton_layer_norm_mul_dropout_bwd(
        dy=dout,
        x=x,
        u=u,
        weight=weight,
        bias=bias,
        mean=mean,
        rstd=rstd,
        BLOCK_D=block_d,
        num_warps=num_warps,
        seed=seed,
        random_mask=random_mask,
        **options,
    )
    grads = (dx, du, dweight, dbias)
    reference, ref_grads = outputs_and_grads(
        lambda *xs: pytorch_norm_mul_dropout(*xs, **options), [x, u, weight, bias], dout
    )
    compare("fused/fwd", actual, reference)
    for name, grad, ref_grad in zip(
        ("dx", "du", "dweight", "dbias"), grads, ref_grads, strict=True
    ):
        compare("fused/" + name, grad, ref_grad)


def attention_inputs(
    interleaved: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    lengths = [93, 117, 128]
    total, heads, dim = sum(lengths), 4, 128
    fused = torch.empty(total, 4 * heads * dim, device="cuda", dtype=DTYPE).uniform_(
        -0.1, 0.1
    )
    _, v, q, k = [part.view(total, heads, dim) for part in fused.chunk(4, dim=1)]
    if not interleaved:
        q, k, v = [x.contiguous() for x in (q, k, v)]
    offsets = torch.tensor([0, 93, 210, total], device="cuda", dtype=torch.int64)
    targets = torch.tensor([7, 13, 16], device="cuda", dtype=torch.int64)
    return q, k, v, offsets, targets


def check_attention(
    interleaved: bool, with_targets: bool, window: int, contextual: int
) -> None:
    q, k, v, offsets, targets = attention_inputs(interleaved)
    references = [x.detach().clone().contiguous().requires_grad_() for x in (q, k, v)]
    options = {
        "alpha": 128**-0.5,
        "seq_offsets": offsets,
        "num_targets": targets if with_targets else None,
        "max_attn_len": window,
        "contextual_seq_len": contextual,
    }
    reference = pytorch_hstu_mha(
        max_seq_len=128, q=references[0], k=references[1], v=references[2], **options
    )
    dout = torch.randn_like(reference)
    ref_grads = torch.autograd.grad(reference, references, dout)
    # Preserve the actual interleaved strides, rather than cloning them away.
    actual_inputs = [x.detach().requires_grad_() for x in (q, k, v)]
    actual = triton_hstu_mha(
        N=128,
        q=actual_inputs[0],
        k=actual_inputs[1],
        v=actual_inputs[2],
        enable_tma=False,
        **options,
    )
    grads = torch.autograd.grad(actual, actual_inputs, dout)
    compare("attention/fwd", actual, reference, atol=1e-5)
    for name, grad, ref_grad in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        compare("attention/" + name, grad, ref_grad, atol=1e-5)


def check_cached_attention(
    interleaved: bool, with_targets: bool, delta: int, window: int, contextual: int
) -> None:
    q, k, v, offsets, targets = attention_inputs(interleaved)
    # Cached queries correspond to the last delta positions of each sequence.
    delta_q = torch.cat([q[end - delta : end] for end in (93, 210, 338)], dim=0)
    options = {
        "alpha": 128**-0.5,
        "delta_q": delta_q,
        "k": k,
        "v": v,
        "seq_offsets": offsets,
        "num_targets": targets if with_targets else None,
        "max_attn_len": window,
        "contextual_seq_len": contextual,
    }
    reference = pytorch_cached_hstu_mha(max_seq_len=128, **options)
    actual = triton_cached_hstu_mha(N=128, enable_tma=False, **options)
    compare("cached_attention/fwd", actual, reference, atol=1e-5)


@unittest.skipIf(*gpu_unavailable)
class TritonPointerTest(unittest.TestCase):
    """Exercise masked tails and tile offsets after block-pointer removal."""

    def setUp(self) -> None:
        # Multi-row norm kernels are selected by the autotuned path.
        self.addCleanup(set_dev_mode, is_dev_mode())
        set_dev_mode(False)

    def test_weighted_norms(self) -> None:
        for rows, dim in ((1000, 512), (37, 200)):
            for kind in ("layer_norm", "swish_layer_norm", "rms_norm", "rms_norm_silu"):
                with self.subTest(rows=rows, dim=dim, kind=kind):
                    check_weighted_norm(rows, dim, kind)

    def test_unweighted_norm(self) -> None:
        for rows, dim in ((1000, 512), (37, 200)):
            with self.subTest(rows=rows, dim=dim):
                check_unweighted_norm(rows, dim)

    def test_fused_training(self) -> None:
        for rows, dim in ((1000, 512), (37, 200)):
            for concat_u, concat_x, silu_u in itertools.product(
                (False, True), repeat=3
            ):
                with self.subTest(
                    rows=rows,
                    dim=dim,
                    concat_u=concat_u,
                    concat_x=concat_x,
                    silu_u=silu_u,
                ):
                    check_fused_gradients(rows, dim, concat_u, concat_x, silu_u)

    def test_attention(self) -> None:
        for interleaved, with_targets in itertools.product((False, True), repeat=2):
            for window, contextual in ((0, 0), (32, 0), (0, 4)):
                with self.subTest(
                    interleaved=interleaved,
                    with_targets=with_targets,
                    window=window,
                    contextual=contextual,
                ):
                    check_attention(interleaved, with_targets, window, contextual)

    def test_cached_attention(self) -> None:
        for interleaved, with_targets in itertools.product((False, True), repeat=2):
            for delta, (window, contextual) in itertools.product(
                (1, 7), ((0, 0), (32, 4))
            ):
                with self.subTest(
                    interleaved=interleaved,
                    with_targets=with_targets,
                    delta=delta,
                    window=window,
                    contextual=contextual,
                ):
                    check_cached_attention(
                        interleaved, with_targets, delta, window, contextual
                    )


if __name__ == "__main__":
    unittest.main()
