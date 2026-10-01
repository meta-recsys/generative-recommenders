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

"""Public FP8 operations used by HSTU Ultra."""

from __future__ import annotations

import torch
from generative_recommenders.ops.pytorch.pt_fp8 import (
    pytorch_fp8_rowwise_addmm,
    pytorch_quantize_fp8_per_row,
)


def quantize_fp8_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to E4M3 with one dequantization scale per row."""
    return pytorch_quantize_fp8_per_row(x)


def fp8_rowwise_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor | None = None,
    mat2: torch.Tensor | None = None,
    mat1_fp8: torch.Tensor | None = None,
    mat1_scale: torch.Tensor | None = None,
    mat2_fp8: torch.Tensor | None = None,
    mat2_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = True,
    is_inference: bool = False,
) -> torch.Tensor:
    """Compute ``mat1 @ mat2 + input`` using row-wise E4M3 inputs.

    Prequantized tensors may be supplied for inference. Weights use the
    ``[N, K]`` layout after quantization, while unquantized weights use
    ``[K, N]``.
    """
    if mat1_fp8 is None or mat1_scale is None:
        if mat1 is None:
            raise ValueError("mat1 is required when mat1_fp8 and mat1_scale are absent")
        mat1_fp8, mat1_scale = quantize_fp8_per_row(mat1)
    if mat2_fp8 is None or mat2_scale is None:
        if mat2 is None:
            raise ValueError("mat2 is required when mat2_fp8 and mat2_scale are absent")
        mat2_fp8, mat2_scale = quantize_fp8_per_row(mat2.t().contiguous())
    return pytorch_fp8_rowwise_addmm(
        input=input,
        mat1=mat1,
        mat2=mat2,
        mat1_fp8=mat1_fp8,
        mat2_fp8=mat2_fp8,
        mat1_scale=mat1_scale,
        mat2_scale=mat2_scale,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
        is_inference=is_inference,
    )
