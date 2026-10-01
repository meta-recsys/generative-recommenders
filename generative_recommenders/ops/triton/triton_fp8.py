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

"""Triton FP8 kernels used by HSTU Ultra."""

from __future__ import annotations

from typing import Optional

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import switch_to_contiguous_if_needed
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_weighted_layer_norm_bwd,
)


_FP8_MAX: float = 448.0
_FP8_EPS: float = 1e-12


@triton.jit
# Triton TR001: BLOCK_D is fixed to the next power of two of the row width.
def _quantize_fp8_per_row_kernel(  # noqa: TR001
    X,
    X_FP8,
    SCALE,
    D,
    stride_x,
    BLOCK_D: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    values = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=mask, other=0.0)
    row_max = tl.maximum(tl.max(tl.abs(values)), FP8_EPS)
    inverse_scale = FP8_MAX / row_max
    tl.store(SCALE + row, 1.0 / inverse_scale)
    tl.store(
        X_FP8 + row.to(tl.int64) * stride_x + cols,
        (values * inverse_scale).to(X_FP8.dtype.element_ty),
        mask=mask,
    )


def triton_quantize_fp8_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a contiguous 2-D tensor with one E4M3 scale per row."""
    if x.ndim != 2:
        raise ValueError("row-wise FP8 quantization requires a 2D tensor")
    x = switch_to_contiguous_if_needed(x)
    rows, width = x.shape
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows == 0:
        return quantized, scale
    block_width = triton.next_power_of_2(width)
    num_warps = min(max(block_width // 256, 1), 8)
    _quantize_fp8_per_row_kernel[(rows,)](
        x,
        quantized,
        scale,
        width,
        x.stride(0),
        BLOCK_D=block_width,
        FP8_MAX=_FP8_MAX,  # pyrefly: ignore [bad-argument-type]
        FP8_EPS=_FP8_EPS,  # pyrefly: ignore [bad-argument-type]
        num_warps=num_warps,  # pyrefly: ignore [unexpected-keyword]
    )
    return quantized, scale


@triton.jit
# Triton TR001: BLOCK_D is fixed by the one-program-per-row reduction.
def _layer_norm_fp8_quantize_kernel(  # noqa: TR001
    X,
    Y,
    Y_FP8,
    WEIGHT,
    BIAS,
    MEAN,
    RSTD,
    SCALE,
    D,
    eps,
    stride_x,
    stride_y,
    BLOCK_D: tl.constexpr,
    SAVE_Y: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    x = tl.load(X + row.to(tl.int64) * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.sum(x, axis=0) / D
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / D
    rstd = tl.rsqrt(variance + eps)
    weight = tl.load(WEIGHT + cols, mask=mask).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=mask).to(tl.float32)
    y = centered * rstd * weight + bias

    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)
    if SAVE_Y:
        tl.store(
            Y + row.to(tl.int64) * stride_y + cols,
            y.to(Y.dtype.element_ty),
            mask=mask,
        )

    row_max = tl.maximum(tl.max(tl.abs(y)), FP8_EPS)
    inverse_scale = FP8_MAX / row_max
    tl.store(SCALE + row, 1.0 / inverse_scale)
    tl.store(
        Y_FP8 + row.to(tl.int64) * stride_y + cols,
        (y * inverse_scale).to(Y_FP8.dtype.element_ty),
        mask=mask,
    )


def _layer_norm_fp8_quantize_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    save_y: bool,
) -> tuple[
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    if x.ndim != 2:
        raise ValueError("fused LayerNorm FP8 quantization requires a 2D tensor")
    x = switch_to_contiguous_if_needed(x)
    rows, width = x.shape
    if weight.shape != (width,) or bias.shape != (width,):
        raise ValueError("LayerNorm weight and bias must match the input width")
    y_storage = torch.empty_like(x) if save_y else torch.empty(0, device=x.device)
    y_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    block_width = triton.next_power_of_2(width)
    max_fused_width = 65536 // x.element_size()
    if width > max_fused_width:
        raise RuntimeError("fused LayerNorm does not support feature sizes >= 64KB")
    num_warps = min(max(block_width // 256, 1), 8)
    if rows > 0:
        _layer_norm_fp8_quantize_kernel[(rows,)](
            x,
            y_storage,
            y_fp8,
            weight,
            bias,
            mean,
            rstd,
            scale,
            width,
            eps,
            x.stride(0),
            x.stride(0) if not save_y else y_storage.stride(0),
            BLOCK_D=block_width,  # pyrefly: ignore [bad-argument-type]
            SAVE_Y=save_y,  # pyrefly: ignore [bad-argument-type]
            FP8_MAX=_FP8_MAX,  # pyrefly: ignore [bad-argument-type]
            FP8_EPS=_FP8_EPS,  # pyrefly: ignore [bad-argument-type]
            num_warps=num_warps,  # pyrefly: ignore [unexpected-keyword]
        )
    return (y_storage if save_y else None), y_fp8, scale, mean, rstd, block_width


class _LayerNormFP8Quantize(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        save_y: bool,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        y, y_fp8, scale, mean, rstd, block_width = _layer_norm_fp8_quantize_fwd(
            x=x,
            weight=weight,
            bias=bias,
            eps=eps,
            save_y=save_y,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.eps = eps
        ctx.block_width = block_width
        return y, y_fp8, scale

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx,
        grad_y: torch.Tensor,
        grad_y_fp8: Optional[torch.Tensor],
        grad_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        x, weight, bias, mean, rstd = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = triton_weighted_layer_norm_bwd(
            dy=grad_y,
            x=x,
            weight=weight,
            bias=bias,
            mean=mean,
            rstd=rstd,
            learnable=True,
            eps=ctx.eps,
            BLOCK_D=ctx.block_width,
        )
        assert grad_weight is not None and grad_bias is not None
        return grad_x, grad_weight, grad_bias, None, None


def triton_layer_norm_fp8_quantize(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    save_y: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Fuse LayerNorm and row-wise FP8 quantization."""
    return _LayerNormFP8Quantize.apply(x, weight, bias, eps, save_y)


@triton.jit
# Triton TR001: BLOCK_D is fixed by the one-program-per-row reduction.
def _hstu_output_fp8_quantize_kernel(  # noqa: C901, TR001
    X,
    U,
    Y,
    Y_FP8,
    WEIGHT,
    BIAS,
    MEAN,
    RSTD,
    SCALE,
    D,
    eps,
    seed,
    dropout_ratio,
    stride_x,
    stride_u,
    stride_y,
    SILU_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TRAINING: tl.constexpr,
    CONCAT_U: tl.constexpr,
    CONCAT_X: tl.constexpr,
    MUL_U_ACTIVATION_TYPE: tl.constexpr,
    SAVE_Y: tl.constexpr,
    FP8_MAX: tl.constexpr,
    FP8_EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    row_offset = row.to(tl.int64)
    x = tl.load(X + row_offset * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U + row_offset * stride_u + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / D
    rstd = tl.rsqrt(variance + eps)
    weight = tl.load(WEIGHT + cols, mask=mask).to(tl.float32)
    bias = tl.load(BIAS + cols, mask=mask).to(tl.float32)
    y = centered * rstd * weight + bias
    sigmoid_u = tl.sigmoid(u)
    silu_u = u * sigmoid_u
    if MUL_U_ACTIVATION_TYPE == "silu":
        y *= silu_u
    elif MUL_U_ACTIVATION_TYPE == "sigmoid":
        y *= sigmoid_u
    else:
        y *= u
    if CONCAT_U and SILU_U:
        u = silu_u

    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)
    if TRAINING:
        random_offsets = 3 * row * BLOCK_D + cols
        if CONCAT_U and CONCAT_X:
            u = tl.where(
                tl.rand(seed, random_offsets) > dropout_ratio,
                u / (1.0 - dropout_ratio),
                0.0,
            )
            x = tl.where(
                tl.rand(seed, random_offsets + BLOCK_D) > dropout_ratio,
                x / (1.0 - dropout_ratio),
                0.0,
            )
            y = tl.where(
                tl.rand(seed, random_offsets + 2 * BLOCK_D) > dropout_ratio,
                y / (1.0 - dropout_ratio),
                0.0,
            )
        elif CONCAT_U:
            u = tl.where(
                tl.rand(seed, random_offsets) > dropout_ratio,
                u / (1.0 - dropout_ratio),
                0.0,
            )
            y = tl.where(
                tl.rand(seed, random_offsets + BLOCK_D) > dropout_ratio,
                y / (1.0 - dropout_ratio),
                0.0,
            )
        elif CONCAT_X:
            x = tl.where(
                tl.rand(seed, random_offsets) > dropout_ratio,
                x / (1.0 - dropout_ratio),
                0.0,
            )
            y = tl.where(
                tl.rand(seed, random_offsets + BLOCK_D) > dropout_ratio,
                y / (1.0 - dropout_ratio),
                0.0,
            )
        else:
            y = tl.where(
                tl.rand(seed, random_offsets) > dropout_ratio,
                y / (1.0 - dropout_ratio),
                0.0,
            )

    row_max = tl.max(tl.abs(y))
    if CONCAT_U:
        row_max = tl.maximum(row_max, tl.max(tl.abs(u)))
    if CONCAT_X:
        row_max = tl.maximum(row_max, tl.max(tl.abs(x)))
    row_max = tl.maximum(row_max, FP8_EPS)
    inverse_scale = FP8_MAX / row_max
    tl.store(SCALE + row, 1.0 / inverse_scale)

    output_offset = row_offset * stride_y
    if CONCAT_U and CONCAT_X:
        if SAVE_Y:
            tl.store(Y + output_offset + cols, u.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + output_offset + D + cols, x.to(Y.dtype.element_ty), mask=mask)
            tl.store(
                Y + output_offset + 2 * D + cols,
                y.to(Y.dtype.element_ty),
                mask=mask,
            )
        tl.store(
            Y_FP8 + output_offset + cols,
            (u * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            Y_FP8 + output_offset + D + cols,
            (x * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            Y_FP8 + output_offset + 2 * D + cols,
            (y * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
    elif CONCAT_U:
        if SAVE_Y:
            tl.store(Y + output_offset + cols, u.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + output_offset + D + cols, y.to(Y.dtype.element_ty), mask=mask)
        tl.store(
            Y_FP8 + output_offset + cols,
            (u * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            Y_FP8 + output_offset + D + cols,
            (y * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
    elif CONCAT_X:
        if SAVE_Y:
            tl.store(Y + output_offset + cols, x.to(Y.dtype.element_ty), mask=mask)
            tl.store(Y + output_offset + D + cols, y.to(Y.dtype.element_ty), mask=mask)
        tl.store(
            Y_FP8 + output_offset + cols,
            (x * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            Y_FP8 + output_offset + D + cols,
            (y * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )
    else:
        if SAVE_Y:
            tl.store(Y + output_offset + cols, y.to(Y.dtype.element_ty), mask=mask)
        tl.store(
            Y_FP8 + output_offset + cols,
            (y * inverse_scale).to(Y_FP8.dtype.element_ty),
            mask=mask,
        )


def triton_hstu_output_fp8_quantize_fwd(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dropout_ratio: float,
    training: bool,
    silu_u: bool,
    concat_u: bool,
    concat_x: bool,
    mul_u_activation_type: str,
    seed: Optional[int],
    save_y: bool,
) -> tuple[
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
]:
    """Fuse HSTU output postprocessing and row-wise FP8 quantization."""
    if x.ndim != 2:
        raise ValueError("fused HSTU output quantization requires a 2D tensor")
    x = switch_to_contiguous_if_needed(x)
    u = switch_to_contiguous_if_needed(u)
    rows, width = x.shape
    output_width = width * (1 + int(concat_u) + int(concat_x))
    y = (
        torch.empty((rows, output_width), dtype=x.dtype, device=x.device)
        if save_y
        else torch.empty(0, dtype=x.dtype, device=x.device)
    )
    y_fp8 = torch.empty(
        (rows, output_width), dtype=torch.float8_e4m3fn, device=x.device
    )
    scale = torch.empty((rows,), dtype=torch.float32, device=x.device)
    mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    block_width = triton.next_power_of_2(width)
    max_fused_width = 65536 // x.element_size()
    if width > max_fused_width:
        raise RuntimeError("fused HSTU output does not support feature sizes >= 64KB")
    if seed is None:
        seed = (
            int(torch.randint(0, 2**62, (1,), dtype=torch.int64).item())
            if training
            else 0
        )
    num_warps = min(max(block_width // 256, 1), 8)
    if rows > 0:
        _hstu_output_fp8_quantize_kernel[(rows,)](
            x,
            u,
            y,
            y_fp8,
            weight,
            bias,
            mean,
            rstd,
            scale,
            width,
            eps,
            seed,
            dropout_ratio,
            x.stride(0),
            u.stride(0),
            output_width,
            SILU_U=silu_u,  # pyrefly: ignore [bad-argument-type]
            BLOCK_D=block_width,  # pyrefly: ignore [bad-argument-type]
            TRAINING=training,  # pyrefly: ignore [bad-argument-type]
            CONCAT_U=concat_u,  # pyrefly: ignore [bad-argument-type]
            CONCAT_X=concat_x,  # pyrefly: ignore [bad-argument-type]
            MUL_U_ACTIVATION_TYPE=mul_u_activation_type,  # pyrefly: ignore [bad-argument-type]
            SAVE_Y=save_y,  # pyrefly: ignore [bad-argument-type]
            FP8_MAX=_FP8_MAX,  # pyrefly: ignore [bad-argument-type]
            FP8_EPS=_FP8_EPS,  # pyrefly: ignore [bad-argument-type]
            num_warps=num_warps,  # pyrefly: ignore [unexpected-keyword]
        )
    return (
        y if save_y else None,
        y_fp8,
        scale,
        mean,
        rstd,
        block_width,
        num_warps,
        seed,
    )


# Adapted from FBGEMM GPU's experimental row-wise FP8 Triton GEMM.
_FP8_MATMUL_CONFIGS: list[triton.Config] = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
        num_stages=4,
        num_warps=4,
    ),
]


@triton.autotune(configs=_FP8_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fp8_rowwise_addmm_kernel(
    A,
    B,
    C,
    A_SCALE,
    B_SCALE,
    BIAS,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bias_m,
    stride_bias_n,
    NUM_SMS: tl.constexpr,
    USE_INT64: tl.constexpr,
    USE_1D_BIAS: tl.constexpr,
    USE_2D_BIAS: tl.constexpr,
    FAST_ACCUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_M * num_pid_n
    index_dtype = tl.int64 if USE_INT64 else tl.int32

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        # pyrefly: ignore [unsupported-operation]
        pid_m = first_pid_m + tile_id % group_size_m
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        safe_m = tl.max_contiguous(
            tl.multiple_of(tl.where(offs_m < M, offs_m, 0), BLOCK_M), BLOCK_M
        )
        safe_n = tl.max_contiguous(
            tl.multiple_of(tl.where(offs_n < N, offs_n, 0), BLOCK_N), BLOCK_N
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(
                A
                + safe_m[:, None].to(index_dtype) * stride_am
                + offs_k[None, :] * stride_ak,
                mask=offs_k[None, :] < K,
                other=0.0,
            )
            b = tl.load(
                B + safe_n[None, :] * stride_bn + offs_k[:, None] * stride_bk,
                # Triton TR003: guard both tiled dimensions explicitly.
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            if FAST_ACCUM:
                # Triton TR011: keep Tensor Core behavior stable across versions.
                accumulator = tl.dot(a, b, accumulator, allow_tf32=True)
            else:
                # Triton TR011: keep Tensor Core behavior stable across versions.
                accumulator += tl.dot(a, b, allow_tf32=True)

        output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        a_scale = tl.load(A_SCALE + offs_m, mask=offs_m < M)
        b_scale = tl.load(B_SCALE + offs_n, mask=offs_n < N)
        accumulator *= a_scale[:, None] * b_scale[None, :]
        if USE_1D_BIAS:
            accumulator += tl.load(BIAS + offs_n, mask=offs_n < N)[None, :]
        if USE_2D_BIAS:
            accumulator += tl.load(
                BIAS
                + offs_m[:, None].to(index_dtype) * stride_bias_m
                + offs_n[None, :] * stride_bias_n,
                mask=output_mask,
            )
        tl.store(
            C
            + offs_m[:, None].to(index_dtype) * stride_cm
            + offs_n[None, :] * stride_cn,
            accumulator.to(C.dtype.element_ty),
            mask=output_mask,
        )


def triton_fp8_rowwise_addmm_fwd(
    input: torch.Tensor,
    mat1_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat2_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """Persistent row-wise FP8 GEMM with vector or matrix bias."""
    mat1_fp8 = mat1_fp8.view(-1, mat1_fp8.shape[-1])
    rows, reduction = mat1_fp8.shape
    columns, weight_reduction = mat2_fp8.shape
    if reduction != weight_reduction:
        raise ValueError("FP8 inputs must have the same reduction dimension")
    if mat1_scale.shape != (rows,) or mat2_scale.shape != (columns,):
        raise ValueError("FP8 scales must contain one value per row")
    if input.shape not in ((columns,), (rows, columns)):
        raise ValueError("input must be a vector or matrix matching the output")
    output = torch.empty((rows, columns), dtype=out_dtype, device=mat1_fp8.device)
    if rows == 0 or columns == 0:
        return output + input
    num_sms = torch.cuda.get_device_properties(mat1_fp8.device).multi_processor_count
    use_int64 = mat1_fp8.numel() > 2**31 - 1 or output.numel() > 2**31 - 1
    grid = lambda meta: (  # noqa: E731
        min(
            num_sms,
            triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(columns, meta["BLOCK_N"]),
        ),
    )
    _fp8_rowwise_addmm_kernel[grid](
        mat1_fp8,
        mat2_fp8,
        output,
        mat1_scale,
        mat2_scale,
        input,
        rows,
        columns,
        reduction,
        mat1_fp8.stride(0),
        mat1_fp8.stride(1),
        mat2_fp8.stride(0),
        mat2_fp8.stride(1),
        output.stride(0),
        output.stride(1),
        input.stride(0) if input.ndim == 2 else 0,
        input.stride(1) if input.ndim == 2 else 0,
        NUM_SMS=num_sms,
        USE_INT64=use_int64,
        USE_1D_BIAS=input.ndim == 1,
        USE_2D_BIAS=input.ndim == 2,
        FAST_ACCUM=use_fast_accum,
    )
    return output


class _TritonFP8Addmm(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        input: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        mat1_fp8: torch.Tensor,
        mat2_fp8: torch.Tensor,
        mat1_scale: torch.Tensor,
        mat2_scale: torch.Tensor,
        out_dtype: torch.dtype,
        use_fast_accum: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(mat1, mat2)
        ctx.input_shape = input.shape
        return triton_fp8_rowwise_addmm_fwd(
            input=input,
            mat1_fp8=mat1_fp8,
            mat2_fp8=mat2_fp8,
            mat1_scale=mat1_scale,
            mat2_scale=mat2_scale,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        mat1, mat2 = ctx.saved_tensors
        return (
            grad_output.sum_to_size(ctx.input_shape),
            torch.mm(grad_output, mat2.t()),
            torch.mm(mat1.t(), grad_output),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_fp8_rowwise_addmm(
    input: torch.Tensor,
    mat1: Optional[torch.Tensor],
    mat2: Optional[torch.Tensor],
    mat1_fp8: torch.Tensor,
    mat2_fp8: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat2_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    use_fast_accum: bool = True,
    is_inference: bool = False,
) -> torch.Tensor:
    """Run the Triton FP8 GEMM with a full-precision training backward."""
    if is_inference:
        return triton_fp8_rowwise_addmm_fwd(
            input,
            mat1_fp8,
            mat2_fp8,
            mat1_scale,
            mat2_scale,
            out_dtype,
            use_fast_accum,
        )
    if mat1 is None or mat2 is None:
        raise ValueError("training requires unquantized mat1 and mat2 tensors")
    return _TritonFP8Addmm.apply(
        input,
        mat1,
        mat2,
        mat1_fp8,
        mat2_fp8,
        mat1_scale,
        mat2_scale,
        out_dtype,
        use_fast_accum,
    )
