# pyre-strict

import torch
from generative_recommenders.common import next_power_of_2, should_trigger_eager_impl
from generative_recommenders.ops.pytorch.pt_hstu_linear import pytorch_norm_mul_dropout
from generative_recommenders.ops.triton.triton_hstu_linear import (
    _group_norm_mul_dropout_fwd,
)
from generative_recommenders.ops.triton_aot.types import triton_aot

_group_norm_mul_dropout_fwd = triton_aot(
    annotations={
        "D": ("i32", 16),
        "eps": "fp32",
        "seed": "i64",
        "dropout_ratio": "fp32",
        "stride_x": ("i32", 16),
        "stride_u": ("i32", 16),
        "stride_y": ("i32", 16),
    },
    # pyrefly: ignore [bad-argument-type]
)(_group_norm_mul_dropout_fwd)


@torch.jit.unused
@torch.fx.wrap
def _triton_aot_group_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    silu_u: bool,
    concat_ux: bool,
    num_heads: int,
    linear_dim: int,
) -> torch.Tensor:
    x = x.contiguous()
    u = u.contiguous()
    N, _ = x.shape
    if concat_ux:
        y = torch.empty((N, 3 * num_heads * linear_dim), dtype=x.dtype, device=x.device)
    else:
        y = torch.empty((N, num_heads * linear_dim), dtype=x.dtype, device=x.device)
    mean = torch.empty((N * num_heads,), dtype=x.dtype, device=x.device)
    rstd = torch.empty((N * num_heads,), dtype=x.dtype, device=x.device)

    BLOCK_D = next_power_of_2(linear_dim)
    BLOCK_H = next_power_of_2(num_heads)

    seed = 0
    dropout_ratio = 0.0

    grid = (N,)
    # pyrefly: ignore [not-callable]
    _group_norm_mul_dropout_fwd[grid](
        x,  # X
        u,  # U
        y,  # Y
        weight,  # W
        bias,  # B
        mean,  # Mean
        rstd,  # Rstd
        linear_dim,  # D
        num_heads,  # Heads
        eps,  # eps
        seed,  # seed
        dropout_ratio,  # dropout_ratio
        x.stride(0),  # stride_x
        u.stride(0),  # stride_u
        y.stride(0),  # stride_y
        # pyrefly: ignore [bad-argument-type]
        SILU_U=silu_u,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_D=BLOCK_D,
        # pyrefly: ignore [bad-argument-type]
        BLOCK_H=BLOCK_H,
        # pyrefly: ignore [bad-argument-type]
        TRAINING=False,
        # pyrefly: ignore [bad-argument-type]
        CONCAT_UX=concat_ux,
    )
    return y


@torch.fx.wrap
def aot_triton_kernel_wrapper_group_norm_mul_dropout(
    x: torch.Tensor,
    u: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    silu_u: bool,
    concat_ux: bool,
    num_heads: int,
    linear_dim: int,
) -> torch.Tensor:
    if should_trigger_eager_impl():
        return pytorch_norm_mul_dropout(
            x=x,
            u=u,
            weight=weight,
            bias=bias,
            eps=eps,
            dropout_ratio=0.0,
            training=False,
            silu_u=silu_u,
            concat_u=concat_ux,
            concat_x=concat_ux,
            group_norm=True,
            num_heads=num_heads,
            linear_dim=linear_dim,
        )
    return _triton_aot_group_norm_mul_dropout(
        x=x,
        u=u,
        weight=weight,
        bias=bias,
        eps=eps,
        silu_u=silu_u,
        concat_ux=concat_ux,
        num_heads=num_heads,
        linear_dim=linear_dim,
    )
