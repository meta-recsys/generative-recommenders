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


from typing import List, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    tlx = None
    HAS_TLX = False

from generative_recommenders.common import triton_autotune, triton_cc
from generative_recommenders.ops.utils import is_sm100

try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    TMA_AVAILABLE = False
    pass


ENABLE_FULL_TURNING_SPACE = False


def _check_tma_alignment(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, min_alignment: int = 16
) -> bool:
    """Check if tensors meet TMA alignment requirements.

    TMA (Tensor Memory Accelerator) on H100 requires:
    1. Base addresses to be 64-byte aligned
    2. Dimensions to be multiples of 64 for optimal performance
    3. Contiguous inner dimensions (stride=1)

    Args:
        x: Input tensor [M, K]
        w: Weight tensor [K, N]
        y: Bias tensor [N] or [M, N]
        min_alignment: Minimum alignment requirement (default: 64)

    Returns:
        True if all tensors meet TMA alignment requirements
    """
    _, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    return (K % min_alignment == 0) and (N % min_alignment == 0)


def get_mm_configs(pre_hook=None) -> List[triton.Config]:
    if torch.version.hip:
        if ENABLE_FULL_TURNING_SPACE:
            block_m_range = [32, 64, 128, 256]
            block_n_range = [32, 64, 128, 256]
            block_k_range = [32, 64]
            group_m_range = [4, 8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [1, 2]
            num_warps_range = [4, 8]
            num_stage_range = [2] if triton.__version__ >= "3.2.0" else [0]
        else:
            block_m_range = [256]
            block_n_range = [256]
            block_k_range = [32]
            group_m_range = [8]
            matrix_instr_nonkdim_range = [16]
            waves_per_eu_range = [0]
            kpack_range = [2]
            num_warps_range = [8]
            num_stage_range = [2] if triton.__version__ >= "3.2.0" else [0]

        return [
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "BLOCK_K": block_k,
                    "GROUP_M": group_m,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                    "waves_per_eu": waves_per_eu,
                    "kpack": kpack,
                },
                num_stages=num_stages,
                num_warps=num_warps,
                pre_hook=pre_hook,
            )
            for block_m in block_m_range
            for block_n in block_n_range
            for block_k in block_k_range
            for group_m in group_m_range
            for matrix_instr_nonkdim in matrix_instr_nonkdim_range
            for waves_per_eu in waves_per_eu_range
            for kpack in kpack_range
            for num_stages in num_stage_range
            for num_warps in num_warps_range
        ]
    else:
        block_m_range = [32, 64, 128, 256]
        block_n_range = [32, 64, 128, 256]
        block_k_range = [32, 64]
        group_m_range = [4, 8]
        # WARP_SPECIALIZE only works with num_warps >=4
        num_warps_range = [4, 8] if is_sm100() else [2, 4, 8]
        num_stage_range = [2, 3, 4, 5]
        if ENABLE_FULL_TURNING_SPACE:
            return [
                triton.Config(
                    {
                        "BLOCK_M": block_m,
                        "BLOCK_N": block_n,
                        "BLOCK_K": block_k,
                        "GROUP_M": group_m,
                    },
                    num_stages=num_stages,
                    num_warps=num_warps,
                    pre_hook=pre_hook,
                )
                for block_m in block_m_range
                for block_n in block_n_range
                for block_k in block_k_range
                for group_m in group_m_range
                for num_stages in num_stage_range
                for num_warps in num_warps_range
            ]
        else:
            configs = [
                triton.Config(
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=5,
                    num_warps=2,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 256,
                        "BLOCK_K": 64,
                        "GROUP_M": 8,
                    },
                    num_stages=3,
                    num_warps=8,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 256,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 128,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 64,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 128,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 32,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                    pre_hook=pre_hook,
                ),
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 32,
                        "BLOCK_K": 32,
                        "GROUP_M": 8,
                    },
                    num_stages=5,
                    num_warps=2,
                    pre_hook=pre_hook,
                ),
            ]
            if is_sm100():
                configs += [
                    triton.Config(
                        {
                            "BLOCK_M": 128,
                            "BLOCK_N": 256,
                            "BLOCK_K": 64,
                            "GROUP_M": 8,
                        },
                        num_stages=3,
                        num_warps=4,
                        pre_hook=pre_hook,
                    ),
                ]
                return [c for c in configs if c.num_warps >= 4]

            return configs


@triton_cc(
    annotations={
        "M": "i32",
        "N": ("i32", 16),
        "K": ("i32", 16),
        "stride_xm": ("i32", 16),
        "stride_xk": ("i32", 1),
        "stride_wk": ("i32", 16),
        "stride_wn": ("i32", 1),
        "stride_ym": ("i32", 16),
        "stride_yn": ("i32", 1),
        "stride_zm": ("i32", 16),
        "stride_zn": ("i32", 1),
    },
)
@triton_autotune(
    configs=get_mm_configs(),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


def _addmm_tma_set_block_size_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    nargs["x_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["w_desc"].block_shape = [BLOCK_K, BLOCK_N]
    nargs["z_desc"].block_shape = [BLOCK_M, BLOCK_N]
    if nargs["BROADCAST_Y"]:
        nargs["y_desc"].block_shape = [1, BLOCK_N]
    else:
        nargs["y_desc"].block_shape = [BLOCK_M, BLOCK_N]


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["N", "K", "WARP_SPECIALIZE"],
)
@triton.jit
def _addmm_fwd_tma_persistent(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_M * num_pid_n

    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
        )
        offs_xm = pid_m * BLOCK_M
        offs_wn = pid_n * BLOCK_N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, k_tiles, warp_specialize=WARP_SPECIALIZE):
            offs_k = k * BLOCK_K
            x = x_desc.load([offs_xm, offs_k])
            w = w_desc.load([offs_k, offs_wn])
            accumulator = tl.dot(x, w, accumulator, allow_tf32=ALLOW_TF32)
        if BROADCAST_Y:
            y = y_desc.load([0, offs_wn])
        else:
            y = y_desc.load([offs_xm, offs_wn])
        z = (accumulator + y.to(tl.float32)).to(z_desc.dtype)
        z_desc.store([offs_xm, offs_wn], z)


@triton_autotune(
    configs=get_mm_configs(pre_hook=_addmm_tma_set_block_size_hook),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd_tma_ws(
    x_desc,
    w_desc,
    y_desc,
    z_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
):
    x_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)
    w_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), w_desc.dtype, NUM_SMEM_BUFFERS)
    acc_tmem_buffer = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, tl.constexpr(1), tlx.storage_kind.tmem
    )

    if BROADCAST_Y:
        y_buffer = tlx.local_alloc((1, BLOCK_N), y_desc.dtype, tl.constexpr(1))
    else:
        y_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), y_desc.dtype, tl.constexpr(1))
    z_buffer = tlx.local_alloc((BLOCK_M, BLOCK_N), z_desc.dtype, tl.constexpr(1))

    smem_full_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    smem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_SMEM_BUFFERS, arrive_count=1)
    y_load_barrier = tlx.alloc_barriers(num_barriers=1, arrive_count=1)

    with tlx.async_tasks():
        # Producer task: TMA loads
        with tlx.async_task("default"):
            pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
            pid = pid_0 * tl.num_programs(axis=1) + pid_1
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_xm = pid_m * BLOCK_M
            offs_wn = pid_n * BLOCK_N
            k_tiles = tl.cdiv(K, BLOCK_K)

            load_phase = 0
            for k in range(0, k_tiles):
                buf = k % int(NUM_SMEM_BUFFERS)

                # Wait for buffer to be free
                if k >= NUM_SMEM_BUFFERS:
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                offs_k = k * BLOCK_K
                tlx.barrier_expect_bytes(
                    smem_full_bars[buf],
                    2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N),
                )
                tlx.async_descriptor_load(
                    x_desc, x_buffers[buf], [offs_xm, offs_k], smem_full_bars[buf]
                )
                tlx.async_descriptor_load(
                    w_desc, w_buffers[buf], [offs_k, offs_wn], smem_full_bars[buf]
                )

                load_phase = load_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

        # Consumer task: async_dot MMA
        with tlx.async_task(num_warps=4, num_regs=232):
            pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
            pid = pid_0 * tl.num_programs(axis=1) + pid_1
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            pid_m = first_pid_m + (pid % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            offs_xm = pid_m * BLOCK_M
            offs_wn = pid_n * BLOCK_N
            k_tiles = tl.cdiv(K, BLOCK_K)

            # Start async load of y early
            y_buf_view = tlx.local_view(y_buffer, 0)
            y_load_bar = tlx.local_view(y_load_barrier, 0)
            if BROADCAST_Y:
                tlx.barrier_expect_bytes(y_load_bar, 1 * BLOCK_N * 2)
                tlx.async_descriptor_load(y_desc, y_buf_view, [0, offs_wn], y_load_bar)
            else:
                tlx.barrier_expect_bytes(y_load_bar, BLOCK_M * BLOCK_N * 2)
                tlx.async_descriptor_load(
                    y_desc, y_buf_view, [offs_xm, offs_wn], y_load_bar
                )

            dot_phase = 0
            for k in range(0, k_tiles):
                buf = k % int(NUM_SMEM_BUFFERS)
                tlx.barrier_wait(smem_full_bars[buf], dot_phase)

                tlx.async_dot(
                    x_buffers[buf],
                    w_buffers[buf],
                    acc_tmem_buffer[0],
                    use_acc=k > 0,
                    mBarriers=[smem_empty_bars[buf]],
                    out_dtype=tl.float32,
                )

                dot_phase = dot_phase ^ (buf == NUM_SMEM_BUFFERS - 1)

            last_buf = (k_tiles - 1) % NUM_SMEM_BUFFERS
            last_dot_phase = dot_phase ^ (last_buf == NUM_SMEM_BUFFERS - 1)
            tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

            tmem_result = tlx.local_load(acc_tmem_buffer[0])

            tlx.barrier_wait(y_load_bar, 0)
            y = tlx.local_load(y_buf_view)

            z = (tmem_result + y.to(tl.float32)).to(z_desc.dtype)
            z_buf_view = tlx.local_view(z_buffer, 0)
            tlx.local_store(z_buf_view, z)
            tlx.async_descriptor_store(z_desc, z_buf_view, [offs_xm, offs_wn])
            tlx.async_descriptor_store_wait(0)


@torch.fx.wrap
def triton_addmm_fwd_tma_persistent(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    warp_specialize: bool = False,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    def grid(meta):
        nonlocal x_desc, w_desc, z_desc
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    _addmm_fwd_tma_persistent[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
        WARP_SPECIALIZE=warp_specialize,
        NUM_SMS=NUM_SMS,
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd_tma_ws_tlx(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    _, N = w.shape

    is_y_1d = y.dim() == 1

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    # A dummy block value that will be overwritten when we have the real block size
    dummy_block = [1, 1]
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    w_desc = TensorDescriptor(w, w.shape, w.stride(), dummy_block)
    y = y.reshape(1, -1) if is_y_1d else y
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    y_desc = TensorDescriptor(y, y.shape, y.stride(), dummy_block)
    # pyre-ignore[6]: In call `TensorDescriptor.__init__`, for 2nd positional
    # argument, expected `List[int]` but got `Size`
    z_desc = TensorDescriptor(z, z.shape, z.stride(), dummy_block)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            triton.cdiv(M, BLOCK_M),
            triton.cdiv(N, BLOCK_N),
        )

    _addmm_fwd_tma_ws[grid](
        x_desc,
        w_desc,
        y_desc,
        z_desc,
        M,
        N,
        K,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
        NUM_SMEM_BUFFERS=2,  # Double buffering
    )
    return z


@torch.fx.wrap
def triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f"incompatible dimensions {K}, {KB}"

    is_y_1d = y.dim() == 1
    NY = y.shape[0] if is_y_1d else y.shape[1]
    assert N == NY, f"incompatible dimensions {N}, {NY}"

    # Allocate output
    z = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return z

    grid = lambda meta: (  # noqa E731
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _addmm_fwd[grid](
        x,
        w,
        y,
        z,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0) if not is_y_1d else 0,
        y.stride(1) if not is_y_1d else y.stride(0),
        z.stride(0),
        z.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=is_y_1d,
    )
    return z


def triton_addmm_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    dz: torch.Tensor,
    is_y_1d: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_y_1d:
        dy = torch.sum(dz, dim=0)
    else:
        dy = dz
    dw = torch.mm(x.t(), dz)
    dx = torch.mm(dz, w.t())

    return dx, dw, dy


@torch.fx.wrap
def maybe_triton_addmm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    # triton addmm is slower than torch (cublas) on AMD/Blackwell.
    # Default to pytorch addmm on AMD/Blackwell for now.
    if is_sm100() or torch.version.hip is not None:
        return torch.addmm(y, x, w)
    else:
        return triton_addmm_fwd(x=x, w=w, y=y)


class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, w)
        ctx.is_y_1d = y.dim() == 1
        if is_sm100() and TMA_AVAILABLE and _check_tma_alignment(x, w, y):
            if x.dtype == torch.float32:
                # use TMA persistent kernel on sm100
                return triton_addmm_fwd_tma_persistent(x, w, y, warp_specialize=True)
            else:
                return triton_addmm_fwd_tma_ws_tlx(
                    x, w, y
                )  # tlx.async_dot doesn't support fp32 inputs because of WGMMA requirements
        else:
            return triton_addmm_fwd(x, w, y)

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dz: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (x, w) = ctx.saved_tensors
        return triton_addmm_bwd(x, w, dz, ctx.is_y_1d)


def triton_addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> torch.Tensor:
    return _AddMmFunction.apply(mat1, mat2, input)
