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

# pyre-unsafe

from typing import List

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl
from generative_recommenders.common import triton_autotune
from generative_recommenders.ops.utils import is_sm100_plus

TMA_AVAILABLE = False
try:
    # @manual=//triton:triton
    from triton.tools.tensor_descriptor import TensorDescriptor

    TMA_AVAILABLE = True
except ImportError:
    pass

HAS_TLX = False
try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    pass


def is_triton_swiglu_supported() -> bool:
    return is_sm100_plus() and TMA_AVAILABLE and HAS_TLX


def _swiglu_tma_set_block_size_hook(nargs) -> None:
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    BLOCK_K = nargs["BLOCK_K"]
    EPILOGUE_SUBTILE = nargs.get("EPILOGUE_SUBTILE", 1)

    nargs["x_desc"].block_shape = [BLOCK_M, BLOCK_K]
    nargs["w_gate_desc"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["w_up_desc"].block_shape = [BLOCK_N, BLOCK_K]
    nargs["out_desc"].block_shape = [BLOCK_M, BLOCK_N // EPILOGUE_SUBTILE]


def get_swiglu_configs(pre_hook) -> List[triton.Config]:
    return [
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        ),
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 64,
                "GROUP_M": 8,
            },
            num_stages=1,
            num_warps=4,
            pre_hook=pre_hook,
        ),
    ]


@triton.jit
def _compute_pid_swiglu(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton_autotune(
    configs=get_swiglu_configs(pre_hook=_swiglu_tma_set_block_size_hook),
    key=["M_BLOCK", "N", "K"],
)
@triton.jit
def _swiglu_fwd_tma_ws_persistent(
    x_desc,
    w_gate_desc,
    w_up_desc,
    out_desc,
    M,
    N,
    K,
    M_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMEM_BUFFERS: tl.constexpr,
    NUM_TMEM_BUFFERS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
):
    # Allocate SMEM buffers
    x_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), x_desc.dtype, NUM_SMEM_BUFFERS)

    # Allocate SMEM buffers for W_gate and W_up
    w_gate_buffers = tlx.local_alloc(
        (BLOCK_N, BLOCK_K), w_gate_desc.dtype, NUM_SMEM_BUFFERS
    )
    w_up_buffers = tlx.local_alloc(
        (BLOCK_N, BLOCK_K), w_up_desc.dtype, NUM_SMEM_BUFFERS
    )

    # Allocate TMEM for accumulators
    tmem_gate_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )
    tmem_up_buffers = tlx.local_alloc(
        (BLOCK_M, BLOCK_N), tl.float32, NUM_TMEM_BUFFERS, tlx.storage_kind.tmem
    )

    # Barriers for Producer <-> MMA synchronization
    smem_full_bars_x_gate = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=1,  # pyre-ignore[6]
    )
    smem_full_bars_up = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=1,  # pyre-ignore[6]
    )
    # Empty barriers: arrive_count=2 because both GEMM1 and GEMM2 signal completion
    smem_empty_bars = tlx.alloc_barriers(
        num_barriers=NUM_SMEM_BUFFERS,
        arrive_count=2,  # pyre-ignore[6]
    )

    # Barriers for MMA <-> Epilogue synchronization
    # pyre-ignore[6]
    tmem_full_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)
    # pyre-ignore[6]
    tmem_empty_bars = tlx.alloc_barriers(num_barriers=NUM_TMEM_BUFFERS, arrive_count=1)

    with tlx.async_tasks():
        # Epilogue Consumer: Reads from TMEM, applies SwiGLU, and stores to output
        with tlx.async_task("default"):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            # Initialize buffer tracking
            processed_tiles = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid_swiglu(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )
                offs_m = pid_m * BLOCK_M
                offs_n = pid_n * BLOCK_N

                cur_tmem_buf = processed_tiles % int(NUM_TMEM_BUFFERS)
                tmem_read_phase = (processed_tiles // int(NUM_TMEM_BUFFERS)) & 1

                # Wait for MMA to finish writing to TMEM
                # pyre-ignore[16]
                tlx.barrier_wait(tmem_full_bars[cur_tmem_buf], tmem_read_phase)

                # Load gate and up results from TMEM
                # pyre-ignore[16]
                gate_tmem = tmem_gate_buffers[cur_tmem_buf]
                up_tmem = tmem_up_buffers[cur_tmem_buf]

                if EPILOGUE_SUBTILE > 1:
                    # Process tile in subtiles
                    slice_size: tl.constexpr = BLOCK_N // EPILOGUE_SUBTILE
                    for slice_id in tl.static_range(EPILOGUE_SUBTILE):
                        gate_subslice = tlx.local_slice(
                            gate_tmem,
                            [0, slice_id * slice_size],
                            # pyre-ignore[6]
                            [BLOCK_M, slice_size],
                        )
                        up_subslice = tlx.local_slice(
                            up_tmem,
                            [0, slice_id * slice_size],
                            # pyre-ignore[6]
                            [BLOCK_M, slice_size],
                        )

                        gate = tlx.local_load(gate_subslice).to(out_desc.dtype)
                        up = tlx.local_load(up_subslice).to(out_desc.dtype)

                        gate_fp32 = gate.to(tl.float32)
                        silu_gate = (gate_fp32 * tl.sigmoid(gate_fp32)).to(
                            out_desc.dtype
                        )
                        result = silu_gate * up

                        out_desc.store([offs_m, offs_n + slice_id * slice_size], result)
                else:
                    # Process full tile
                    gate = tlx.local_load(gate_tmem).to(out_desc.dtype)
                    up = tlx.local_load(up_tmem).to(out_desc.dtype)

                    gate_fp32 = gate.to(tl.float32)
                    silu_gate = (gate_fp32 * tl.sigmoid(gate_fp32)).to(out_desc.dtype)
                    result = silu_gate * up

                    out_desc.store([offs_m, offs_n], result)

                # Signal MMA that TMEM buffer is free
                # pyre-ignore[6]
                tlx.barrier_arrive(tmem_empty_bars[cur_tmem_buf], 1)

                processed_tiles += 1

        # MMA Consumer: Computes both GEMMs: gate = X @ W_gate, up = X @ W_up
        with tlx.async_task(num_warps=4, num_regs=232):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            processed_k_iters = 0
            processed_tiles = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid_swiglu(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )

                cur_tmem_buf = processed_tiles % int(NUM_TMEM_BUFFERS)
                tmem_write_phase = (processed_tiles // int(NUM_TMEM_BUFFERS)) & 1

                # Wait for epilogue to finish
                tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase ^ 1)

                # Perform K-dimension reduction for both GEMMs
                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)

                    total_iters = processed_k_iters + k
                    dot_phase = (total_iters // int(NUM_SMEM_BUFFERS)) & 1

                    # Wait for x and w_gate to be loaded, then start GEMM1
                    tlx.barrier_wait(smem_full_bars_x_gate[buf], dot_phase)

                    # Transpose weight buffer for MMA
                    w_gate_trans = tlx.local_trans(w_gate_buffers[buf])

                    # GEMM 1: gate = X @ W_gate.T
                    tlx.async_dot(
                        x_buffers[buf],
                        w_gate_trans,
                        tmem_gate_buffers[cur_tmem_buf],
                        # pyre-ignore[6]
                        use_acc=(k > 0),
                        mBarriers=[smem_empty_bars[buf]],
                        out_dtype=tl.float32,
                    )

                    # Wait for w_up to be loaded before starting GEMM2
                    tlx.barrier_wait(smem_full_bars_up[buf], dot_phase)

                    w_up_trans = tlx.local_trans(w_up_buffers[buf])

                    # GEMM 2: up = X @ W_up.T
                    tlx.async_dot(
                        x_buffers[buf],
                        w_up_trans,
                        tmem_up_buffers[cur_tmem_buf],
                        # pyre-ignore[6]
                        use_acc=(k > 0),
                        mBarriers=[smem_empty_bars[buf]],
                        out_dtype=tl.float32,
                    )

                # Wait for last MMA to complete
                last_buf = (processed_k_iters + k_tiles - 1) % int(NUM_SMEM_BUFFERS)
                last_total_iters = processed_k_iters + k_tiles - 1
                last_dot_phase = (last_total_iters // int(NUM_SMEM_BUFFERS)) & 1
                tlx.barrier_wait(smem_empty_bars[last_buf], last_dot_phase)

                # Signal epilogue that results are ready
                # pyre-ignore[6]
                tlx.barrier_arrive(tmem_full_bars[cur_tmem_buf], 1)

                processed_tiles += 1
                processed_k_iters += k_tiles

        # Producer: TMA loads for X, W_gate, W_up
        with tlx.async_task(num_warps=1, num_regs=24):
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n
            k_tiles = tl.cdiv(K, BLOCK_K)

            # Initialize phase tracking
            processed_k_iters = 0

            for tile_id in range(start_pid, num_tiles, NUM_SMS):
                pid_m, pid_n = _compute_pid_swiglu(
                    tile_id, num_pid_in_group, num_pid_m, GROUP_M, NUM_SMS
                )
                offs_m = pid_m * BLOCK_M
                offs_n = pid_n * BLOCK_N

                for k in range(0, k_tiles):
                    buf = (processed_k_iters + k) % int(NUM_SMEM_BUFFERS)

                    total_iters = processed_k_iters + k
                    load_phase = (total_iters // int(NUM_SMEM_BUFFERS)) & 1

                    # Wait for buffer to be free
                    tlx.barrier_wait(smem_empty_bars[buf], load_phase ^ 1)

                    offs_k = k * BLOCK_K

                    # Set expected bytes for x+w_gate barrier
                    tlx.barrier_expect_bytes(
                        smem_full_bars_x_gate[buf],
                        # pyre-ignore[6]
                        2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N),
                    )

                    # Set expected bytes for w_up barrier
                    tlx.barrier_expect_bytes(
                        smem_full_bars_up[buf],
                        # pyre-ignore[6]
                        2 * (BLOCK_K * BLOCK_N),
                    )

                    # Load x and w_gate first, signal smem_full_bars_x_gate
                    tlx.async_descriptor_load(
                        x_desc,
                        x_buffers[buf],
                        [offs_m, offs_k],
                        smem_full_bars_x_gate[buf],
                    )

                    # Weights are in [N, K] layout, load with [offs_n, offs_k]
                    tlx.async_descriptor_load(
                        w_gate_desc,
                        w_gate_buffers[buf],
                        [offs_n, offs_k],
                        smem_full_bars_x_gate[buf],
                    )

                    # Load w_up separately, signal smem_full_bars_up
                    tlx.async_descriptor_load(
                        w_up_desc,
                        w_up_buffers[buf],
                        [offs_n, offs_k],
                        smem_full_bars_up[buf],
                    )

                processed_k_iters += k_tiles


@torch.fx.wrap
def triton_swiglu_fwd_tma_ws_persistent_tlx(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    M, K = x.shape
    N, K_gate = w_gate.shape
    N_up, K_up = w_up.shape

    # Only bf16/fp16 supported by the kernel
    supported_dtypes = (torch.bfloat16, torch.float16)
    assert x.dtype in supported_dtypes, (
        f"x.dtype must be bfloat16 or float16, got {x.dtype}"
    )
    assert w_gate.dtype in supported_dtypes, (
        f"w_gate.dtype must be bfloat16 or float16, got {w_gate.dtype}"
    )
    assert w_up.dtype in supported_dtypes, (
        f"w_up.dtype must be bfloat16 or float16, got {w_up.dtype}"
    )

    assert K == K_gate, f"Incompatible dimensions: x.K={K}, w_gate.K={K_gate}"
    assert K == K_up, f"Incompatible dimensions: x.K={K}, w_up.K={K_up}"
    assert N == N_up, f"Incompatible dimensions: w_gate.N={N}, w_up.N={N_up}"

    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return out

    M_BLOCK = triton.next_power_of_2(M)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # A dummy block value that will be overwritten by the hook
    dummy_block = [1, 1]

    # pyre-ignore[6]
    x_desc = TensorDescriptor(x, x.shape, x.stride(), dummy_block)
    # pyre-ignore[6]
    w_gate_desc = TensorDescriptor(w_gate, w_gate.shape, w_gate.stride(), dummy_block)
    # pyre-ignore[6]
    w_up_desc = TensorDescriptor(w_up, w_up.shape, w_up.stride(), dummy_block)
    # pyre-ignore[6]
    out_desc = TensorDescriptor(out, out.shape, out.stride(), dummy_block)

    def grid(meta):
        BLOCK_M = meta["BLOCK_M"]
        BLOCK_N = meta["BLOCK_N"]
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    _swiglu_fwd_tma_ws_persistent[grid](
        x_desc,
        w_gate_desc,
        w_up_desc,
        out_desc,
        M,
        N,
        K,
        M_BLOCK,
        NUM_SMS=NUM_SMS,
        NUM_SMEM_BUFFERS=4,
        NUM_TMEM_BUFFERS=2,
        EPILOGUE_SUBTILE=2,
    )
    return out


def triton_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> torch.Tensor:
    assert is_sm100_plus() and TMA_AVAILABLE and HAS_TLX, (
        "triton_swiglu requires Blackwell (SM100+) with TMA and TLX support"
    )

    _, K = x.shape
    N, _ = w_gate.shape
    assert K % 16 == 0 and N % 16 == 0, (
        f"K ({K}) and N ({N}) must be divisible by 16 for TMA alignment"
    )

    return triton_swiglu_fwd_tma_ws_persistent_tlx(x, w_gate, w_up)
