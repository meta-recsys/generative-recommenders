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

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


@triton.jit
def _copy_u_grad_and_column_sum_kernel(
    DU,
    DZ,
    PARTIAL,
    T,
    N,
    U,
    stride_du,
    stride_dz,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    is_u = cols < U
    row_start = pid_t.to(tl.int64) * ROWS_PER_PROGRAM
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for r in range(0, ROWS_PER_PROGRAM, BLOCK_T):
        rows = row_start + r + tl.arange(0, BLOCK_T).to(tl.int64)
        row_mask = rows < T
        mask = row_mask[:, None] & col_mask[None, :]
        u_mask = row_mask[:, None] & (is_u & col_mask)[None, :]
        dz_mask = row_mask[:, None] & ((~is_u) & col_mask)[None, :]
        u_vals = tl.load(
            DU + rows[:, None] * stride_du + cols[None, :], mask=u_mask, other=0.0
        )
        tl.store(DZ + rows[:, None] * stride_dz + cols[None, :], u_vals, mask=u_mask)
        z_vals = tl.load(
            DZ + rows[:, None] * stride_dz + cols[None, :], mask=dz_mask, other=0.0
        )
        vals = tl.where(mask, u_vals.to(tl.float32) + z_vals.to(tl.float32), 0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(PARTIAL + pid_t.to(tl.int64) * N + cols, acc, mask=col_mask)


def copy_u_grad_and_column_sum(
    du: torch.Tensor,
    duvqk: torch.Tensor,
    rows_per_program: int = 4096,
    block_t: int = 64,
    block_n: int = 128,
) -> torch.Tensor:
    T, N = duvqk.shape
    U = du.shape[1]
    assert du.shape[0] == T and U <= N
    assert duvqk.stride(1) == 1 and du.stride(1) == 1
    assert du.dtype == duvqk.dtype
    if T == 0:
        return torch.zeros((N,), dtype=duvqk.dtype, device=duvqk.device)
    num_row_programs = triton.cdiv(T, rows_per_program)
    partial = torch.empty(
        (num_row_programs, N), dtype=torch.float32, device=duvqk.device
    )
    grid = (num_row_programs, triton.cdiv(N, block_n))
    _copy_u_grad_and_column_sum_kernel[grid](
        du,
        duvqk,
        partial,
        T,
        N,
        U,
        du.stride(0),
        duvqk.stride(0),
        ROWS_PER_PROGRAM=rows_per_program,
        BLOCK_T=block_t,
        BLOCK_N=block_n,
    )
    return partial.sum(dim=0).to(duvqk.dtype)
