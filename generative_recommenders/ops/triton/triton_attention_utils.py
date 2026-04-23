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

# pyre-unsafe

#!/usr/bin/env python3


# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


@triton.jit
def acc_dq(
    dq_ptrs_trans,
    start_m,
    stride_dqm,
    k,
    dqk_trans,
    alpha,
    mask_m,
    MAX_SEQ_LEN,
    LOCK,
    BLOCK_M: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(
        dq_ptrs_trans + start_m * stride_dqm,
        mask=mask_m[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans.to(k.dtype)
    tl.store(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_trans,
        mask=mask_m[None, :],
        eviction_policy="evict_last",
    )
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)  # pyre-ignore [61]
