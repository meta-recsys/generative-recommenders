// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `generative_recommenders/ops/ck/hstu_attention/generate_fwd_instances.py`

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "hstu_attention_batched_forward_dispatch.hpp"
#include "hstu_attention_params.hpp"

template void run_batched_forward_dispatch<
    ck_tile::bf16_t,
    true,
    true,
    true,
    false,
    true,
    256>(HstuAttentionNoGroupFwdParams& param, hipStream_t stream);
