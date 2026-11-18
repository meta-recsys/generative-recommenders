// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `generative_recommenders/ops/ck/hstu_attention/generate_bwd_instances.py`

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "hstu_attention_batched_backward_dispatch.hpp"
#include "hstu_attention_params.hpp"

template void
run_batched_backward_dispatch<ck_tile::bf16_t, false, false, false, true, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);
