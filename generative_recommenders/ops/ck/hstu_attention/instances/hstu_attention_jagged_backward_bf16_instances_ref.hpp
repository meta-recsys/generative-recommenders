// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `generative_recommenders/ops/ck/hstu_attention/generate_bwd_instances.py`

#include <ck_tile/core/numeric/bfloat16.hpp>
#include "hstu_attention_jagged_backward_dispatch.hpp"
#include "hstu_attention_params.hpp"

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, true, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, true, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, true, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, true, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, false, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, false, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, false, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, false, 64>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, true, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, true, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, true, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, true, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, false, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, false, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, false, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, false, 96>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, true, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, true, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, true, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, true, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, false, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, false, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, false, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, false, 128>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, true, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, true, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, true, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, true, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, true, false, false, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, true, false, false, false, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, true, false, false, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);

extern template void
run_jagged_backward_dispatch<ck_tile::bf16_t, false, false, false, false, 256>(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream);
