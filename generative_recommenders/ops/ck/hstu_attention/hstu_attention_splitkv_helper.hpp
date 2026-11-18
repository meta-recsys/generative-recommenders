// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ck_tile/core/numeric/math.hpp>

#include <cstdlib>

#include "hstu_attention_host_util.hpp"

static int get_hstu_attention_fwd_mtile(
    int num_batches,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_kv,
    int window_size = 0) {
  int num_CUs = get_number_of_cu();
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  if (max_seqlen_q <= 64)
    return 64;

  // A tile wider than the window cannot attend to more data, it only adds
  // masked work.
  if (window_size > 0 && window_size < 128)
    return 64;

  // for cross-attention where max_seqlen_kv is much bigger than max_seqlen_q,
  // we always use mtile_size 128, not to worry about the CU coverage, since
  // split-kv can help us to solve
  if (max_seqlen_q >= 128 &&
      static_cast<float>(max_seqlen_kv) / max_seqlen_q >= 5.0)
    return 128;

  int nbatch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, 128);

  // assuming each CU is assigned two work-groups
  if (nbatch_nhead_mblocks >= static_cast<int>(0.85f * num_CUs * 2.0f))
    return 128;

  // currently, only hdim-128 actually uses mtile-64, for other hdim, the
  // settings for mtile-64 can be added through tuning/verification
  return 64;
};

static float get_estimated_cu_coverage_ratio(
    int num_batches,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_kv,
    int window_size = 0) {
  int num_CUs = get_number_of_cu();
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  int m_tile_size = get_hstu_attention_fwd_mtile(
      num_batches, num_heads, max_seqlen_q, max_seqlen_kv, window_size);

  int nbatch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, m_tile_size);

  // assume each CU can run two work-groups, common cases for hdim128
  return static_cast<float>(nbatch_nhead_mblocks) / (2.0f * num_CUs);
};

static bool shall_use_splitkv(
    int num_batches,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_kv) {
  // Please tune the threshold here
  const float threshold = (max_seqlen_kv >= 8192) ? 3.0f : 0.8f;

  if (get_estimated_cu_coverage_ratio(
          num_batches, num_heads, max_seqlen_q, max_seqlen_kv) < threshold)
    return true;
  return false;
};

static int get_suggested_num_splits(
    bool almost_invariant_seqlen,
    int num_batches,
    int num_heads,
    int max_seqlen_q,
    int max_seqlen_kv,
    int kv_tile_size = 0) {
  static const int forced_splits = []() {
    const char* env_p = std::getenv("HSTU_NUM_SPLITS");
    return env_p == nullptr ? 0 : atoi(env_p);
  }();
  if (forced_splits > 0)
    return ck_tile::min(forced_splits, 64);

  // When seqlen is not almost invariant, num_splits is used as the x dim of the
  // dispatched work-group grid, so stepping by 8 keeps work-groups of the same
  // i_split on the same xcd.
  const int step = almost_invariant_seqlen ? 1 : 8;
  int i = almost_invariant_seqlen ? 2 : 8;

  // Please tune the threshold here
  const float threshold = 3.0f;
  while (get_estimated_cu_coverage_ratio(
             num_batches, num_heads, max_seqlen_q, max_seqlen_kv) *
             i <
         threshold)
    i += step;

  // A split owning no KV tile still launches a workgroup, writes its o_acc
  // slice and is read back by the combine kernel. Two tiles per split also
  // keeps the main loop deep enough to prefetch.
  int max_splits = 64;
  if (kv_tile_size > 0) {
    const int num_kv_tiles = (max_seqlen_kv + kv_tile_size - 1) / kv_tile_size;
    max_splits = ck_tile::min(max_splits, ck_tile::max(1, num_kv_tiles / 2));
    i = ck_tile::min(i, max_splits);
  }

  // o_acc is laid out [batch, max_seqlen_q, num_head, num_splits, hdim_v], so
  // num_splits sets the per-row block size and a non-power-of-two count
  // misaligns every row. Ties round up, since rounding down costs parallelism
  // the shape already asked for, but never past the cap.
  int pow2 = 1;
  while (pow2 * 2 <= i)
    pow2 *= 2;
  if ((i - pow2) >= (pow2 * 2 - i) && pow2 * 2 <= max_splits)
    pow2 *= 2;

  // the num_splits shall not be bigger than 64
  return ck_tile::min(pow2, 64);
};

struct SplitkvWorkspace {
  int num_splits;
  void* o_acc_ptr;
  void* lse_acc_ptr; // only used when softmax is used
};

static void* splitkv_workspace_alloc(
    const HstuWorkspaceAllocator& allocator,
    size_t bytes,
    hipStream_t stream) {
  if (allocator.alloc != nullptr)
    return allocator.alloc(bytes, stream);

  void* ptr = nullptr;
  HIP_CHECK_ERROR(hipMallocAsync(&ptr, bytes, stream));
  return ptr;
}

static void splitkv_workspace_free(
    const HstuWorkspaceAllocator& allocator,
    void* ptr,
    hipStream_t stream) {
  if (allocator.free != nullptr)
    allocator.free(ptr, stream);
  else
    HIP_CHECK_ERROR(hipFreeAsync(ptr, stream));
}
