// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

#include <ck_tile/core/numeric/integer.hpp>

// Allocates the split-KV workspace. Left null, the workspace comes from
// hipMallocAsync and goes back with hipFreeAsync; a framework can route it
// through its own caching allocator instead, which avoids contending on the
// device default memory pool.
struct HstuWorkspaceAllocator {
  void* (*alloc)(size_t bytes, hipStream_t stream) = nullptr;
  void (*free)(void* ptr, hipStream_t stream) = nullptr;
};

struct HstuAttentionNoGroupFwdParams {
  // for self-attention (is_cross_attention = false), we requires
  // 1) either seqlen_kv == 0 or seqlen_kv == seqlen_q
  // 2) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr ==
  // seq_q_offsets_ptr
  bool is_cross_attention;

  bool is_jagged;

  bool use_softmax;

  bool is_training;

  ck_tile::index_t num_batch;
  ck_tile::index_t seqlen_q; // batched mode only
  ck_tile::index_t seqlen_kv; // batched mode only
  const void* seq_q_offsets_ptr; // jagged mode only
  const void* seq_kv_offsets_ptr; // jagged mode only
  ck_tile::index_t max_seqlen_q; // jagged mode only
  ck_tile::index_t max_seqlen_kv; // jagged mode only
  ck_tile::index_t min_seqlen_q; // jagged mode only
  ck_tile::index_t min_seqlen_kv; // jagged mode only

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* bias_ptr;
  void* o_ptr;
  void* lse_ptr; // only used when both is_training and use_softmax be true

  ck_tile::index_t hdim_qk;
  ck_tile::index_t hdim_v;
  ck_tile::index_t num_head;
  float scale_s; // scaling factor exerted on the immediate Q@K result
  float attn_scale; // scaling factor exerted on the SiLU result

  ck_tile::index_t seq_stride_q;
  ck_tile::index_t seq_stride_k;
  ck_tile::index_t seq_stride_v;
  ck_tile::index_t seq_stride_bias;
  ck_tile::index_t seq_stride_o;
  ck_tile::index_t
      seq_stride_lse; // not needed if lse layout is [nhead, seqlen_q]

  ck_tile::index_t nhead_stride_q;
  ck_tile::index_t nhead_stride_k;
  ck_tile::index_t nhead_stride_v;
  ck_tile::index_t nhead_stride_bias;
  ck_tile::index_t nhead_stride_o;
  ck_tile::index_t
      nhead_stride_lse; // not needed if lse layout is [seqlen_q, nhead]

  // batched mode only parameters
  ck_tile::index_t batch_stride_q;
  ck_tile::index_t batch_stride_k;
  ck_tile::index_t batch_stride_v;
  ck_tile::index_t batch_stride_bias;
  ck_tile::index_t batch_stride_o;
  ck_tile::index_t batch_stride_lse;

  const void* num_targets_ptr;

  bool use_causal;
  // parameters used by Non-Group HSTU
  ck_tile::index_t window_size;
  ck_tile::index_t contextual_seqlen;
  ck_tile::index_t min_full_attn_seqlen;

  float p_drop;
  uint64_t philox_seed;
  uint64_t philox_offset;

  HstuWorkspaceAllocator workspace_allocator;

  // Device pointer to attn_scale, when the caller supplied it as a Tensor.
  // Reading it on the device avoids a blocking D2H copy per call; the scalar
  // `attn_scale` above stays as the fallback when this is null.
  const void* attn_scale_ptr = nullptr;

  // jagged mode only: total Q/KV tokens across the batch, known on the host
  // without a sync. 0 means unknown, which disables the strategy.
  ck_tile::index_t total_seqlen_q = 0;
  ck_tile::index_t total_seqlen_kv = 0;

  // jagged mode only: > 0 keeps kv_truncate_len keys of each sequence, the last
  // ones when kv_keep_last, else the first ones (forward only).
  ck_tile::index_t kv_truncate_len = 0;
  bool kv_keep_last = false;

  // Set by run_jagged_forward_dispatch from a strategy-chosen config; negative
  // leaves the choice to the dispatchers' heuristics. grid_order: 1
  // almost-invariant (batch-leading), 0 length-aware.
  int num_splits_override = -1;
  int grid_order_override = -1;
  const char* config_source = "heuristic";
  int config_rule = -1;
};

struct HstuAttentionGroupFwdParams {
  // for self-attention (is_cross_attention = false), we requires
  // 1) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr ==
  // seq_q_offsets_ptr
  bool is_cross_attention;

  bool use_softmax;

  bool is_training;

  ck_tile::index_t num_group;
  ck_tile::index_t num_batch;
  const void* seq_q_offsets_ptr;
  const void* seq_kv_offsets_ptr;
  ck_tile::index_t max_seqlen_q; // the maximum of all the groups' max_seqlen_q
  ck_tile::index_t
      max_seqlen_kv; // the maximum of all the groups' max_seqlen_kv
  ck_tile::index_t min_seqlen_q; // jagged mode only
  ck_tile::index_t min_seqlen_kv; // jagged mode only

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* bias_ptr;
  void* o_ptr;
  void* lse_ptr; // only used when both is_training and use_softmax be true

  ck_tile::index_t hdim_qk;
  ck_tile::index_t hdim_v;
  ck_tile::index_t num_head;
  float scale_s; // scaling factor exerted on the immediate Q@K result

  ck_tile::index_t seq_stride_q;
  ck_tile::index_t seq_stride_k;
  ck_tile::index_t seq_stride_v;
  ck_tile::index_t seq_stride_bias;
  ck_tile::index_t seq_stride_o;
  ck_tile::index_t
      seq_stride_lse; // not needed if lse layout is [nhead, seqlen_q]

  ck_tile::index_t nhead_stride_q;
  ck_tile::index_t nhead_stride_k;
  ck_tile::index_t nhead_stride_v;
  ck_tile::index_t nhead_stride_bias;
  ck_tile::index_t nhead_stride_o;
  ck_tile::index_t
      nhead_stride_lse; // not needed if lse layout is [seqlen_q, nhead]

  const void* num_targets_ptr;

  bool use_causal;

  // parameters used by Group HSTU
  const void* group_attn_scale_ptr;
  const void* group_max_seqlen_q_ptr; // use for setting attn_scales
  const void* group_window_size_ptr;
  const void* group_contextual_seqlen_ptr;
  const void* group_min_full_attn_seqlen_ptr;

  float p_drop;
  uint64_t philox_seed;
  uint64_t philox_offset;
  HstuWorkspaceAllocator workspace_allocator;
};

struct HstuGenerateRandUniformNumbersParams {
  bool is_jagged;

  void* rand_val_ptr;

  ck_tile::index_t num_batch;
  ck_tile::index_t seqlen_q; // batched mode only
  ck_tile::index_t seqlen_k; // batched mode only
  const void* seq_q_offsets_ptr; // jagged mode only
  const void* seq_k_offsets_ptr; // jagged mode only
  ck_tile::index_t max_seqlen_q; // jagged mode only

  ck_tile::index_t num_head;

  ck_tile::index_t stride_seqlen;
  ck_tile::index_t stride_nhead;
  ck_tile::index_t stride_batch; // batched mode only

  uint64_t philox_seed;
  uint64_t philox_offset;
};

struct HstuAttentionNoGroupBwdParams {
  // for self-attention (is_cross_attention = false), we requires
  // 1) either seqlen_kv == 0 or seqlen_kv == seqlen_q
  // 2) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr ==
  // seq_q_offsets_ptr
  bool is_cross_attention;

  bool is_jagged;

  bool use_softmax;

  ck_tile::index_t num_batch;
  ck_tile::index_t seqlen_q; // batched mode only
  ck_tile::index_t seqlen_kv; // batched mode only
  const void* seq_q_offsets_ptr; // jagged mode only
  const void* seq_kv_offsets_ptr; // jagged mode only
  ck_tile::index_t max_seqlen_q; // jagged mode only
  ck_tile::index_t
      max_seqlen_kv; // jagged mode only; needed for kernel 2 GridSize()
  ck_tile::index_t min_seqlen_q; // jagged mode only
  ck_tile::index_t min_seqlen_kv; // jagged mode only

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* bias_ptr;
  const void*
      o_ptr; // forward output, needed for D[sq] = dO row(.) O in softmax path
  const void* do_ptr; // upstream gradient
  const void* lse_ptr; // read-only in backward (was output in forward)
  void* dq_ptr;
  void* dk_ptr;
  void* dv_ptr;

  ck_tile::index_t hdim_qk;
  ck_tile::index_t hdim_v;
  ck_tile::index_t num_head;
  float scale_s; // scaling factor exerted on the immediate Q@K result
  float attn_scale; // scaling factor exerted on the SiLU result

  ck_tile::index_t seq_stride_q;
  ck_tile::index_t seq_stride_k;
  ck_tile::index_t seq_stride_v;
  ck_tile::index_t seq_stride_bias;
  ck_tile::index_t seq_stride_o;
  ck_tile::index_t seq_stride_do;
  ck_tile::index_t
      seq_stride_lse; // not needed if lse layout is [nhead, seqlen_q]
  ck_tile::index_t seq_stride_dq;
  ck_tile::index_t seq_stride_dk;
  ck_tile::index_t seq_stride_dv;

  ck_tile::index_t nhead_stride_q;
  ck_tile::index_t nhead_stride_k;
  ck_tile::index_t nhead_stride_v;
  ck_tile::index_t nhead_stride_bias;
  ck_tile::index_t nhead_stride_o;
  ck_tile::index_t nhead_stride_do;
  ck_tile::index_t
      nhead_stride_lse; // not needed if lse layout is [seqlen_q, nhead]
  ck_tile::index_t nhead_stride_dq;
  ck_tile::index_t nhead_stride_dk;
  ck_tile::index_t nhead_stride_dv;

  // batched mode only parameters
  ck_tile::index_t batch_stride_q;
  ck_tile::index_t batch_stride_k;
  ck_tile::index_t batch_stride_v;
  ck_tile::index_t batch_stride_bias;
  ck_tile::index_t batch_stride_o;
  ck_tile::index_t batch_stride_do;
  ck_tile::index_t batch_stride_lse;
  ck_tile::index_t batch_stride_dq; // set to zero in jagged mode
  ck_tile::index_t batch_stride_dk; // set to zero in jagged mode
  ck_tile::index_t batch_stride_dv; // set to zero in jagged mode

  const void* num_targets_ptr;

  bool use_causal;
  // parameters used by Non-Group HSTU
  ck_tile::index_t window_size;
  ck_tile::index_t contextual_seqlen;
  ck_tile::index_t min_full_attn_seqlen;

  // dropout parameters: used to replay the fwd dropout mask so that dS = P * dP
  // can be computed
  float p_drop;
  uint64_t philox_seed;
  uint64_t philox_offset;
};

struct HstuAttentionGroupBwdParams {
  // for self-attention (is_cross_attention = false), we requires
  // 1) either seq_kv_offsets_ptr == nullptr, or seq_kv_offsets_ptr ==
  // seq_q_offsets_ptr
  bool is_cross_attention;

  bool use_softmax;

  ck_tile::index_t num_group;
  ck_tile::index_t num_batch;
  const void* seq_q_offsets_ptr;
  const void* seq_kv_offsets_ptr;
  ck_tile::index_t max_seqlen_q; // the maximum of all the groups' max_seqlen_q
  ck_tile::index_t
      max_seqlen_kv; // the maximum of all the groups' max_seqlen_kv; needed for
                     // kernel 2 GridSize()
  ck_tile::index_t min_seqlen_q; // jagged mode only
  ck_tile::index_t min_seqlen_kv; // jagged mode only

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* bias_ptr;
  const void*
      o_ptr; // forward output, needed for D[sq] = dO row(.) O in softmax path
  const void* do_ptr; // upstream gradient
  const void* lse_ptr; // read-only in backward (was output in forward)
  void* dq_ptr;
  void* dk_ptr;
  void* dv_ptr;

  ck_tile::index_t hdim_qk;
  ck_tile::index_t hdim_v;
  ck_tile::index_t num_head;
  float scale_s; // scaling factor exerted on the immediate Q@K result

  ck_tile::index_t seq_stride_q;
  ck_tile::index_t seq_stride_k;
  ck_tile::index_t seq_stride_v;
  ck_tile::index_t seq_stride_bias;
  ck_tile::index_t seq_stride_o;
  ck_tile::index_t seq_stride_do;
  ck_tile::index_t
      seq_stride_lse; // not needed if lse layout is [nhead, seqlen_q]
  ck_tile::index_t seq_stride_dq;
  ck_tile::index_t seq_stride_dk;
  ck_tile::index_t seq_stride_dv;

  ck_tile::index_t nhead_stride_q;
  ck_tile::index_t nhead_stride_k;
  ck_tile::index_t nhead_stride_v;
  ck_tile::index_t nhead_stride_bias;
  ck_tile::index_t nhead_stride_o;
  ck_tile::index_t nhead_stride_do;
  ck_tile::index_t
      nhead_stride_lse; // not needed if lse layout is [seqlen_q, nhead]
  ck_tile::index_t nhead_stride_dq;
  ck_tile::index_t nhead_stride_dk;
  ck_tile::index_t nhead_stride_dv;

  const void* num_targets_ptr;

  bool use_causal;

  // parameters used by Group HSTU
  const void* group_attn_scale_ptr;
  const void* group_max_seqlen_q_ptr; // use for setting attn_scales
  const void* group_window_size_ptr;
  const void* group_contextual_seqlen_ptr;
  const void* group_min_full_attn_seqlen_ptr;

  // dropout parameters: used to replay the fwd dropout mask so that dS = P * dP
  // can be computed
  float p_drop;
  uint64_t philox_seed;
  uint64_t philox_offset;
};
