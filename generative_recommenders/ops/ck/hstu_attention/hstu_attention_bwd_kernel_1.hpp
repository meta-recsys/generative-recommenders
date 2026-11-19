// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include <string>
#include <type_traits>

#include "hstu_attention_bool_switch_return.hpp"
#include "hstu_attention_kernel_util.hpp"
#include "hstu_block_masking.hpp"

#ifndef HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM
#define HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM 1
#endif

// Backward Kernel 1: computes dQ for each (batch, head, sq_tile).
// For the softmax path it also computes D[sq] = dO row(.) O and writes it to
// delta_ptr. Grid: one workgroup per (batch, head, sq_tile) -- same as the
// forward kernel.

namespace ck_tile {

template <typename HstuAttentionBwdPipeline_, typename EpiloguePipeline_>
struct HstuAttentionBwdKernel1 {
  using HstuAttentionBwdPipeline = remove_cvref_t<HstuAttentionBwdPipeline_>;
  using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

  static constexpr index_t kBlockSize = HstuAttentionBwdPipeline::kBlockSize;
  static constexpr index_t kBlockPerCu = HstuAttentionBwdPipeline::kBlockPerCu;
  static_assert(kBlockPerCu > 0);

  using QKVDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::QKVDataType>;
  using BiasDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::BiasDataType>;
  using OGradDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::OGradDataType>;
  using QGradAccDataType = remove_cvref_t<
      typename HstuAttentionBwdPipeline::Problem::QGradAccDataType>;
  using QGradDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::QGradDataType>;
  using CompDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::CompDataType>;

  static constexpr bool kIsCrossAttention =
      HstuAttentionBwdPipeline::Problem::kIsCrossAttention;
  static constexpr bool kUseGroup =
      HstuAttentionBwdPipeline::Problem::kUseGroup;
  static constexpr bool kIsJagged =
      HstuAttentionBwdPipeline::Problem::kIsJagged;
  static constexpr bool kHasBias = HstuAttentionBwdPipeline::Problem::kHasBias;
  static constexpr bool kHasCausal =
      HstuAttentionBwdPipeline::Problem::kHasCausal;
  static constexpr bool kUseSoftmax =
      HstuAttentionBwdPipeline::Problem::kUseSoftmax;
  static constexpr bool kHasDropout =
      HstuAttentionBwdPipeline::Problem::kHasDropout;

  static constexpr bool kPadSeqLenQ = HstuAttentionBwdPipeline::kPadSeqLenQ;
  static constexpr bool kPadSeqLenK = HstuAttentionBwdPipeline::kPadSeqLenK;
  static constexpr bool kPadHeadDimQK = HstuAttentionBwdPipeline::kPadHeadDimQK;
  static constexpr bool kPadHeadDimV = HstuAttentionBwdPipeline::kPadHeadDimV;

  static constexpr bool kUseNoGroupBatched = (!kUseGroup && !kIsJagged);
  static constexpr bool kUseNoGroupJagged = (!kUseGroup && kIsJagged);

  // -------------------------------------------------------------------------
  // Kargs helpers
  // -------------------------------------------------------------------------

  template <index_t I>
  struct EmptyKargs {};

  // --- Base kargs shared by batched and jagged no-group mode ---

  struct HstuBwdKernel1NoGroupBatchedBaseKargs {
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_dq;

    const int32_t* num_targets_ptr;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr; // forward output -- needed for D[sq] in softmax path
    const void* do_ptr;
    void* dq_ptr;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_dq;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_kv;
    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t seq_stride_dq;

    ck_tile::index_t num_head;
    float scale_s; // scaling value exerted on the immediate Q@K result
    float scale_p; // scaling value exerted on the SiLU result

    bool almost_invariant_seqlen; // should always be true for batched mode

    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t window_size;
    ck_tile::index_t min_full_attn_seqlen;
  };

  struct HstuBwdKernel1NoGroupJaggedBaseKargs {
    const int32_t* seq_q_offsets_ptr;
    const int32_t* seq_kv_offsets_ptr;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t seq_stride_dq;

    const int32_t* num_targets_ptr;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr;
    const void* do_ptr;
    void* dq_ptr;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_dq;

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_kv;

    ck_tile::index_t num_head;
    float scale_s; // scaling value exerted on the immediate Q@K result
    float scale_p; // scaling value exerted on the SiLU result

    bool almost_invariant_seqlen;

    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t window_size;
    ck_tile::index_t min_full_attn_seqlen;
  };

  struct HstuBwdKernel1GroupBaseKargs {
    ck_tile::index_t num_batch_per_group;

    const int32_t* seq_q_offsets_ptr;
    const int32_t* seq_kv_offsets_ptr;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t seq_stride_dq;

    const int32_t* num_targets_ptr;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* o_ptr;
    const void* do_ptr;
    void* dq_ptr;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_dq;

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;

    ck_tile::index_t seqlen_q; // set at runtime from seq_q_offsets_ptr
    ck_tile::index_t seqlen_kv; // set at runtime from seq_kv_offsets_ptr

    ck_tile::index_t num_head;
    float scale_s;
    float scale_p; // set at runtime from group_attn_scale_ptr

    bool almost_invariant_seqlen;

    int32_t
        contextual_seqlen; // set at runtime from group_contextual_seqlen_ptr
    int32_t window_size; // set at runtime from group_window_size_ptr
    int32_t min_full_attn_seqlen; // set at runtime from
                                  // group_min_full_attn_seqlen_ptr

    const int32_t* group_max_seqlen_q_ptr;
    const int32_t* group_contextual_seqlen_ptr;
    const int32_t* group_window_size_ptr;
    const int32_t* group_min_full_attn_seqlen_ptr;
    const float* group_attn_scale_ptr;
  };

  struct HstuBwdKernel1BatchedBiasKargs {
    const void* bias_ptr;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t batch_stride_bias;
  };

  struct HstuBwdKernel1JaggedBiasKargs {
    const void* bias_ptr;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t nhead_stride_bias;
  };

  struct HstuBwdKernel1BatchedLSEDeltaKargs {
    const void* lse_ptr; // read-only in kernel 1; written by the forward pass
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t nhead_stride_lse;

    void* delta_ptr; // written by kernel 1 (D[sq] = dO row(.) O)
    ck_tile::index_t batch_stride_delta;
    ck_tile::index_t seq_stride_delta;
    ck_tile::index_t nhead_stride_delta;
  };

  struct HstuBwdKernel1JaggedLSEDeltaKargs {
    const void* lse_ptr; // read-only in kernel 1; written by the forward pass
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t nhead_stride_lse;

    void* delta_ptr; // written by kernel 1 (D[sq] = dO row(.) O)
    ck_tile::index_t seq_stride_delta;
    ck_tile::index_t nhead_stride_delta;
  };

  struct HstuBwdKernel1CommonDropoutKargs {
    void init_dropout(float p_drop, uint64_t seed, uint64_t offset) {
      float p_undrop = 1.0 - p_drop;
      p_undrop_in_uint8_t =
          uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
      rp_undrop = 1.0 / p_undrop;

      this->drop_seed = seed;
      this->drop_offset = offset;
    }

    // used for generating random numbers
    uint64_t drop_seed;
    uint64_t drop_offset;

    float rp_undrop = 1;
    uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
  };

  struct HstuBwdKernel1NoGroupBatchedKargs
      : HstuBwdKernel1NoGroupBatchedBaseKargs,
        std::conditional_t<
            kHasBias,
            HstuBwdKernel1BatchedBiasKargs,
            EmptyKargs<1>>,
        std::conditional_t<
            kUseSoftmax,
            HstuBwdKernel1BatchedLSEDeltaKargs,
            EmptyKargs<2>>,
        std::conditional_t<
            kHasDropout,
            HstuBwdKernel1CommonDropoutKargs,
            EmptyKargs<3>> {};

  struct HstuBwdKernel1NoGroupJaggedKargs
      : HstuBwdKernel1NoGroupJaggedBaseKargs,
        std::conditional_t<
            kHasBias,
            HstuBwdKernel1JaggedBiasKargs,
            EmptyKargs<1>>,
        std::conditional_t<
            kUseSoftmax,
            HstuBwdKernel1JaggedLSEDeltaKargs,
            EmptyKargs<2>>,
        std::conditional_t<
            kHasDropout,
            HstuBwdKernel1CommonDropoutKargs,
            EmptyKargs<3>> {};

  struct HstuBwdKernel1GroupKargs : HstuBwdKernel1GroupBaseKargs,
                                    std::conditional_t<
                                        kHasBias,
                                        HstuBwdKernel1JaggedBiasKargs,
                                        EmptyKargs<1>>,
                                    std::conditional_t<
                                        kUseSoftmax,
                                        HstuBwdKernel1JaggedLSEDeltaKargs,
                                        EmptyKargs<2>>,
                                    std::conditional_t<
                                        kHasDropout,
                                        HstuBwdKernel1CommonDropoutKargs,
                                        EmptyKargs<3>> {};

  using Kargs = std::conditional_t<
      kUseGroup,
      HstuBwdKernel1GroupKargs,
      std::conditional_t<
          kIsJagged,
          HstuBwdKernel1NoGroupJaggedKargs,
          HstuBwdKernel1NoGroupBatchedKargs>>;

  // -------------------------------------------------------------------------
  // MakeKargs factory functions
  // -------------------------------------------------------------------------

  // Overload 1: NoGroup + Batched (kUseNoGroupBatched == true)
  template <bool Cond = kUseNoGroupBatched>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      const void* bias_ptr,
      const void* o_ptr,
      const void* do_ptr,
      void* dq_ptr,
      const void* lse_ptr,
      void* delta_ptr,
      ck_tile::index_t seqlen_q,
      ck_tile::index_t seqlen_kv,
      ck_tile::index_t hdim_qk,
      ck_tile::index_t hdim_v,
      ck_tile::index_t num_head,
      float scale_s,
      float attn_scale,
      ck_tile::index_t seq_stride_q,
      ck_tile::index_t seq_stride_k,
      ck_tile::index_t seq_stride_v,
      ck_tile::index_t seq_stride_o,
      ck_tile::index_t seq_stride_do,
      ck_tile::index_t seq_stride_dq,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t seq_stride_delta,
      ck_tile::index_t seq_stride_bias,
      ck_tile::index_t nhead_stride_q,
      ck_tile::index_t nhead_stride_k,
      ck_tile::index_t nhead_stride_v,
      ck_tile::index_t nhead_stride_o,
      ck_tile::index_t nhead_stride_do,
      ck_tile::index_t nhead_stride_dq,
      ck_tile::index_t nhead_stride_lse,
      ck_tile::index_t nhead_stride_delta,
      ck_tile::index_t nhead_stride_bias,
      ck_tile::index_t batch_stride_q,
      ck_tile::index_t batch_stride_k,
      ck_tile::index_t batch_stride_v,
      ck_tile::index_t batch_stride_o,
      ck_tile::index_t batch_stride_do,
      ck_tile::index_t batch_stride_dq,
      ck_tile::index_t batch_stride_lse,
      ck_tile::index_t batch_stride_delta,
      ck_tile::index_t batch_stride_bias,
      const void* num_targets_ptr,
      ck_tile::index_t contextual_seqlen,
      ck_tile::index_t window_size,
      ck_tile::index_t min_full_attn_seqlen,
      float p_drop,
      uint64_t philox_seed,
      uint64_t philox_offset) {
    Kargs kargs{
        {batch_stride_q,
         batch_stride_k,
         batch_stride_v,
         batch_stride_o,
         batch_stride_do,
         batch_stride_dq,
         reinterpret_cast<const int32_t*>(num_targets_ptr),
         q_ptr,
         k_ptr,
         v_ptr,
         o_ptr,
         do_ptr,
         dq_ptr,
         nhead_stride_q,
         nhead_stride_k,
         nhead_stride_v,
         nhead_stride_o,
         nhead_stride_do,
         nhead_stride_dq,
         seqlen_q,
         seqlen_kv,
         hdim_qk,
         hdim_v,
         seq_stride_q,
         seq_stride_k,
         seq_stride_v,
         seq_stride_o,
         seq_stride_do,
         seq_stride_dq,
         num_head,
         scale_s,
         attn_scale ? attn_scale : 1.0f / static_cast<float>(seqlen_q),
         true, // almost_invariant_seqlen
         contextual_seqlen,
         window_size,
         min_full_attn_seqlen}, // base kargs
        {}, // placeholder for bias
        {}, // placeholder for lse_delta
    };

    if constexpr (kHasBias) {
      kargs.bias_ptr = bias_ptr;
      kargs.seq_stride_bias = seq_stride_bias;
      kargs.nhead_stride_bias = nhead_stride_bias;
      kargs.batch_stride_bias = batch_stride_bias;
    }

    if constexpr (kUseSoftmax) {
      kargs.lse_ptr = lse_ptr;
      kargs.batch_stride_lse = batch_stride_lse;
      kargs.seq_stride_lse = seq_stride_lse;
      kargs.nhead_stride_lse = nhead_stride_lse;

      kargs.delta_ptr = delta_ptr;
      kargs.batch_stride_delta = batch_stride_delta;
      kargs.seq_stride_delta = seq_stride_delta;
      kargs.nhead_stride_delta = nhead_stride_delta;
    }

    if constexpr (kHasDropout) {
      kargs.init_dropout(p_drop, philox_seed, philox_offset);
    }

    return kargs;
  }

  // Overload 2: NoGroup + Jagged (kUseNoGroupJagged == true)
  template <bool Cond = kUseNoGroupJagged>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      const void* bias_ptr,
      const void* o_ptr,
      const void* do_ptr,
      void* dq_ptr,
      const void* lse_ptr,
      void* delta_ptr,
      const void* seq_q_offsets_ptr,
      const void* seq_kv_offsets_ptr,
      ck_tile::index_t max_seqlen_q,
      ck_tile::index_t hdim_qk,
      ck_tile::index_t hdim_v,
      ck_tile::index_t num_head,
      float scale_s,
      float attn_scale,
      bool almost_invariant_seqlen,
      ck_tile::index_t seq_stride_q,
      ck_tile::index_t seq_stride_k,
      ck_tile::index_t seq_stride_v,
      ck_tile::index_t seq_stride_o,
      ck_tile::index_t seq_stride_do,
      ck_tile::index_t seq_stride_dq,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t seq_stride_delta,
      ck_tile::index_t seq_stride_bias,
      ck_tile::index_t nhead_stride_q,
      ck_tile::index_t nhead_stride_k,
      ck_tile::index_t nhead_stride_v,
      ck_tile::index_t nhead_stride_o,
      ck_tile::index_t nhead_stride_do,
      ck_tile::index_t nhead_stride_dq,
      ck_tile::index_t nhead_stride_lse,
      ck_tile::index_t nhead_stride_delta,
      ck_tile::index_t nhead_stride_bias,
      const void* num_targets_ptr,
      ck_tile::index_t contextual_seqlen,
      ck_tile::index_t window_size,
      ck_tile::index_t min_full_attn_seqlen,
      float p_drop,
      uint64_t philox_seed,
      uint64_t philox_offset) {
    Kargs kargs{
        {reinterpret_cast<const int32_t*>(seq_q_offsets_ptr),
         reinterpret_cast<const int32_t*>(seq_kv_offsets_ptr),
         seq_stride_q,
         seq_stride_k,
         seq_stride_v,
         seq_stride_o,
         seq_stride_do,
         seq_stride_dq,
         reinterpret_cast<const int32_t*>(num_targets_ptr),
         q_ptr,
         k_ptr,
         v_ptr,
         o_ptr,
         do_ptr,
         dq_ptr,
         nhead_stride_q,
         nhead_stride_k,
         nhead_stride_v,
         nhead_stride_o,
         nhead_stride_do,
         nhead_stride_dq,
         hdim_qk,
         hdim_v,
         -1, // seqlen_q: set at runtime from seq_q_offsets_ptr
         -1, // seqlen_kv: set at runtime from seq_kv_offsets_ptr
         num_head,
         scale_s,
         attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q),
         almost_invariant_seqlen,
         contextual_seqlen,
         window_size,
         min_full_attn_seqlen}, // base kargs
        {}, // placeholder for bias
        {}, // placeholder for lse_delta
    };

    if constexpr (kHasBias) {
      kargs.bias_ptr = bias_ptr;
      kargs.seq_stride_bias = seq_stride_bias;
      kargs.nhead_stride_bias = nhead_stride_bias;
    }

    if constexpr (kUseSoftmax) {
      kargs.lse_ptr = lse_ptr;
      kargs.seq_stride_lse = seq_stride_lse;
      kargs.nhead_stride_lse = nhead_stride_lse;

      kargs.delta_ptr = delta_ptr;
      kargs.seq_stride_delta = seq_stride_delta;
      kargs.nhead_stride_delta = nhead_stride_delta;
    }

    if constexpr (kHasDropout) {
      kargs.init_dropout(p_drop, philox_seed, philox_offset);
    }

    return kargs;
  }

  // Overload 3: Group (kUseGroup == true)
  template <bool Cond = kUseGroup>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      const void* bias_ptr,
      const void* o_ptr,
      const void* do_ptr,
      void* dq_ptr,
      const void* lse_ptr,
      void* delta_ptr,
      ck_tile::index_t num_batch_per_group,
      const void* seq_q_offsets_ptr,
      const void* seq_kv_offsets_ptr,
      const void* group_max_seqlen_q_ptr,
      const void* group_contextual_seqlen_ptr,
      const void* group_window_size_ptr,
      const void* group_min_full_attn_seqlen_ptr,
      const void* group_attn_scale_ptr,
      ck_tile::index_t hdim_qk,
      ck_tile::index_t hdim_v,
      ck_tile::index_t num_head,
      float scale_s,
      bool almost_invariant_seqlen,
      ck_tile::index_t seq_stride_q,
      ck_tile::index_t seq_stride_k,
      ck_tile::index_t seq_stride_v,
      ck_tile::index_t seq_stride_o,
      ck_tile::index_t seq_stride_do,
      ck_tile::index_t seq_stride_dq,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t seq_stride_delta,
      ck_tile::index_t seq_stride_bias,
      ck_tile::index_t nhead_stride_q,
      ck_tile::index_t nhead_stride_k,
      ck_tile::index_t nhead_stride_v,
      ck_tile::index_t nhead_stride_o,
      ck_tile::index_t nhead_stride_do,
      ck_tile::index_t nhead_stride_dq,
      ck_tile::index_t nhead_stride_lse,
      ck_tile::index_t nhead_stride_delta,
      ck_tile::index_t nhead_stride_bias,
      const void* num_targets_ptr,
      float p_drop,
      uint64_t philox_seed,
      uint64_t philox_offset) {
    Kargs kargs{
        {num_batch_per_group,
         reinterpret_cast<const int32_t*>(seq_q_offsets_ptr),
         reinterpret_cast<const int32_t*>(seq_kv_offsets_ptr),
         seq_stride_q,
         seq_stride_k,
         seq_stride_v,
         seq_stride_o,
         seq_stride_do,
         seq_stride_dq,
         reinterpret_cast<const int32_t*>(num_targets_ptr),
         q_ptr,
         k_ptr,
         v_ptr,
         o_ptr,
         do_ptr,
         dq_ptr,
         nhead_stride_q,
         nhead_stride_k,
         nhead_stride_v,
         nhead_stride_o,
         nhead_stride_do,
         nhead_stride_dq,
         hdim_qk,
         hdim_v,
         -1, // seqlen_q: set at runtime from seq_q_offsets_ptr
         -1, // seqlen_kv: set at runtime from seq_kv_offsets_ptr
         num_head,
         scale_s,
         1.0f, // scale_p: set at runtime from group_attn_scale_ptr
         almost_invariant_seqlen,
         0, // contextual_seqlen: set at runtime from
            // group_contextual_seqlen_ptr
         0, // window_size: set at runtime from group_window_size_ptr
         0, // min_full_attn_seqlen: set at runtime from
            // group_min_full_attn_seqlen_ptr
         reinterpret_cast<const int32_t*>(group_max_seqlen_q_ptr),
         reinterpret_cast<const int32_t*>(group_contextual_seqlen_ptr),
         reinterpret_cast<const int32_t*>(group_window_size_ptr),
         reinterpret_cast<const int32_t*>(group_min_full_attn_seqlen_ptr),
         reinterpret_cast<const float*>(group_attn_scale_ptr)}, // base kargs
        {}, // placeholder for bias
        {}, // placeholder for lse_delta
    };

    if constexpr (kHasBias) {
      kargs.bias_ptr = bias_ptr;
      kargs.seq_stride_bias = seq_stride_bias;
      kargs.nhead_stride_bias = nhead_stride_bias;
    }

    if constexpr (kUseSoftmax) {
      kargs.lse_ptr = lse_ptr;
      kargs.seq_stride_lse = seq_stride_lse;
      kargs.nhead_stride_lse = nhead_stride_lse;

      kargs.delta_ptr = delta_ptr;
      kargs.seq_stride_delta = seq_stride_delta;
      kargs.nhead_stride_delta = nhead_stride_delta;
    }

    if constexpr (kHasDropout) {
      kargs.init_dropout(p_drop, philox_seed, philox_offset);
    }

    return kargs;
  }

  // -------------------------------------------------------------------------
  // Grid / block sizing
  // -------------------------------------------------------------------------

  CK_TILE_HOST static constexpr auto GridSize(
      ck_tile::index_t batch_size,
      ck_tile::index_t nhead,
      ck_tile::index_t seqlen_q,
      bool almost_invariant_seqlen,
      bool has_minfull_attn_seqlen = false) {
    // Q sequence may be split into two parts when min_full_attn_seqlen > 0:
    //   [0, seqlen_q - num_target - min_full_attn_seqlen)  -- first split
    //   [seqlen_q - num_target - min_full_attn_seqlen, seqlen_q)  -- second
    //   split
    // An extra sentinel tile is added to cover the second split.
    ck_tile::index_t num_tile_m =
        ck_tile::integer_divide_ceil(seqlen_q, HstuAttentionBwdPipeline::kM0);

    // when kHasDropout is false, the second split starts just from position
    // seqlen_q - num_target - min_full_attn_seqlen, so both the first split and
    // the second split could have a incomplete last tile, thus one additional
    // workgroup should be allocated for each seqlen_q
    if constexpr (!kHasDropout) {
      if constexpr (kUseGroup)
        num_tile_m += 1; // extra sentinel tile: groups have variable seqlen_q
      else {
        if (has_minfull_attn_seqlen)
          num_tile_m += 1;
      }
    }

    if (almost_invariant_seqlen)
      return dim3(batch_size, nhead, num_tile_m);
    else
      return dim3(num_tile_m, nhead, batch_size);
  }

  CK_TILE_HOST static constexpr auto BlockSize() {
    if (is_wave32())
      return dim3(kBlockSize / get_warp_size() * 32);
    else
      return dim3(kBlockSize);
  }

  CK_TILE_DEVICE static constexpr index_t GetSmemSize() {
    return ck_tile::max(
        HstuAttentionBwdPipeline::GetSmemSize(),
        EpiloguePipeline::GetSmemSize());
  }

  // -------------------------------------------------------------------------
  // Tile index helper
  // -------------------------------------------------------------------------

  CK_TILE_DEVICE static auto GetTileIndex(const Kargs& kargs) {
    if (kargs.almost_invariant_seqlen) {
      const index_t i_batch = blockIdx.x;
      const index_t i_nhead = blockIdx.y;
      const index_t i_tile_m =
          gridDim.z - 1 - blockIdx.z; // reverse so tile 0 is last
                                      //
      return make_tuple(i_tile_m, i_nhead, i_batch);
    } else {
      const index_t i_tile_m = blockIdx.x;
      const index_t i_nhead = blockIdx.y;
      const index_t i_batch = blockIdx.z;

      return make_tuple(i_tile_m, i_nhead, i_batch);
    }
  }

  // -------------------------------------------------------------------------
  // Device operator
  // -------------------------------------------------------------------------

  CK_TILE_DEVICE void operator()(Kargs kargs) const {
    __shared__ char smem_ptr[GetSmemSize()];

    const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex(kargs);

    // ---- Batch offsets ----
    long_index_t batch_offset_q = 0;
    long_index_t batch_offset_k = 0;
    long_index_t batch_offset_v = 0;
    long_index_t batch_offset_o = 0;
    long_index_t batch_offset_do = 0;
    long_index_t batch_offset_dq = 0;
    long_index_t batch_offset_lse = 0;
    long_index_t batch_offset_delta = 0;
    long_index_t batch_offset_bias = 0;

    if constexpr (kIsJagged) {
      const long_index_t query_start = kargs.seq_q_offsets_ptr[i_batch];
      const long_index_t key_start = kargs.seq_kv_offsets_ptr[i_batch];

      batch_offset_q = query_start * kargs.seq_stride_q;
      batch_offset_k = key_start * kargs.seq_stride_k;
      batch_offset_v = key_start * kargs.seq_stride_v;
      batch_offset_o = query_start * kargs.seq_stride_o;
      batch_offset_do = query_start * kargs.seq_stride_do;
      batch_offset_dq = query_start * kargs.seq_stride_dq;

      if constexpr (kUseSoftmax) {
        batch_offset_lse = query_start * kargs.seq_stride_lse;
        batch_offset_delta = query_start * kargs.seq_stride_delta;
      }
      if constexpr (kHasBias)
        batch_offset_bias = query_start * kargs.seq_stride_bias;

      kargs.seqlen_q = kargs.seq_q_offsets_ptr[i_batch + 1] -
          kargs.seq_q_offsets_ptr[i_batch];
      kargs.seqlen_kv = kargs.seq_kv_offsets_ptr[i_batch + 1] -
          kargs.seq_kv_offsets_ptr[i_batch];

      // read per-group mask and scaling parameters from device memory
      if constexpr (kUseGroup) {
        const index_t i_group =
            __builtin_amdgcn_readfirstlane(i_batch / kargs.num_batch_per_group);
        const float attn_scale = kargs.group_attn_scale_ptr[i_group];
        const index_t max_seqlen_q = kargs.group_max_seqlen_q_ptr[i_group];
        kargs.scale_p =
            attn_scale ? attn_scale : 1.0f / static_cast<float>(max_seqlen_q);
        kargs.contextual_seqlen = kargs.group_contextual_seqlen_ptr[i_group];
        kargs.window_size = kargs.group_window_size_ptr[i_group];
        kargs.min_full_attn_seqlen =
            kargs.group_min_full_attn_seqlen_ptr[i_group];
      }
    } else {
      batch_offset_q =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
      batch_offset_k =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
      batch_offset_v =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
      batch_offset_o =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
      batch_offset_do =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
      batch_offset_dq =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq;
      if constexpr (kUseSoftmax) {
        batch_offset_lse =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
        batch_offset_delta =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_delta;
      }
      if constexpr (kHasBias)
        batch_offset_bias =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
    }

    const int num_target =
        (kargs.num_targets_ptr == nullptr) ? 0 : kargs.num_targets_ptr[i_batch];

    index_t seqlen_in_upper_scope = kargs.seqlen_q;
    bool is_tile_in_upper_scope = true;
    index_t i_m0;

    // 1) when kHasDropout is false, the second split starts just from position
    // seqlen_q - num_target - min_full_attn_seqlen; 2) when kHasDropout is
    // true, the second split must start from the first kM0 aligned position
    // bigger or equal to seqlen_q - num_target - min_full_attn_seqlen, since
    // BlockDropout requires each workgroup has a kM0 aligned i_m0 position
    if (kargs.min_full_attn_seqlen > 0) {
      if (kargs.seqlen_q - num_target > kargs.min_full_attn_seqlen) {
        seqlen_in_upper_scope =
            kargs.seqlen_q - num_target - kargs.min_full_attn_seqlen;

        const index_t num_tile_in_upper_scope = __builtin_amdgcn_readfirstlane(
            ck_tile::integer_divide_ceil(
                seqlen_in_upper_scope, HstuAttentionBwdPipeline::kM0));

        is_tile_in_upper_scope = (i_tile_m < num_tile_in_upper_scope);

        if constexpr (!kHasDropout) {
          // be careful that i_m0 for second_split could be not aligned on kM0
          i_m0 = is_tile_in_upper_scope
              ? __builtin_amdgcn_readfirstlane(
                    i_tile_m * HstuAttentionBwdPipeline::kM0)
              : __builtin_amdgcn_readfirstlane(
                    (i_tile_m - num_tile_in_upper_scope) *
                    HstuAttentionBwdPipeline::kM0) +
                  seqlen_in_upper_scope;
        } else
          i_m0 = __builtin_amdgcn_readfirstlane(
              i_tile_m * HstuAttentionBwdPipeline::kM0);
      } else {
        seqlen_in_upper_scope = 0;
        is_tile_in_upper_scope = false;

        // adjust min_full_attn_seqlen passed to HstuBlockMask constructor
        kargs.min_full_attn_seqlen = kargs.seqlen_q - num_target;

        i_m0 = __builtin_amdgcn_readfirstlane(
            i_tile_m * HstuAttentionBwdPipeline::kM0);
      }
    } else
      i_m0 = __builtin_amdgcn_readfirstlane(
          i_tile_m * HstuAttentionBwdPipeline::kM0);

    index_t seqlen_q_in_ctrl = kHasDropout
        ? kargs.seqlen_q
        : (is_tile_in_upper_scope ? seqlen_in_upper_scope : kargs.seqlen_q);

    if (seqlen_q_in_ctrl <= i_m0)
      return;

    // ---- Data pointers ----
    const QKVDataType* q_ptr =
        reinterpret_cast<const QKVDataType*>(kargs.q_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
        batch_offset_q;
    const QKVDataType* k_ptr =
        reinterpret_cast<const QKVDataType*>(kargs.k_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_k +
        batch_offset_k;
    const QKVDataType* v_ptr =
        reinterpret_cast<const QKVDataType*>(kargs.v_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_v +
        batch_offset_v;
    const QKVDataType* o_ptr =
        reinterpret_cast<const QKVDataType*>(kargs.o_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
        batch_offset_o;
    const OGradDataType* do_ptr =
        reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
        batch_offset_do;
    QGradDataType* dq_ptr = reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq +
        batch_offset_dq;

    // ---- DRAM views ----
    const auto q_dram = [&]() {
      const auto naive = make_naive_tensor_view<address_space_enum::global>(
          q_ptr,
          make_tuple(seqlen_q_in_ctrl, kargs.hdim_qk),
          make_tuple(kargs.seq_stride_q, 1),
          number<HstuAttentionBwdPipeline::kAlignmentQ>{},
          number<1>{});
      return pad_tensor_view(
          naive,
          make_tuple(
              number<HstuAttentionBwdPipeline::kM0>{},
              number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
          sequence<false, kPadHeadDimQK>{});
    }();

    const auto k_dram = [&]() {
      const auto k_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              k_ptr,
              make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
              make_tuple(kargs.seq_stride_k, 1),
              number<HstuAttentionBwdPipeline::kAlignmentK>{},
              number<1>{});

      if constexpr (detail::is_using_tdm_v<HstuAttentionBwdPipeline>) {
        return k_dram_naive;
      } else {
        const auto k_dram_pad = pad_tensor_view(
            k_dram_naive,
            make_tuple(
                number<HstuAttentionBwdPipeline::kN0>{},
                number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
            sequence<false, kPadHeadDimQK>{});

        if constexpr (detail::is_direct_lds_pipeline_v<
                          HstuAttentionBwdPipeline>)
          return detail::make_direct_lds_xor_dram_view<
              HstuAttentionBwdPipeline::kQKHeaddim,
              HstuAttentionBwdPipeline::Policy::
                  template GetKDirectLdsSwizzleUnit<
                      typename HstuAttentionBwdPipeline::Problem>()>(
              k_dram_pad);
        else
          return k_dram_pad;
      }
    }();

    // V stored as [seqlen_kv, hdim_v]; tile shape [kN0, kVHeaddim]
    const auto v_dram = [&]() {
      const auto v_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              v_ptr,
              make_tuple(kargs.seqlen_kv, kargs.hdim_v),
              make_tuple(kargs.seq_stride_v, 1),
              number<HstuAttentionBwdPipeline::kAlignmentV>{},
              number<1>{});

      if constexpr (detail::is_using_tdm_v<HstuAttentionBwdPipeline>) {
        return v_dram_naive;
      } else {
        const auto v_dram_pad = pad_tensor_view(
            v_dram_naive,
            make_tuple(
                number<HstuAttentionBwdPipeline::kN0>{},
                number<HstuAttentionBwdPipeline::kVHeaddim>{}),
            sequence<false, kPadHeadDimV>{});

        if constexpr (detail::is_direct_lds_pipeline_v<
                          HstuAttentionBwdPipeline>)
          return detail::make_direct_lds_xor_dram_view<
              HstuAttentionBwdPipeline::kVHeaddim,
              HstuAttentionBwdPipeline::Policy::
                  template GetVDirectLdsSwizzleUnit<
                      typename HstuAttentionBwdPipeline::Problem>()>(
              v_dram_pad);
        else
          return v_dram_pad;
      }
    }();

    // dO -- [seqlen_q_in_ctrl, hdim_v]; tile shape [kM0, kVHeaddim]
    const auto do_dram = [&]() {
      const auto naive = make_naive_tensor_view<address_space_enum::global>(
          do_ptr,
          make_tuple(seqlen_q_in_ctrl, kargs.hdim_v),
          make_tuple(kargs.seq_stride_do, 1),
          number<HstuAttentionBwdPipeline::kAlignmentOGrad>{},
          number<1>{});
      return pad_tensor_view(
          naive,
          make_tuple(
              number<HstuAttentionBwdPipeline::kM0>{},
              number<HstuAttentionBwdPipeline::kVHeaddim>{}),
          sequence<false, kPadHeadDimV>{});
    }();

    // ---- DRAM windows ----
    auto q_dram_window = make_tile_window(
        q_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kM0>{},
            number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
        {i_m0, 0});

    auto k_dram_window = make_tile_window(
        k_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kN0>{},
            number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
        {0, 0});

    auto v_dram_window = make_tile_window(
        v_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kN0>{},
            number<HstuAttentionBwdPipeline::kVHeaddim>{}),
        {0, 0});

    // dO DRAM window -- first parameter to pipeline operator() per spec
    auto do_dram_window = make_tile_window(
        do_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kM0>{},
            number<HstuAttentionBwdPipeline::kVHeaddim>{}),
        {i_m0, 0});

    // Bias DRAM window
    const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
      constexpr auto lengths = make_tuple(
          number<HstuAttentionBwdPipeline::kM0>{},
          number<HstuAttentionBwdPipeline::kN0>{});
      if constexpr (kHasBias) {
        const BiasDataType* bias_ptr =
            reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
            static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
            batch_offset_bias;
        const auto bias_dram_naive =
            make_naive_tensor_view<address_space_enum::global>(
                bias_ptr,
                make_tuple(seqlen_q_in_ctrl, kargs.seqlen_kv),
                make_tuple(kargs.seq_stride_bias, 1),
                number<1>{},
                number<1>{});
        const auto bias_dram_padded = pad_tensor_view(
            bias_dram_naive, lengths, sequence<false, kPadSeqLenK>{});
        return make_tile_window(bias_dram_padded, lengths, {i_m0, 0});
      } else {
        return make_null_tile_window(lengths);
      }
    }();

    auto null_randval_window = [&]() {
      if constexpr (kHasDropout) {
        // need to make a tile window from this null_randval_dram since the
        // null_tile_window does not have store_tile() over-loaded, which will
        // cause compiling issue when used inside BlockDropout::Run(). Also we
        // need this dram window to provide start_m0_idx used in
        // BlockDropout::Run()
        const auto null_randval_dram = [&]() {
          const auto null_dram_naive =
              make_naive_tensor_view<address_space_enum::global>(
                  static_cast<uint8_t*>(nullptr),
                  make_tuple(seqlen_q_in_ctrl, kargs.seqlen_kv),
                  make_tuple(kargs.seqlen_kv, 1),
                  number<1>{},
                  number<1>{});

          return pad_tensor_view(
              null_dram_naive,
              make_tuple(
                  number<HstuAttentionBwdPipeline::kM0>{},
                  number<HstuAttentionBwdPipeline::kN0>{}),
              sequence<true, true>{});
        }();

        return make_tile_window(
            null_randval_dram,
            make_tuple(
                number<HstuAttentionBwdPipeline::kM0>{},
                number<HstuAttentionBwdPipeline::kN0>{}),
            {i_m0, 0});
      } else
        return make_null_tile_window(make_tuple(number<1>{}, number<1>{}));
    }();

    auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
      if constexpr (kHasDropout) {
        // no need to save rand_val since we have separate kernel to generate
        // them for the host
        return BlockDropout{
            i_batch_,
            i_nhead_,
            kargs.num_head,
            kargs.drop_seed,
            kargs.drop_offset,
            kargs.rp_undrop,
            kargs.p_undrop_in_uint8_t,
            false};
      } else {
        return NullBlockDropout{};
      };
    }();

    // ---- Build HSTU mask and run pipeline ----
    // Runtime branch on window_size selects the compile-time local/non-local
    // mask type, matching the pattern used in the forward kernel. Kernel 1
    // iterates over Q row tiles, so is_tile_in_upper_scope is meaningful.
    const auto run_pipeline = [&](const auto& mask) {
      const auto [seqlen_k_start, seqlen_k_end] = [&]() {
        if constexpr (std::remove_cvref_t<decltype(mask)>::kUseLocal)
          return mask.GetTileRangeAlongX(
              bool_constant<kHasDropout>{},
              i_m0,
              number<HstuAttentionBwdPipeline::kM0>{},
              number<HstuAttentionBwdPipeline::kN0>{});
        else
          return mask.GetTileRangeAlongX(
              i_m0,
              number<HstuAttentionBwdPipeline::kM0>{},
              number<HstuAttentionBwdPipeline::kN0>{});
      }();

      if constexpr (!kUseSoftmax) {
        return HstuAttentionBwdPipeline{}(
            do_dram_window,
            q_dram_window,
            k_dram_window,
            v_dram_window,
            bias_dram_window,
            null_randval_window,
            seqlen_k_start,
            seqlen_k_end,
            mask,
            kargs.scale_s,
            kargs.scale_p,
            smem_ptr,
            dropout);
      } else {
        // Build O, LSE, and delta windows for the softmax path
        const auto o_dram = [&]() {
          const auto naive = make_naive_tensor_view<address_space_enum::global>(
              o_ptr,
              make_tuple(seqlen_q_in_ctrl, kargs.hdim_v),
              make_tuple(kargs.seq_stride_o, 1),
              number<HstuAttentionBwdPipeline::kAlignmentOGrad>{},
              number<1>{});
          return pad_tensor_view(
              naive,
              make_tuple(
                  number<HstuAttentionBwdPipeline::kM0>{},
                  number<HstuAttentionBwdPipeline::kVHeaddim>{}),
              sequence<false, kPadHeadDimV>{});
        }();

        auto o_dram_window = make_tile_window(
            o_dram,
            make_tuple(
                number<HstuAttentionBwdPipeline::kM0>{},
                number<HstuAttentionBwdPipeline::kVHeaddim>{}),
            {i_m0, 0});

        const CompDataType* lse_ptr =
            reinterpret_cast<const CompDataType*>(kargs.lse_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lse +
            batch_offset_lse;

        const auto lse_dram = [&]() {
          const auto naive = make_naive_tensor_view<address_space_enum::global>(
              lse_ptr,
              make_tuple(seqlen_q_in_ctrl),
              make_tuple(kargs.seq_stride_lse),
              number<1>{},
              number<1>{});
          return pad_tensor_view(
              naive,
              make_tuple(number<HstuAttentionBwdPipeline::kM0>{}),
              sequence<false>{});
        }();

        auto lse_dram_window = make_tile_window(
            lse_dram,
            make_tuple(number<HstuAttentionBwdPipeline::kM0>{}),
            {i_m0});

        CompDataType* delta_ptr =
            reinterpret_cast<CompDataType*>(kargs.delta_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_delta +
            batch_offset_delta;

        const auto delta_dram = [&]() {
          const auto naive = make_naive_tensor_view<address_space_enum::global>(
              delta_ptr,
              make_tuple(seqlen_q_in_ctrl),
              make_tuple(kargs.seq_stride_delta),
              number<1>{},
              number<1>{});
          return pad_tensor_view(
              naive,
              make_tuple(number<HstuAttentionBwdPipeline::kM0>{}),
              sequence<false>{});
        }();

        auto delta_dram_window = make_tile_window(
            delta_dram,
            make_tuple(number<HstuAttentionBwdPipeline::kM0>{}),
            {i_m0});

        return HstuAttentionBwdPipeline{}(
            do_dram_window,
            o_dram_window,
            lse_dram_window,
            q_dram_window,
            k_dram_window,
            v_dram_window,
            bias_dram_window,
            null_randval_window,
            delta_dram_window,
            seqlen_k_start,
            seqlen_k_end,
            mask,
            kargs.scale_s,
            smem_ptr,
            dropout);
      }
    };

    auto dq_acc_tile = [&]() {
      bool use_local = kargs.window_size > 0;

      return BOOL_SWITCH_RETURN(use_local, kUseLocal, [&]() {
        using HstuMaskType = typename ck_tile::
            HstuBlockMasking<kIsCrossAttention, kHasCausal, kUseLocal>::Type;

        if constexpr (kUseLocal) {
          auto mask = [&]() {
            if constexpr (kIsCrossAttention)
              return make_hstu_cross_attention_block_mask_with_local<
                  HstuMaskType>(
                  is_tile_in_upper_scope,
                  kargs.seqlen_q,
                  kargs.seqlen_kv,
                  kargs.contextual_seqlen,
                  num_target,
                  kargs.window_size,
                  kargs.min_full_attn_seqlen);
            else
              return make_hstu_self_attention_block_mask_with_local<
                  HstuMaskType>(
                  is_tile_in_upper_scope,
                  kargs.seqlen_q,
                  kargs.contextual_seqlen,
                  num_target,
                  kargs.window_size,
                  kargs.min_full_attn_seqlen);
          }();

          return run_pipeline(mask);
        } else {
          auto mask = [&]() {
            if constexpr (kIsCrossAttention)
              return make_hstu_cross_attention_block_mask_without_local<
                  HstuMaskType>(
                  kargs.seqlen_q,
                  kargs.seqlen_kv,
                  kargs.contextual_seqlen,
                  num_target);
            else
              return make_hstu_self_attention_block_mask_without_local<
                  HstuMaskType>(
                  kargs.seqlen_q, kargs.contextual_seqlen, num_target);
          }();

          return run_pipeline(mask);
        }
      });
    }();

    // ---- dQ output DRAM window -- tile storing window for the epilogue ----
    const auto dq_dram = [&]() {
      const auto naive = make_naive_tensor_view<address_space_enum::global>(
          dq_ptr,
          make_tuple(seqlen_q_in_ctrl, kargs.hdim_qk),
          make_tuple(kargs.seq_stride_dq, 1),
          number<HstuAttentionBwdPipeline::kAlignmentQGrad>{},
          number<1>{});
      return pad_tensor_view(
          naive,
          make_tuple(
              number<HstuAttentionBwdPipeline::kM0>{},
              number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
          sequence<false, kPadHeadDimQK>{});
    }();

    auto dq_dram_window = make_tile_window(
        dq_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kM0>{},
            number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
        {i_m0, 0});

    // Epilogue: write dq_acc -> dQ DRAM
    constexpr index_t NumRepN = HstuAttentionBwdPipeline::kQKHeaddim /
        HstuAttentionBwdPipeline::kGemm4SingleRepN;
    EpiloguePipeline{}(dq_dram_window, dq_acc_tile, number<NumRepN>{});
  }
};

} // namespace ck_tile
