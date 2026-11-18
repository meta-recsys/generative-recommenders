// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include <string>
#include <type_traits>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_kernel_util.hpp"
#include "hstu_bwd_block_masking.hpp"

#ifndef HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM
#define HSTU_SCHED_BATCH_AS_FIRST_GRID_DIM 1
#endif

// Backward Kernel 2: computes dK and dV for each (batch, head, sk_tile).
// K and V are loaded once into registers outside the main loop; Q and dO are
// streamed from DRAM through LDS at each iteration.
// Grid: one workgroup per (batch, head, sk_tile) -- columns instead of rows.

namespace ck_tile {

template <
    typename HstuAttentionBwdPipeline_,
    typename EpiloguePipelineKGrad_,
    typename EpiloguePipelineVGrad_>
struct HstuAttentionBwdKernel2 {
  using HstuAttentionBwdPipeline = remove_cvref_t<HstuAttentionBwdPipeline_>;
  using EpiloguePipelineKGrad = remove_cvref_t<EpiloguePipelineKGrad_>;
  using EpiloguePipelineVGrad = remove_cvref_t<EpiloguePipelineVGrad_>;

  static constexpr index_t kBlockSize = HstuAttentionBwdPipeline::kBlockSize;
  static constexpr index_t kBlockPerCu = HstuAttentionBwdPipeline::kBlockPerCu;
  static_assert(kBlockPerCu > 0);

  using QKVDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::QKVDataType>;
  using BiasDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::BiasDataType>;
  using OGradDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::OGradDataType>;
  using KGradAccDataType = remove_cvref_t<
      typename HstuAttentionBwdPipeline::Problem::KGradAccDataType>;
  using VGradAccDataType = remove_cvref_t<
      typename HstuAttentionBwdPipeline::Problem::VGradAccDataType>;
  using KGradDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::KGradDataType>;
  using VGradDataType =
      remove_cvref_t<typename HstuAttentionBwdPipeline::Problem::VGradDataType>;
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

  static constexpr bool IsWarpGemm32 =
      HstuAttentionBwdPipeline::Problem::IsWarpGemm32;

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

  // --- Base kargs for batched no-group mode ---
  struct HstuBwdKernel2NoGroupBatchedBaseKargs {
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_dk; // set to zero in jagged mode
    ck_tile::index_t batch_stride_dv; // set to zero in jagged mode
    ck_tile::index_t batch_stride_lse; // only meaningful when kUseSoftmax
    ck_tile::index_t batch_stride_delta; // only meaningful when kUseSoftmax

    const int32_t* num_targets_ptr;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* do_ptr;
    void* dk_ptr;
    void* dv_ptr;
    const void* lse_ptr; // read-only; only used when kUseSoftmax
    const void* delta_ptr; // read-only; D[sq] written by kernel 1, only when
                           // kUseSoftmax

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_delta;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_kv;
    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t seq_stride_dk;
    ck_tile::index_t seq_stride_dv;
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t seq_stride_delta;

    ck_tile::index_t num_head;
    float scale_s; // scaling value exerted on the immediate Q@K result
    float scale_p; // scaling value exerted on the SiLU result

    bool almost_invariant_seqlen; // should always be true for batched mode

    ck_tile::index_t contextual_seqlen;
    ck_tile::index_t window_size;
    ck_tile::index_t min_full_attn_seqlen;
  };

  // --- Base kargs for jagged no-group mode ---
  struct HstuBwdKernel2NoGroupJaggedBaseKargs {
    const int32_t* seq_q_offsets_ptr;
    const int32_t* seq_kv_offsets_ptr;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t seq_stride_dk;
    ck_tile::index_t seq_stride_dv;
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t seq_stride_delta;

    const int32_t* num_targets_ptr;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* do_ptr;
    void* dk_ptr;
    void* dv_ptr;
    const void* lse_ptr;
    const void* delta_ptr;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_delta;

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

  // --- Base kargs for group mode ---
  struct HstuBwdKernel2GroupBaseKargs {
    ck_tile::index_t num_batch_per_group;

    const int32_t* seq_q_offsets_ptr;
    const int32_t* seq_kv_offsets_ptr;

    ck_tile::index_t seq_stride_q;
    ck_tile::index_t seq_stride_k;
    ck_tile::index_t seq_stride_v;
    ck_tile::index_t seq_stride_do;
    ck_tile::index_t seq_stride_dk;
    ck_tile::index_t seq_stride_dv;
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t seq_stride_delta;

    const int32_t* num_targets_ptr;

    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* do_ptr;
    void* dk_ptr;
    void* dv_ptr;
    const void* lse_ptr;
    const void* delta_ptr;

    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_delta;

    ck_tile::index_t hdim_qk;
    ck_tile::index_t hdim_v;

    ck_tile::index_t seqlen_q; // set at runtime from seq_q_offsets_ptr
    ck_tile::index_t seqlen_kv; // set at runtime from seq_kv_offsets_ptr

    ck_tile::index_t num_head;
    float scale_s;
    float scale_p; // set at runtime from group_attn_scale_ptr

    bool almost_invariant_seqlen;

    int32_t contextual_seqlen; // set at runtime
    int32_t window_size; // set at runtime
    int32_t min_full_attn_seqlen; // set at runtime

    const int32_t* group_max_seqlen_q_ptr;
    const int32_t* group_contextual_seqlen_ptr;
    const int32_t* group_window_size_ptr;
    const int32_t* group_min_full_attn_seqlen_ptr;
    const float* group_attn_scale_ptr;
  };

  struct HstuBwdKernel2BatchedBiasKargs {
    const void* bias_ptr;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t batch_stride_bias;
  };

  struct HstuBwdKernel2JaggedBiasKargs {
    const void* bias_ptr;
    ck_tile::index_t seq_stride_bias;
    ck_tile::index_t nhead_stride_bias;
  };

  struct HstuBwdKernel2CommonDropoutKargs {
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

  struct HstuBwdKernel2NoGroupBatchedKargs
      : HstuBwdKernel2NoGroupBatchedBaseKargs,
        std::conditional_t<
            kHasBias,
            HstuBwdKernel2BatchedBiasKargs,
            EmptyKargs<1>>,
        std::conditional_t<
            kHasDropout,
            HstuBwdKernel2CommonDropoutKargs,
            EmptyKargs<2>> {};

  struct HstuBwdKernel2NoGroupJaggedKargs
      : HstuBwdKernel2NoGroupJaggedBaseKargs,
        std::conditional_t<
            kHasBias,
            HstuBwdKernel2JaggedBiasKargs,
            EmptyKargs<1>>,
        std::conditional_t<
            kHasDropout,
            HstuBwdKernel2CommonDropoutKargs,
            EmptyKargs<2>> {};

  struct HstuBwdKernel2GroupKargs : HstuBwdKernel2GroupBaseKargs,
                                    std::conditional_t<
                                        kHasBias,
                                        HstuBwdKernel2JaggedBiasKargs,
                                        EmptyKargs<1>>,
                                    std::conditional_t<
                                        kHasDropout,
                                        HstuBwdKernel2CommonDropoutKargs,
                                        EmptyKargs<2>> {};

  using Kargs = std::conditional_t<
      kUseGroup,
      HstuBwdKernel2GroupKargs,
      std::conditional_t<
          kIsJagged,
          HstuBwdKernel2NoGroupJaggedKargs,
          HstuBwdKernel2NoGroupBatchedKargs>>;

  // -------------------------------------------------------------------------
  // MakeKargs factory functions
  // -------------------------------------------------------------------------

  // Overload 1: NoGroup + Batched
  template <bool Cond = kUseNoGroupBatched>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      const void* bias_ptr,
      const void* do_ptr,
      void* dk_ptr,
      void* dv_ptr,
      const void* lse_ptr,
      const void* delta_ptr,
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
      ck_tile::index_t seq_stride_do,
      ck_tile::index_t seq_stride_dk,
      ck_tile::index_t seq_stride_dv,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t seq_stride_delta,
      ck_tile::index_t seq_stride_bias,
      ck_tile::index_t nhead_stride_q,
      ck_tile::index_t nhead_stride_k,
      ck_tile::index_t nhead_stride_v,
      ck_tile::index_t nhead_stride_do,
      ck_tile::index_t nhead_stride_dk,
      ck_tile::index_t nhead_stride_dv,
      ck_tile::index_t nhead_stride_lse,
      ck_tile::index_t nhead_stride_delta,
      ck_tile::index_t nhead_stride_bias,
      ck_tile::index_t batch_stride_q,
      ck_tile::index_t batch_stride_k,
      ck_tile::index_t batch_stride_v,
      ck_tile::index_t batch_stride_do,
      ck_tile::index_t batch_stride_dk,
      ck_tile::index_t batch_stride_dv,
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
         batch_stride_do,
         batch_stride_dk,
         batch_stride_dv,
         batch_stride_lse,
         batch_stride_delta,
         reinterpret_cast<const int32_t*>(num_targets_ptr),
         q_ptr,
         k_ptr,
         v_ptr,
         do_ptr,
         dk_ptr,
         dv_ptr,
         lse_ptr,
         delta_ptr,
         nhead_stride_q,
         nhead_stride_k,
         nhead_stride_v,
         nhead_stride_do,
         nhead_stride_dk,
         nhead_stride_dv,
         nhead_stride_lse,
         nhead_stride_delta,
         seqlen_q,
         seqlen_kv,
         hdim_qk,
         hdim_v,
         seq_stride_q,
         seq_stride_k,
         seq_stride_v,
         seq_stride_do,
         seq_stride_dk,
         seq_stride_dv,
         seq_stride_lse,
         seq_stride_delta,
         num_head,
         scale_s,
         attn_scale ? attn_scale : 1.0f / static_cast<float>(seqlen_q),
         true, // almost_invariant_seqlen
         contextual_seqlen,
         window_size,
         min_full_attn_seqlen}, // base kargs
        {}, // placeholder for bias
    };

    if constexpr (kHasBias) {
      kargs.bias_ptr = bias_ptr;
      kargs.seq_stride_bias = seq_stride_bias;
      kargs.nhead_stride_bias = nhead_stride_bias;
      kargs.batch_stride_bias = batch_stride_bias;
    }

    if constexpr (kHasDropout) {
      kargs.init_dropout(p_drop, philox_seed, philox_offset);
    }

    return kargs;
  }

  // Overload 2: NoGroup + Jagged
  template <bool Cond = kUseNoGroupJagged>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      const void* bias_ptr,
      const void* do_ptr,
      void* dk_ptr,
      void* dv_ptr,
      const void* lse_ptr,
      const void* delta_ptr,
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
      ck_tile::index_t seq_stride_do,
      ck_tile::index_t seq_stride_dk,
      ck_tile::index_t seq_stride_dv,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t seq_stride_delta,
      ck_tile::index_t seq_stride_bias,
      ck_tile::index_t nhead_stride_q,
      ck_tile::index_t nhead_stride_k,
      ck_tile::index_t nhead_stride_v,
      ck_tile::index_t nhead_stride_do,
      ck_tile::index_t nhead_stride_dk,
      ck_tile::index_t nhead_stride_dv,
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
         seq_stride_do,
         seq_stride_dk,
         seq_stride_dv,
         seq_stride_lse,
         seq_stride_delta,
         reinterpret_cast<const int32_t*>(num_targets_ptr),
         q_ptr,
         k_ptr,
         v_ptr,
         do_ptr,
         dk_ptr,
         dv_ptr,
         lse_ptr,
         delta_ptr,
         nhead_stride_q,
         nhead_stride_k,
         nhead_stride_v,
         nhead_stride_do,
         nhead_stride_dk,
         nhead_stride_dv,
         nhead_stride_lse,
         nhead_stride_delta,
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
    };

    if constexpr (kHasBias) {
      kargs.bias_ptr = bias_ptr;
      kargs.seq_stride_bias = seq_stride_bias;
      kargs.nhead_stride_bias = nhead_stride_bias;
    }

    if constexpr (kHasDropout) {
      kargs.init_dropout(p_drop, philox_seed, philox_offset);
    }

    return kargs;
  }

  // Overload 3: Group
  template <bool Cond = kUseGroup>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* q_ptr,
      const void* k_ptr,
      const void* v_ptr,
      const void* bias_ptr,
      const void* do_ptr,
      void* dk_ptr,
      void* dv_ptr,
      const void* lse_ptr,
      const void* delta_ptr,
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
      ck_tile::index_t seq_stride_do,
      ck_tile::index_t seq_stride_dk,
      ck_tile::index_t seq_stride_dv,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t seq_stride_delta,
      ck_tile::index_t seq_stride_bias,
      ck_tile::index_t nhead_stride_q,
      ck_tile::index_t nhead_stride_k,
      ck_tile::index_t nhead_stride_v,
      ck_tile::index_t nhead_stride_do,
      ck_tile::index_t nhead_stride_dk,
      ck_tile::index_t nhead_stride_dv,
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
         seq_stride_do,
         seq_stride_dk,
         seq_stride_dv,
         seq_stride_lse,
         seq_stride_delta,
         reinterpret_cast<const int32_t*>(num_targets_ptr),
         q_ptr,
         k_ptr,
         v_ptr,
         do_ptr,
         dk_ptr,
         dv_ptr,
         lse_ptr,
         delta_ptr,
         nhead_stride_q,
         nhead_stride_k,
         nhead_stride_v,
         nhead_stride_do,
         nhead_stride_dk,
         nhead_stride_dv,
         nhead_stride_lse,
         nhead_stride_delta,
         hdim_qk,
         hdim_v,
         -1, // seqlen_q: set at runtime
         -1, // seqlen_kv: set at runtime
         num_head,
         scale_s,
         1.0f, // scale_p: set at runtime from group_attn_scale_ptr
         almost_invariant_seqlen,
         0, // contextual_seqlen: set at runtime
         0, // window_size: set at runtime
         0, // min_full_attn_seqlen: set at runtime
         reinterpret_cast<const int32_t*>(group_max_seqlen_q_ptr),
         reinterpret_cast<const int32_t*>(group_contextual_seqlen_ptr),
         reinterpret_cast<const int32_t*>(group_window_size_ptr),
         reinterpret_cast<const int32_t*>(group_min_full_attn_seqlen_ptr),
         reinterpret_cast<const float*>(group_attn_scale_ptr)}, // base kargs
        {}, // placeholder for bias
    };

    if constexpr (kHasBias) {
      kargs.bias_ptr = bias_ptr;
      kargs.seq_stride_bias = seq_stride_bias;
      kargs.nhead_stride_bias = nhead_stride_bias;
    }

    if constexpr (kHasDropout) {
      kargs.init_dropout(p_drop, philox_seed, philox_offset);
    }

    return kargs;
  }

  // -------------------------------------------------------------------------
  // Grid / block sizing
  // -------------------------------------------------------------------------

  // Grid: one block per (batch, head, sk_tile).
  // seqlen_kv is the K/V sequence length; each block owns one kN0-wide column
  // tile.
  CK_TILE_HOST static constexpr auto GridSize(
      ck_tile::index_t batch_size,
      ck_tile::index_t nhead,
      ck_tile::index_t seqlen_kv,
      bool almost_invariant_seqlen) {
    const ck_tile::index_t num_tile_n =
        ck_tile::integer_divide_ceil(seqlen_kv, HstuAttentionBwdPipeline::kN0);
    if (almost_invariant_seqlen)
      return dim3(batch_size, nhead, num_tile_n);
    else
      return dim3(num_tile_n, nhead, batch_size);
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
        max(EpiloguePipelineKGrad::GetSmemSize(),
            EpiloguePipelineVGrad::GetSmemSize()));
  }

  // -------------------------------------------------------------------------
  // Tile index helper
  // -------------------------------------------------------------------------

  CK_TILE_DEVICE static auto GetTileIndex(const Kargs& kargs) {
    if (kargs.almost_invariant_seqlen) {
      const index_t i_batch = blockIdx.x;
      const index_t i_nhead = blockIdx.y;
      const index_t i_tile_n = blockIdx.z;

      return make_tuple(i_tile_n, i_nhead, i_batch);
    } else {
      const index_t i_tile_n = blockIdx.x;
      const index_t i_nhead = blockIdx.y;
      const index_t i_batch = blockIdx.z;

      return make_tuple(i_tile_n, i_nhead, i_batch);
    }
  }

  // -------------------------------------------------------------------------
  // Device operator
  // -------------------------------------------------------------------------

  CK_TILE_DEVICE void operator()(Kargs kargs) const {
    __shared__ char smem_ptr[GetSmemSize()];

    const auto [i_tile_n, i_nhead, i_batch] = GetTileIndex(kargs);

    // ---- Batch offsets ----
    long_index_t batch_offset_q = 0;
    long_index_t batch_offset_k = 0;
    long_index_t batch_offset_v = 0;
    long_index_t batch_offset_do = 0;
    long_index_t batch_offset_dk = 0;
    long_index_t batch_offset_dv = 0;
    long_index_t batch_offset_lse = 0;
    long_index_t batch_offset_delta = 0;
    long_index_t batch_offset_bias = 0;

    if constexpr (kIsJagged) {
      const long_index_t query_start = kargs.seq_q_offsets_ptr[i_batch];
      const long_index_t key_start = kargs.seq_kv_offsets_ptr[i_batch];

      batch_offset_q = query_start * kargs.seq_stride_q;
      batch_offset_k = key_start * kargs.seq_stride_k;
      batch_offset_v = key_start * kargs.seq_stride_v;
      batch_offset_do = query_start * kargs.seq_stride_do;
      batch_offset_dk = key_start * kargs.seq_stride_dk;
      batch_offset_dv = key_start * kargs.seq_stride_dv;

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
      batch_offset_do =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
      batch_offset_dk =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
      batch_offset_dv =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
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

    // Column origin for this block in the K/V sequence dimension
    const index_t i_n0 = __builtin_amdgcn_readfirstlane(
        i_tile_n * HstuAttentionBwdPipeline::kN0);

    // Guard: do not launch beyond seqlen_kv
    if (kargs.seqlen_kv <= i_n0)
      return;

    const int num_target =
        (kargs.num_targets_ptr == nullptr) ? 0 : kargs.num_targets_ptr[i_batch];

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
    const OGradDataType* do_ptr =
        reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
        batch_offset_do;
    KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk +
        batch_offset_dk;
    VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv +
        batch_offset_dv;

    // ---- Q DRAM view [seqlen_q, hdim_qk] ----
    const auto q_dram = [&]() {
      const auto q_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              q_ptr,
              make_tuple(kargs.seqlen_q, kargs.hdim_qk),
              make_tuple(kargs.seq_stride_q, 1),
              number<HstuAttentionBwdPipeline::kAlignmentQ>{},
              number<1>{});

      if constexpr (detail::is_using_tdm_v<HstuAttentionBwdPipeline>) {
        return q_dram_naive;
      } else {
        const auto q_dram_pad = pad_tensor_view(
            q_dram_naive,
            make_tuple(
                number<HstuAttentionBwdPipeline::kM0>{},
                number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
            sequence<false, kPadHeadDimQK>{});

        if constexpr (detail::is_direct_lds_pipeline_v<
                          HstuAttentionBwdPipeline>)
          return detail::make_direct_lds_xor_dram_view<
              HstuAttentionBwdPipeline::kQKHeaddim,
              HstuAttentionBwdPipeline::Policy::
                  template GetQDirectLdsSwizzleUnit<
                      typename HstuAttentionBwdPipeline::Problem>()>(
              q_dram_pad);
        else
          return q_dram_pad;
      }
    }();

    // ---- K DRAM view [seqlen_kv, hdim_qk] -- loaded once into registers ----
    const auto k_dram = [&]() {
      const auto naive = make_naive_tensor_view<address_space_enum::global>(
          k_ptr,
          make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
          make_tuple(kargs.seq_stride_k, 1),
          number<HstuAttentionBwdPipeline::kAlignmentK>{},
          number<1>{});
      return pad_tensor_view(
          naive,
          make_tuple(
              number<HstuAttentionBwdPipeline::kN0>{},
              number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
          sequence<false, kPadHeadDimQK>{});
    }();

    // ---- V DRAM view [seqlen_kv, hdim_v] -- loaded once into registers ----
    const auto v_dram = [&]() {
      const auto naive = make_naive_tensor_view<address_space_enum::global>(
          v_ptr,
          make_tuple(kargs.seqlen_kv, kargs.hdim_v),
          make_tuple(kargs.seq_stride_v, 1),
          number<HstuAttentionBwdPipeline::kAlignmentV>{},
          number<1>{});
      return pad_tensor_view(
          naive,
          make_tuple(
              number<HstuAttentionBwdPipeline::kN0>{},
              number<HstuAttentionBwdPipeline::kVHeaddim>{}),
          sequence<false, kPadHeadDimV>{});
    }();

    // ---- dO DRAM view [seqlen_q, hdim_v] ----
    const auto do_dram = [&]() {
      const auto do_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              do_ptr,
              make_tuple(kargs.seqlen_q, kargs.hdim_v),
              make_tuple(kargs.seq_stride_do, 1),
              number<HstuAttentionBwdPipeline::kAlignmentOGrad>{},
              number<1>{});

      if constexpr (detail::is_using_tdm_v<HstuAttentionBwdPipeline>) {
        return do_dram_naive;
      } else {
        const auto do_dram_pad = pad_tensor_view(
            do_dram_naive,
            make_tuple(
                number<HstuAttentionBwdPipeline::kM0>{},
                number<HstuAttentionBwdPipeline::kVHeaddim>{}),
            sequence<false, kPadHeadDimV>{});

        if constexpr (detail::is_direct_lds_pipeline_v<
                          HstuAttentionBwdPipeline>)
          return detail::make_direct_lds_xor_dram_view<
              HstuAttentionBwdPipeline::kVHeaddim,
              HstuAttentionBwdPipeline::Policy::
                  template GetOGradDirectLdsSwizzleUnit<
                      typename HstuAttentionBwdPipeline::Problem>()>(
              do_dram_pad);
        else
          return do_dram_pad;
      }
    }();

    // ---- DRAM windows ----
    auto q_dram_window = make_tile_window(
        q_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kM0>{},
            number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
        {0, 0});

    // K and V: windows at this block's column tile origin
    auto k_dram_window = make_tile_window(
        k_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kN0>{},
            number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
        {i_n0, 0},
        HstuAttentionBwdPipeline::Policy::template MakeKRegTileDistribution<
            typename HstuAttentionBwdPipeline::Problem>());

    auto v_dram_window = make_tile_window(
        v_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kN0>{},
            number<HstuAttentionBwdPipeline::kVHeaddim>{}),
        {i_n0, 0},
        HstuAttentionBwdPipeline::Policy::template MakeVRegTileDistribution<
            typename HstuAttentionBwdPipeline::Problem>());

    auto do_dram_window = make_tile_window(
        do_dram,
        make_tuple(
            number<HstuAttentionBwdPipeline::kM0>{},
            number<HstuAttentionBwdPipeline::kVHeaddim>{}),
        {0, 0});

    // ---- Bias DRAM window ----
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
                make_tuple(kargs.seqlen_q, kargs.seqlen_kv),
                make_tuple(kargs.seq_stride_bias, 1),
                number<1>{},
                number<1>{});
        const auto bias_dram_padded = pad_tensor_view(
            bias_dram_naive, lengths, sequence<false, kPadSeqLenK>{});
        return make_tile_window(bias_dram_padded, lengths, {0, i_n0});
      } else {
        return make_null_tile_window(lengths);
      }
    }();

    // ---- Build HSTU epilogue helpers (shared by both mask branches) ----
    const auto run_epilogue = [&](const auto& dk_acc, const auto& dv_acc) {
      // ---- dK output DRAM window ----
      const auto dk_dram = [&]() {
        const auto naive = make_naive_tensor_view<address_space_enum::global>(
            dk_ptr,
            make_tuple(kargs.seqlen_kv, kargs.hdim_qk),
            make_tuple(kargs.seq_stride_dk, 1),
            number<HstuAttentionBwdPipeline::kAlignmentKGrad>{},
            number<1>{});
        return pad_tensor_view(
            naive,
            make_tuple(
                number<HstuAttentionBwdPipeline::kN0>{},
                number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
            sequence<false, kPadHeadDimQK>{});
      }();

      auto dk_dram_window = make_tile_window(
          dk_dram,
          make_tuple(
              number<HstuAttentionBwdPipeline::kN0>{},
              number<HstuAttentionBwdPipeline::kQKHeaddim>{}),
          {i_n0, 0});

      // ---- dV output DRAM window ----
      const auto dv_dram = [&]() {
        const auto naive = make_naive_tensor_view<address_space_enum::global>(
            dv_ptr,
            make_tuple(kargs.seqlen_kv, kargs.hdim_v),
            make_tuple(kargs.seq_stride_dv, 1),
            number<HstuAttentionBwdPipeline::kAlignmentVGrad>{},
            number<1>{});
        return pad_tensor_view(
            naive,
            make_tuple(
                number<HstuAttentionBwdPipeline::kN0>{},
                number<HstuAttentionBwdPipeline::kVHeaddim>{}),
            sequence<false, kPadHeadDimV>{});
      }();

      auto dv_dram_window = make_tile_window(
          dv_dram,
          make_tuple(
              number<HstuAttentionBwdPipeline::kN0>{},
              number<HstuAttentionBwdPipeline::kVHeaddim>{}),
          {i_n0, 0});

      constexpr index_t NumRepN_K = HstuAttentionBwdPipeline::kQKHeaddim /
          HstuAttentionBwdPipeline::kGemm3SingleRepN;
      constexpr index_t NumRepN_V = HstuAttentionBwdPipeline::kVHeaddim /
          HstuAttentionBwdPipeline::kGemm1SingleRepN;

      EpiloguePipelineKGrad{}(dk_dram_window, dk_acc, number<NumRepN_K>{});
      EpiloguePipelineVGrad{}(dv_dram_window, dv_acc, number<NumRepN_V>{});
    };

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
                  make_tuple(kargs.seqlen_q, kargs.seqlen_kv),
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
            {0, i_n0});
      } else
        return make_null_tile_window(make_tuple(number<1>{}, number<1>{}));
    }();

    auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
      if constexpr (kHasDropout) {
        // no need to save rand_val since we have separate kernel to generate
        // them for the host. Kernel-2's Gemm0 P tile is M-major, so use the
        // backward (M-major) dropout variant that matches
        // HstuAttentionNoSoftmaxBwdPipelineKRVRQS_dK_dV::DropoutType.
        return BlockDropoutBwd<true, IsWarpGemm32, false>{
            i_batch_,
            i_nhead_,
            kargs.num_head,
            kargs.drop_seed,
            kargs.drop_offset,
            kargs.rp_undrop,
            kargs.p_undrop_in_uint8_t};
      } else {
        return NullBlockDropout{};
      };
    }();

    // ---- Build HSTU mask and run pipeline ----
    // Runtime branch on window_size selects the compile-time local/non-local
    // mask type, Kernel 2 iterates over K/V col tiles, so
    // is_tile_in_upper_scope is always true.
    const auto run_pipeline = [&](const auto& mask) {
      const auto [seqlen_q_start, seqlen_q_end] = mask.GetTileRangeAlongY(
          i_n0,
          number<HstuAttentionBwdPipeline::kN0>{},
          number<HstuAttentionBwdPipeline::kM0>{});

      if constexpr (!kUseSoftmax) {
        const auto [dk_acc, dv_acc] = HstuAttentionBwdPipeline{}(
            q_dram_window,
            do_dram_window,
            bias_dram_window,
            k_dram_window,
            v_dram_window,
            null_randval_window,
            seqlen_q_start,
            seqlen_q_end,
            i_n0,
            mask,
            kargs.scale_s,
            kargs.scale_p,
            smem_ptr,
            dropout);
        run_epilogue(dk_acc, dv_acc);
      } else {
        // Build LSE and delta DRAM windows
        const CompDataType* lse_ptr =
            reinterpret_cast<const CompDataType*>(kargs.lse_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lse +
            batch_offset_lse;

        const auto lse_dram = [&]() {
          const auto naive = make_naive_tensor_view<address_space_enum::global>(
              lse_ptr,
              make_tuple(kargs.seqlen_q),
              make_tuple(kargs.seq_stride_lse),
              number<1>{},
              number<1>{});
          return pad_tensor_view(
              naive,
              make_tuple(number<HstuAttentionBwdPipeline::kM0>{}),
              sequence<false>{});
        }();

        auto lse_dram_window = make_tile_window(
            lse_dram, make_tuple(number<HstuAttentionBwdPipeline::kM0>{}), {0});

        const CompDataType* delta_ptr =
            reinterpret_cast<const CompDataType*>(kargs.delta_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_delta +
            batch_offset_delta;

        const auto delta_dram = [&]() {
          const auto naive = make_naive_tensor_view<address_space_enum::global>(
              delta_ptr,
              make_tuple(kargs.seqlen_q),
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
            {0});

        const auto [dk_acc, dv_acc] = HstuAttentionBwdPipeline{}(
            q_dram_window,
            do_dram_window,
            lse_dram_window,
            delta_dram_window,
            bias_dram_window,
            k_dram_window,
            v_dram_window,
            null_randval_window,
            seqlen_q_start,
            seqlen_q_end,
            i_n0,
            mask,
            kargs.scale_s,
            smem_ptr,
            dropout);
        run_epilogue(dk_acc, dv_acc);
      }
    };

    bool use_local = kargs.window_size > 0;

    return BOOL_SWITCH(use_local, kUseLocal, [&]() {
      using HstuMaskType = typename ck_tile::
          HstuBwdBlockMasking<kIsCrossAttention, kHasCausal, kUseLocal>::Type;

      if constexpr (kUseLocal) {
        auto mask = [&]() {
          if constexpr (kIsCrossAttention)
            return make_hstu_bwd_cross_attention_block_mask_with_local<
                HstuMaskType>(
                kargs.seqlen_q,
                kargs.seqlen_kv,
                kargs.contextual_seqlen,
                num_target,
                kargs.window_size,
                kargs.min_full_attn_seqlen);
          else
            return make_hstu_bwd_self_attention_block_mask_with_local<
                HstuMaskType>(
                kargs.seqlen_q,
                kargs.contextual_seqlen,
                num_target,
                kargs.window_size,
                kargs.min_full_attn_seqlen);
        }();

        run_pipeline(mask);
      } else {
        auto mask = [&]() {
          if constexpr (kIsCrossAttention)
            return make_hstu_bwd_cross_attention_block_mask_without_local<
                HstuMaskType>(
                kargs.seqlen_q,
                kargs.seqlen_kv,
                kargs.contextual_seqlen,
                num_target);
          else
            return make_hstu_bwd_self_attention_block_mask_without_local<
                HstuMaskType>(
                kargs.seqlen_q, kargs.contextual_seqlen, num_target);
        }();

        run_pipeline(mask);
      }
    });
  }
};

} // namespace ck_tile
