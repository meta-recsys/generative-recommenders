// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/common.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

#include "hstu_block_masking.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] @ V^T[hdim_v, seqlen_k]

namespace ck_tile {

template <typename HstuAttentionPipeline_, typename EpiloguePipeline_>
struct HstuAttentionFwdSplitKVCombineKernel {
  using HstuAttentionPipeline = ck_tile::remove_cvref_t<HstuAttentionPipeline_>;
  using EpiloguePipeline = ck_tile::remove_cvref_t<EpiloguePipeline_>;
  static constexpr ck_tile::index_t kBlockSize =
      HstuAttentionPipeline::kBlockSize;
  static constexpr ck_tile::index_t kBlockPerCu =
      HstuAttentionPipeline::kBlockPerCu;
  static_assert(kBlockPerCu > 0);

  using OaccDataType = ck_tile::remove_cvref_t<
      typename HstuAttentionPipeline::Problem::OaccDataType>;
  using ODataType = ck_tile::remove_cvref_t<
      typename HstuAttentionPipeline::Problem::ODataType>;
  using LSEDataType = ck_tile::remove_cvref_t<
      typename HstuAttentionPipeline::Problem::LSEDataType>;

  static constexpr bool kIsJagged = HstuAttentionPipeline::Problem::kIsJagged;
  static constexpr bool kUseSoftmax =
      HstuAttentionPipeline::Problem::kUseSoftmax;
  static constexpr bool kStoreLSE = HstuAttentionPipeline::Problem::kStoreLSE;

  static constexpr bool kPadSeqLenQ = HstuAttentionPipeline::kPadSeqLenQ;
  static constexpr bool kPadHeadDimO = HstuAttentionPipeline::kPadHeadDimO;
  static constexpr bool kPadNumSplits = HstuAttentionPipeline::kPadNumSplits;

  template <ck_tile::index_t I> // to avoid duplicated base class problem,
                                // introduce an template arg
  struct HstuAttentionCombineEmptyKargs {};

  // kargs use aggregate initializer, so no constructor will provided
  // use inheritance to minimize karg size
  // user need to use MakeKargs() function to create kargs.
  struct HstuAttentionBatchedCombineBaseKargs {
    const void* o_acc_ptr;
    void* o_ptr;

    ck_tile::index_t batch_stride_o;
    ck_tile::index_t seq_stride_o;
    ck_tile::index_t nhead_stride_o;

    ck_tile::index_t seqlen_q;
    ck_tile::index_t num_head;
    ck_tile::index_t num_splits;
    ck_tile::index_t hdim_v;

    bool almost_invariant_seqlen; // should always be true for batched mode
  };

  struct HstuAttentionJaggedCombineBaseKargs {
    const void* o_acc_ptr;
    void* o_ptr;

    ck_tile::index_t seq_stride_o;
    ck_tile::index_t nhead_stride_o;

    const int32_t* seq_q_offsets_ptr;
    ck_tile::index_t num_head;
    ck_tile::index_t num_splits;
    ck_tile::index_t hdim_v;

    ck_tile::index_t seqlen_q;

    bool almost_invariant_seqlen;
  };

  struct HstuAttentionCombineSoftmaxKargs {
    const void* lse_acc_ptr = nullptr;
  };

  struct HstuAttentionBatchedCombineLSEKargs {
    void* lse_ptr;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t nhead_stride_lse;
  };

  struct HstuAttentionJaggedCombineLSEKargs {
    void* lse_ptr;
    ck_tile::index_t seq_stride_lse;
    ck_tile::index_t nhead_stride_lse;
  };

  struct HstuAttentionBatchedCombineKargs
      : HstuAttentionBatchedCombineBaseKargs,
        std::conditional_t<
            kUseSoftmax,
            HstuAttentionCombineSoftmaxKargs,
            HstuAttentionCombineEmptyKargs<1>>,
        std::conditional_t<
            kStoreLSE,
            HstuAttentionBatchedCombineLSEKargs,
            HstuAttentionCombineEmptyKargs<2>>

  {};

  struct HstuAttentionJaggedCombineKargs
      : HstuAttentionJaggedCombineBaseKargs,
        std::conditional_t<
            kUseSoftmax,
            HstuAttentionCombineSoftmaxKargs,
            HstuAttentionCombineEmptyKargs<1>>,
        std::conditional_t<
            kStoreLSE,
            HstuAttentionJaggedCombineLSEKargs,
            HstuAttentionCombineEmptyKargs<2>> {};

  using Kargs = std::conditional_t<
      kIsJagged,
      HstuAttentionJaggedCombineKargs,
      HstuAttentionBatchedCombineKargs>;

  template <bool Cond = !kIsJagged>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* o_acc_ptr, // workspace for accumulation of o
      const void* lse_acc_ptr, // workspace for accummulation of lse
      void* o_ptr,
      void* lse_ptr,
      ck_tile::index_t batch_stride_o,
      ck_tile::index_t batch_stride_lse,
      ck_tile::index_t seq_stride_o,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t nhead_stride_o,
      ck_tile::index_t nhead_stride_lse,
      ck_tile::index_t seqlen_q,
      ck_tile::index_t num_head,
      ck_tile::index_t num_splits, // number of split seqlen_kv
      ck_tile::index_t hdim_v) {
    Kargs kargs{
        {o_acc_ptr,
         o_ptr,
         batch_stride_o,
         seq_stride_o,
         nhead_stride_o,
         seqlen_q,
         num_head,
         num_splits,
         hdim_v,
         true}, // almost_invariant_seqlen
        {}, // place holder for softmax
        {}, // place holder for LSE
    };

    if constexpr (kUseSoftmax) {
      kargs.lse_acc_ptr = lse_acc_ptr;
    }
    if constexpr (kStoreLSE) {
      kargs.lse_ptr = lse_ptr;
      kargs.batch_stride_lse = batch_stride_lse;
      kargs.seq_stride_lse = seq_stride_lse;
      kargs.nhead_stride_lse = nhead_stride_lse;
    }

    return kargs;
  }

  template <bool Cond = kIsJagged>
  CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs> MakeKargs(
      const void* o_acc_ptr, // workspace for accumulation of o
      const void* lse_acc_ptr, // workspace for accummulation of lse
      void* o_ptr,
      void* lse_ptr,
      ck_tile::index_t seq_stride_o,
      ck_tile::index_t seq_stride_lse,
      ck_tile::index_t nhead_stride_o,
      ck_tile::index_t nhead_stride_lse,
      const void* seq_q_offsets_ptr,
      ck_tile::index_t num_head,
      ck_tile::index_t num_splits, // number of split seqlen_kv
      ck_tile::index_t hdim_v,
      bool almost_invariant_seqlen) {
    Kargs kargs{
        {o_acc_ptr,
         o_ptr,
         seq_stride_o,
         nhead_stride_o,
         reinterpret_cast<const int32_t*>(seq_q_offsets_ptr),
         num_head,
         num_splits,
         hdim_v,
         0, /* seqlen_q will be updated later*/
         almost_invariant_seqlen},
        {}, // place holder for softmax
        {}, // place holder for LSE
    };

    if constexpr (kUseSoftmax) {
      kargs.lse_acc_ptr = lse_acc_ptr;
    }
    if constexpr (kStoreLSE) {
      kargs.lse_ptr = lse_ptr;
      kargs.seq_stride_lse = seq_stride_lse;
      kargs.nhead_stride_lse = nhead_stride_lse;
    }

    return kargs;
  }

  CK_TILE_HOST static constexpr auto GridSize(
      ck_tile::index_t batch_size_,
      ck_tile::index_t nhead_,
      ck_tile::index_t seqlen_,
      bool almost_invariant_seqlen) {
    ck_tile::index_t num_tile_in_seqlen =
        ck_tile::integer_divide_ceil(seqlen_, HstuAttentionPipeline::kM);

    if (almost_invariant_seqlen) {
      return dim3(batch_size_, nhead_, num_tile_in_seqlen);
    } else {
      return dim3(num_tile_in_seqlen, nhead_, batch_size_);
    }
  }

  CK_TILE_DEVICE static constexpr auto GetTileIndex(const Kargs& kargs) {
    if (kargs.almost_invariant_seqlen) {
      const index_t i_batch = blockIdx.x;
      const index_t i_nhead = blockIdx.y;
      const index_t i_tile_m = blockIdx.z;
      return ck_tile::make_tuple(i_tile_m, i_nhead, i_batch);
    } else {
      const index_t i_tile_m = blockIdx.x;
      const index_t i_nhead = blockIdx.y;
      const index_t i_batch = blockIdx.z;
      return ck_tile::make_tuple(i_tile_m, i_nhead, i_batch);
    }
  }

  CK_TILE_HOST static constexpr auto BlockSize() {
    if (is_wave32()) {
      // it looks get_warp_size() always return 64 when called from host, so
      // halfing is needed to get actual BlockSize
      return dim3(kBlockSize / get_warp_size() * 32);
    } else
      return dim3(kBlockSize);
  }

  CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return ck_tile::max(
        HstuAttentionPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
  }

  CK_TILE_DEVICE void operator()(Kargs kargs) const {
    // allocate LDS
    __shared__ char smem_ptr[GetSmemSize()];

    const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex(kargs);

    long_index_t batch_offset_o_acc = 0;
    long_index_t batch_offset_o = 0;
    long_index_t batch_offset_lse_acc = 0;
    long_index_t batch_offset_lse = 0;

    if constexpr (kIsJagged) {
      // get starting offset for each batch
      const long_index_t query_start = kargs.seq_q_offsets_ptr[i_batch];

      // assume o_acc is in compact shape of [batch_size, max_seqlen_q,
      // num_head, num_splits, hdim]
      batch_offset_o_acc =
          query_start * kargs.num_head * kargs.num_splits * kargs.hdim_v;

      batch_offset_o = query_start * kargs.seq_stride_o;

      // assume lse_acc is in compact shape of [batch_size, max_seqlen_q,
      // num_head, num_splits]
      if constexpr (kUseSoftmax) {
        batch_offset_lse_acc = query_start * kargs.num_head * kargs.num_splits;
      }
      if constexpr (kStoreLSE) {
        batch_offset_lse = query_start * kargs.seq_stride_lse;
      }

      kargs.seqlen_q = kargs.seq_q_offsets_ptr[i_batch + 1] -
          kargs.seq_q_offsets_ptr[i_batch];
    } else {
      // assume o_acc is in compact shape of [batch_size, seqlen_q, num_head,
      // num_splits, hdim]
      batch_offset_o_acc = static_cast<long_index_t>(i_batch) * kargs.seqlen_q *
          kargs.num_head * kargs.num_splits * kargs.hdim_v;

      batch_offset_o =
          static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;

      // assume l_acc is in compact shape of [batch_size, seqlen_q, num_head,
      // num_splits]
      if constexpr (kUseSoftmax) {
        batch_offset_lse_acc = static_cast<long_index_t>(i_batch) *
            kargs.seqlen_q * kargs.num_head * kargs.num_splits;
      }
      if constexpr (kStoreLSE) {
        batch_offset_lse =
            static_cast<long_index_t>(i_batch) * kargs.batch_stride_lse;
      }
    }

    index_t i_m;

    i_m = __builtin_amdgcn_readfirstlane(i_tile_m * HstuAttentionPipeline::kM);

    if (kargs.seqlen_q <= i_m)
      return;

    // assume o_acc is in compact shape of [batch_size, seqlen, num_head,
    // num_splits, hdim]
    const OaccDataType* o_acc_ptr =
        reinterpret_cast<const OaccDataType*>(kargs.o_acc_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.num_splits * kargs.hdim_v +
        batch_offset_o_acc;
    ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
        static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
        batch_offset_o;

    // Oacc DRAM and Oacc DRAM window
    auto seq_stride_o_acc = kargs.num_head * kargs.num_splits * kargs.hdim_v;
    const auto o_acc_dram_naive =
        make_naive_tensor_view<address_space_enum::global>(
            o_acc_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(seq_stride_o_acc, 1),
            number<HstuAttentionPipeline::kAlignmentO>{},
            number<1>{});

    const auto o_acc_dram = pad_tensor_view(
        o_acc_dram_naive,
        make_tuple(
            number<HstuAttentionPipeline::kM>{},
            number<HstuAttentionPipeline::kOHeaddim>{}),
        sequence<false, kPadHeadDimO>{});

    auto o_acc_dram_window = make_tile_window(
        o_acc_dram,
        make_tuple(
            number<HstuAttentionPipeline::kM>{},
            number<HstuAttentionPipeline::kOHeaddim>{}),
        {i_m, 0});

    auto o_acc_tile = [&]() {
      if constexpr (kUseSoftmax) {
        // assume l_acc is in compact shape of [batch_size, seqlen_q, num_head,
        // num_splits]
        const LSEDataType* lse_acc_ptr =
            reinterpret_cast<const LSEDataType*>(kargs.lse_acc_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.num_splits +
            batch_offset_lse_acc;

        // LSEacc DRAM and LSEacc DRAM window
        auto seq_stride_lse_acc = kargs.num_head * kargs.num_splits;

        auto lse_acc_desc = make_naive_tensor_descriptor(
            make_tuple(kargs.seqlen_q, kargs.num_splits),
            make_tuple(seq_stride_lse_acc, 1),
            number<HstuAttentionPipeline::kAlignmentLSEacc>{},
            number<1>{});

        auto lse_acc_buf_view = make_buffer_view<address_space_enum::global>(
            lse_acc_ptr,
            lse_acc_desc.get_element_space_size(),
            -numeric<LSEDataType>::infinity());

        auto lse_acc_dram_naive =
            tensor_view<decltype(lse_acc_buf_view), decltype(lse_acc_desc)>{
                lse_acc_buf_view, lse_acc_desc};

        const auto lse_acc_dram = pad_tensor_view(
            lse_acc_dram_naive,
            make_tuple(
                number<HstuAttentionPipeline::kM>{},
                number<HstuAttentionPipeline::kMaxSplits>{}),
            sequence<false, kPadNumSplits>{});

        auto lse_acc_dram_window = make_tile_window(
            lse_acc_dram,
            make_tuple(
                number<HstuAttentionPipeline::kM>{},
                number<HstuAttentionPipeline::kMaxSplits>{}),
            {i_m, 0});

        auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
          constexpr auto lse_dram_window_lengths =
              make_tuple(number<HstuAttentionPipeline::kM>{});
          if constexpr (kStoreLSE) {
            LSEDataType* lse_ptr =
                reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse +
                batch_offset_lse;

            const auto lse_dram = [&]() {
              const auto lse_dram_naive =
                  make_naive_tensor_view<address_space_enum::global>(
                      lse_ptr,
                      make_tuple(kargs.seqlen_q),
                      make_tuple(kargs.seq_stride_lse),
                      number<1>{},
                      number<1>{});

              return pad_tensor_view(
                  lse_dram_naive, lse_dram_window_lengths, sequence<false>{});
            }();

            return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m});
          } else {
            return make_null_tile_window(lse_dram_window_lengths);
          }
        }();

        return HstuAttentionPipeline{}(
            lse_acc_dram_window,
            o_acc_dram_window,
            lse_dram_window,
            kargs.hdim_v,
            kargs.num_splits,
            smem_ptr);
      } else {
        return HstuAttentionPipeline{}(
            o_acc_dram_window, kargs.hdim_v, kargs.num_splits);
      }
    }();

    // O DRAM and O DRAM window
    auto o_dram = [&]() {
      const auto o_dram_naive =
          make_naive_tensor_view<address_space_enum::global>(
              o_ptr,
              make_tuple(kargs.seqlen_q, kargs.hdim_v),
              make_tuple(kargs.seq_stride_o, 1),
              number<HstuAttentionPipeline::kAlignmentO>{},
              number<1>{});

      return pad_tensor_view(
          o_dram_naive,
          make_tuple(
              number<HstuAttentionPipeline::kM>{},
              number<HstuAttentionPipeline::kOHeaddim>{}),
          sequence<false, kPadHeadDimO>{});
    }();

    auto o_dram_window = make_tile_window(
        o_dram,
        make_tuple(
            number<HstuAttentionPipeline::kM>{},
            number<HstuAttentionPipeline::kOHeaddim>{}),
        {i_m, 0});

    EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
  }
};

} // namespace ck_tile
