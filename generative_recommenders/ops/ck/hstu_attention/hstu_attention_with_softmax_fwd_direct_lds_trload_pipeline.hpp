// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"
#include "hstu_attention_pipeline_helper.hpp"

namespace ck_tile {

template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
struct HstuAttentionWithSoftmaxFwdPipelineQRKSVSDirectLdsTrLoad {
  using Problem = remove_cvref_t<Problem_>;
  using Traits = remove_cvref_t<Traits_>;
  using Policy = remove_cvref_t<Policy_>;
  using QKVDataType = remove_cvref_t<typename Problem::InOutDataType>;
  using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
  using CompDataType = remove_cvref_t<typename Problem::CompDataType>;
  using BiasDataType = remove_cvref_t<typename Problem::BiasDataType>;
  using PDataType = remove_cvref_t<typename Problem::InOutDataType>;
  using ODataType = remove_cvref_t<typename Problem::InOutDataType>;

  using HstuAttentionTileSetting =
      remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

  static constexpr index_t kBlockSize = Problem::kBlockSize;

  static constexpr index_t kM0 = HstuAttentionTileSetting::kM0;
  static constexpr index_t kN0 = HstuAttentionTileSetting::kN0;
  static constexpr index_t kN0Sub = HstuAttentionTileSetting::kN0Sub;
  static constexpr index_t kN1 = HstuAttentionTileSetting::kN1;
  static constexpr index_t kK1 = HstuAttentionTileSetting::kK1;
  static constexpr index_t kQKHeaddim = HstuAttentionTileSetting::kQKHeaddim;

  static_assert(
      kQKHeaddim <= 256,
      "hdim bigger than 256 is not suitable for this pipeline!");

  static_assert(
      Problem::kUseSoftmax == true,
      "This pipeline only works with using softmax");

  static constexpr bool kIsJagged = Problem::kIsJagged;
  static constexpr auto kHasBias = Problem::kHasBias;
  static constexpr bool kHasDropout = Problem::kHasDropout;
  static constexpr bool kHasCausal = Problem::kHasCausal;

  static constexpr bool kUseTrLoad = true;

  static constexpr bool kPadSeqLenQ = Traits::kPadSeqLenQ;
  static constexpr bool kPadSeqLenK = Traits::kPadSeqLenK;
  static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
  static constexpr bool kPadHeadDimV = Traits::kPadHeadDimV;

  static_assert(
      kPadHeadDimQK == false && kPadHeadDimV == false,
      "Check failed!");

  // use direct-lds buffer_load_dword instructions to load K & V data to Lds
  // from Dram
  static constexpr bool kUseDirectLds = true;

  // gemm_0 over both kN0Sub halves and gemm_1 over both kK1 slices each run as
  // one LDS-read stream with kGemm{0,1}Prefetch B warp tiles in flight. Only
  // with 16x16x32 warp gemms (the jagged mtile-128 softmax tile): the
  // vote-gated softmax's fp16 row sum needs each lane to hold at most 16 P
  // values per row.
  static constexpr bool kGemmRunAhead = kQKHeaddim == 128 && kM0 == 128 &&
      std::is_same_v<typename Problem::HstuAttentionTileSetting::Gemm0WarpTile,
                     sequence<16, 16, 32>>;
  static constexpr index_t kGemm0Prefetch = 4;
  static constexpr index_t kGemm1Prefetch = 4;

  // Softmax main loop: l is kept as per-lane partial sums and reduced across
  // lanes once after the loop; the exact cross-lane rowmax (and the o_acc/l
  // rescale) only runs on KV tiles where some lane sees a value that defeats
  // the "keep the previous maximum" heuristic, which is decided by a wave-wide
  // vote on the lane-local maxima.
  static constexpr bool kVoteGatedSoftmax = kGemmRunAhead && !kHasBias &&
      !kHasDropout && std::is_same_v<CompDataType, float>;
  // With kVoteGatedSoftmax and fp16 P, l is accumulated in fp32 on the matrix
  // pipe from the fp16 P that feeds gemm_1 (see rowsum_mfma) instead of from
  // the fp32 values with v_pk_add_f32.
  static constexpr bool kSoftmaxMfmaRowsum =
      kVoteGatedSoftmax && std::is_same_v<PDataType, half_t>;

  // last dimension vector length used to create tensor view(and decide
  // buffer_load vector length)
  // ... together with tensor distribution. tensor dist should able to overwrite
  // this
  static constexpr index_t kAlignmentQ =
      Policy::template GetAlignmentQ<Problem>();
  static constexpr index_t kAlignmentK =
      Policy::template GetAlignmentK<Problem>();
  static constexpr index_t kAlignmentV =
      Policy::template GetAlignmentV<Problem, true /*kUseTrLoad*/>();

  // since dim num_splits is the fastest dim and each workgroup accesses only
  // one value in this dim
  static constexpr index_t kAlignmentLSEacc = 1;

  static constexpr index_t kAlignmentO =
      Policy::template GetAlignmentO<Problem, true /*kUseTrLoad*/>();
  static constexpr index_t kAlignmentBias =
      kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

  // used by NRepetitions2DEpilogue
  static constexpr index_t kGemm1SingleRepN =
      Policy::template GetPVTBlockGemmSingleRepN<Problem>();

  static constexpr index_t kBlockPerCu = []() {
    if constexpr (Traits::kBlockPerCu != -1)
      return Traits::kBlockPerCu;
    else {
      if constexpr (kQKHeaddim == 64) {
        if constexpr (kM0 == 128)
          return 3;
        if constexpr (kM0 == 64)
          return 3;

        return 2;
      } else if constexpr (kQKHeaddim == 96 || kQKHeaddim == 128) {
        if constexpr (kHasBias)
          return 2;
        else
          return 2;
      } else if constexpr (kQKHeaddim == 256) {
        return 1;
      } else {
        return 1;
      };
    }
  }();

  using DropoutType =
      std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

  CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return Policy::template GetSmemSize<
        Problem,
        true /*kPipelineUseTrLoad*/,
        true /*kUseDirectLds*/>();
  }

  // One KV tile of the online softmax (see kVoteGatedSoftmax). On return
  // pcomp_tile holds exp2(s * exp_scale + nml), m the stabilizing row max, nml
  // = -m * exp_scale (0 while m is -inf, whose rows only hold -inf scores) and
  // l the per-lane partial row sums (with kSoftmaxMfmaRowsum, lacc holds the
  // row sums and l is untouched). The keep-m_old test is monotone in the row
  // max, so "every lane keeps m_old with its own lane-local max" is exactly
  // "the row keeps m_old with the exact row max".
  template <typename PTile, typename MLTile, typename OTile, typename LAcc>
  CK_TILE_DEVICE static void softmax_vote_gated(
      PTile& pcomp_tile,
      MLTile& m,
      MLTile& nml,
      MLTile& l,
      LAcc& lacc,
      OTile& o_acc,
      float exp_scale,
      float m_reuse_threshold) {
    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
    constexpr float kNegInf = -numeric<float>::infinity();
    constexpr index_t kRowLn =
        PTile::get_thread_buffer_size() / MLTile::get_thread_buffer_size();
    static_assert(
        kRowLn % 4 == 0 && (kRowLn & (kRowLn - 1)) == 0, "Check failed!");

    const auto m_old = m;
    auto m_loc = m;
    block_tile_reduce(m_loc, pcomp_tile, sequence<1>{}, f_max);

    // m_old == -inf always votes (-inf - x is -inf or NaN)
    constexpr auto m_spans = MLTile::get_distributed_spans();
    bool need_exact = false;
    sweep_tile_span(m_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      need_exact |= !(m_old[i_idx] - m_loc[i_idx] >= m_reuse_threshold);
    });

    if (__builtin_expect(__builtin_amdgcn_ballot_w64(need_exact) != 0, 0)) {
      block_tile_reduce_sync(m_loc, f_max, bool_constant<false>{});

      constexpr auto o_spans = OTile::get_distributed_spans();
      sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);
        const float m_new = (m_old[i_idx] - m_loc[i_idx] >= m_reuse_threshold)
            ? m_old[i_idx]
            : m_loc[i_idx];
        m(i_idx) = m_new;
        nml(i_idx) = m_new == kNegInf ? 0.f : -m_new * exp_scale;
        if (m_new > m_old[i_idx]) {
          const float tmp =
              __builtin_amdgcn_exp2f((m_old[i_idx] - m_new) * exp_scale);
          if constexpr (kSoftmaxMfmaRowsum) {
            constexpr index_t r = row_slot<MLTile, decltype(i_idx)>();
            lacc(r) = lacc[r] * tmp;
          } else
            l(i_idx) *= tmp;
          sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
            o_acc(make_tuple(idx0, idx1)) *= tmp;
          });
        }
      });
    }

    constexpr auto p_spans = PTile::get_distributed_spans();
    sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      {
        const fp32x2_t c = {nml[i_idx], nml[i_idx]};
        fp32x2_t e[kRowLn / 2];
        index_t k = 0;
        sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
          e[k / 2][k % 2] = pcomp_tile[make_tuple(idx0, idx1)];
          k++;
        });
        static_for<0, kRowLn / 2, 1>{}([&](auto j) {
          e[j] = __builtin_elementwise_fma(
              e[j], fp32x2_t{exp_scale, exp_scale}, c);
        });
        static_for<0, kRowLn / 2, 1>{}([&](auto j) {
          e[j] = fp32x2_t{
              __builtin_amdgcn_exp2f(e[j].x), __builtin_amdgcn_exp2f(e[j].y)};
        });
        k = 0;
        sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
          pcomp_tile(make_tuple(idx0, idx1)) = e[k / 2][k % 2];
          k++;
        });
      }

      if constexpr (!kSoftmaxMfmaRowsum) {
        fp32x2_t e[kRowLn / 2];
        index_t k = 0;
        sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
          e[k / 2][k % 2] = pcomp_tile[make_tuple(idx0, idx1)];
          k++;
        });
        static_for<0, integer_log2_floor(kRowLn / 2), 1>{}([&](auto lvl) {
          constexpr index_t half = (kRowLn / 4) >> lvl;
          static_for<0, half, 1>{}([&](auto j) { e[j] = e[j] + e[j + half]; });
        });
        l(i_idx) += e[0].x + e[0].y;
      }
    });
  }

  // Thread-buffer slot of row i_idx of an [kM0] row tile.
  template <typename MLTile, typename IIdx>
  CK_TILE_DEVICE static constexpr index_t row_slot() {
    constexpr auto y_idx =
        MLTile::get_tile_distribution().get_y_indices_from_distributed_indices(
            IIdx{});
    return typename MLTile::ThreadTensorDesc{}.calculate_offset(y_idx);
  }

  // lacc[r] += rowsum(P) of the lane's row r (kSoftmaxMfmaRowsum).
  // v_mfma_f32_16x16x32_f16 with A = 1 and B = P: B column n is row n of the
  // 16-row block (lanes n, n + 16, n + 32, n + 48 hold its P values), so every
  // D element a lane holds is the full row sum of its own row, accumulated in
  // fp32 from the fp16 P that gemm_1 consumes. ones is the all-1 A operand,
  // kept in VGPRs across the KV loop by the caller.
  template <typename PTile, typename MLTile, typename LAcc>
  CK_TILE_DEVICE static void
  rowsum_mfma(const PTile& p_tile, LAcc& lacc, const fp16x8_t& ones) {
    constexpr index_t kRowLn =
        PTile::get_thread_buffer_size() / MLTile::get_thread_buffer_size();
    static_assert(kRowLn % 8 == 0, "Check failed!");
    constexpr index_t kRows = MLTile::get_thread_buffer_size();
    constexpr auto p_spans = PTile::get_distributed_spans();
    fp16x8_t b[kRows][kRowLn / 8];
    sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      constexpr index_t r = row_slot<MLTile, decltype(i_idx)>();
      index_t k = 0;
      sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
        b[r][k / 8][k % 8] = p_tile[make_tuple(idx0, idx1)];
        k++;
      });
    });
    // row-interleaved, so consecutive MFMAs do not chain on the same
    // accumulator
    static_for<0, kRowLn / 8, 1>{}([&](auto j) {
      static_for<0, kRows, 1>{}([&](auto r) {
        lacc(r) = __builtin_amdgcn_mfma_f32_16x16x32_f16(
            ones, b[r][j], lacc[r], 0, 0, 0);
      });
    });
  }

  template <
      typename QDramBlockWindowTmp,
      typename KDramBlockWindowTmp,
      typename VDramBlockWindowTmp,
      typename BiasDramBlockWindowTmp,
      typename LSEorLSEaccDramBlockWindow,
      typename NullRandValDramWindowTmp,
      typename HstuMask>
  CK_TILE_DEVICE auto operator()(
      const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*kQKHeaddim tile
      const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*kQKHeaddim tile
      const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
      const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
      LSEorLSEaccDramBlockWindow& lse_or_lse_acc_dram_block_window, // M0 tile
      NullRandValDramWindowTmp& null_randval_window_tmp, // M0*N0 tile
      index_t seqlen_k_start,
      index_t seqlen_k_end,
      HstuMask& mask,
      float scale_s, // scaling value exerted on the immediate Q@K result
      float scale_p, // scaling value exerted on the SiLu result
      void* smem_ptr,
      DropoutType& dropout) const {
    ignore = scale_p;
    ignore = null_randval_window_tmp;

    static_assert(
        std::is_same_v<
            QKVDataType,
            remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                QKVDataType,
                remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
            std::is_same_v<
                QKVDataType,
                remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
        "wrong!");

    static_assert(
        kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kQKHeaddim ==
                KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
            kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
            kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
        "wrong!");

    constexpr index_t n0_loops = kN0 / kN0Sub;
    constexpr index_t k1_loops = kN0 / kK1;

    static_assert(
        (k1_loops % n0_loops == 0) || (n0_loops % k1_loops == 0),
        "k1_loops % n0_loops == 0 or n0_loops % k1_loops == 0 required by this pipeline");

    constexpr index_t k1loops_per_n0sub = (k1_loops / n0_loops);
    constexpr index_t n0loops_per_k1 = (n0_loops / k1_loops);

    // Non-causal cross attention without a local window admits every (row, col)
    // pair (IsTokenPairInsideMask is identically true), so only the kv tail
    // needs masking.
    using MaskType = remove_cvref_t<HstuMask>;
    constexpr bool kKvBoundOnlyMask = kVoteGatedSoftmax &&
        MaskType::kIsCrossAttention && !MaskType::kUseLocal &&
        !MaskType::IsMasking;

    constexpr auto NumKLdsBuffers =
        Policy::template GetNumKLdsBuffers<Problem, true /*kUseDirectLds*/>();
    constexpr auto NumVLdsBuffers =
        Policy::template GetNumVLdsBuffers<Problem, true /*kUseDirectLds*/>();

    static_assert(NumKLdsBuffers == n0_loops, "Check failed!");
    static_assert(NumVLdsBuffers == k1_loops, "Check failed!");

    // Block GEMM
    constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
    constexpr auto gemm_1 = Policy::
        template GetPVTBlockGemm<Problem, true /*kPipelineUseTrLoad*/>();

    using Gemm0Combined =
        decltype(Policy::template GetQKCombinedBlockGemm<Problem>());

    // SaccBlockTile size is [kM0, kN0Sub]
    // PcompBlockTile size is [kM0, kN0]
    using SaccBlockTileType =
        decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
    using CombineSaccBlockTileType =
        decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
    using PcompBlockTileType =
        decltype(cast_tile<CompDataType>(CombineSaccBlockTileType{}));

    SaccBlockTileType sacc_tile;
    PcompBlockTileType pcomp_tile;

    // reduction function for softmax
    const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    using MLBlockTileType = decltype(block_tile_reduce<CompDataType>(
        PcompBlockTileType{}, sequence<1>{}, f_max, CompDataType{0}));

    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};

    using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
    OaccBlockTileType o_acc;

    if (seqlen_k_end <= seqlen_k_start) {
      clear_tile(o_acc);

      if constexpr (!is_null_tile_window_v<LSEorLSEaccDramBlockWindow>) {
        auto lse_or_lse_acc = make_static_distributed_tensor<CompDataType>(
            m.get_tile_distribution());

        set_tile(lse_or_lse_acc, -numeric<CompDataType>::infinity());

        store_tile(lse_or_lse_acc_dram_block_window, lse_or_lse_acc);
      }

      return o_acc;
    };

    auto q_dram_window = make_tile_window(
        q_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
        q_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeQRegTileDistribution<Problem>());

    auto q_tile = load_tile(q_dram_window);

    const auto q_origin = q_dram_window.get_window_origin();

    // K/V are DMAed from fixed dram windows at the sequence start, the row
    // offset going in as a scalar dram offset: the XOR of the dram view is on
    // the row within the window, i.e. the row within an LDS buffer
    const auto k_dram_window = make_tile_window_raw(
        k_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
        {0, 0},
        Policy::template MakeKDramTileDistribution<
            Problem,
            true /*kUseDirectLds*/>());

    const auto& k_dram_desc = k_dram_block_window_tmp.get_bottom_tensor_view()
                                  .get_tensor_descriptor();

    // K tile in LDS
    QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);

    auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
        k_lds_ptr,
        Policy::template MakeKLdsBlockDescriptor<
            Problem,
            true /*kUseDirectLds*/>());

    auto k_lds_write_monolithic_window = make_tile_window(
        k_lds_write_view,
        Policy::
            template MakeKLdsBlockDescriptor<Problem, true /*kUseDirectLds*/>()
                .get_lengths(),
        {0, 0});

    // the dram view is XOR-swizzled per row: group 1 of row 1 maps back to
    // group 0, so the row stride is its offset
    constexpr index_t kKSwizzleUnit =
        Policy::template GetKDirectLdsSwizzleUnit<Problem>();
    static_assert(kQKHeaddim / kKSwizzleUnit >= 2, "Check failed!");
    const index_t k_row_stride =
        k_dram_desc.calculate_offset(make_tuple(1, kKSwizzleUnit)) -
        k_dram_desc.calculate_offset(make_tuple(0, 0));
    // elements per DMA of each thread: the KPerThread of the K dram tile
    // distribution
    constexpr index_t kKDmaVector =
        remove_cvref_t<decltype(k_dram_window)>::Traits::ScalarPerVector;
    static_assert(
        kKDmaVector * sizeof(QKVDataType) == 4 * sizeof(int32_t),
        "the raw direct-to-LDS DMA moves dwordx4");
    index_t k_row_offset =
        amd_wave_read_first_lane(seqlen_k_start * k_row_stride);
    const index_t k_row_step = kN0Sub * k_row_stride;

    const auto k_issues = generate_tuple(
        [&](auto i_buf) {
          return detail::
              MakeDirectLdsIssues<NumKLdsBuffers, kBlockSize / get_warp_size()>(
                  k_lds_write_monolithic_window,
                  number<i_buf % NumKLdsBuffers>{},
                  k_dram_window);
        },
        number<n0_loops>{});

    __builtin_amdgcn_sched_barrier(
        LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

    // provide partition_index for LDS tile window so that warp_id is in vgpr
    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
        k_lds_ptr,
        Policy::template MakeKLdsBlockDescriptor<
            Problem,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>());
    auto k_lds_read_monolithic_window = make_tile_window(
        k_lds_read_view,
        Policy::template MakeKLdsBlockDescriptor<
            Problem,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>()
            .get_lengths(),
        {0, 0});

    using k_lds_read_window_type = decltype(get_slice_tile(
        k_lds_read_monolithic_window,
        sequence<0, 0>{},
        sequence<kN0Sub, kQKHeaddim>{}));

    statically_indexed_array<k_lds_read_window_type, n0_loops>
        k_lds_read_windows;

    static_for<0, n0_loops, 1>{}([&](auto i_buf) {
      k_lds_read_windows[i_buf] = get_slice_tile(
          k_lds_read_monolithic_window,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
    });

    static_for<0, n0_loops, 1>{}([&](auto i_n0) {
      detail::IssueDirectLds(
          static_cast<QKVDataType*>(smem_ptr),
          k_dram_window,
          k_issues(number<i_n0>{}),
          k_row_offset);
      k_row_offset += k_row_step;
    });

    __builtin_amdgcn_sched_barrier(
        LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

    // V tile in LDS
    QKVDataType* v_lds_ptr = reinterpret_cast<QKVDataType*>(
        reinterpret_cast<char*>(smem_ptr) +
        Policy::template GetSmemSizeK<Problem, true /*kUseDirectLds*/>());

    auto v_lds_write_view = make_tensor_view<address_space_enum::lds>(
        v_lds_ptr,
        Policy::template MakeVLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>());
    auto v_lds_write_monolithic_window = make_tile_window(
        v_lds_write_view,
        Policy::template MakeVLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>()
            .get_lengths(),
        {0, 0});

    auto v_lds_trload_view = make_tensor_view<address_space_enum::lds>(
        v_lds_ptr,
        Policy::template MakeVLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>());
    auto v_lds_trload_monolithic_window = make_tile_window(
        v_lds_trload_view,
        Policy::template MakeVLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>()
            .get_lengths(),
        {0, 0});

    using v_lds_trload_window_type = decltype(get_slice_tile(
        v_lds_trload_monolithic_window,
        sequence<0, 0>{},
        sequence<kK1, kN1>{}));

    statically_indexed_array<v_lds_trload_window_type, k1_loops>
        v_lds_trload_windows;

    static_for<0, k1_loops, 1>{}([&](auto i_buf) {
      v_lds_trload_windows[i_buf] = get_slice_tile(
          v_lds_trload_monolithic_window,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kN1>{});
    });

    const auto v_dram_window = make_tile_window_raw(
        v_dram_block_window_tmp.get_bottom_tensor_view(),
        v_dram_block_window_tmp.get_window_lengths(),
        {0, 0},
        Policy::template MakeVDramTileDistribution<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>());

    const auto& v_dram_desc = v_dram_block_window_tmp.get_bottom_tensor_view()
                                  .get_tensor_descriptor();
    // the dram view is XOR-swizzled per row: group 1 of row 1 maps back to
    // group 0, so the row stride is its offset
    constexpr index_t kVSwizzleUnit =
        Policy::template GetVDirectLdsSwizzleUnit<Problem>();
    static_assert(kN1 / kVSwizzleUnit >= 2, "Check failed!");
    const index_t v_row_stride =
        v_dram_desc.calculate_offset(make_tuple(1, kVSwizzleUnit)) -
        v_dram_desc.calculate_offset(make_tuple(0, 0));
    // elements per DMA of each thread: the NPerThread of the V dram tile
    // distribution
    constexpr index_t kVDmaVector =
        remove_cvref_t<decltype(v_dram_window)>::Traits::ScalarPerVector;
    static_assert(
        kVDmaVector * sizeof(QKVDataType) == 4 * sizeof(int32_t),
        "the raw direct-to-LDS DMA moves dwordx4");
    index_t v_row_offset =
        amd_wave_read_first_lane(seqlen_k_start * v_row_stride);
    const index_t v_row_step = kK1 * v_row_stride;

    const auto v_issues = generate_tuple(
        [&](auto i_buf) {
          return detail::
              MakeDirectLdsIssues<NumVLdsBuffers, kBlockSize / get_warp_size()>(
                  v_lds_write_monolithic_window,
                  number<i_buf % NumVLdsBuffers>{},
                  v_dram_window);
        },
        number<k1_loops>{});

    [[maybe_unused]] const auto f_exp = [&](CompDataType x) {
      if constexpr (std::is_same_v<CompDataType, float>) {
        return __expf(x);
      } else {
        return exp(x);
      }
    };

    // Softmax numerator in the log2 domain: exp(s * scale_s - m) is evaluated
    // as exp2(s * exp_scale - m * exp_scale), i.e. one v_fma_f32 plus one
    // v_exp_f32 per element. That lets S leave gemm_0 unscaled and removes the
    // separate `pcomp *= scale_s` pass over the whole kM0 x kN0 tile on every
    // KV block. A bias has to be added in the already-scaled S domain, so that
    // configuration keeps applying scale_s up front.
    constexpr bool kFoldScaleIntoExp = !kHasBias;
    const CompDataType exp_scale = type_convert<CompDataType>(
        (kFoldScaleIntoExp ? scale_s : 1.0f) * ck_tile::log2e_v<float>);
    // m/m_old live in the unscaled S domain when the scale is folded, so the
    // "keep the previous maximum" heuristic threshold has to be unscaled too.
    const CompDataType m_reuse_threshold = type_convert<CompDataType>(
        kFoldScaleIntoExp ? (-8.0f / scale_s) : -8.0f);
    const auto f_exp2 = [](CompDataType x) {
      if constexpr (std::is_same_v<CompDataType, float>) {
        return __builtin_amdgcn_exp2f(x);
      } else {
        return type_convert<CompDataType>(
            __builtin_amdgcn_exp2f(type_convert<float>(x)));
      }
    };

    const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
    auto bias_dram_window = make_tile_window(
        bias_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kK1>{}),
        {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
        Policy::template MakeBiasDramTileDistribution<Problem>());

    auto null_randval_window =
        dropout.template MakeRandvalDramWindow<Gemm0Combined>(
            null_randval_window_tmp, seqlen_k_start);

    clear_tile(o_acc);
    set_tile(m, -numeric<CompDataType>::infinity());
    clear_tile(l);

    // kVoteGatedSoftmax state: -m * exp_scale (l holds per-lane partial row
    // sums)
    auto nml = MLBlockTileType{};
    if constexpr (kVoteGatedSoftmax)
      clear_tile(nml);

    // kSoftmaxMfmaRowsum row sums, indexed by row_slot
    array<fp32x4_t, MLBlockTileType::get_thread_buffer_size()> lacc;
    fp16x8_t mfma_ones = {1, 1, 1, 1, 1, 1, 1, 1};
    if constexpr (kSoftmaxMfmaRowsum) {
      static_for<0, MLBlockTileType::get_thread_buffer_size(), 1>{}(
          [&](auto r) { lacc(r) = fp32x4_t{0.f, 0.f, 0.f, 0.f}; });
      // kept in VGPRs instead of being rematerialized every tile
      asm volatile("" : "+v"(mfma_ones));
    }

    auto seqlen_k_curr = seqlen_k_start;

    do {
      // wait for both the Direct-Lds K copying and reading from v_lds to be
      // done
      s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>();

      // STAGE 1, Gemm_0 ( S = Q@K )
      if constexpr (kGemmRunAhead) {
        // this tile's V DMA is issued under the first K reads of gemm_0
        CombineSaccBlockTileType s_tile;
        gemm_0.template run_ahead<kGemm0Prefetch>(
            s_tile, q_tile, k_lds_read_monolithic_window, [&]() {
              static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                detail::IssueDirectLds(
                    static_cast<QKVDataType*>(smem_ptr),
                    v_dram_window,
                    v_issues(number<i_k1>{}),
                    v_row_offset);
                v_row_offset += v_row_step;
                // one SGPR row offset, not a VGPR per issue (loop strength
                // reduction)
                asm volatile("" : "+s"(v_row_offset));
              });
            });
        pcomp_tile = cast_tile<CompDataType>(s_tile);
      } else {
        static_for<0, n0_loops, 1>{}([&](auto i_n0) {
          // load v_tiles used in current iteration
          if constexpr (k1_loops >= n0_loops) {
            static_for<0, k1loops_per_n0sub, 1>{}([&](auto j) {
              constexpr index_t i_k1 = i_n0 * k1loops_per_n0sub + j;

              detail::IssueDirectLds(
                  static_cast<QKVDataType*>(smem_ptr),
                  v_dram_window,
                  v_issues(number<i_k1>{}),
                  v_row_offset);
              v_row_offset += v_row_step;
            });
          } else {
            if constexpr ((i_n0 + 1) % n0loops_per_k1 == 0) {
              constexpr index_t i_k1 = (i_n0 + 1) / n0loops_per_k1 - 1;

              detail::IssueDirectLds(
                  static_cast<QKVDataType*>(smem_ptr),
                  v_dram_window,
                  v_issues(number<i_k1>{}),
                  v_row_offset);
              v_row_offset += v_row_step;
            }
          }

          // execute current unroll of gemm_0
          gemm_0(sacc_tile, q_tile, k_lds_read_windows[number<i_n0>{}]);

          auto tmp_tile = cast_tile<CompDataType>(sacc_tile);

          set_slice_tile(
              pcomp_tile,
              tmp_tile,
              sequence<0, i_n0 * kN0Sub>{},
              sequence<kM0, (i_n0 + 1) * kN0Sub>{});
        });
      }

      __builtin_amdgcn_sched_barrier(
          LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

      // STAGE 2, scale_s, add bias, mask, siLU
      if constexpr (kHasBias) {
        const auto bias_tile = load_tile(bias_dram_window);

        tile_elementwise_inout(
            [&scale_s](auto& x, const auto& y) {
              x = x * scale_s + type_convert<CompDataType>(y);
            },
            pcomp_tile,
            bias_tile);

        move_tile_window(bias_dram_window, {0, kN0});
      } else if constexpr (!kFoldScaleIntoExp) {
        tile_elementwise_inout(
            [&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
      }

      if constexpr (kKvBoundOnlyMask) {
        // the only masked scores are the columns past seqlen_k_end of the last
        // tile
        if (seqlen_k_curr + kN0 > seqlen_k_end) {
          const index_t kv_left = seqlen_k_end - seqlen_k_curr;
          constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
          sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
              const auto tile_idx = get_x_indices_from_distributed_indices(
                  pcomp_tile.get_tile_distribution(),
                  make_tuple(idx0, idx1),
                  partition_index);
              constexpr auto i_j_idx = make_tuple(idx0, idx1);
              pcomp_tile(i_j_idx) = tile_idx.at(number<1>{}) < kv_left
                  ? pcomp_tile[i_j_idx]
                  : -numeric<CompDataType>::infinity();
            });
          });
        }
      } else if (!mask.IsFullTileInsideMask(
                     q_origin.at(number<0>{}),
                     seqlen_k_curr,
                     number<kN0>{},
                     number<kM0>{})) {
        constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                pcomp_tile.get_tile_distribution(),
                make_tuple(idx0, idx1),
                partition_index);

            const auto row =
                q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
            const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
            constexpr auto i_j_idx = make_tuple(idx0, idx1);

            if (!mask.IsTokenPairInsideMask(row, col) || col >= seqlen_k_end) {
              pcomp_tile(i_j_idx) = -numeric<CompDataType>::infinity();
            };
          });
        });
      }

      if constexpr (kVoteGatedSoftmax) {
        softmax_vote_gated(
            pcomp_tile, m, nml, l, lacc, o_acc, exp_scale, m_reuse_threshold);
      } else {
        const auto m_old = m;

        block_tile_reduce(m, pcomp_tile, sequence<1>{}, f_max);
        block_tile_reduce_sync(m, f_max, bool_constant<false>{});

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        constexpr auto p_spans = decltype(pcomp_tile)::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);

          if (__builtin_expect(
                  m[i_idx] == -numeric<CompDataType>::infinity(),
                  0)) // unlikely
          {
            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
              constexpr auto i_j_idx = make_tuple(idx0, idx1);
              pcomp_tile(i_j_idx) = type_convert<CompDataType>(0.0f);
            });
          } else {
            // use the m_old[i] as the m-for-stablization if m_old[i] - m[i] >=
            // -8.0f and still keep the m-for-stablization in m[]
            if (m_old[i_idx] - m[i_idx] >= m_reuse_threshold)
              m(i_idx) = m_old[i_idx];

            const CompDataType m_log2 = m[i_idx] * exp_scale;

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
              constexpr auto i_j_idx = make_tuple(idx0, idx1);
              pcomp_tile(i_j_idx) =
                  f_exp2(pcomp_tile[i_j_idx] * exp_scale - m_log2);
            });
          }
        });

        auto rowsum_p = block_tile_reduce<CompDataType>(
            pcomp_tile, sequence<1>{}, f_sum, CompDataType{0});

        block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

        // adjust o_acc[] according to the update between m and m_old
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);

          if (__builtin_expect(
                  m[i_idx] == -numeric<CompDataType>::infinity(),
                  0)) // unlikely
          {
            l(i_idx) = rowsum_p[i_idx];
          } else if (m[i_idx] > m_old[i_idx]) {
            const auto tmp = f_exp2((m_old[i_idx] - m[i_idx]) * exp_scale);
            l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
              constexpr auto i_j_idx = make_tuple(idx0, idx1);
              o_acc(i_j_idx) *= tmp;
            });
          } else
            l(i_idx) = l[i_idx] + rowsum_p[i_idx];
        });
      }

      if constexpr (kHasDropout) {
        auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) +
            Policy::template GetSmemSizeK<Problem, true /*kUseDirectLds*/>() +
            Policy::template GetSmemSizeV<
                                   Problem,
                                   true /*kPipelineUseTrLoad*/,
                                   true /*kUseDirectLds*/>();

        dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
            randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);
      }

      seqlen_k_curr += kN0;

      auto p = cast_tile<PDataType>(pcomp_tile);

      // wait for both the Direct-Lds V copying and reading from k_lds to be
      // done
      s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>();

      // STAGE 3, Gemm_1 ( O = P@V )
      if constexpr (kGemmRunAhead) {
        // the next tile's K DMA is issued under the first V reads of gemm_1
        gemm_1.template run_ahead<kGemm1Prefetch>(
            o_acc, p, v_lds_trload_monolithic_window, [&]() {
              static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                detail::IssueDirectLds(
                    static_cast<QKVDataType*>(smem_ptr),
                    k_dram_window,
                    k_issues(number<i_n0>{}),
                    k_row_offset);
                k_row_offset += k_row_step;
                // one SGPR row offset, not a VGPR per issue (loop strength
                // reduction)
                asm volatile("" : "+s"(k_row_offset));
              });
            });
        if constexpr (kSoftmaxMfmaRowsum)
          rowsum_mfma<decltype(p), MLBlockTileType>(p, lacc, mfma_ones);
      } else {
        static_for<0, k1_loops, 1>{}([&](auto i_k1) {
          if constexpr (k1_loops >= n0_loops) {
            if constexpr ((i_k1 + 1) % k1loops_per_n0sub == 0) {
              constexpr index_t i_n0 = (i_k1 + 1) / k1loops_per_n0sub - 1;

              detail::IssueDirectLds(
                  static_cast<QKVDataType*>(smem_ptr),
                  k_dram_window,
                  k_issues(number<i_n0>{}),
                  k_row_offset);
              k_row_offset += k_row_step;
            }
          } else {
            static_for<0, n0loops_per_k1, 1>{}([&](auto j) {
              constexpr index_t i_n0 = i_k1 * n0loops_per_k1 + j;

              detail::IssueDirectLds(
                  static_cast<QKVDataType*>(smem_ptr),
                  k_dram_window,
                  k_issues(number<i_n0>{}),
                  k_row_offset);
              k_row_offset += k_row_step;
            });
          }

          gemm_1(
              o_acc,
              get_slice_tile(
                  p,
                  sequence<0, i_k1 * kK1>{},
                  sequence<kM0, (i_k1 + 1) * kK1>{}),
              v_lds_trload_windows[number<i_k1>{}]);
        });
      }
    } while (seqlen_k_curr < seqlen_k_end);

    if constexpr (kSoftmaxMfmaRowsum) {
      constexpr auto l_spans = MLBlockTileType::get_distributed_spans();
      sweep_tile_span(l_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);
        l(i_idx) = lacc[row_slot<MLBlockTileType, decltype(i_idx)>()][0];
      });
    } else if constexpr (kVoteGatedSoftmax) {
      block_tile_reduce_sync(l, f_sum, bool_constant<false>{});
    }

    // if pipeline is called from splitkv_kernel, the window shall not be null;
    // if pipeline is called from non-splitkv kernel, the window is null if
    // kStoreLSE is false
    if constexpr (!is_null_tile_window_v<LSEorLSEaccDramBlockWindow>) {
      // m[] is tracked on the unscaled S values when the scale is folded into
      // the exp; lse must still be reported in the scaled domain because the
      // combine kernel compares lse across splits.
      const CompDataType m_to_scaled_domain =
          type_convert<CompDataType>(kFoldScaleIntoExp ? scale_s : 1.0f);

      // store lse or lse_acc
      auto lse_or_lse_acc = make_static_distributed_tensor<CompDataType>(
          m.get_tile_distribution());

      constexpr auto lse_or_lse_acc_spans =
          decltype(lse_or_lse_acc)::get_distributed_spans();
      sweep_tile_span(
          lse_or_lse_acc_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            lse_or_lse_acc(i_idx) =
                m_[i_idx] * m_to_scaled_domain + log(l_[i_idx]);
          });

      store_tile(lse_or_lse_acc_dram_block_window, lse_or_lse_acc);
    }

    constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

    if constexpr (kVoteGatedSoftmax) {
      // one reciprocal per row instead of a guarded division per element
      sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);
        const CompDataType inv_l =
            m[i_idx] == -numeric<CompDataType>::infinity() ? 0.0f
                                                           : 1.0f / l[i_idx];
        sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
          o_acc(make_tuple(idx0, idx1)) *= inv_l;
        });
      });

      return o_acc;
    }

    sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
        constexpr auto i_j_idx = make_tuple(idx0, idx1);

        if (__builtin_expect(
                m[i_idx] == -numeric<CompDataType>::infinity(), 0)) // unlikely
        {
          o_acc(i_j_idx) = 0.0f;
        } else
          o_acc(i_j_idx) *= 1.0f / l[i_idx];
      });
    });

    return o_acc;
  }
};

} // namespace ck_tile
