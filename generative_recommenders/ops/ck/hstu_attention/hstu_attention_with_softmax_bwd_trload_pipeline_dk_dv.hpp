// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_bwd_kernel_2_pipeline_policy.hpp"

namespace ck_tile {

// Kernel 2 backward pipeline -- Softmax path (kUseSoftmax == true).
// Iterates over Q blocks for a fixed K/V tile; accumulates dK and dV.
// Naming: KR = K and V are register-resident, QS = Q and dO staged through LDS.
//
// Four static_for loops:
//   Loop 1 (m0_loops): Gemm0 (S = Q@K); Q prefetched and written to q_lds.
//   Loop 2 (m0_loops): Gemm2 (dP = dO@V); dO prefetched and written to do_lds.
//   Loop 3 (k1_loops): Gemm1 (dV += P^T @ dO^T); dO^T is transpose-loaded from
//   do_lds. Loop 4 (k1_loops): Gemm3 (dK += dS^T @ Q^T); Q^T is
//   transpose-loaded from q_lds.
//
// LDS layout (fix regions):
//   q_lds   : [kM0Sub, kQKHeaddim] x m0_loops, normal write/read and transposed
//   loading do_lds  : [kM0Sub, kVHeaddim] x m0_loops, normal write/read and
//   transposed loading
template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionBwdKernel2PipelinePolicy>
struct HstuAttentionWithSoftmaxBwdTrLoadPipelineKRVRQS_dK_dV {
  using Problem = remove_cvref_t<Problem_>;
  using Traits = remove_cvref_t<Traits_>;
  using Policy = remove_cvref_t<Policy_>;
  using QKVDataType = remove_cvref_t<typename Problem::QKVDataType>;
  using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
  using CompDataType = remove_cvref_t<typename Problem::CompDataType>;
  using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
  using KGradAccDataType = remove_cvref_t<typename Problem::KGradAccDataType>;
  using VGradAccDataType = remove_cvref_t<typename Problem::VGradAccDataType>;
  using PDataType = remove_cvref_t<typename Problem::PDataType>;

  using HstuAttentionTileSetting =
      remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

  static constexpr index_t kBlockSize = Problem::kBlockSize;

  static constexpr index_t kM0 = HstuAttentionTileSetting::kM0;
  static constexpr index_t kN0 = HstuAttentionTileSetting::kN;
  static constexpr index_t kM0Sub = HstuAttentionTileSetting::kM0Sub;
  static constexpr index_t kQKHeaddim = HstuAttentionTileSetting::kQKHeaddim;
  static constexpr index_t kVHeaddim = HstuAttentionTileSetting::kVHeaddim;
  static constexpr index_t kK1 = HstuAttentionTileSetting::kK1;

  static constexpr bool IsWarpGemm32 = HstuAttentionTileSetting::IsWarpGemm32;

  static_assert(
      Problem::kUseSoftmax == true,
      "This pipeline only works with the softmax path");

  static constexpr bool kIsJagged = Problem::kIsJagged;
  static constexpr bool kHasBias = Problem::kHasBias;
  static constexpr bool kHasCausal = Problem::kHasCausal;
  static constexpr bool kHasDropout = Problem::kHasDropout;

  static constexpr bool kPadSeqLenQ = Traits::kPadSeqLenQ;
  static constexpr bool kPadSeqLenK = Traits::kPadSeqLenK;
  static constexpr bool kPadHeadDimQK = Traits::kPadHeadDimQK;
  static constexpr bool kPadHeadDimV = Traits::kPadHeadDimV;

  static constexpr index_t kAlignmentQ =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
  static constexpr index_t kAlignmentK =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
  static constexpr index_t kAlignmentV =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentV<Problem>();
  static constexpr index_t kAlignmentOGrad =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentOGrad<Problem>();
  static constexpr index_t kAlignmentKGrad =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentKGrad<Problem>();
  static constexpr index_t kAlignmentVGrad =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentVGrad<Problem>();

  static constexpr index_t kGemm3SingleRepN =
      Policy::template GetSGradTQTBlockGemmSingleRepN<Problem>();
  static constexpr index_t kGemm1SingleRepN =
      Policy::template GetPTOGradTBlockGemmSingleRepN<Problem>();

  static constexpr index_t kBlockPerCu = []() {
    if constexpr (Traits::kBlockPerCuForKernel2 != -1)
      return Traits::kBlockPerCuForKernel2;
    else {
      if constexpr (kQKHeaddim <= 128)
        return 2;
      else
        return 1;
    }
  }();

  // Kernel-2's Gemm0 uses an M-major warp layout (block-warps <1,4,1>),
  // producing a P tile whose warp C-vector lies on the M axis. The forward
  // BlockDropout::Run assumes an N-major P tile, so use the backward (M-major)
  // BlockDropoutBwd::Run here instead. This must match the dropout instance
  // constructed in HstuAttentionBwdKernel2. Template args: <IsDropout=true,
  // IsWG32 (don't-care; derived internally), IsStoreRandval=false>.
  using DropoutType = std::conditional_t<
      kHasDropout,
      BlockDropoutBwd<true, IsWarpGemm32, false>,
      NullBlockDropout>;

  CK_TILE_DEVICE static constexpr index_t GetSmemSize() {
    return Policy::template GetSmemSize<Problem, true /* kUseTrLoad */>();
  }

  // operator() accumulates dK and dV into the provided accumulator tiles.
  //
  // Parameters:
  //   q_dram_block_window_tmp    : Q  tile window [kM0, kQKHeaddim]
  //   do_dram_block_window_tmp   : dO tile window [kM0, kVHeaddim]
  //   lse_dram_block_window_tmp  : LSE tile window [kM0] (for P = exp(S - LSE))
  //   delta_dram_block_window_tmp: delta tile window [kM0] (D[sq], read-only)
  //   bias_dram_block_window_tmp : optional bias [kM0, kN0]
  //   k_dram_block_window_tmp    : K  DRAM window [kN0, kQKHeaddim] (loaded
  //   into registers here) v_dram_block_window_tmp    : V  DRAM window [kN0,
  //   kVHeaddim]  (loaded into registers here) seqlen_q_start / seqlen_q_end
  //   i_n0                       : column origin in K/V sequence dimension
  //   mask / scale_s / smem_ptr
  template <
      typename QDramBlockWindowTmp,
      typename OGradDramBlockWindowTmp,
      typename LSEDramBlockWindowTmp,
      typename DeltaDramBlockWindowTmp,
      typename BiasDramBlockWindowTmp,
      typename KDramBlockWindowTmp,
      typename VDramBlockWindowTmp,
      typename NullRandValDramWindowTmp,
      typename HstuMask>
  CK_TILE_DEVICE auto operator()(
      const QDramBlockWindowTmp& q_dram_block_window_tmp,
      const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
      const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
      const DeltaDramBlockWindowTmp& delta_dram_block_window_tmp,
      const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
      const KDramBlockWindowTmp& k_dram_block_window_tmp,
      const VDramBlockWindowTmp& v_dram_block_window_tmp,
      const NullRandValDramWindowTmp& null_randval_window_tmp,
      index_t seqlen_q_start,
      index_t seqlen_q_end,
      index_t i_n0,
      HstuMask& mask,
      float scale_s,
      void* smem_ptr,
      DropoutType& dropout) const {
    constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
    constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
    constexpr auto gemm_1 =
        Policy::template GetPTOGradTBlockGemm<Problem, true /* kUseTrLoad */>();
    constexpr auto gemm_3 =
        Policy::template GetSGradTQTBlockGemm<Problem, true /* kUseTrLoad */>();

    using Gemm0Combined =
        decltype(Policy::template GetQKCombinedBlockGemm<Problem>());

    // ---- Load K and V into registers (register-resident for the entire Q
    // loop) ----
    auto k_tile = load_tile(k_dram_block_window_tmp);
    auto v_tile = load_tile(v_dram_block_window_tmp);

    // ---- Initialize dK and dV accumulators ----
    auto dk_acc = gemm_3.MakeCBlockTile();
    auto dv_acc = gemm_1.MakeCBlockTile();

    if (seqlen_q_start >= seqlen_q_end) {
      clear_tile(dk_acc);
      clear_tile(dv_acc);
      return make_tuple(dk_acc, dv_acc);
    };

    constexpr index_t m0_loops = Policy::template GetNumM0Loops<Problem>();
    constexpr index_t k1_loops = Policy::template GetNumK1Loops<Problem>();

    constexpr auto NumQOGradPrefetches = min(m0_loops, 2);

    // ---- Tile type declarations ----
    using SaccBlockTileType =
        decltype(gemm_0.template MakeCBlockTile<kM0Sub, kN0>());
    using PGradaccBlockTileType =
        decltype(gemm_2.template MakeCBlockTile<kM0Sub, kN0>());
    using CombinedTileType =
        decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
    using PcompBlockTileType =
        decltype(cast_tile<CompDataType>(CombinedTileType{}));
    using PGradcompBlockTileType = PcompBlockTileType;

    SaccBlockTileType sacc_tile;
    PGradaccBlockTileType dpacc_tile;
    PcompBlockTileType pcomp_tile;
    PGradcompBlockTileType dscomp_tile;

    auto pt_tile = make_static_distributed_tensor<QKVDataType>(
        Policy::template MakePTRegTileDistribution<Problem>());
    auto dst_tile = make_static_distributed_tensor<QKVDataType>(
        Policy::template MakeSGradTRegTileDistribution<Problem>());

    // ---- LDS setup ----
    // Two LDS regions in order:
    //   [q_lds | do_lds]
    // q_lds  : complete-buffered [kM0Sub, kQKHeaddim] x m0_loops, invariant
    // view for normal write/read and transposed loading do_lds :
    // complete-buffered [kM0Sub, kVHeaddim] x m0_loops, invariant view for
    // normal write/read and transposed loading
    constexpr index_t q_smem_size =
        Policy::template GetSmemSizeQ<Problem, true /*kUseTrLoad*/>();

    // q_lds, the same tensor_view for write/read
    QKVDataType* q_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
    auto q_lds = make_tensor_view<address_space_enum::lds>(
        q_lds_ptr,
        Policy::
            template MakeQLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>());
    auto q_lds_monolithic_window = make_tile_window(
        q_lds,
        Policy::template MakeQLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
            .get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeQLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
                .get_lengths()[number<0>{}] == kM0,
        "Check failed!");
    static_assert(
        Policy::template MakeQLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
                .get_lengths()[number<1>{}] == kQKHeaddim,
        "Check failed!");

    // q_lds windows for normal write and normal read
    using q_lds_window_type = decltype(get_slice_tile(
        q_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kM0Sub, kQKHeaddim>{}));
    statically_indexed_array<q_lds_window_type, m0_loops> q_lds_windows;
    static_for<0, m0_loops, 1>{}([&](auto i_buf) {
      q_lds_windows[i_buf] = get_slice_tile(
          q_lds_monolithic_window,
          sequence<i_buf * kM0Sub, 0>{},
          sequence<(i_buf + 1) * kM0Sub, kQKHeaddim>{});
    });

    // q_lds windows for trload read
    using q_lds_trload_window_type = decltype(get_slice_tile(
        q_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kK1, kQKHeaddim>{}));
    statically_indexed_array<q_lds_trload_window_type, m0_loops>
        q_lds_trload_windows;
    static_for<0, k1_loops, 1>{}([&](auto i_buf) {
      q_lds_trload_windows[i_buf] = get_slice_tile(
          q_lds_monolithic_window,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kQKHeaddim>{});
    });

    // do_lds, the same tensor_view for write/read
    QKVDataType* do_lds_ptr = reinterpret_cast<QKVDataType*>(
        static_cast<char*>(smem_ptr) + q_smem_size);
    auto do_lds = make_tensor_view<address_space_enum::lds>(
        do_lds_ptr,
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/>());
    auto do_lds_monolithic_window = make_tile_window(
        do_lds,
        Policy::
            template MakeOGradLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
                .get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/>()
                .get_lengths()[number<0>{}] == kM0,
        "Check failed!");
    static_assert(
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/>()
                .get_lengths()[number<1>{}] == kVHeaddim,
        "Check failed!");

    // do_lds windows for normal write and normal read
    using do_lds_window_type = decltype(get_slice_tile(
        do_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kM0Sub, kVHeaddim>{}));
    statically_indexed_array<do_lds_window_type, m0_loops> do_lds_windows;
    static_for<0, m0_loops, 1>{}([&](auto i_buf) {
      do_lds_windows[i_buf] = get_slice_tile(
          do_lds_monolithic_window,
          sequence<i_buf * kM0Sub, 0>{},
          sequence<(i_buf + 1) * kM0Sub, kVHeaddim>{});
    });

    // do_lds windows for trload read
    using do_lds_trload_window_type = decltype(get_slice_tile(
        do_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kK1, kVHeaddim>{}));
    statically_indexed_array<do_lds_trload_window_type, m0_loops>
        do_lds_trload_windows;
    static_for<0, m0_loops, 1>{}([&](auto i_buf) {
      do_lds_trload_windows[i_buf] = get_slice_tile(
          do_lds_monolithic_window,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kVHeaddim>{});
    });

    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    // ---- Q DRAM window (per-iteration tile [kM0Sub, kQKHeaddim]) ----
    auto q_dram_window = make_tile_window(
        q_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0Sub>{}, number<kQKHeaddim>{}),
        {seqlen_q_start, 0},
        Policy::template MakeQDramTileDistribution<Problem>());

    // ---- dO DRAM window (per-iteration tile [kM0Sub, kVHeaddim]) ----
    auto do_dram_window = make_tile_window(
        do_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0Sub>{}, number<kVHeaddim>{}),
        {seqlen_q_start, 0},
        Policy::template MakeOGradDramTileDistribution<Problem>());

    // ---- LSE DRAM window [kM0] -- loaded once per outer iteration ----
    auto lse_dram_window = make_tile_window(
        lse_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}),
        {seqlen_q_start},
        Policy::template MakeLSETileDistribution<Problem>());

    // ---- Delta DRAM window [kM0] -- loaded once per outer iteration ----
    auto delta_dram_window = make_tile_window(
        delta_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}),
        {seqlen_q_start},
        Policy::template MakeDeltaTileDistribution<Problem>());

    // ---- Bias DRAM window ----
    const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
    auto bias_dram_window = make_tile_window(
        bias_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kN0>{}),
        {seqlen_q_start, bias_origin.at(number<1>{})},
        Policy::template MakeBiasDramTileDistribution<Problem>());

    auto null_randval_window =
        dropout.template MakeRandvalDramWindow<Gemm0Combined>(
            null_randval_window_tmp, i_n0);

    const auto f_exp = [](CompDataType x) -> CompDataType {
      if constexpr (std::is_same_v<CompDataType, float>)
        return __expf(x);
      else
        return exp(x);
    };

    // ---- Prefetch first round of Q tiles (for Loop 1 / Gemm0) ----
    using q_tile_type = decltype(load_tile(q_dram_window));
    using do_tile_type = decltype(load_tile(do_dram_window));
    statically_indexed_array<q_tile_type, NumQOGradPrefetches> q_tiles;
    statically_indexed_array<do_tile_type, NumQOGradPrefetches> do_tiles;

    // Prefetch Q and dO for the *next* outer do-while iteration
    q_tiles[number<0>{}] = load_tile(q_dram_window);
    move_tile_window(q_dram_window, {kM0Sub, 0});

    __builtin_amdgcn_sched_barrier(0);

    clear_tile(dk_acc);
    clear_tile(dv_acc);

    do_tiles[number<0>{}] = load_tile(do_dram_window);
    move_tile_window(do_dram_window, {kM0Sub, 0});

    __builtin_amdgcn_sched_barrier(0);

    // Full [kM0] row-scalar tiles for LSE and delta
    using LSETileType = decltype(block_tile_reduce<CompDataType>(
        decltype(gemm_0.template MakeCBlockTile<kM0, kN0>()){},
        sequence<1>{},
        [](auto a, auto b) { return a + b; },
        CompDataType{0}));
    using DeltaTileType = LSETileType;
    LSETileType lse_tile;
    DeltaTileType delta_tile;

    auto seqlen_q_curr = seqlen_q_start;

    do {
      // Load LSE and delta for the full kM0 tile
      lse_tile = cast_tile<CompDataType>(load_tile(lse_dram_window));
      delta_tile = cast_tile<CompDataType>(load_tile(delta_dram_window));
      move_tile_window(lse_dram_window, {kM0});
      move_tile_window(delta_dram_window, {kM0});

      // =======================================================================
      // Loop 1: Gemm0 (S = Q@K); prefetch Q to LDS
      // Loop 2: Gemm2 (dP = dO@V); prefetch dO to LDS
      // =======================================================================
      // Ensure the trloading of q in previous itertation completely done by all
      // warps
      __builtin_amdgcn_s_barrier();

      static_for<0, m0_loops, 1>{}([&](auto i_m0) {
        constexpr auto i_current_buf = number<i_m0 % NumQOGradPrefetches>{};
        constexpr auto i_prefetch_buf =
            number<(i_m0 + 1) % NumQOGradPrefetches>{};

        // Store Q to q_lds
        store_tile(
            q_lds_windows[i_m0], q_tiles[i_current_buf], partition_index);

        // Store dO to do_lds
        store_tile(
            do_lds_windows[i_m0], do_tiles[i_current_buf], partition_index);

        // Prefetch next Q tile while current stores are in flight
        if constexpr (i_m0 + 1 < m0_loops) {
          q_tiles[i_prefetch_buf] = load_tile(q_dram_window);
          move_tile_window(q_dram_window, {kM0Sub, 0});
        }

        // Prefetch next dO tile while current stores are in flight
        if constexpr (i_m0 + 1 < m0_loops) {
          do_tiles[i_prefetch_buf] = load_tile(do_dram_window);
          move_tile_window(do_dram_window, {kM0Sub, 0});
        }

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::SALU |
            LLVMSchedGroupMask::TRANS);

        // Ensure all LDS stores are visible before Gemm0 reads
        block_sync_lds();

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::SALU |
            LLVMSchedGroupMask::TRANS);

        // Gemm0: sacc_tile = Q_sub @ K
        gemm_0(sacc_tile, q_lds_windows[i_m0], k_tile);
        auto s_tmp = cast_tile<CompDataType>(sacc_tile);
        // Place the [kM0Sub, kN0] sub-tile into the combined [kM0, kN0]
        // pcomp_tile by a direct thread-buffer copy. M (the sub-tiled dim) is
        // the outermost Y-dim of the C distribution, so sub-tile i_m0 occupies
        // a contiguous thread-buffer block at offset i_m0 * sacc_tbuf_size.
        // This avoids set_slice_tile, which miscomputes the write offsets when
        // slicing the register-interleaved M axis of the MFMA C fragment.
        constexpr index_t sacc_tbuf_size =
            decltype(s_tmp)::get_thread_buffer_size();
        static_for<0, sacc_tbuf_size, 1>{}([&](auto j) {
          pcomp_tile.get_thread_buffer()(number<i_m0 * sacc_tbuf_size + j>{}) =
              s_tmp.get_thread_buffer()(number<j>{});
        });

        // Gemm2: dpacc_tile = dO_sub @ V
        gemm_2(dpacc_tile, do_lds_windows[i_m0], v_tile);
        auto dp_tmp = cast_tile<CompDataType>(dpacc_tile);
        // Direct thread-buffer copy of the [kM0Sub, kN0] sub-tile into
        // dscomp_tile (see the pcomp_tile note above for why set_slice_tile is
        // not used).
        constexpr index_t dpacc_tbuf_size =
            decltype(dp_tmp)::get_thread_buffer_size();
        static_for<0, dpacc_tbuf_size, 1>{}([&](auto j) {
          dscomp_tile.get_thread_buffer()(
              number<i_m0 * dpacc_tbuf_size + j>{}) =
              dp_tmp.get_thread_buffer()(number<j>{});
        });
      });

      __builtin_amdgcn_sched_barrier(
          LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::SALU |
          LLVMSchedGroupMask::TRANS);

      // ---- Scale, optional bias, mask ----
      if constexpr (kHasBias) {
        const auto bias_tile = load_tile(bias_dram_window);
        tile_elementwise_inout(
            [&scale_s](auto& x, const auto& y) {
              x = x * scale_s + type_convert<CompDataType>(y);
            },
            pcomp_tile,
            bias_tile);
        move_tile_window(bias_dram_window, {kM0, 0});
      } else {
        tile_elementwise_inout(
            [&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
      }

      // Softmax path mask: set masked-out S to -inf.
      // The `row >= seqlen_q_end` guard excludes padded tail Q rows of the last
      // tile (seqlen_q not a multiple of kM0). Kernel 2 reduces over Q (rows),
      // so a padded row that slips through the mask (IsTokenPairInsideMask
      // clamps and may return true for it) would pollute every dK/dV column;
      // forcing S=-inf makes P=0 (hence dS=0) there.
      if (!mask.IsFullTileInsideMask(
              seqlen_q_curr, i_n0, number<kN0>{}, number<kM0>{})) {
        constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType lse_val = lse_tile[i_idx];
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                pcomp_tile.get_tile_distribution(),
                make_tuple(idx0, idx1),
                partition_index);
            const auto row = seqlen_q_curr + tile_idx.at(number<0>{});
            const auto col = i_n0 + tile_idx.at(number<1>{});
            constexpr auto ij = make_tuple(idx0, idx1);
            if (!mask.IsTokenPairInsideMask(row, col) || row >= seqlen_q_end)
              pcomp_tile(ij) = type_convert<CompDataType>(0.0f);
            else
              pcomp_tile(ij) = f_exp(pcomp_tile[ij] - lse_val);
          });
        });
      } else {
        // Fully inside the mask, but the last tile may still contain padded
        // tail rows (row >= seqlen_q_end) that must be excluded from the dK/dV
        // reduction.
        constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType lse_val = lse_tile[i_idx];
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                pcomp_tile.get_tile_distribution(),
                make_tuple(idx0, idx1),
                partition_index);
            const auto row = seqlen_q_curr + tile_idx.at(number<0>{});
            constexpr auto ij = make_tuple(idx0, idx1);
            if (row >= seqlen_q_end)
              pcomp_tile(ij) = type_convert<CompDataType>(0.0f);
            else
              pcomp_tile(ij) = f_exp(pcomp_tile[ij] - lse_val);
          });
        });
      }

      if constexpr (kHasDropout) {
        // Build a per-element keep/drop mask WITHOUT disturbing P. Feeding a +1
        // sentinel tile to BlockDropoutBwd::Run yields +1 (kept) / -1
        // (dropped). The mask is applied below as drop_scale = rp_undrop (kept)
        // / 0 (dropped) to P (-> dV) and to dP inside the softmax dS jacobian
        // (-> dK), matching reference_hstu_attention_bwd.hpp:
        //   dS = P * (drop_scale*dP - D),  P_drop = drop_scale * P   (P is the
        //   PURE softmax).
        // int8_t sentinel (1 byte/elem) keeps drop_mask cheap in VGPRs; Run
        // only flips its sign (+1 kept / -1 dropped), so 8 bits are enough.
        auto drop_mask = make_static_distributed_tensor<int8_t>(
            pcomp_tile.get_tile_distribution());

        tile_elementwise_inout(
            [](auto& x) { x = type_convert<int8_t>(1); }, drop_mask);

        // BlockDropoutBwd::Run is M-major and needs no LDS scratch (no
        // transpose). Signature: Run<BlockGemm,
        // RandValOutputDataType>(start_m0_idx, start_n0_idx,
        //                                                   p_compute,
        //                                                   randval_dram_window).
        dropout.template Run<Gemm0Combined, uint8_t>(
            seqlen_q_curr, i_n0, drop_mask, null_randval_window);

        move_tile_window(null_randval_window, {kM0, 0});

        // ---- dS = P * (drop_scale*dP - D[sq]);  P_drop = drop_scale * P (for
        // dV) ---- The dropout mask propagates onto dP (and onto P for dV); the
        // softmax jacobian's outer factor stays the PURE P. drop_scale =
        // rp_undrop (kept) / 0 (dropped). Note the D term is NOT scaled by
        // drop_scale -- dropped positions still get dS = -P*D through the
        // softmax coupling (see reference derivation).
        constexpr auto ds_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType delta_val = delta_tile[i_idx];
          sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
            constexpr auto ij = make_tuple(idx0, idx1);
            const CompDataType p_pure = pcomp_tile[ij];
            CompDataType dp = dscomp_tile[ij];
            CompDataType p_drop = p_pure;

            if (drop_mask[ij] > 0) {
              dp = dp * dropout.rp_undrop;
              p_drop = p_pure * dropout.rp_undrop;
            } else {
              dp = type_convert<CompDataType>(0.0f);
              p_drop = type_convert<CompDataType>(0.0f);
            }
            // dK path: dS = P_pure * (drop_scale*dP - D)
            dscomp_tile(ij) = p_pure * (dp - delta_val);
            // dV path: P_drop = drop_scale * P
            pcomp_tile(ij) = p_drop;
          });
        });
      } else {
        constexpr auto ds_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType delta_val = delta_tile[i_idx];
          sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
            constexpr auto ij = make_tuple(idx0, idx1);
            CompDataType dp = dscomp_tile[ij];

            // dK path: dS = P * (drop_scale*dP - D)
            dscomp_tile(ij) = pcomp_tile[ij] * (dp - delta_val);
          });
        });
      }

      auto p_gemm_tile = cast_tile<QKVDataType>(pcomp_tile);

      Policy::template PTFromGemm0CToGemm1A<
          Problem,
          Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}) ==
              32>(pt_tile, p_gemm_tile);

      // =======================================================================
      // Loop 3: Gemm1  dV += P^T @ dO^T
      // =======================================================================
      static_for<0, k1_loops, 1>{}([&](auto i_k1) {
        // Gemm1: dV += P^T_sub @ dO^T_sub
        auto pt_slice = get_slice_tile(
            pt_tile,
            sequence<0, i_k1 * kK1>{},
            sequence<kN0, (i_k1 + 1) * kK1>{});
        gemm_1(dv_acc, pt_slice, do_lds_trload_windows[i_k1]);
      });

      // Prefetch Q and dO for the *next* outer do-while iteration
      q_tiles[number<0>{}] = load_tile(q_dram_window);
      move_tile_window(q_dram_window, {kM0Sub, 0});

      do_tiles[number<0>{}] = load_tile(do_dram_window);
      move_tile_window(do_dram_window, {kM0Sub, 0});

      auto ds_gemm_tile = cast_tile<QKVDataType>(dscomp_tile);

      Policy::template SGradTFromGemm2CToGemm3A<
          Problem,
          Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{}) ==
              32>(dst_tile, ds_gemm_tile);

      // =======================================================================
      // Loop 4: Gemm3  dK += dS^T @ Q^T
      // =======================================================================
      static_for<0, k1_loops, 1>{}([&](auto i_k1) {
        // Gemm3: dK += dS^T_sub @ Q^T_sub
        auto dst_slice = get_slice_tile(
            dst_tile,
            sequence<0, i_k1 * kK1>{},
            sequence<kN0, (i_k1 + 1) * kK1>{});
        gemm_3(dk_acc, dst_slice, q_lds_trload_windows[i_k1]);
      });

      seqlen_q_curr += kM0;
    } while (seqlen_q_curr < seqlen_q_end);

    // Apply alpha scaling to accumulated dK
    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, dk_acc);

    return make_tuple(dk_acc, dv_acc);
  }
};

} // namespace ck_tile
