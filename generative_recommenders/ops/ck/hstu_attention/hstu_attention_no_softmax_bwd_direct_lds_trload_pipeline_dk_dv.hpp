// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_bwd_kernel_2_pipeline_policy.hpp"
#include "hstu_attention_pipeline_helper.hpp"

namespace ck_tile {

// Kernel 2 backward pipeline -- SiLU path (kUseSoftmax == false).
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
struct HstuAttentionNoSoftmaxBwdDirectLdsTrLoadPipelineKRVRQS_dK_dV {
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

  static constexpr bool kUseDirectLds = true;
  static constexpr bool kUseTrLoad = true;

  static_assert(
      Problem::kUseSoftmax == false,
      "This pipeline only works with the SiLU (no-softmax) path");

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

  // dK epilogue uses Gemm3's single-rep-N (SGradTQT: dK += dS^T @ Q^T)
  static constexpr index_t kGemm3SingleRepN =
      Policy::template GetSGradTQTBlockGemmSingleRepN<Problem>();

  // dV epilogue uses Gemm1's single-rep-N (PTOGradT: dV += P^T @ dO^T)
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
  // BlockDropoutBwd::Run here instead.
  using DropoutType = std::conditional_t<
      kHasDropout,
      BlockDropoutBwd<true, IsWarpGemm32, false>,
      NullBlockDropout>;

  CK_TILE_DEVICE static constexpr index_t GetSmemSize() {
    return Policy::template GetSmemSize<
        Problem,
        true /*kUseTrLoad*/,
        true /*kUseDirectLds*/>();
  }

  // operator() loads K and V into registers, initialises dK/dV accumulators,
  // runs the Q-loop, and returns the accumulated (dk_acc, dv_acc) tiles.
  //
  // Parameters:
  //   q_dram_block_window_tmp    : Q  tile window [kM0, kQKHeaddim]
  //   do_dram_block_window_tmp   : dO tile window [kM0, kVHeaddim]
  //   bias_dram_block_window_tmp : optional bias  [kM0, kN0]
  //   k_dram_block_window_tmp    : K  DRAM window [kN0, kQKHeaddim] (loaded
  //   into registers here) v_dram_block_window_tmp    : V  DRAM window [kN0,
  //   kVHeaddim]  (loaded into registers here) seqlen_q_start / seqlen_q_end :
  //   Q-range for this K/V tile i_n0                       : column origin in
  //   the K/V sequence dimension mask                       : HSTU block mask
  //   scale_s                    : alpha
  //   scale_p                    : applied to SiLU result
  //   smem_ptr                   : shared memory pointer
  template <
      typename QDramBlockWindowTmp,
      typename OGradDramBlockWindowTmp,
      typename BiasDramBlockWindowTmp,
      typename KDramBlockWindowTmp,
      typename VDramBlockWindowTmp,
      typename NullRandValDramWindowTmp,
      typename HstuMask>
  CK_TILE_DEVICE auto operator()(
      const QDramBlockWindowTmp& q_dram_block_window_tmp,
      const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
      const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
      const KDramBlockWindowTmp& k_dram_block_window_tmp,
      const VDramBlockWindowTmp& v_dram_block_window_tmp,
      const NullRandValDramWindowTmp& null_randval_window_tmp,
      index_t seqlen_q_start,
      index_t seqlen_q_end,
      index_t i_n0,
      HstuMask& mask,
      float scale_s,
      float scale_p,
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
    constexpr index_t q_smem_size = Policy::template GetSmemSizeQ<
        Problem,
        true /*kUseTrLoad*/,
        true /*kUseDirectLds*/>();
    constexpr index_t do_smem_size = Policy::template GetSmemSizeOGrad<
        Problem,
        true /*kUseTrLoad*/,
        true /*kUseDirectLds*/>();

    // q_lds, the same tensor_view for write/read
    QKVDataType* q_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
    auto q_lds_write_view = make_tensor_view<address_space_enum::lds>(
        q_lds_ptr,
        Policy::template MakeQLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad */,
            true /*kUseDirectLds*/>());

    auto q_lds_write_monolithic_window_0 = make_tile_window(
        q_lds_write_view,
        Policy::template MakeQLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad */,
            true /*kUseDirectLds*/>()
            .get_lengths(),
        {0, 0});

    auto q_lds_write_monolithic_window_1 = q_lds_write_monolithic_window_0;
    q_lds_write_monolithic_window_1.set_bottom_tensor_view_data_ptr(
        q_lds_ptr + q_smem_size / sizeof(QKVDataType));

    static_assert(
        Policy::template MakeQLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>()
                .get_lengths()[number<0>{}] == kM0,
        "Check failed!");
    static_assert(
        Policy::template MakeQLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>()
                .get_lengths()[number<1>{}] == kQKHeaddim,
        "Check failed!");

    auto q_lds_read_view = make_tensor_view<address_space_enum::lds>(
        q_lds_ptr,
        Policy::template MakeQLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad */,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>());

    auto q_lds_read_monolithic_window_0 = make_tile_window(
        q_lds_read_view,
        Policy::template MakeQLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad */,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>()
            .get_lengths(),
        {0, 0});

    auto q_lds_read_monolithic_window_1 = q_lds_read_monolithic_window_0;
    q_lds_read_monolithic_window_1.set_bottom_tensor_view_data_ptr(
        q_lds_ptr + q_smem_size / sizeof(QKVDataType));

    // q_lds windows for normal read
    using q_lds_read_window_type = decltype(get_slice_tile(
        q_lds_read_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kM0Sub, kQKHeaddim>{}));
    statically_indexed_array<q_lds_read_window_type, m0_loops>
        q_lds_read_windows_0;
    statically_indexed_array<q_lds_read_window_type, m0_loops>
        q_lds_read_windows_1;
    static_for<0, m0_loops, 1>{}([&](auto i_buf) {
      q_lds_read_windows_0[i_buf] = get_slice_tile(
          q_lds_read_monolithic_window_0,
          sequence<i_buf * kM0Sub, 0>{},
          sequence<(i_buf + 1) * kM0Sub, kQKHeaddim>{});
      q_lds_read_windows_1[i_buf] = get_slice_tile(
          q_lds_read_monolithic_window_1,
          sequence<i_buf * kM0Sub, 0>{},
          sequence<(i_buf + 1) * kM0Sub, kQKHeaddim>{});
    });

    // q_lds windows for trload read
    using q_lds_trload_window_type = decltype(get_slice_tile(
        q_lds_read_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kK1, kQKHeaddim>{}));
    statically_indexed_array<q_lds_trload_window_type, k1_loops>
        q_lds_trload_windows_0;
    statically_indexed_array<q_lds_trload_window_type, k1_loops>
        q_lds_trload_windows_1;
    static_for<0, k1_loops, 1>{}([&](auto i_buf) {
      q_lds_trload_windows_0[i_buf] = get_slice_tile(
          q_lds_read_monolithic_window_0,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kQKHeaddim>{});
      q_lds_trload_windows_1[i_buf] = get_slice_tile(
          q_lds_read_monolithic_window_1,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kQKHeaddim>{});
    });

    // do_lds, the same tensor_view for write/read
    QKVDataType* do_lds_ptr = reinterpret_cast<QKVDataType*>(
        static_cast<char*>(smem_ptr) + q_smem_size * 2);
    auto do_lds_write_view = make_tensor_view<address_space_enum::lds>(
        do_lds_ptr,
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>());
    auto do_lds_write_monolithic_window_0 = make_tile_window(
        do_lds_write_view,
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>()
            .get_lengths(),
        {0, 0});

    auto do_lds_write_monolithic_window_1 = do_lds_write_monolithic_window_0;
    do_lds_write_monolithic_window_1.set_bottom_tensor_view_data_ptr(
        do_lds_ptr + do_smem_size / sizeof(QKVDataType));

    static_assert(
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>()
                .get_lengths()[number<0>{}] == kM0,
        "Check failed!");
    static_assert(
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/>()
                .get_lengths()[number<1>{}] == kVHeaddim,
        "Check failed!");

    auto do_lds_read_view = make_tensor_view<address_space_enum::lds>(
        do_lds_ptr,
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>());
    auto do_lds_read_monolithic_window_0 = make_tile_window(
        do_lds_read_view,
        Policy::template MakeOGradLdsBlockDescriptor<
            Problem,
            true /*kUseTrLoad*/,
            true /*kUseDirectLds*/,
            true /*kUseXor*/>()
            .get_lengths(),
        {0, 0});

    auto do_lds_read_monolithic_window_1 = do_lds_read_monolithic_window_0;
    do_lds_read_monolithic_window_1.set_bottom_tensor_view_data_ptr(
        do_lds_ptr + do_smem_size / sizeof(QKVDataType));

    // do_lds windows for normal read
    using do_lds_read_window_type = decltype(get_slice_tile(
        do_lds_read_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kM0Sub, kVHeaddim>{}));
    statically_indexed_array<do_lds_read_window_type, m0_loops>
        do_lds_read_windows_0;
    statically_indexed_array<do_lds_read_window_type, m0_loops>
        do_lds_read_windows_1;
    static_for<0, m0_loops, 1>{}([&](auto i_buf) {
      do_lds_read_windows_0[i_buf] = get_slice_tile(
          do_lds_read_monolithic_window_0,
          sequence<i_buf * kM0Sub, 0>{},
          sequence<(i_buf + 1) * kM0Sub, kVHeaddim>{});
      do_lds_read_windows_1[i_buf] = get_slice_tile(
          do_lds_read_monolithic_window_1,
          sequence<i_buf * kM0Sub, 0>{},
          sequence<(i_buf + 1) * kM0Sub, kVHeaddim>{});
    });

    // do_lds windows for trload read
    using do_lds_trload_window_type = decltype(get_slice_tile(
        do_lds_read_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kK1, kVHeaddim>{}));
    statically_indexed_array<do_lds_trload_window_type, k1_loops>
        do_lds_trload_windows_0;
    statically_indexed_array<do_lds_trload_window_type, k1_loops>
        do_lds_trload_windows_1;
    static_for<0, k1_loops, 1>{}([&](auto i_buf) {
      do_lds_trload_windows_0[i_buf] = get_slice_tile(
          do_lds_read_monolithic_window_0,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kVHeaddim>{});
      do_lds_trload_windows_1[i_buf] = get_slice_tile(
          do_lds_read_monolithic_window_1,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kVHeaddim>{});
    });

    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    // ---- Q DRAM window (per-iteration tile [kM0Sub, kQKHeaddim]) ----
    auto q_dram_window = make_tile_window_raw(
        q_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0Sub>{}, number<kQKHeaddim>{}),
        {0, 0},
        Policy::template MakeQDramTileDistribution<Problem>());

    // ---- dO DRAM window (per-iteration tile [kM0Sub, kVHeaddim]) ----
    auto do_dram_window = make_tile_window_raw(
        do_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0Sub>{}, number<kVHeaddim>{}),
        {0, 0},
        Policy::template MakeOGradDramTileDistribution<Problem>());

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

    const auto& q_dram_desc =
        q_dram_window.get_bottom_tensor_view().get_tensor_descriptor();

    // the dram view is XOR-swizzled per row: group 1 of row 1 maps back to
    // group 0, so the row stride is its offset
    constexpr index_t kQSwizzleUnit =
        Policy::template GetQDirectLdsSwizzleUnit<Problem>();
    const index_t q_row_stride =
        q_dram_desc.calculate_offset(make_tuple(1, kQSwizzleUnit)) -
        q_dram_desc.calculate_offset(make_tuple(0, 0));
    // elements per DMA of each thread: the MPerThread of the Q dram tile
    // distribution
    index_t q_row_offset =
        amd_wave_read_first_lane(seqlen_q_start * q_row_stride);
    const index_t q_row_step = kM0Sub * q_row_stride;

    const auto q_issues_0 = generate_tuple(
        [&](auto i_buf) {
          return detail::
              MakeDirectLdsIssues<m0_loops, kBlockSize / get_warp_size()>(
                  q_lds_write_monolithic_window_0,
                  number<i_buf % m0_loops>{},
                  q_dram_window);
        },
        number<m0_loops>{});

    const auto q_issues_1 = generate_tuple(
        [&](auto i_buf) {
          return detail::
              MakeDirectLdsIssues<m0_loops, kBlockSize / get_warp_size()>(
                  q_lds_write_monolithic_window_1,
                  number<i_buf % m0_loops>{},
                  q_dram_window);
        },
        number<m0_loops>{});

    const auto& do_dram_desc =
        do_dram_window.get_bottom_tensor_view().get_tensor_descriptor();

    // the dram view is XOR-swizzled per row: group 1 of row 1 maps back to
    // group 0, so the row stride is its offset
    constexpr index_t kDoSwizzleUnit =
        Policy::template GetOGradDirectLdsSwizzleUnit<Problem>();
    const index_t do_row_stride =
        do_dram_desc.calculate_offset(make_tuple(1, kDoSwizzleUnit)) -
        do_dram_desc.calculate_offset(make_tuple(0, 0));
    // elements per DMA of each thread: the MPerThread of the dO dram tile
    // distribution
    index_t do_row_offset =
        amd_wave_read_first_lane(seqlen_q_start * do_row_stride);
    const index_t do_row_step = kM0Sub * do_row_stride;

    const auto do_issues_0 = generate_tuple(
        [&](auto i_buf) {
          return detail::
              MakeDirectLdsIssues<m0_loops, kBlockSize / get_warp_size()>(
                  do_lds_write_monolithic_window_0,
                  number<i_buf % m0_loops>{},
                  do_dram_window);
        },
        number<m0_loops>{});

    const auto do_issues_1 = generate_tuple(
        [&](auto i_buf) {
          return detail::
              MakeDirectLdsIssues<m0_loops, kBlockSize / get_warp_size()>(
                  do_lds_write_monolithic_window_1,
                  number<i_buf % m0_loops>{},
                  do_dram_window);
        },
        number<m0_loops>{});

    detail::IssueDirectLds(
        static_cast<QKVDataType*>(smem_ptr),
        q_dram_window,
        q_issues_0(number<0>{}),
        q_row_offset);
    q_row_offset += q_row_step;

    detail::IssueDirectLds(
        static_cast<QKVDataType*>(smem_ptr),
        do_dram_window,
        do_issues_0(number<0>{}),
        do_row_offset);
    do_row_offset += do_row_step;

    clear_tile(dk_acc);
    clear_tile(dv_acc);

    // SiLU activation
    const auto f_dsilu = [](CompDataType x) -> CompDataType {
      const auto one = type_convert<CompDataType>(1.0f);
      CompDataType sig;
      if constexpr (std::is_same_v<CompDataType, float>)
        sig = __builtin_amdgcn_rcpf(one + __expf(-x));
      else
        sig = one / (one + exp(-x));
      return sig * (one + x * (one - sig));
    };

    const auto f_silu = [](CompDataType x) -> CompDataType {
      const auto one = type_convert<CompDataType>(1.0f);
      if constexpr (std::is_same_v<CompDataType, float>)
        return x * __builtin_amdgcn_rcpf(one + __expf(-x));
      else
        return x / (one + exp(-x));
    };

    auto seqlen_q_curr = seqlen_q_start;

    auto main_loop_func = [&](auto EvenIter) {
      // =======================================================================
      // Loop 1: Gemm0 (S = Q@K); prefetch Q to LDS
      // Loop 2: Gemm2 (dP = dO@V); prefetch dO to LDS
      // =======================================================================
      static_for<0, m0_loops, 1>{}([&](auto i_m0) {
        // wait the last Direct-Lds q/do transferring to be done
        s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt>();

        // Gemm0: sacc_tile = Q_sub @ K
        if constexpr (EvenIter) {
          gemm_0(sacc_tile, q_lds_read_windows_0[i_m0], k_tile);
          if constexpr (i_m0 < m0_loops - 1) {
            detail::IssueDirectLds(
                static_cast<QKVDataType*>(smem_ptr),
                q_dram_window,
                q_issues_0(number<i_m0 + 1>{}),
                q_row_offset);
            q_row_offset += q_row_step;
          }
        } else {
          gemm_0(sacc_tile, q_lds_read_windows_1[i_m0], k_tile);
          if constexpr (i_m0 < m0_loops - 1) {
            detail::IssueDirectLds(
                static_cast<QKVDataType*>(smem_ptr),
                q_dram_window,
                q_issues_1(number<i_m0 + 1>{}),
                q_row_offset);
            q_row_offset += q_row_step;
          }
        }

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
        if constexpr (EvenIter) {
          gemm_2(dpacc_tile, do_lds_read_windows_0[i_m0], v_tile);
          if constexpr (i_m0 < m0_loops - 1) {
            detail::IssueDirectLds(
                static_cast<QKVDataType*>(smem_ptr),
                do_dram_window,
                do_issues_0(number<i_m0 + 1>{}),
                do_row_offset);
            do_row_offset += do_row_step;
          }
        } else {
          gemm_2(dpacc_tile, do_lds_read_windows_1[i_m0], v_tile);
          if constexpr (i_m0 < m0_loops - 1) {
            detail::IssueDirectLds(
                static_cast<QKVDataType*>(smem_ptr),
                do_dram_window,
                do_issues_1(number<i_m0 + 1>{}),
                do_row_offset);
            do_row_offset += do_row_step;
          }
        }

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
          LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

      if constexpr (EvenIter) {
        detail::IssueDirectLds(
            static_cast<QKVDataType*>(smem_ptr),
            q_dram_window,
            q_issues_1(number<0>{}),
            q_row_offset);
        q_row_offset += q_row_step;

        detail::IssueDirectLds(
            static_cast<QKVDataType*>(smem_ptr),
            do_dram_window,
            do_issues_1(number<0>{}),
            do_row_offset);
        do_row_offset += do_row_step;
      } else {
        detail::IssueDirectLds(
            static_cast<QKVDataType*>(smem_ptr),
            q_dram_window,
            q_issues_0(number<0>{}),
            q_row_offset);
        q_row_offset += q_row_step;

        detail::IssueDirectLds(
            static_cast<QKVDataType*>(smem_ptr),
            do_dram_window,
            do_issues_0(number<0>{}),
            do_row_offset);
        do_row_offset += do_row_step;
      }

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

      const bool need_mask = !mask.IsFullTileInsideMask(
          seqlen_q_curr, i_n0, number<kN0>{}, number<kM0>{});

      if constexpr (kHasDropout) {
        auto combined_mask = make_static_distributed_tensor<int8_t>(
            pcomp_tile.get_tile_distribution());

        tile_elementwise_inout(
            [](auto& x) { x = type_convert<int8_t>(1); }, combined_mask);

        // BlockDropoutBwd::Run is M-major and needs no LDS scratch (no
        // transpose). Signature: Run<BlockGemm,
        // RandValOutputDataType>(start_m0_idx, start_n0_idx,
        //                                                   p_compute,
        //                                                   randval_dram_window).
        dropout.template Run<Gemm0Combined, uint8_t>(
            seqlen_q_curr, i_n0, combined_mask, null_randval_window);

        constexpr auto spans = PcompBlockTileType::get_distributed_spans();

        if (need_mask) {
          sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
              const auto tile_idx = get_x_indices_from_distributed_indices(
                  pcomp_tile.get_tile_distribution(),
                  make_tuple(idx0, idx1),
                  partition_index);
              const auto row = seqlen_q_curr + tile_idx.at(number<0>{});
              const auto col = i_n0 + tile_idx.at(number<1>{});
              constexpr auto ij = make_tuple(idx0, idx1);
              if (!mask.IsTokenPairInsideMask(row, col))
                combined_mask(ij) = type_convert<int8_t>(-1);
            });
          });
        }

        // ---- Compute P = silu(S) * scale_p and dS = dP * scale_p * dsilu(S)
        // ---- Corrections applied here:
        //  - masked-out pairs: P is already 0 (silu(0) = 0), but dsilu(0) = 0.5
        //  != 0, so dS
        //    must be explicitly forced to 0 (reference: locals_dS = 0 for
        //    masked-out).
        //  - padded tail Q rows (row >= seqlen_q_end): Kernel 2 reduces over Q,
        //  so these rows
        //    must contribute nothing; force BOTH P (-> dV) and dS (-> dK) to 0.
        //    Their S was not zeroed above (IsTokenPairInsideMask can clamp and
        //    accept padded rows), so P would otherwise be silu(garbage) != 0.
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
          sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
            constexpr auto ij = make_tuple(idx0, idx1);
            const CompDataType s = pcomp_tile[ij];
            const CompDataType dp = dscomp_tile[ij];
            CompDataType p = f_silu(s) * type_convert<CompDataType>(scale_p);
            CompDataType ds =
                dp * type_convert<CompDataType>(scale_p) * f_dsilu(s);
            // Dropout propagates through the chain rule: kept -> *rp_undrop on
            // BOTH P (-> dV) and dS (-> dK); dropped -> 0. drop_mask > 0 means
            // kept.
            if (combined_mask[ij] > 0) {
              p = p * dropout.rp_undrop;
              ds = ds * dropout.rp_undrop;
            } else {
              p = type_convert<CompDataType>(0.0f);
              ds = type_convert<CompDataType>(0.0f);
            }
            // P stored back into pcomp_tile for dV (Gemm1)
            pcomp_tile(ij) = p;
            // dS stored into dscomp_tile for dK (Gemm3)
            dscomp_tile(ij) = ds;
          });
        });
      } else {
        auto hstu_mask = make_static_distributed_tensor<int8_t>(
            pcomp_tile.get_tile_distribution());

        tile_elementwise_inout(
            [](auto& x) { x = type_convert<int8_t>(1); }, hstu_mask);

        constexpr auto spans = PcompBlockTileType::get_distributed_spans();

        if (need_mask) {
          sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
              const auto tile_idx = get_x_indices_from_distributed_indices(
                  pcomp_tile.get_tile_distribution(),
                  make_tuple(idx0, idx1),
                  partition_index);
              const auto row = seqlen_q_curr + tile_idx.at(number<0>{});
              const auto col = i_n0 + tile_idx.at(number<1>{});
              constexpr auto ij = make_tuple(idx0, idx1);
              if (!mask.IsTokenPairInsideMask(row, col))
                hstu_mask(ij) = type_convert<int8_t>(0);
            });
          });
        }

        // ---- Compute P = silu(S) * scale_p and dS = dP * scale_p * dsilu(S)
        // ---- Corrections applied here:
        //  - masked-out pairs: P is already 0 (silu(0) = 0), but dsilu(0) = 0.5
        //  != 0, so dS
        //    must be explicitly forced to 0 (reference: locals_dS = 0 for
        //    masked-out).
        //  - padded tail Q rows (row >= seqlen_q_end): Kernel 2 reduces over Q,
        //  so these rows
        //    must contribute nothing; force BOTH P (-> dV) and dS (-> dK) to 0.
        //    Their S was not zeroed above (IsTokenPairInsideMask can clamp and
        //    accept padded rows), so P would otherwise be silu(garbage) != 0.
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
          sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
            constexpr auto ij = make_tuple(idx0, idx1);
            const CompDataType s = pcomp_tile[ij];
            const CompDataType dp = dscomp_tile[ij];
            CompDataType p = f_silu(s) * type_convert<CompDataType>(scale_p);
            CompDataType ds =
                dp * type_convert<CompDataType>(scale_p) * f_dsilu(s);

            pcomp_tile(ij) = p * type_convert<CompDataType>(hstu_mask(ij));
            dscomp_tile(ij) = ds * type_convert<CompDataType>(hstu_mask(ij));
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
        if constexpr (EvenIter) {
          gemm_1(dv_acc, pt_slice, do_lds_trload_windows_0[i_k1]);
        } else {
          gemm_1(dv_acc, pt_slice, do_lds_trload_windows_1[i_k1]);
        }
      });

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
        if constexpr (EvenIter) {
          gemm_3(dk_acc, dst_slice, q_lds_trload_windows_0[i_k1]);
        } else {
          gemm_3(dk_acc, dst_slice, q_lds_trload_windows_1[i_k1]);
        }
      });
    };

    while (seqlen_q_curr + kM0 < seqlen_q_end) {
      main_loop_func(bool_constant<true>{});
      seqlen_q_curr += kM0;

      main_loop_func(bool_constant<false>{});
      seqlen_q_curr += kM0;
    }

    if (seqlen_q_curr < seqlen_q_end)
      main_loop_func(bool_constant<true>{});

    // Apply alpha scaling to accumulated dK
    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, dk_acc);

    return make_tuple(dk_acc, dv_acc);
  }
};

} // namespace ck_tile
