// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_bwd_kernel_1_pipeline_policy.hpp"

namespace ck_tile {

// Kernel 1 backward pipeline -- Softmax path (kUseSoftmax == true).
// Iterates over K/V blocks for a fixed Q tile; accumulates dQ and also
// computes D[sq] = dO row(.) O, storing it to delta_dram_window inside the
// pipeline. Naming: QR = Q/dO/O are register-resident, KS = K LDS-staged, VS =
// V LDS-staged.
template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionBwdKernel1PipelinePolicy>
struct HstuAttentionWithSoftmaxBwdPipelineQRKSVS_dQ_D {
  using Problem = remove_cvref_t<Problem_>;
  using Traits = remove_cvref_t<Traits_>;
  using Policy = remove_cvref_t<Policy_>;
  using QKVDataType = remove_cvref_t<typename Problem::QKVDataType>;
  using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
  using CompDataType = remove_cvref_t<typename Problem::CompDataType>;
  using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
  using QGradAccDataType = remove_cvref_t<typename Problem::QGradAccDataType>;
  using PDataType = remove_cvref_t<typename Problem::PDataType>;

  using HstuAttentionTileSetting =
      remove_cvref_t<typename Problem::HstuAttentionTileSetting>;

  static constexpr index_t kBlockSize = Problem::kBlockSize;

  static constexpr index_t kM0 = HstuAttentionTileSetting::kM;
  static constexpr index_t kN0 = HstuAttentionTileSetting::kN0;
  static constexpr index_t kN0Sub = HstuAttentionTileSetting::kN0Sub;
  static constexpr index_t kQKHeaddim = HstuAttentionTileSetting::kQKHeaddim;
  static constexpr index_t kVHeaddim = HstuAttentionTileSetting::kVHeaddim;
  static constexpr index_t kK1 = HstuAttentionTileSetting::kK1;

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
  static constexpr index_t kAlignmentQGrad =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentQGrad<Problem>();

  static constexpr index_t kGemm4SingleRepN =
      Policy::template GetSGradKTBlockGemmSingleRepN<Problem>();

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

  using DropoutType =
      std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

  CK_TILE_DEVICE static constexpr index_t GetSmemSize() {
    return Policy::template GetSmemSize<Problem>();
  }

  // operator() returns dq_acc tile.
  // D[sq] = dO row(.) O is computed and written to delta_dram_window inside
  // this pipeline.
  //
  // Parameters (in order):
  //   do_dram_block_window_tmp   : dO tile [kM0, kVHeaddim]  (first parameter
  //   per spec) o_dram_block_window_tmp    : O  tile [kM0, kVHeaddim]  (for
  //   D[sq] = dO row(.) O) lse_dram_block_window_tmp  : LSE tile [kM0] (for P =
  //   exp(S - LSE)) q_dram_block_window_tmp    : Q  tile [kM0, kQKHeaddim]
  //   k_dram_block_window_tmp    : K  tile [kN0, kQKHeaddim]
  //   v_dram_block_window_tmp    : V  tile [kN0, kVHeaddim]
  //   bias_dram_block_window_tmp : optional bias [kM0, kN0]
  //   delta_dram_block_window    : output window for D[sq] [kM0] (written
  //   inside) seqlen_k_start / seqlen_k_end mask / scale_s / smem_ptr
  template <
      typename DODramBlockWindowTmp,
      typename DeltaDramBlockWindow,
      typename ODramBlockWindowTmp,
      typename LSEDramBlockWindowTmp,
      typename QDramBlockWindowTmp,
      typename KDramBlockWindowTmp,
      typename VDramBlockWindowTmp,
      typename BiasDramBlockWindowTmp,
      typename NullRandValDramWindowTmp,
      typename HstuMask>
  CK_TILE_DEVICE auto operator()(
      const DODramBlockWindowTmp& do_dram_block_window_tmp, // kM0*kVHeaddim
      const ODramBlockWindowTmp& o_dram_block_window_tmp, // kM0*kVHeaddim
      const LSEDramBlockWindowTmp& lse_dram_block_window_tmp, // kM0
      const QDramBlockWindowTmp& q_dram_block_window_tmp, // kM0*kQKHeaddim
      const KDramBlockWindowTmp& k_dram_block_window_tmp, // kN0*kQKHeaddim
      const VDramBlockWindowTmp& v_dram_block_window_tmp, // kN0*kVHeaddim
      const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // kM0*kN0
      const NullRandValDramWindowTmp& null_randval_window_tmp, // M0*N0 tile
      DeltaDramBlockWindow& delta_dram_block_window, // kM0  (output)
      index_t seqlen_k_start,
      index_t seqlen_k_end,
      HstuMask& mask,
      float scale_s,
      void* smem_ptr,
      DropoutType& dropout) const {
    // ---- Gemm objects ----
    constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
    constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
    constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

    using Gemm0Combined =
        decltype(Policy::template GetQKCombinedBlockGemm<Problem>());

    constexpr index_t n0_loops = Policy::template GetNumN0Loops<Problem>();
    constexpr index_t k1_loops = Policy::template GetNumK1Loops<Problem>();

    constexpr auto NumKVPrefetches = min(n0_loops, 2);
    constexpr auto NumKVLdsBuffers =
        Policy::template GetNumKVLdsBuffers<Problem>();

    // ---- Tile type declarations ----
    using SaccBlockTileType =
        decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
    using PGradaccBlockTileType =
        decltype(gemm_2.template MakeCBlockTile<kM0, kN0Sub>());
    using CombinedTileType =
        decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
    using PcompBlockTileType =
        decltype(cast_tile<CompDataType>(CombinedTileType{}));
    using PGradcompBlockTileType = PcompBlockTileType;
    using QGradaccBlockTileType = decltype(gemm_4.MakeCBlockTile());

    // 1-D tile type for per-row scalars (LSE and delta share the same
    // distribution since both reduce the same [kM0, kN0Sub] row shape --
    // SaccBlockTileType)
    using MLBlockTileType = decltype(block_tile_reduce<CompDataType>(
        SaccBlockTileType{},
        sequence<1>{},
        [](auto a, auto b) { return a + b; },
        CompDataType{0}));

    SaccBlockTileType sacc_tile;
    PGradaccBlockTileType dpacc_tile;
    PcompBlockTileType pcomp_tile;
    PGradcompBlockTileType dpcomp_tile;
    QGradaccBlockTileType dq_acc;

    clear_tile(dq_acc);

    if (seqlen_k_start >= seqlen_k_end)
      return dq_acc;

    // ---- LDS setup ----
    // Three LDS regions in order:
    //   [k_lds | v_lds | kt_lds]
    // k_lds  and v_lds  : double-buffered [kN0Sub, kQKHeaddim], invariant view
    // for write/read kt_lds: complete-buffered [kQKHeaddim, kN0Sub], transposed
    // view [kN0Sub, kQKHeaddim] for write
    constexpr index_t k_smem_size = Policy::template GetSmemSizeK<Problem>();
    constexpr index_t v_smem_size = Policy::template GetSmemSizeV<Problem>();
    constexpr index_t kt_smem_size = Policy::template GetSmemSizeKT<Problem>();

    QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
    auto k_lds = make_tensor_view<address_space_enum::lds>(
        k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
    auto k_lds_monolithic_window = make_tile_window(
        k_lds,
        Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeKLdsBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == NumKVLdsBuffers * kN0Sub,
        "Check failed!");
    static_assert(
        Policy::template MakeKLdsBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kQKHeaddim,
        "Check failed!");

    using k_lds_window_type = decltype(get_slice_tile(
        k_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kN0Sub, kQKHeaddim>{}));
    statically_indexed_array<k_lds_window_type, NumKVLdsBuffers> k_lds_windows;
    static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
      k_lds_windows[i_buf] = get_slice_tile(
          k_lds_monolithic_window,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
    });

    QKVDataType* v_lds_ptr = reinterpret_cast<QKVDataType*>(
        static_cast<char*>(smem_ptr) + k_smem_size);
    auto v_lds = make_tensor_view<address_space_enum::lds>(
        v_lds_ptr, Policy::template MakeVLdsBlockDescriptor<Problem>());
    auto v_lds_monolithic_window = make_tile_window(
        v_lds,
        Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == NumKVLdsBuffers * kN0Sub,
        "Check failed!");
    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kVHeaddim,
        "Check failed!");

    using v_lds_window_type = decltype(get_slice_tile(
        v_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kN0Sub, kVHeaddim>{}));
    statically_indexed_array<v_lds_window_type, NumKVLdsBuffers> v_lds_windows;
    static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
      v_lds_windows[i_buf] = get_slice_tile(
          v_lds_monolithic_window,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kVHeaddim>{});
    });

    QKVDataType* kt_lds_ptr = reinterpret_cast<QKVDataType*>(
        static_cast<char*>(smem_ptr) + k_smem_size + v_smem_size);
    auto kt_write_lds = make_tensor_view<address_space_enum::lds>(
        kt_lds_ptr, Policy::template MakeKTLdsWriteBlockDescriptor<Problem>());
    auto kt_lds_write_monolithic_window = make_tile_window(
        kt_write_lds,
        Policy::template MakeKTLdsWriteBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeKTLdsWriteBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == kN0,
        "Check failed!");
    static_assert(
        Policy::template MakeKTLdsWriteBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kQKHeaddim,
        "Check failed!");

    using kt_lds_write_window_type = decltype(get_slice_tile(
        kt_lds_write_monolithic_window,
        sequence<0, 0>{},
        sequence<kN0Sub, kQKHeaddim>{}));
    statically_indexed_array<kt_lds_write_window_type, n0_loops>
        kt_lds_write_windows;
    static_for<0, n0_loops, 1>{}([&](auto i_buf) {
      kt_lds_write_windows[i_buf] = get_slice_tile(
          kt_lds_write_monolithic_window,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
    });

    auto kt_read_lds = make_tensor_view<address_space_enum::lds>(
        kt_lds_ptr, Policy::template MakeKTLdsReadBlockDescriptor<Problem>());
    auto kt_lds_read_monolithic_window = make_tile_window(
        kt_read_lds,
        Policy::template MakeKTLdsReadBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeKTLdsReadBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == kQKHeaddim,
        "Check failed!");
    static_assert(
        Policy::template MakeKTLdsReadBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kN0,
        "Check failed!");

    using kt_lds_read_window_type = decltype(get_slice_tile(
        kt_lds_read_monolithic_window,
        sequence<0, 0>{},
        sequence<kQKHeaddim, kK1>{}));
    statically_indexed_array<kt_lds_read_window_type, k1_loops>
        kt_lds_read_windows;
    static_for<0, k1_loops, 1>{}([&](auto i_buf) {
      kt_lds_read_windows[i_buf] = get_slice_tile(
          kt_lds_read_monolithic_window,
          sequence<0, i_buf * kK1>{},
          sequence<kQKHeaddim, (i_buf + 1) * kK1>{});
    });

    // ---- Delta LDS staging buffer (reuses K LDS space at smem_ptr) ----
    // delta is computed and consumed entirely before the main K/V loop, so it
    // is safe to alias the beginning of the K LDS region for the temporary
    // shuffle.
    CompDataType* delta_lds_ptr = static_cast<CompDataType*>(smem_ptr);
    auto delta_lds = make_tensor_view<address_space_enum::lds>(
        delta_lds_ptr, Policy::template MakeDeltaLdsBlockDescriptor<Problem>());
    auto delta_lds_write_window =
        make_tile_window(delta_lds, make_tuple(number<kM0>{}), {0});
    auto delta_lds_read_window = make_tile_window(
        delta_lds,
        make_tuple(number<kM0>{}),
        {0},
        MLBlockTileType::get_tile_distribution());

    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    // ---- Load Q into registers ----
    auto q_dram_window = make_tile_window(
        q_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
        q_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeQRegTileDistribution<Problem>());
    auto q_tile = load_tile(q_dram_window);

    // ---- Load dO into registers ----
    auto do_dram_window = make_tile_window(
        do_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kVHeaddim>{}),
        do_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeOGradRegTileDistribution<Problem>());
    auto do_tile = load_tile(do_dram_window);

    // ---- Load O into registers (used once for D[sq] = dO row(.) O) ----
    auto o_dram_window = make_tile_window(
        o_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kVHeaddim>{}),
        o_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeORegTileDistribution<Problem>());
    auto o_tile = load_tile(o_dram_window);

    // ---- Load LSE from DRAM [kM0] -- loaded once ----
    auto lse_dram_window = make_tile_window(
        lse_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}),
        lse_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeLSETileDistribution<Problem>());
    auto lse_tile = load_tile(lse_dram_window);

    const auto q_origin = q_dram_window.get_window_origin();

    // ---- Compute D[sq] = dO row(.) O via cross-lane reduction ----
    // do_o_tile[i,j] = dO[i,j] * O[i,j] (element-wise product, CompDataType)
    const auto f_sum = [](auto a, auto b) { return a + b; };

    auto do_o_tile = tile_elementwise_in(
        [](auto dov, auto ov) {
          return type_convert<CompDataType>(dov) *
              type_convert<CompDataType>(ov);
        },
        do_tile,
        o_tile);

    // Row-reduce do_o_tile: each thread holds a partial sum; cross-lane sync
    // completes it.
    auto tmp_delta_tile = block_tile_reduce<CompDataType>(
        do_o_tile, sequence<1>{}, f_sum, CompDataType{0});
    block_tile_reduce_sync(tmp_delta_tile, f_sum, bool_constant<false>{});

    // Shuffle through LDS to rebroadcast delta into the MLBlockTileType
    // distribution.
    store_tile(delta_lds_write_window, tmp_delta_tile);
    block_sync_lds();
    auto delta_tile = load_tile(delta_lds_read_window);

    // Store D[sq] to device memory (spec: written inside pipeline)
    store_tile(
        delta_dram_block_window,
        cast_tile<typename DeltaDramBlockWindow::DataType>(delta_tile));

    // ---- K DRAM window (per-iteration tile [kN0Sub, kQKHeaddim]) ----
    auto k_dram_window = make_tile_window(
        k_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
        {seqlen_k_start, 0},
        Policy::template MakeKDramTileDistribution<Problem>());

    // ---- V DRAM window (per-iteration tile [kN0Sub, kVHeaddim]) ----
    auto v_dram_window = make_tile_window(
        v_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kN0Sub>{}, number<kVHeaddim>{}),
        {seqlen_k_start, 0},
        Policy::template MakeVDramTileDistribution<Problem>());

    // ---- Bias DRAM window ----
    const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
    auto bias_dram_window = make_tile_window(
        bias_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kN0>{}),
        {bias_origin.at(number<0>{}), seqlen_k_start},
        Policy::template MakeBiasDramTileDistribution<Problem>());

    auto null_randval_window =
        dropout.template MakeRandvalDramWindow<Gemm0Combined>(
            null_randval_window_tmp, seqlen_k_start);

    const auto f_exp = [](CompDataType x) -> CompDataType {
      if constexpr (std::is_same_v<CompDataType, float>)
        return __expf(x);
      else
        return exp(x);
    };

    // ---- Prefetch first round of K and V tiles ----
    using k_tile_type = decltype(load_tile(k_dram_window));
    using v_tile_type = decltype(load_tile(v_dram_window));
    statically_indexed_array<k_tile_type, NumKVPrefetches> k_tiles;
    statically_indexed_array<v_tile_type, NumKVPrefetches> v_tiles;

    k_tiles[number<0>{}] = load_tile(k_dram_window);
    move_tile_window(k_dram_window, {kN0Sub, 0});

    __builtin_amdgcn_sched_barrier(
        LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

    v_tiles[number<0>{}] = load_tile(v_dram_window);
    move_tile_window(v_dram_window, {kN0Sub, 0});

    __builtin_amdgcn_sched_barrier(
        LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

    auto seqlen_k_curr = seqlen_k_start;

    // ensure loading access of delta data have been done by all warps
    // so that K/V and reuse the delta_lds space
    block_sync_lds();

    do {
      // === STAGE 1: Gemm0 (S = Q@K) ===
      static_for<0, n0_loops, 1>{}([&](auto i_n0) {
        constexpr auto i_current_buf = number<i_n0 % NumKVPrefetches>{};
        constexpr auto i_prefetch_buf = number<(i_n0 + 1) % NumKVPrefetches>{};
        constexpr auto i_lds_buf_0 = number<i_n0 % NumKVLdsBuffers>{};
        constexpr auto i_lds_buf_1 = i_n0;

        // Store K to k_lds
        store_tile(
            k_lds_windows[i_lds_buf_0],
            k_tiles[i_current_buf],
            partition_index);

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        // Prefetch next K tile while current stores are in flight
        if constexpr (i_n0 + 1 < n0_loops) {
          k_tiles[i_prefetch_buf] = load_tile(k_dram_window);
          move_tile_window(k_dram_window, {kN0Sub, 0});
        }

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        // Ensure all LDS stores are visible before Gemm0 reads
        block_sync_lds();

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        // Gemm0: sacc_tile = Q @ K_sub
        gemm_0(sacc_tile, q_tile, k_lds_windows[i_lds_buf_0]);

        // Store K to kt_lds (transposed, for Gemm4 dQ)
        store_tile(
            kt_lds_write_windows[i_lds_buf_1],
            k_tiles[i_current_buf],
            partition_index);

        auto s_tmp = cast_tile<CompDataType>(sacc_tile);
        set_slice_tile(
            pcomp_tile,
            s_tmp,
            sequence<0, i_n0 * kN0Sub>{},
            sequence<kM0, (i_n0 + 1) * kN0Sub>{});
      });

      // === STAGE 2: Gemm2 (dP = dO@V) ===
      static_for<0, n0_loops, 1>{}([&](auto i_n0) {
        constexpr auto i_current_buf = number<i_n0 % NumKVPrefetches>{};
        constexpr auto i_prefetch_buf = number<(i_n0 + 1) % NumKVPrefetches>{};
        constexpr auto i_lds_buf_0 = number<i_n0 % NumKVLdsBuffers>{};

        store_tile(
            v_lds_windows[i_lds_buf_0],
            v_tiles[i_current_buf],
            partition_index);

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        // Prefetch next V tile while current stores are in flight
        if constexpr (i_n0 + 1 < n0_loops) {
          v_tiles[i_prefetch_buf] = load_tile(v_dram_window);
          move_tile_window(v_dram_window, {kN0Sub, 0});
        }

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        // Ensure all LDS stores are visible before Gemm2 reads
        block_sync_lds();

        // Gemm2: dpacc_tile = dO @ V_sub
        gemm_2(dpacc_tile, do_tile, v_lds_windows[i_lds_buf_0]);
        auto dp_tmp = cast_tile<CompDataType>(dpacc_tile);
        set_slice_tile(
            dpcomp_tile,
            dp_tmp,
            sequence<0, i_n0 * kN0Sub>{},
            sequence<kM0, (i_n0 + 1) * kN0Sub>{});
      });

      __builtin_amdgcn_sched_barrier(
          LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

      // === STAGE 3: scale, bias, mask, then compute P = exp(S - LSE) ===

      if constexpr (kHasBias) {
        const auto bias_tile = load_tile(bias_dram_window);
        tile_elementwise_inout(
            [&scale_s](auto& x, const auto& y) {
              x = x * scale_s + type_convert<CompDataType>(y);
            },
            pcomp_tile,
            bias_tile);
        move_tile_window(bias_dram_window, {0, kN0});
      } else {
        tile_elementwise_inout(
            [&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
      }

      // Softmax path mask: set masked-out S to -inf.
      // The `col >= seqlen_k_end` guard excludes padded tail K columns of the
      // last tile (seqlen_kv not a multiple of kN0). IsTokenPairInsideMask
      // clamps out-of-range columns and can return true for them, so without
      // this guard those padded columns would get S=0 -> P=exp(-LSE)!=0 and
      // pollute dQ.
      if (!mask.IsFullTileInsideMask(
              q_origin.at(number<0>{}),
              seqlen_k_curr,
              number<kN0>{},
              number<kM0>{})) {
        constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType lse_val =
              type_convert<CompDataType>(lse_tile[i_idx]);
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                pcomp_tile.get_tile_distribution(),
                make_tuple(idx0, idx1),
                partition_index);
            const auto row =
                q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
            const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
            constexpr auto ij = make_tuple(idx0, idx1);
            if (!mask.IsTokenPairInsideMask(row, col) || col >= seqlen_k_end)
              pcomp_tile(ij) = type_convert<CompDataType>(0.0f);
            else
              pcomp_tile(ij) = f_exp(pcomp_tile[ij] - lse_val);
          });
        });
      } else {
        // Fully inside the mask, but the last tile may still contain padded
        // tail columns (col >= seqlen_k_end) that must be excluded.
        constexpr auto p_spans = PcompBlockTileType::get_distributed_spans();
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType lse_val =
              type_convert<CompDataType>(lse_tile[i_idx]);
          sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
            const auto tile_idx = get_x_indices_from_distributed_indices(
                pcomp_tile.get_tile_distribution(),
                make_tuple(idx0, idx1),
                partition_index);
            const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
            constexpr auto ij = make_tuple(idx0, idx1);
            if (col >= seqlen_k_end)
              pcomp_tile(ij) = type_convert<CompDataType>(0.0f);
            else
              pcomp_tile(ij) = f_exp(pcomp_tile[ij] - lse_val);
          });
        });
      }

      if constexpr (kHasDropout) {
        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) + k_smem_size +
            v_smem_size + kt_smem_size;

        // Dropout propagates through the chain rule onto dP, NOT onto P. The
        // forward is
        //   O = dropout(P) @ V,   P = exp(S - LSE)
        // so  dS = P * (drop_scale . dP - D),  drop_scale = rp_undrop (kept) /
        // 0 (dropped), where the OUTER P must be the PURE softmax value.
        // BlockDropout::Run applies the mask (kept -> *rp_undrop, dropped -> 0)
        // to dpcomp_tile (= dP), leaving pcomp_tile
        // (= pure P) intact as the outer factor in Stage 4. Matches
        // reference_hstu_attention_bwd.hpp: locals_dS = P * (drop_scale*dP -
        // D).
        dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
            randval_lds_ptr, seqlen_k_curr, dpcomp_tile, null_randval_window);

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);
      }

      // === STAGE 4: dS = P * (dP - D[sq]), then dQ += alpha * dS @ K^T ===
      {
        constexpr auto ds_spans =
            PGradcompBlockTileType::get_distributed_spans();
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
          constexpr auto i_idx = make_tuple(idx0);
          const CompDataType delta_val = delta_tile[i_idx];
          sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
            constexpr auto ij = make_tuple(idx0, idx1);
            const CompDataType p = pcomp_tile[ij];
            const CompDataType dp = dpcomp_tile[ij];
            dpcomp_tile(ij) = p * (dp - delta_val);
          });
        });
      }

      // ensure kt is completely available on Lds
      block_sync_lds();

      k_tiles[number<0>{}] = load_tile(k_dram_window);
      move_tile_window(k_dram_window, {kN0Sub, 0});

      v_tiles[number<0>{}] = load_tile(v_dram_window);
      move_tile_window(v_dram_window, {kN0Sub, 0});

      // Gemm4: dQ += alpha * dS @ K^T
      // K^T is already staged in kt_lds_write_windows from Stage 1.
      // The last block_sync_lds() in Stage 1 guarantees all KT stores are
      // visible.
      static_for<0, k1_loops, 1>{}([&](auto i_k1) {
        auto ds_slice = cast_tile<QKVDataType>(get_slice_tile(
            dpcomp_tile,
            sequence<0, i_k1 * kK1>{},
            sequence<kM0, (i_k1 + 1) * kK1>{}));

        // dQ += dS_sub @ KT_sub
        gemm_4(dq_acc, ds_slice, kt_lds_read_windows[i_k1]);
      });

      seqlen_k_curr += kN0;
    } while (seqlen_k_curr < seqlen_k_end);

    // Apply alpha scaling to dQ
    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, dq_acc);

    return dq_acc;
  }
};

} // namespace ck_tile
