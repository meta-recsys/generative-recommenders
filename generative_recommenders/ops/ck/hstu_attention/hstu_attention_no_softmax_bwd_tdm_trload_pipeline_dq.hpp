// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_bwd_kernel_1_pipeline_policy.hpp"

namespace ck_tile {

// Kernel 1 backward pipeline -- SiLU path (kUseSoftmax == false).
// Iterates over K/V blocks for a fixed Q tile; accumulates dQ.
// Naming: QR = Q and dO are register-resident, KS = K staged through LDS,
//         VS = V staged through LDS (same layout as K, no transpose needed).
template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionBwdKernel1PipelinePolicy>
struct HstuAttentionNoSoftmaxBwdTdmTrLoadPipelineQRKSVS_dQ {
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
  static constexpr index_t kVHeaddim =
      kQKHeaddim; // V shares head dim with K in HSTU
  static constexpr index_t kK1 = HstuAttentionTileSetting::kK1;

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
  // Parameters:
  //   do_dram_block_window_tmp   : dO tile window [kM0, kVHeaddim]  (first per
  //   spec) q_dram_block_window_tmp    : Q  tile window [kM0, kQKHeaddim]
  //   k_dram_block_window_tmp    : K  tile window [kN0, kQKHeaddim]
  //   v_dram_block_window_tmp    : V  tile window [kN0, kVHeaddim]
  //   bias_dram_block_window_tmp : optional bias  [kM0, kN0]
  //   seqlen_k_start / seqlen_k_end : K-range for this Q tile
  //   mask       : HSTU block mask
  //   scale_s    : alpha -- scaling on Q@K result
  //   scale_p    : scale_p -- applied to silu output
  //   smem_ptr   : shared memory pointer
  template <
      typename DODramBlockWindowTmp,
      typename QDramBlockWindowTmp,
      typename KDramBlockWindowTmp,
      typename VDramBlockWindowTmp,
      typename BiasDramBlockWindowTmp,
      typename NullRandValDramWindowTmp,
      typename HstuMask>
  CK_TILE_DEVICE auto operator()(
      const DODramBlockWindowTmp& do_dram_block_window_tmp, // kM0*kVHeaddim
      const QDramBlockWindowTmp& q_dram_block_window_tmp, // kM0*kQKHeaddim
      const KDramBlockWindowTmp& k_dram_block_window_tmp, // kN0*kQKHeaddim
      const VDramBlockWindowTmp& v_dram_block_window_tmp, // kN0*kVHeaddim
      const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // kM0*kN0
      NullRandValDramWindowTmp& null_randval_window_tmp, // M0*N0 tile
      index_t seqlen_k_start,
      index_t seqlen_k_end,
      HstuMask& mask,
      float scale_s,
      float scale_p,
      void* smem_ptr,
      DropoutType& dropout) const {
    // ---- Gemm objects ----
    // Gemm0: S  = Q  @ K   [kM0, kN0Sub]  (A=[kM0,kQKHeaddim],
    // B=[kN0Sub,kQKHeaddim]) Gemm2: dP = dO @ V   [kM0, kN0Sub]  (same shape as
    // Gemm0, separate loop) Gemm4: dQ += dS @ K^T  [kM0, kQKHeaddim]
    constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
    constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
    constexpr auto gemm_4 =
        Policy::template GetSGradKTBlockGemm<Problem, true /* kUseTrLoad */>();

    using Gemm0Combined =
        decltype(Policy::template GetQKCombinedBlockGemm<Problem>());

    constexpr index_t n0_loops = Policy::template GetNumN0Loops<Problem>();
    constexpr index_t k1_loops = Policy::template GetNumK1Loops<Problem>();

    constexpr auto NumVLdsBuffers =
        Policy::template GetNumKVLdsBuffers<Problem>();

    static_assert(NumVLdsBuffers <= n0_loops, "Check failed!");

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

    SaccBlockTileType sacc_tile;
    PGradaccBlockTileType dpacc_tile;
    PcompBlockTileType pcomp_tile;
    PGradcompBlockTileType dpcomp_tile;
    QGradaccBlockTileType dq_acc;

    clear_tile(dq_acc);

    if (seqlen_k_start >= seqlen_k_end)
      return dq_acc;

    // ---- LDS setup ----
    // Two LDS regions in order:
    //   [k_lds | v_lds]
    // k_lds : two complete-buffered [kN0Sub, kQKHeaddim], each used for tdm
    // write/normal read and transposed read using trload; one used for even
    // iteration, the other used for odd iteration:w
    //
    // v_lds : double-buffered [kQKHeaddim, kN0Sub], invariant view for normal
    // write/read
    constexpr index_t k_smem_size = Policy::template GetSmemSizeK<Problem>();
    constexpr index_t v_smem_size = Policy::template GetSmemSizeV<Problem>();

    QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
    auto k_lds = make_tensor_view<address_space_enum::lds>(
        k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
    auto k_lds_monolithic_window_0 = make_tile_window(
        k_lds,
        Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    auto k_lds_monolithic_window_1 = k_lds_monolithic_window_0;
    k_lds_monolithic_window_1.set_bottom_tensor_view_data_ptr(
        k_lds_ptr + k_smem_size / sizeof(QKVDataType));

    static_assert(
        Policy::template MakeKLdsBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == kN0,
        "Check failed!");
    static_assert(
        Policy::template MakeKLdsBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kQKHeaddim,
        "Check failed!");

    // k_lds windows for tdm write and normal read
    using k_lds_window_type = decltype(get_slice_tile(
        k_lds_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kN0Sub, kQKHeaddim>{}));
    statically_indexed_array<k_lds_window_type, n0_loops> k_lds_windows_0;
    statically_indexed_array<k_lds_window_type, n0_loops> k_lds_windows_1;
    static_for<0, n0_loops, 1>{}([&](auto i_buf) {
      k_lds_windows_0[i_buf] = get_slice_tile(
          k_lds_monolithic_window_0,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
      k_lds_windows_1[i_buf] = get_slice_tile(
          k_lds_monolithic_window_1,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
    });

    // k_lds windows for trload read
    using k_lds_trload_window_type = decltype(get_slice_tile(
        k_lds_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kK1, kQKHeaddim>{}));
    statically_indexed_array<k_lds_trload_window_type, k1_loops>
        k_lds_trload_windows_0;
    statically_indexed_array<k_lds_trload_window_type, k1_loops>
        k_lds_trload_windows_1;
    static_for<0, k1_loops, 1>{}([&](auto i_buf) {
      k_lds_trload_windows_0[i_buf] = get_slice_tile(
          k_lds_monolithic_window_0,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kQKHeaddim>{});
      k_lds_trload_windows_1[i_buf] = get_slice_tile(
          k_lds_monolithic_window_1,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kQKHeaddim>{});
    });

    QKVDataType* v_lds_ptr = reinterpret_cast<QKVDataType*>(
        static_cast<char*>(smem_ptr) + k_smem_size * 2);
    auto v_lds = make_tensor_view<address_space_enum::lds>(
        v_lds_ptr, Policy::template MakeVLdsBlockDescriptor<Problem>());
    auto v_lds_monolithic_window_0 = make_tile_window(
        v_lds,
        Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    auto v_lds_monolithic_window_1 = v_lds_monolithic_window_0;
    v_lds_monolithic_window_1.set_bottom_tensor_view_data_ptr(
        v_lds_ptr + v_smem_size / sizeof(QKVDataType));

    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == NumVLdsBuffers * kN0Sub,
        "Check failed!");
    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kVHeaddim,
        "Check failed!");

    using v_lds_window_type = decltype(get_slice_tile(
        v_lds_monolithic_window_0,
        sequence<0, 0>{},
        sequence<kN0Sub, kVHeaddim>{}));
    statically_indexed_array<v_lds_window_type, NumVLdsBuffers> v_lds_windows_0;
    statically_indexed_array<v_lds_window_type, NumVLdsBuffers> v_lds_windows_1;
    static_for<0, NumVLdsBuffers, 1>{}([&](auto i_buf) {
      v_lds_windows_0[i_buf] = get_slice_tile(
          v_lds_monolithic_window_0,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kVHeaddim>{});
      v_lds_windows_1[i_buf] = get_slice_tile(
          v_lds_monolithic_window_1,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kVHeaddim>{});
    });

    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    // ---- Load Q and dO into registers (register-resident for the entire loop)
    // ----
    auto q_dram_window = make_tile_window(
        q_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
        q_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeQRegTileDistribution<Problem>());
    auto q_tile = load_tile(q_dram_window);

    auto do_dram_window = make_tile_window(
        do_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM0>{}, number<kVHeaddim>{}),
        do_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeOGradRegTileDistribution<Problem>());
    auto do_tile = load_tile(do_dram_window);

    const auto q_origin = q_dram_window.get_window_origin();

    // ---- K DRAM window (per-iteration tile [kN0Sub, kQKHeaddim]) ----
    auto k_dram_window = make_tile_window(
        k_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
        {seqlen_k_start, 0},
        Policy::template MakeKDramTileDistribution<Problem>());

    // ---- V DRAM window (per-iteration tile [kN0Sub, kVHeaddim], mirrors K)
    // ----
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

    TDMConfig tdm_config_k;
    constexpr auto pad_cfg_k = Policy::template GetKLdsPaddingConfig<Problem>();
    tdm_config_k.pad_enable = pad_cfg_k[number<0>{}];
    tdm_config_k.pad_config.pad_amount = pad_cfg_k[number<1>{}];
    tdm_config_k.pad_config.pad_interval = pad_cfg_k[number<2>{}];

    TDMConfig tdm_config_v;
    constexpr auto pad_cfg_v = Policy::template GetVLdsPaddingConfig<Problem>();
    tdm_config_v.pad_enable = pad_cfg_v[number<0>{}];
    tdm_config_v.pad_config.pad_amount = pad_cfg_v[number<1>{}];
    tdm_config_v.pad_config.pad_interval = pad_cfg_v[number<2>{}];

    load_tile_tdm(tdm_config_k, k_lds_windows_0[number<0>{}], k_dram_window);
    move_tile_window(k_dram_window, {kN0Sub, 0});

    load_tile_tdm(tdm_config_v, v_lds_windows_0[number<0>{}], v_dram_window);
    move_tile_window(v_dram_window, {kN0Sub, 0});

    // SiLU activation: dsilu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    const auto f_dsilu = [](CompDataType x) -> CompDataType {
      const auto one = type_convert<CompDataType>(1.0f);
      CompDataType sig;
      if constexpr (std::is_same_v<CompDataType, float>)
        sig = __builtin_amdgcn_rcpf(one + __expf(-x));
      else
        sig = one / (one + exp(-x));
      return sig * (one + x * (one - sig));
    };

    auto seqlen_k_curr = seqlen_k_start;

    auto main_loop_func = [&](auto EvenIter) {
      // === STAGE 1: Gemm0 (S = Q@K) ===
      // === STAGE 2: Gemm2 (dP = dO@V) ===
      static_for<0, n0_loops, 1>{}([&](auto i_n0) {
        constexpr auto i_lds_buf_0 = number<i_n0 % NumVLdsBuffers>{};
        constexpr auto i_lds_buf_1 = number<(i_n0 + 1) % NumVLdsBuffers>{};

        s_wait_tensorcnt_barrier<0>();

        __builtin_amdgcn_sched_barrier(
            LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

        // Gemm0: sacc_tile = Q @ K_sub
        // Gemm2: dpacc_tile = dO @ V_sub
        if constexpr (EvenIter) {
          if constexpr (i_n0 < n0_loops - 1) {
            load_tile_tdm(
                tdm_config_k,
                k_lds_windows_0[number<i_n0 + 1>{}],
                k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
            load_tile_tdm(
                tdm_config_v, v_lds_windows_0[i_lds_buf_1], v_dram_window);
            move_tile_window(v_dram_window, {kN0Sub, 0});
          } else {
            load_tile_tdm(
                tdm_config_k, k_lds_windows_1[number<0>{}], k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
            load_tile_tdm(
                tdm_config_v, v_lds_windows_1[number<0>{}], v_dram_window);
            move_tile_window(v_dram_window, {kN0Sub, 0});
          }

          __builtin_amdgcn_sched_barrier(
              LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

          gemm_0(sacc_tile, q_tile, k_lds_windows_0[i_n0]);
          gemm_2(dpacc_tile, do_tile, v_lds_windows_0[i_lds_buf_0]);
        } else {
          if constexpr (i_n0 < n0_loops - 1) {
            load_tile_tdm(
                tdm_config_k,
                k_lds_windows_1[number<i_n0 + 1>{}],
                k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
            load_tile_tdm(
                tdm_config_v, v_lds_windows_1[i_lds_buf_1], v_dram_window);
            move_tile_window(v_dram_window, {kN0Sub, 0});
          } else {
            load_tile_tdm(
                tdm_config_k, k_lds_windows_0[number<0>{}], k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
            load_tile_tdm(
                tdm_config_v, v_lds_windows_0[number<0>{}], v_dram_window);
            move_tile_window(v_dram_window, {kN0Sub, 0});
          }

          __builtin_amdgcn_sched_barrier(
              LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

          gemm_0(sacc_tile, q_tile, k_lds_windows_1[i_n0]);
          gemm_2(dpacc_tile, do_tile, v_lds_windows_1[i_lds_buf_0]);
        }

        auto s_tmp = cast_tile<CompDataType>(sacc_tile);
        set_slice_tile(
            pcomp_tile,
            s_tmp,
            sequence<0, i_n0 * kN0Sub>{},
            sequence<kM0, (i_n0 + 1) * kN0Sub>{});

        auto dp_tmp = cast_tile<CompDataType>(dpacc_tile);
        set_slice_tile(
            dpcomp_tile,
            dp_tmp,
            sequence<0, i_n0 * kN0Sub>{},
            sequence<kM0, (i_n0 + 1) * kN0Sub>{});
      });

      // === STAGE 3: scale, optional bias, mask, then compute P and dS ===

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

      const bool need_mask = !mask.IsFullTileInsideMask(
          q_origin.at(number<0>{}),
          seqlen_k_curr,
          number<kN0>{},
          number<kM0>{});

      if constexpr (kHasDropout) {
        __builtin_amdgcn_sched_barrier(0);

        auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) +
            k_smem_size * 2 + v_smem_size * 2;

        // Dropout propagates through the chain rule onto dP, NOT onto S. The
        // forward is
        //   O = dropout(P) @ V,   P = silu(S) * scale_p
        // so  dS = (drop_scale . dP) * scale_p * dsilu(S),  drop_scale =
        // rp_undrop (kept) / 0 (dropped). BlockDropout::Run applies exactly
        // that mask (kept -> *rp_undrop, dropped -> 0) to dpcomp_tile (= dP),
        // leaving pcomp_tile (= S) intact for the dsilu(S) in Stage 4. Matches
        // reference_hstu_attention_bwd.hpp:
        //   locals_dS = drop_scale * dP * scale_p * dsilu(S).
        dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
            randval_lds_ptr, seqlen_k_curr, dpcomp_tile, null_randval_window);

        __builtin_amdgcn_sched_barrier(0);
      }

      // === STAGE 4: dS = dP * scale_p * dsilu(S), then dQ += alpha * dS @ K^T
      // === dS[sq,sk] = dP[sq,sk] * scale_p * dsilu(S[sq,sk]) Correction vs. a
      // naive dS = dp*scale_p*dsilu(s):
      //  - dsilu(0) != 0 (dsilu(0) = 0.5), so masking S to 0 above is NOT
      //  enough to zero
      //    dS at masked-out pairs; force dS = 0 outside the mask (reference:
      //    locals_dS = 0).
      // Padded columns (col >= seqlen_k_end) in the last K tile need no
      // explicit handling: that column of dP = dO @ V comes from OOB row `col`
      // of V, which buffer_load zeroes (the seqlen dim is intentionally not
      // pad_tensor_view'd), so dp == 0 exactly there and dS = 0 * scale_p *
      // dsilu(s) = 0 (s is also 0 via the same K OOB zeroing, so dsilu(s) is
      // finite -- no 0*inf hazard). They cannot leak into the dQ reduction.
      constexpr auto ds_spans = PGradcompBlockTileType::get_distributed_spans();

      sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
          constexpr auto ij = make_tuple(idx0, idx1);
          const CompDataType s = pcomp_tile[ij];
          const CompDataType dp = dpcomp_tile[ij];
          CompDataType ds =
              dp * type_convert<CompDataType>(scale_p) * f_dsilu(s);

          dpcomp_tile(ij) = ds;
        });
      });

      if (need_mask) {
        sweep_tile_span(ds_spans[number<0>{}], [&](auto idx0) {
          sweep_tile_span(ds_spans[number<1>{}], [&](auto idx1) {
            constexpr auto ij = make_tuple(idx0, idx1);
            const auto tile_idx = get_x_indices_from_distributed_indices(
                dpcomp_tile.get_tile_distribution(),
                make_tuple(idx0, idx1),
                partition_index);
            const auto row =
                q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
            const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
            if (!mask.IsTokenPairInsideMask(row, col))
              dpcomp_tile(ij) = type_convert<CompDataType>(0.0f);
          });
        });
      }

      // Gemm4: dQ += alpha * dS @ K^T
      // K^T is already staged in kt_lds_read_windows from Stage 1.
      static_for<0, k1_loops, 1>{}([&](auto i_k1) {
        auto ds_slice = cast_tile<QKVDataType>(get_slice_tile(
            dpcomp_tile,
            sequence<0, i_k1 * kK1>{},
            sequence<kM0, (i_k1 + 1) * kK1>{}));

        // dQ += dS_sub @ KT_sub
        if constexpr (EvenIter) {
          gemm_4(dq_acc, ds_slice, k_lds_trload_windows_0[i_k1]);
        } else {
          gemm_4(dq_acc, ds_slice, k_lds_trload_windows_1[i_k1]);
        }
      });
    };

    while (seqlen_k_curr + kN0 < seqlen_k_end) {
      main_loop_func(bool_constant<true>{});
      seqlen_k_curr += kN0;

      main_loop_func(bool_constant<false>{});
      seqlen_k_curr += kN0;
    }

    if (seqlen_k_curr < seqlen_k_end)
      main_loop_func(bool_constant<true>{});

    // Apply alpha scaling to accumulated dQ
    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, dq_acc);

    return dq_acc;
  }
};

} // namespace ck_tile
