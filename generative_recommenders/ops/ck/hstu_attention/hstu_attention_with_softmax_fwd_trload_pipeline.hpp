// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"

namespace ck_tile {

template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
struct HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad {
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

  // last dimension vector length used to create tensor view(and decide
  // buffer_load vector length)
  // ... together with tensor distribution. tensor dist should able to overwrite
  // this
  static constexpr index_t kAlignmentQ =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
  static constexpr index_t kAlignmentK =
      kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
  static constexpr index_t kAlignmentV = Traits::kPadHeadDimV
      ? 1
      : Policy::template GetAlignmentV<Problem, true /*kUseTrLoad*/>();

  // since dim num_splits is the fastest dim and each workgroup accesses only
  // one value in this dim
  static constexpr index_t kAlignmentLSEacc = 1;

  static constexpr index_t kAlignmentO = kPadHeadDimV
      ? 1
      : Policy::template GetAlignmentO<Problem, true /*kUseTrLoad*/>();
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
    return Policy::template GetSmemSize<Problem, true /*kPipelineUseTrLoad*/>();
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

    constexpr auto NumKLdsBuffers =
        Policy::template GetNumKLdsBuffers<Problem>();
    constexpr auto NumVLdsBuffers =
        Policy::template GetNumVLdsBuffers<Problem>();

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

    auto k_dram_window = make_tile_window(
        k_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
        {seqlen_k_start, 0},
        Policy::template MakeKDramTileDistribution<Problem>());

    using k_tile_type = decltype(load_tile(k_dram_window));

    statically_indexed_array<k_tile_type, n0_loops> k_tiles;

    static_for<0, n0_loops, 1>{}([&](auto i_n0) {
      k_tiles[i_n0] = load_tile(k_dram_window);
      move_tile_window(k_dram_window, {kN0Sub, 0});
    });

    __builtin_amdgcn_sched_barrier(
        LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

    // provide partition_index for LDS tile window so that warp_id is in vgpr
    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    // K tile in LDS
    QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
    auto k_lds = make_tensor_view<address_space_enum::lds>(
        k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
    auto k_lds_monolithic_window = make_tile_window(
        k_lds,
        Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeKLdsBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == NumKLdsBuffers * kN0Sub,
        "Check failed!");
    static_assert(
        Policy::template MakeKLdsBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kQKHeaddim,
        "Check failed!");

    using k_lds_window_type = decltype(get_slice_tile(
        k_lds_monolithic_window,
        sequence<0, 0>{},
        sequence<kN0Sub, kQKHeaddim>{}));

    statically_indexed_array<k_lds_window_type, NumKLdsBuffers> k_lds_windows;

    static_for<0, NumKLdsBuffers, 1>{}([&](auto i_buf) {
      k_lds_windows[i_buf] = get_slice_tile(
          k_lds_monolithic_window,
          sequence<i_buf * kN0Sub, 0>{},
          sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
    });

    // V tile in LDS
    auto v_lds = make_tensor_view<address_space_enum::lds>(
        reinterpret_cast<QKVDataType*>(
            reinterpret_cast<char*>(smem_ptr) +
            Policy::template GetSmemSizeK<Problem>()),
        Policy::
            template MakeVLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>());
    auto v_lds_monolithic_window = make_tile_window(
        v_lds,
        Policy::template MakeVLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
            .get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
                .get_lengths()[number<0>{}] == NumVLdsBuffers * kK1,
        "Check failed!");
    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem, true /*kUseTrLoad*/>()
                .get_lengths()[number<1>{}] == kN1,
        "Check failed!");

    // v_lds for normal write and trload read
    using v_lds_trload_window_type = decltype(get_slice_tile(
        v_lds_monolithic_window, sequence<0, 0>{}, sequence<kK1, kN1>{}));

    statically_indexed_array<v_lds_trload_window_type, NumVLdsBuffers>
        v_lds_trload_windows;

    static_for<0, NumVLdsBuffers, 1>{}([&](auto i_buf) {
      v_lds_trload_windows[i_buf] = get_slice_tile(
          v_lds_monolithic_window,
          sequence<i_buf * kK1, 0>{},
          sequence<(i_buf + 1) * kK1, kN1>{});
    });

    auto v_dram_window = make_tile_window(
        v_dram_block_window_tmp.get_bottom_tensor_view(),
        v_dram_block_window_tmp.get_window_lengths(),
        {seqlen_k_start, 0},
        Policy::
            template MakeVDramTileDistribution<Problem, true /*kUseTrLoad*/>());

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

    auto seqlen_k_curr = seqlen_k_start;

    using v_tile_type = decltype(load_tile(v_dram_window));

    statically_indexed_array<v_tile_type, k1_loops> v_tiles;

    do {
      // STAGE 1, Gemm_0 ( S = Q@K )
      static_for<0, n0_loops, 1>{}([&](auto i_n0) {
        store_tile(
            k_lds_windows[number<i_n0 % NumKLdsBuffers>{}],
            k_tiles[i_n0],
            partition_index);

        // load v_tiles used in current iteration
        if constexpr (k1_loops >= n0_loops) {
          static_for<0, k1loops_per_n0sub, 1>{}([&](auto j) {
            v_tiles[number<i_n0 * k1loops_per_n0sub + j>{}] =
                load_tile(v_dram_window);
            move_tile_window(v_dram_window, {kK1, 0});
          });
        } else {
          if constexpr ((i_n0 + 1) % n0loops_per_k1 == 0) {
            constexpr index_t i_k1 = (i_n0 + 1) / n0loops_per_k1 - 1;

            // load v_tiles used in current iteration
            v_tiles[number<i_k1>{}] = load_tile(v_dram_window);
            move_tile_window(v_dram_window, {kK1, 0});
          }
        }

        block_sync_lds();

        // execute current unroll of gemm_0
        gemm_0(
            sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKLdsBuffers>{}]);

        auto tmp_tile = cast_tile<CompDataType>(sacc_tile);

        set_slice_tile(
            pcomp_tile,
            tmp_tile,
            sequence<0, i_n0 * kN0Sub>{},
            sequence<kM0, (i_n0 + 1) * kN0Sub>{});
      });

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

      if (!mask.IsFullTileInsideMask(
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

      const auto m_old = m;

      block_tile_reduce(m, pcomp_tile, sequence<1>{}, f_max);
      block_tile_reduce_sync(m, f_max, bool_constant<false>{});

      __builtin_amdgcn_sched_barrier(
          LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

      constexpr auto p_spans = decltype(pcomp_tile)::get_distributed_spans();
      sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);

        if (__builtin_expect(
                m[i_idx] == -numeric<CompDataType>::infinity(), 0)) // unlikely
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
                m[i_idx] == -numeric<CompDataType>::infinity(), 0)) // unlikely
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

      if constexpr (kHasDropout) {
        __builtin_amdgcn_sched_barrier(0);

        auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) +
            Policy::template GetSmemSizeK<Problem>() +
            Policy::
                template GetSmemSizeV<Problem, true /*kPipelineUseTrLoad*/>();

        dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
            randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);

        __builtin_amdgcn_sched_barrier(0);
      }

      seqlen_k_curr += kN0;

      auto p = cast_tile<PDataType>(pcomp_tile);

      // STAGE 3, Gemm_1 ( O = P@V )
      static_for<0, k1_loops, 1>{}([&](auto i_k1) {
        store_tile(
            v_lds_trload_windows[number<i_k1 % NumVLdsBuffers>{}],
            v_tiles[number<i_k1>{}],
            partition_index);

        if constexpr (k1_loops >= n0_loops) {
          if constexpr ((i_k1 + 1) % k1loops_per_n0sub == 0) {
            constexpr index_t i_n0 = (i_k1 + 1) / k1loops_per_n0sub - 1;

            k_tiles[number<i_n0>{}] = load_tile(k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
          }
        } else {
          static_for<0, n0loops_per_k1, 1>{}([&](auto j) {
            k_tiles[number<i_k1 * n0loops_per_k1 + j>{}] =
                load_tile(k_dram_window);
            move_tile_window(k_dram_window, {kN0Sub, 0});
          });
        }

        block_sync_lds();

        gemm_1(
            o_acc,
            get_slice_tile(
                p,
                sequence<0, i_k1 * kK1>{},
                sequence<kM0, (i_k1 + 1) * kK1>{}),
            v_lds_trload_windows[number<i_k1 % NumVLdsBuffers>{}]);
      });
    } while (seqlen_k_curr < seqlen_k_end);

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
