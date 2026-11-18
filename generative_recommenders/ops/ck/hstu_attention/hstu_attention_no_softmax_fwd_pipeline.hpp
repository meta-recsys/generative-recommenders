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
struct HstuAttentionNoSoftmaxFwdPipelineQRKSVS {
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
      Problem::kUseSoftmax == false,
      "This pipeline only works with not-using softmax");

  static constexpr bool kIsCrossAttention = Problem::kIsCrossAttention;
  static constexpr bool kIsJagged = Problem::kIsJagged;
  static constexpr auto kHasBias = Problem::kHasBias;
  static constexpr bool kHasDropout = Problem::kHasDropout;
  static constexpr bool kHasCausal = Problem::kHasCausal;

  static constexpr bool kUseTrLoad = false;

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
  static constexpr index_t kAlignmentV =
      Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();

  static constexpr index_t kAlignmentO =
      kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
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
          return 4;
        if constexpr (kM0 == 64 && kIsCrossAttention)
          return 4;
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
    return Policy::template GetSmemSize<Problem>();
  }

  template <
      typename QDramBlockWindowTmp,
      typename KDramBlockWindowTmp,
      typename VDramBlockWindowTmp,
      typename BiasDramBlockWindowTmp,
      typename NullRandValDramWindowTmp,
      typename HstuMask>
  CK_TILE_DEVICE auto operator()(
      const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*kQKHeaddim tile
      const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*kQKHeaddim tile
      const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
      const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
      NullRandValDramWindowTmp& null_randval_window_tmp, // M0*N0 tile
      index_t seqlen_k_start,
      index_t seqlen_k_end,
      HstuMask& mask,
      float scale_s, // scaling value exerted on the immediate Q@K result
      float scale_p, // scaling value exerted on the SiLu result
      void* smem_ptr,
      DropoutType& dropout) const {
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
            kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
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
    constexpr auto gemm_1 = Policy::template GetPVTBlockGemm<Problem>();

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

    using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
    OaccBlockTileType o_acc;

    if (seqlen_k_end <= seqlen_k_start) {
      clear_tile(o_acc);

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
    QKVDataType* v_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
    auto v_lds = make_tensor_view<address_space_enum::lds>(
        v_lds_ptr, Policy::template MakeVLdsBlockDescriptor<Problem>());
    auto v_lds_monolithic_window = make_tile_window(
        v_lds,
        Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
        {0, 0});

    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem>()
                .get_lengths()[number<0>{}] == NumVLdsBuffers * kN1,
        "Check failed!");
    static_assert(
        Policy::template MakeVLdsBlockDescriptor<Problem>()
                .get_lengths()[number<1>{}] == kK1,
        "Check failed!");

    using v_lds_window_type = decltype(get_slice_tile(
        v_lds_monolithic_window, sequence<0, 0>{}, sequence<kN1, kK1>{}));

    statically_indexed_array<v_lds_window_type, NumVLdsBuffers> v_lds_windows;

    static_for<0, NumVLdsBuffers, 1>{}([&](auto i_buf) {
      v_lds_windows[i_buf] = get_slice_tile(
          v_lds_monolithic_window,
          sequence<i_buf * kN1, 0>{},
          sequence<(i_buf + 1) * kN1, kK1>{});
    });

    auto v_dram_window = make_tile_window(
        v_dram_block_window_tmp.get_bottom_tensor_view(),
        v_dram_block_window_tmp.get_window_lengths(),
        {0, seqlen_k_start}, // TODO: hdim split?
        Policy::template MakeVDramTileDistribution<Problem>());

    // reduction function for softmax
    const auto f_silu = [&](CompDataType& x) {
      const auto one = ck_tile::type_convert<CompDataType>(1.0f);

      if constexpr (std::is_same_v<CompDataType, float>) {
        x = x * __builtin_amdgcn_rcpf(one + __expf(-x));
      } else {
        x = x / (one + exp(-x));
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
            move_tile_window(v_dram_window, {0, kK1});
          });
        } else {
          if constexpr ((i_n0 + 1) % n0loops_per_k1 == 0) {
            constexpr index_t i_k1 = (i_n0 + 1) / n0loops_per_k1 - 1;

            // load v_tiles used in current iteration
            v_tiles[number<i_k1>{}] = load_tile(v_dram_window);
            move_tile_window(v_dram_window, {0, kK1});
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
      } else {
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

            if (!mask.IsTokenPairInsideMask(row, col)) {
              pcomp_tile(i_j_idx) = type_convert<CompDataType>(0.0f);
            };
          });
        });
      }

      __builtin_amdgcn_sched_barrier(
          LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

      using v_shuffled_tile_type =
          decltype(make_static_distributed_tensor<QKVDataType>(
              Policy::template MakeShuffledVRegTileDistribution<Problem>()));

      v_shuffled_tile_type v_shuffled_tile;

      shuffle_tile(v_shuffled_tile, v_tiles[number<0>{}]);

      if constexpr (Policy::template IsFirstVBufferOverlapLastKBuffer<
                        Problem>()) {
        // ensure all warps have done their access to k_lds so that v_lds can be
        // stored
        __builtin_amdgcn_s_barrier();
      }

      store_tile(v_lds_windows[number<0>{}], v_shuffled_tile, partition_index);

      __builtin_amdgcn_sched_barrier(
          LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

      tile_elementwise_inout(f_silu, pcomp_tile);

      if constexpr (kHasDropout) {
        __builtin_amdgcn_sched_barrier(0);

        auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) +
            max(Policy::template GetSmemSizeK<Problem>(),
                Policy::template GetSmemSizeV<Problem>());

        dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
            randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);

        __builtin_amdgcn_sched_barrier(0);
      }

      seqlen_k_curr += kN0;

      auto p = cast_tile<PDataType>(pcomp_tile);

      // STAGE 3, Gemm_1 ( O = P@V )
      static_for<0, k1_loops, 1>{}([&](auto i_k1) {
        // load k_tiles used by next iteration

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
            v_lds_windows[number<i_k1 % NumVLdsBuffers>{}]);

        if constexpr (i_k1 < k1_loops - 1) {
          __builtin_amdgcn_sched_barrier(
              LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);

          shuffle_tile(v_shuffled_tile, v_tiles[number<i_k1 + 1>{}]);
          store_tile(
              v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
              v_shuffled_tile,
              partition_index);

          __builtin_amdgcn_sched_barrier(
              LLVMSchedGroupMask::ALU | LLVMSchedGroupMask::TRANS);
        };
      });

      if constexpr (Policy::template IsFirstKBufferOverlapLastVBuffer<
                        Problem>()) {
        // ensure all warps have done their access to v_lds so that k_lds can be
        // stored
        __builtin_amdgcn_s_barrier();
      }
    } while (seqlen_k_curr < seqlen_k_end);

    tile_elementwise_inout(
        [&](auto& x) { x = x * type_convert<CompDataType>(scale_p); }, o_acc);

    return o_acc;
  }
};

} // namespace ck_tile
