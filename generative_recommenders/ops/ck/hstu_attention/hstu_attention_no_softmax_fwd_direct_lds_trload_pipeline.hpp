// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>

#include "hstu_attention_fwd_pipeline_policy.hpp"
#include "hstu_attention_kernel_util.hpp"
#include "hstu_attention_pipeline_helper.hpp"

namespace ck_tile {

template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionFwdPipelineQRKSVSPolicy>
struct HstuAttentionNoSoftmaxFwdPipelineQRKSVSDirectLdsTrLoad {
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

  static constexpr index_t kAlignmentO =
      Policy::template GetAlignmentO<Problem, true /*kUseTrLoad */>();
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
        if constexpr (kM0 == 64)
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
    return Policy::template GetSmemSize<
        Problem,
        true /*kPipelineUseTrLoad*/,
        true /*kUseDirectLds*/>();
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

    constexpr auto NumKLdsBuffers =
        Policy::template GetNumKLdsBuffers<Problem, true /*kUseDirectLds*/>();
    constexpr auto NumVLdsBuffers =
        Policy::template GetNumVLdsBuffers<Problem, true /*kUseDirectLds*/>();

    static_assert(NumKLdsBuffers == n0_loops, "Check failed!");
    static_assert(NumVLdsBuffers == k1_loops, "Check failed!");

    // Block GEMM
    constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
    constexpr auto gemm_1 =
        Policy::template GetPVTBlockGemm<Problem, true /*kUseTrLoad*/>();

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

    do {
      // wait for both the Direct-Lds K copying and reading from v_lds to be
      // done
      s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>();

      // STAGE 1, Gemm_0 ( S = Q@K )
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

      tile_elementwise_inout(f_silu, pcomp_tile);

      if constexpr (kHasDropout) {
        auto randval_lds_ptr = reinterpret_cast<char*>(smem_ptr) +
            Policy::template GetSmemSizeK<Problem, true /*kUseDirectLds*/>() +
            Policy::template GetSmemSizeV<
                                   Problem,
                                   true /*kPipelineUseTrLoad*/,
                                   true /*kUseDirectLds */>();

        dropout.template Run<Gemm0Combined, CompDataType, uint8_t>(
            randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);
      }

      seqlen_k_curr += kN0;

      auto p = cast_tile<PDataType>(pcomp_tile);

      // wait for both the Direct-Lds V copying and reading from k_lds to be
      // done
      s_waitcnt_barrier<0, waitcnt_arg::kMaxExpCnt, 0>();

      // STAGE 3, Gemm_1 ( O = P@V )
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
    } while (seqlen_k_curr < seqlen_k_end);

    detail::scale_tile_in_pack(o_acc, scale_p);

    return o_acc;
  }
};

} // namespace ck_tile
