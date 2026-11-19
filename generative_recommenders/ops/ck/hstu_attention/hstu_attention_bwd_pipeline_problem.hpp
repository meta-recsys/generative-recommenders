// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

#include <ck_tile/core.hpp>

#include "hstu_attention_bwd_tile_setting_define.hpp"
#include "hstu_attention_problem_common_util.hpp"

namespace ck_tile {

namespace detail {

template <typename T>
struct is_hstu_attention_bwd_tile_setting_for_kernel1 : std::false_type {};

template <
    typename BlockTile_,
    typename Gemm0Gemm2BlockWarps_,
    typename Gemm0Gemm2WarpTile_,
    typename Gemm4BlockWarps_,
    typename Gemm4WarpTile_>
struct is_hstu_attention_bwd_tile_setting_for_kernel1<
    HstuAttentionBwdTileSettingClassForKernel1<
        BlockTile_,
        Gemm0Gemm2BlockWarps_,
        Gemm0Gemm2WarpTile_,
        Gemm4BlockWarps_,
        Gemm4WarpTile_>> : std::true_type {};

template <typename T>
struct is_hstu_attention_bwd_tile_setting_for_kernel2 : std::false_type {};

template <
    typename BlockTile_,
    typename Gemm0Gemm2BlockWarps_,
    typename Gemm0Gemm2WarpTile_,
    typename Gemm1BlockWarps_,
    typename Gemm1WarpTile_,
    typename Gemm3BlockWarps_,
    typename Gemm3WarpTile_>
struct is_hstu_attention_bwd_tile_setting_for_kernel2<
    HstuAttentionBwdTileSettingClassForKernel2<
        BlockTile_,
        Gemm0Gemm2BlockWarps_,
        Gemm0Gemm2WarpTile_,
        Gemm1BlockWarps_,
        Gemm1WarpTile_,
        Gemm3BlockWarps_,
        Gemm3WarpTile_>> : std::true_type {};

}; // namespace detail

// Shared template parameters for both backward kernels, excluding tile-sizing
// policy. Analogous to HstuAttentionFwdPipelineProblem but without TileSetting.
template <
    typename InOutDataType_, // fp16 or bf16 -- Q, K, V, O, dO, dQ, dK, dV
    typename GemmAccDataType_, // float -- GEMM accumulator
    typename CompDataType_, // float -- SiLU / softmax intermediate
    bool kIsCrossAttention_,
    bool kUseGroup_,
    bool kIsJagged_,
    bool kHasBias_, // bias added to S (same semantics as forward)
    bool kHasCausal_,
    bool kUseSoftmax_,
    bool kHasDropout_>
struct HstuAttentionBwdPipelineBaseProblem {
  using InOutDataType = remove_cvref_t<InOutDataType_>;
  using QKVDataType = InOutDataType;
  using BiasDataType = InOutDataType;
  using ODataType = InOutDataType;
  using OGradDataType = InOutDataType;
  using QGradDataType = InOutDataType;
  using KGradDataType = InOutDataType;
  using VGradDataType = InOutDataType;
  using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;
  using CompDataType = remove_cvref_t<CompDataType_>;

  using PDataType = QKVDataType;
  using OaccDataType = GemmAccDataType;

  static constexpr bool kIsCrossAttention = kIsCrossAttention_;
  static constexpr bool kUseGroup = kUseGroup_;
  static constexpr bool kIsJagged = kIsJagged_;
  static constexpr bool kHasBias = kHasBias_;
  static constexpr bool kHasCausal = kHasCausal_;
  static constexpr bool kUseSoftmax = kUseSoftmax_;
  static constexpr bool kHasDropout = kHasDropout_;

  static_assert(
      !kUseGroup || (kUseGroup && kIsJagged),
      "Group HSTU is only used with jagged mode!");
};

// Kernel 1: computes dQ (and D[sq] = dO row(.) O for the softmax path).
// Iterates over K/V dimension blocks; one block per (batch, head, sq_tile).
// TileSetting must be an HstuAttentionTileSettingClassForKernel1 instance.
template <
    typename PipelineBaseProblem_, // HstuAttentionBwdPipelineBaseProblem
                                   // instance
    typename TileSetting_> // HstuAttentionTileSettingClassForKernel1 instance
struct HstuAttentionBwdPipelineProblemForKernel1 {
  static_assert(
      detail::is_hstu_attention_bwd_tile_setting_for_kernel1<
          remove_cvref_t<TileSetting_>>::value,
      "TileSetting_ must be an instance of HstuAttentionBwdTileSettingClassForKernel1");

  using BaseProblem = remove_cvref_t<PipelineBaseProblem_>;

  using InOutDataType = typename BaseProblem::InOutDataType;
  using QKVDataType = typename BaseProblem::QKVDataType;
  using ODataType = typename BaseProblem::ODataType;
  using BiasDataType = typename BaseProblem::BiasDataType;
  using OGradDataType = typename BaseProblem::OGradDataType;
  using QGradDataType = typename BaseProblem::QGradDataType;
  using KGradDataType = typename BaseProblem::KGradDataType;
  using VGradDataType = typename BaseProblem::VGradDataType;
  using GemmAccDataType = typename BaseProblem::GemmAccDataType;
  using CompDataType = typename BaseProblem::CompDataType;
  using PDataType = typename BaseProblem::PDataType;
  using QGradAccDataType = typename BaseProblem::GemmAccDataType;

  static constexpr bool kIsCrossAttention = BaseProblem::kIsCrossAttention;
  static constexpr bool kUseGroup = BaseProblem::kUseGroup;
  static constexpr bool kIsJagged = BaseProblem::kIsJagged;
  static constexpr bool kHasBias = BaseProblem::kHasBias;
  static constexpr bool kHasCausal = BaseProblem::kHasCausal;
  static constexpr bool kUseSoftmax = BaseProblem::kUseSoftmax;
  static constexpr bool kHasDropout = BaseProblem::kHasDropout;

  using HstuAttentionTileSetting = remove_cvref_t<TileSetting_>;

  static constexpr bool IsWarpGemm32 = HstuAttentionTileSetting::IsWarpGemm32;

  static constexpr index_t kNumGemm0Gemm2Warps =
      TileSetting_::NumGemm0Gemm2Warps;
  static constexpr index_t kNumGemm4Warps = TileSetting_::NumGemm4Warps;
  static constexpr index_t kBlockSize =
      TileSetting_::NumWarps * get_warp_size();

  // K tile: [kN0Sub, kQKHeaddim]
  CK_TILE_HOST_DEVICE static constexpr auto GetKDramTileAccessMaxVectorSize() {
    constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kNPerBlock,
        kKPerBlock>();
  }

  // V tile: [kN0Sub, kQKHeaddim]
  CK_TILE_HOST_DEVICE static constexpr auto GetVDramTileAccessMaxVectorSize() {
    constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kNPerBlock,
        kKPerBlock>();
  }
};

// Kernel 2: computes dK and dV.
// Iterates over Q dimension blocks; one block per (batch, head, sk_tile).
// TileSetting must be an HstuAttentionTileSettingClassForKernel2 instance.
template <
    typename PipelineBaseProblem_, // HstuAttentionBwdPipelineBaseProblem
                                   // instance
    typename TileSetting_> // HstuAttentionTileSettingClassForKernel2 instance
struct HstuAttentionBwdPipelineProblemForKernel2 {
  static_assert(
      detail::is_hstu_attention_bwd_tile_setting_for_kernel2<
          remove_cvref_t<TileSetting_>>::value,
      "TileSetting_ must be an instance of HstuAttentionBwdTileSettingClassForKernel2");

  using BaseProblem = remove_cvref_t<PipelineBaseProblem_>;

  using InOutDataType = typename BaseProblem::InOutDataType;
  using QKVDataType = typename BaseProblem::QKVDataType;
  using ODataType = typename BaseProblem::ODataType;
  using BiasDataType = typename BaseProblem::BiasDataType;
  using OGradDataType = typename BaseProblem::OGradDataType;
  using QGradDataType = typename BaseProblem::QGradDataType;
  using KGradDataType = typename BaseProblem::KGradDataType;
  using VGradDataType = typename BaseProblem::VGradDataType;
  using GemmAccDataType = typename BaseProblem::GemmAccDataType;
  using CompDataType = typename BaseProblem::CompDataType;
  using PDataType = typename BaseProblem::PDataType;
  using KGradAccDataType = typename BaseProblem::GemmAccDataType;
  using VGradAccDataType = typename BaseProblem::GemmAccDataType;

  static constexpr bool kIsCrossAttention = BaseProblem::kIsCrossAttention;
  static constexpr bool kUseGroup = BaseProblem::kUseGroup;
  static constexpr bool kIsJagged = BaseProblem::kIsJagged;
  static constexpr bool kHasBias = BaseProblem::kHasBias;
  static constexpr bool kHasCausal = BaseProblem::kHasCausal;
  static constexpr bool kUseSoftmax = BaseProblem::kUseSoftmax;
  static constexpr bool kHasDropout = BaseProblem::kHasDropout;

  using HstuAttentionTileSetting = remove_cvref_t<TileSetting_>;

  static constexpr bool IsWarpGemm32 = HstuAttentionTileSetting::IsWarpGemm32;

  static constexpr index_t kNumGemm0Gemm2Warps =
      TileSetting_::NumGemm0Gemm2Warps;
  static constexpr index_t kNumGemm1Warps = TileSetting_::NumGemm1Warps;
  static constexpr index_t kNumGemm3Warps = TileSetting_::NumGemm3Warps;
  static constexpr index_t kBlockSize =
      TileSetting_::NumWarps * get_warp_size();

  // Q tile: [kM0Sub, kQKHeaddim]
  CK_TILE_HOST_DEVICE static constexpr auto GetQDramTileAccessMaxVectorSize() {
    constexpr index_t kMPerBlock = HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kMPerBlock,
        kKPerBlock>();
  }

  // OGrad tile: [kM0Sub, kVHeaddim]
  CK_TILE_HOST_DEVICE static constexpr auto
  GetOGradDramTileAccessMaxVectorSize() {
    constexpr index_t kMPerBlock = HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kVHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kMPerBlock,
        kKPerBlock>();
  }
};

} // namespace ck_tile
