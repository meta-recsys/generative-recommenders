// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

#include <ck_tile/core.hpp>

#include "hstu_attention_fwd_tile_setting_define.hpp"
#include "hstu_attention_problem_common_util.hpp"

namespace ck_tile {

namespace detail {

template <typename T>
struct is_hstu_attention_fwd_tile_setting : std::false_type {};

template <
    typename BlockTile_,
    typename Gemm0BlockWarps_,
    typename Gemm0WarpTile_,
    typename Gemm1BlockWarps_,
    typename Gemm1WarpTile_>
struct is_hstu_attention_fwd_tile_setting<HstuAttentionFwdTileSettingClass<
    BlockTile_,
    Gemm0BlockWarps_,
    Gemm0WarpTile_,
    Gemm1BlockWarps_,
    Gemm1WarpTile_>> : std::true_type {};

template <typename T>
struct is_hstu_attention_fwd_splitkv_combine_tile_setting : std::false_type {};

template <index_t kM_, index_t NumWarps_, index_t kOHeaddim_>
struct is_hstu_attention_fwd_splitkv_combine_tile_setting<
    HstuAttentionFwdSplitKVCombineTileSettingClass<kM_, NumWarps_, kOHeaddim_>>
    : std::true_type {};

}; // namespace detail

// PipelineProblem encodes information not only from the original user-problem,
// but it also contains other information needed by the pipeline, which includes
// TileShape -- which determines how block-layer calculation is done in tiles
// and
//              how warps are allocated on dimensions
template <
    typename InOutDataType_,
    typename GemmAccDataType_,
    typename CompDataType_, // data type for SiLU and other non-linear
                            // calculation
    typename BiasDataType_,
    bool kIsCrossAttention_,
    bool kUseGroup_,
    bool kIsJagged_,
    bool kHasBias_,
    bool kHasDropout_,
    bool kHasCausal_,
    bool kUseSoftmax_,
    bool kStoreLSE_,
    typename TileSetting_>
struct HstuAttentionFwdPipelineProblem {
  using InOutDataType = remove_cvref_t<InOutDataType_>;
  using QKVDataType = InOutDataType;
  using ODataType = InOutDataType;
  using GemmAccDataType = remove_cvref_t<GemmAccDataType_>;

  // DataType used when siLU calculation
  using CompDataType = remove_cvref_t<CompDataType_>;
  using BiasDataType = remove_cvref_t<BiasDataType_>;

  using OaccDataType = GemmAccDataType;
  using PDataType = QKVDataType;

  static constexpr bool kIsCrossAttention = kIsCrossAttention_;
  static constexpr bool kUseGroup = kUseGroup_;
  static constexpr bool kIsJagged = kIsJagged_;
  static constexpr bool kHasBias = kHasBias_;
  static constexpr bool kHasDropout = kHasDropout_;
  static constexpr bool kHasCausal = kHasCausal_;
  static constexpr bool kUseSoftmax = kUseSoftmax_;
  static constexpr bool kStoreLSE = kStoreLSE_;

  static_assert(
      detail::is_hstu_attention_fwd_tile_setting<
          remove_cvref_t<TileSetting_>>::value,
      "TileSetting_ must be an instance of HstuAttentionFwdTileSettingClass!");
  static_assert(
      !kUseGroup || (kUseGroup && kIsJagged),
      "Group HSTU is only used with jagged mode!");
  static_assert(
      !kStoreLSE || (kStoreLSE && kUseSoftmax),
      "Storing Lse is only necessary when softmax is used!");

  using HstuAttentionTileSetting = remove_cvref_t<TileSetting_>;

  static constexpr index_t kNumGemm0Warps = TileSetting_::NumGemm0Warps;
  static constexpr index_t kNumGemm1Warps = TileSetting_::NumGemm1Warps;
  static constexpr index_t kBlockSize =
      TileSetting_::NumWarps * get_warp_size();

  CK_TILE_HOST_DEVICE static constexpr auto GetQDramTileAccessMaxVectorSize() {
    constexpr index_t kMPerBlock = HstuAttentionTileSetting::kM0;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kMPerBlock,
        kKPerBlock,
        false,
        true>();
  }

  template <bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr auto GetKDramTileAccessMaxVectorSize(
      bool_constant<kUseDirectLds> = {}) {
    constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kQKHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kNPerBlock,
        kKPerBlock,
        kUseDirectLds,
        true>();
  }

  template <bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr auto GetVDramTileAccessMaxVectorSize(
      bool_constant<kUseDirectLds> = {}) {
    constexpr index_t kNPerBlock = HstuAttentionTileSetting::kN1;
    constexpr index_t kKPerBlock = HstuAttentionTileSetting::kK1;

    return detail::GetDramTileAccessMaxVectorSize<
        QKVDataType,
        kBlockSize,
        kKPerBlock, // HigerDim
        kNPerBlock, // LowerDim
        kUseDirectLds,
        true>();
  };
};

template <
    typename OaccDataType_,
    typename LSEDataType_,
    typename ODataType_,
    bool kIsJagged_,
    bool kUseSoftmax_,
    bool kStoreLSE_,
    typename TileSetting_,
    index_t kMaxSplits_ = 0>
struct HstuAttentionFwdSplitKVCombinePipelineProblem {
  using OaccDataType = remove_cvref_t<OaccDataType_>;
  using ODataType = remove_cvref_t<ODataType_>;
  using LSEDataType = remove_cvref_t<LSEDataType_>;

  static_assert(
      detail::is_hstu_attention_fwd_splitkv_combine_tile_setting<
          remove_cvref_t<TileSetting_>>::value,
      "TileSetting_ must be an instance of HstuAttentionFwdSplitKVCombineTileSettingClass!");

  static constexpr bool kIsJagged = kIsJagged_;
  static constexpr bool kUseSoftmax = kUseSoftmax_;
  static constexpr bool kStoreLSE = kStoreLSE_;

  static constexpr index_t kM = TileSetting_::kM;
  static constexpr index_t NumWarps = TileSetting_::NumWarps;
  static constexpr index_t kOHeaddim = TileSetting_::kOHeaddim;
  static constexpr index_t kSubOHeaddim = TileSetting_::kSubOHeaddim;
  static constexpr index_t kBlockSize =
      TileSetting_::NumWarps * get_warp_size();
  static constexpr index_t kMaxSplits = kMaxSplits_;

  static_assert(
      (kMaxSplits == 0) || (kM * kMaxSplits >= kBlockSize),
      "Check failed!");

  CK_TILE_HOST_DEVICE static constexpr auto
  GetOaccDramTileAccessMaxVectorSize() {
    constexpr index_t kMPerBlock = kM;
    constexpr index_t kKPerBlock = kOHeaddim;

    constexpr auto VectorSize = detail::GetDramTileAccessMaxVectorSize<
        OaccDataType,
        kBlockSize,
        kMPerBlock,
        kKPerBlock,
        false,
        true>();

    if constexpr (std::is_same_v<OaccDataType, float>) {
      // since o_acc_tile[] is element-wisely converted to o_tile[] in the
      // Default2DEpilogue , they have the same distrbitution, so we should
      // restrict the VectorSize since fp16/bf16 could not use VectorSize 3
      static_assert(
          VectorSize != 3,
          "VectorSize 3 shall not be used for accessing o_acc_tile!");
    }

    return VectorSize;
  };

  CK_TILE_HOST_DEVICE static constexpr auto GetODramTileAccessMaxVectorSize() {
    constexpr index_t kMPerBlock = kM;
    constexpr index_t kKPerBlock = kOHeaddim;

    return detail::GetDramTileAccessMaxVectorSize<
        ODataType,
        kBlockSize,
        kMPerBlock,
        kKPerBlock,
        false,
        true>();
  };

  CK_TILE_HOST_DEVICE static constexpr auto
  GetLSEaccDramTileAccessMaxVectorSize() {
    constexpr index_t kMPerBlock = kM;
    constexpr index_t kKPerBlock = kMaxSplits;

    return detail::GetDramTileAccessMaxVectorSize<
        LSEDataType,
        kBlockSize,
        kMPerBlock,
        kKPerBlock,
        false,
        true>();
  };
};

} // namespace ck_tile
