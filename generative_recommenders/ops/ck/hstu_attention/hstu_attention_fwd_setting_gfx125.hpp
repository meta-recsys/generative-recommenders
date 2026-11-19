// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_config.hpp"
#include "hstu_attention_fwd_tile_setting_define.hpp"

#if defined(BUILD_HSTU_FOR_GFX125)
using WarpTile_16x16x32 = ck_tile::sequence<16, 16, 32>;

using HstuAttentionFwdGemm0Warps = ck_tile::sequence<4, 1, 1>;
using HstuAttentionFwdGemm1Warps = ck_tile::sequence<4, 1, 1>;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (N0 % K1 == 0, MaxK % N1 == 0)
//
// Tdm block tiles. gfx1250 runs every forward instance on the Tdm pipelines
// (see get_hstu_fwd_pipeline_kind()), which need kN0 == kN0Sub == kK1 -- one
// n0/k1 sub-loop, with K and V double-buffered across kv-tiles instead of
// within one.
using HstuAttentionFwdBlockTile_Hdim64_M64_N0_32_Sub32_K1_32_Tdm =
    ck_tile::sequence<64, 32, 32, 64, 32, 64>;
using HstuAttentionFwdBlockTile_Hdim64_M128_N0_64_Sub64_K1_64_Tdm =
    ck_tile::sequence<128, 64, 64, 64, 64, 64>;
using HstuAttentionFwdBlockTile_Hdim96_M64_N0_32_Sub32_K1_32_Tdm =
    ck_tile::sequence<64, 32, 32, 96, 32, 96>;
using HstuAttentionFwdBlockTile_Hdim96_M128_N0_64_Sub64_K1_64_Tdm =
    ck_tile::sequence<128, 64, 64, 96, 64, 96>;
using HstuAttentionFwdBlockTile_Hdim128_M64_N0_32_Sub32_K1_32_Tdm =
    ck_tile::sequence<64, 32, 32, 128, 32, 128>;
using HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub64_K1_64_Tdm =
    ck_tile::sequence<128, 64, 64, 128, 64, 128>;
// The hdim128 MTile-128 slot: a 32-wide KV tile keeps a wave near 330 VGPRs,
// three workgroups per CU instead of the two the 64-wide tile (about 448 VGPRs)
// allows.
using HstuAttentionFwdBlockTile_Hdim128_M128_N0_32_Sub32_K1_32_Tdm =
    ck_tile::sequence<128, 32, 32, 128, 32, 128>;
// hdim256 uses this 64-row tile for both MTile slots: with 128 rows a wave
// needs more than 512 VGPRs (one workgroup per CU), with 64 rows about 336
// (three workgroups per CU).
using HstuAttentionFwdBlockTile_Hdim256_M64_N0_32_Sub32_K1_32_Tdm =
    ck_tile::sequence<64, 32, 32, 256, 32, 256>;

template <
    ck_tile::index_t MaxK,
    ck_tile::index_t ExpectedMTile,
    bool kUseSoftmax,
    bool kUseCausal,
    HstuFwdPipelineKind kPipelineKind = HstuFwdPipelineKind::TdmTrLoad>
static constexpr auto GetHstuAttentionFwdTileSetting() {
  static_assert(
      kPipelineKind == HstuFwdPipelineKind::TdmTrLoad,
      "gfx1250 has Tdm forward tiles only");

  if constexpr (MaxK == 64) {
    if constexpr (ExpectedMTile == 64) {
      return ck_tile::HstuAttentionFwdTileSettingClass<
          HstuAttentionFwdBlockTile_Hdim64_M64_N0_32_Sub32_K1_32_Tdm,
          HstuAttentionFwdGemm0Warps,
          WarpTile_16x16x32,
          HstuAttentionFwdGemm1Warps,
          WarpTile_16x16x32>{};
    } else {
      return ck_tile::HstuAttentionFwdTileSettingClass<
          HstuAttentionFwdBlockTile_Hdim64_M128_N0_64_Sub64_K1_64_Tdm,
          HstuAttentionFwdGemm0Warps,
          WarpTile_16x16x32,
          HstuAttentionFwdGemm1Warps,
          WarpTile_16x16x32>{};
    }
  } else if constexpr (MaxK == 96) {
    if constexpr (ExpectedMTile == 64) {
      return ck_tile::HstuAttentionFwdTileSettingClass<
          HstuAttentionFwdBlockTile_Hdim96_M64_N0_32_Sub32_K1_32_Tdm,
          HstuAttentionFwdGemm0Warps,
          WarpTile_16x16x32,
          HstuAttentionFwdGemm1Warps,
          WarpTile_16x16x32>{};
    } else {
      return ck_tile::HstuAttentionFwdTileSettingClass<
          HstuAttentionFwdBlockTile_Hdim96_M128_N0_64_Sub64_K1_64_Tdm,
          HstuAttentionFwdGemm0Warps,
          WarpTile_16x16x32,
          HstuAttentionFwdGemm1Warps,
          WarpTile_16x16x32>{};
    }
  } else if constexpr (MaxK == 128) {
    if constexpr (ExpectedMTile == 64) {
      return ck_tile::HstuAttentionFwdTileSettingClass<
          HstuAttentionFwdBlockTile_Hdim128_M64_N0_32_Sub32_K1_32_Tdm,
          HstuAttentionFwdGemm0Warps,
          WarpTile_16x16x32,
          HstuAttentionFwdGemm1Warps,
          WarpTile_16x16x32>{};
    } else {
      return ck_tile::HstuAttentionFwdTileSettingClass<
          HstuAttentionFwdBlockTile_Hdim128_M128_N0_32_Sub32_K1_32_Tdm,
          HstuAttentionFwdGemm0Warps,
          WarpTile_16x16x32,
          HstuAttentionFwdGemm1Warps,
          WarpTile_16x16x32>{};
    }
  } else if constexpr (MaxK == 256) {
    return ck_tile::HstuAttentionFwdTileSettingClass<
        HstuAttentionFwdBlockTile_Hdim256_M64_N0_32_Sub32_K1_32_Tdm,
        HstuAttentionFwdGemm0Warps,
        WarpTile_16x16x32,
        HstuAttentionFwdGemm1Warps,
        WarpTile_16x16x32>{};
  } else {
    static_assert(false, "MaxK size not supported!");
  }
}
#endif
