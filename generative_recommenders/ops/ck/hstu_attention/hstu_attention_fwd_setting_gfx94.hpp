// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_config.hpp"
#include "hstu_attention_fwd_tile_setting_define.hpp"

#if defined(BUILD_HSTU_FOR_GFX94)
using WarpTile_16x16x16 = ck_tile::sequence<16, 16, 16>;

using HstuAttentionFwdGemm0Warps = ck_tile::sequence<4, 1, 1>;
using HstuAttentionFwdGemm1Warps = ck_tile::sequence<4, 1, 1>;

// Tile-sizes: M N0 N0Sub N1 K1 MaxK (MaxK % N1 == 0, N0 % K1 == 0)
//
using HstuAttentionFwdBlockTile_Hdim64_M64_N0_64_Sub32_K1_32 =
    ck_tile::sequence<64, 64, 32, 64, 32, 64>;

using HstuAttentionFwdBlockTile_Hdim64_M128_N0_32_Sub16_K1_32 =
    ck_tile::sequence<128, 32, 16, 64, 32, 64>;

using HstuAttentionFwdBlockTile_Hdim96_M128_N0_64_Sub32_K1_32 =
    ck_tile::sequence<128, 64, 32, 96, 32, 96>;

// used by silu/softmax + causal mask/no causal mask situation
using HstuAttentionFwdBlockTile_Hdim128_M64_N0_64_Sub16_K1_32 =
    ck_tile::sequence<64, 64, 16, 128, 32, 128>;

// used by silu + causal mask situation
using HstuAttentionFwdBlockTile_Hdim128_M128_N0_32_Sub16_K1_32_silu =
    ck_tile::sequence<128, 32, 16, 128, 32, 128>;
// used by silu + no causal mask situation
using HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub16_K1_32_silu =
    ck_tile::sequence<128, 64, 16, 128, 32, 128>;

// used by softmax + causal mask situation
using HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub16_K1_16_softmax =
    ck_tile::sequence<128, 64, 16, 128, 16, 128>;
// used by softmax + no causal mask situation
using HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub32_K1_16_softmax =
    ck_tile::sequence<128, 64, 32, 128, 16, 128>;

using HstuAttentionFwdBlockTile_Hdim256_M128_N0_32_Sub16_K1_16 =
    ck_tile::sequence<128, 32, 16, 256, 16, 256>;

template <
    ck_tile::index_t MaxK,
    ck_tile::index_t ExpectedMTile,
    bool kUseSoftmax,
    bool kUseCausal,
    HstuFwdPipelineKind kPipelineKind = HstuFwdPipelineKind::Default>
static constexpr auto GetHstuAttentionFwdTileSetting() {
  if constexpr (MaxK == 64) {
    if constexpr (ExpectedMTile == 64) {
      if constexpr (kUseSoftmax) {
        return ck_tile::HstuAttentionFwdTileSettingClass<
            HstuAttentionFwdBlockTile_Hdim64_M64_N0_64_Sub32_K1_32,
            HstuAttentionFwdGemm0Warps,
            WarpTile_16x16x16,
            HstuAttentionFwdGemm1Warps,
            WarpTile_16x16x16>{};
      } else {
        return ck_tile::HstuAttentionFwdTileSettingClass<
            HstuAttentionFwdBlockTile_Hdim64_M64_N0_64_Sub32_K1_32,
            HstuAttentionFwdGemm0Warps,
            WarpTile_16x16x16,
            HstuAttentionFwdGemm1Warps,
            WarpTile_16x16x16>{};
      }
    } else {
      if constexpr (kUseSoftmax) {
        return ck_tile::HstuAttentionFwdTileSettingClass<
            HstuAttentionFwdBlockTile_Hdim64_M128_N0_32_Sub16_K1_32,
            HstuAttentionFwdGemm0Warps,
            WarpTile_16x16x16,
            HstuAttentionFwdGemm1Warps,
            WarpTile_16x16x16>{};
      } else {
        return ck_tile::HstuAttentionFwdTileSettingClass<
            HstuAttentionFwdBlockTile_Hdim64_M128_N0_32_Sub16_K1_32,
            HstuAttentionFwdGemm0Warps,
            WarpTile_16x16x16,
            HstuAttentionFwdGemm1Warps,
            WarpTile_16x16x16>{};
      }
    }
  } else if constexpr (MaxK == 96) {
    return ck_tile::HstuAttentionFwdTileSettingClass<
        HstuAttentionFwdBlockTile_Hdim96_M128_N0_64_Sub32_K1_32,
        HstuAttentionFwdGemm0Warps,
        WarpTile_16x16x16,
        HstuAttentionFwdGemm1Warps,
        WarpTile_16x16x16>{};
  } else if constexpr (MaxK == 128) {
    if constexpr (ExpectedMTile == 64) {
      if constexpr (kUseCausal) {
        return ck_tile::HstuAttentionFwdTileSettingClass<
            HstuAttentionFwdBlockTile_Hdim128_M64_N0_64_Sub16_K1_32,
            HstuAttentionFwdGemm0Warps,
            WarpTile_16x16x16,
            HstuAttentionFwdGemm1Warps,
            WarpTile_16x16x16>{};
      } else {
        return ck_tile::HstuAttentionFwdTileSettingClass<
            HstuAttentionFwdBlockTile_Hdim128_M64_N0_64_Sub16_K1_32,
            HstuAttentionFwdGemm0Warps,
            WarpTile_16x16x16,
            HstuAttentionFwdGemm1Warps,
            WarpTile_16x16x16>{};
      }
    } else {
      if constexpr (kUseSoftmax) {
        if constexpr (kUseCausal) {
          return ck_tile::HstuAttentionFwdTileSettingClass<
              HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub16_K1_16_softmax,
              HstuAttentionFwdGemm0Warps,
              WarpTile_16x16x16,
              HstuAttentionFwdGemm1Warps,
              WarpTile_16x16x16>{};
        } else {
          return ck_tile::HstuAttentionFwdTileSettingClass<
              HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub32_K1_16_softmax,
              HstuAttentionFwdGemm0Warps,
              WarpTile_16x16x16,
              HstuAttentionFwdGemm1Warps,
              WarpTile_16x16x16>{};
        }
      } else {
        if constexpr (kUseCausal) {
          return ck_tile::HstuAttentionFwdTileSettingClass<
              HstuAttentionFwdBlockTile_Hdim128_M128_N0_32_Sub16_K1_32_silu,
              HstuAttentionFwdGemm0Warps,
              WarpTile_16x16x16,
              HstuAttentionFwdGemm1Warps,
              WarpTile_16x16x16>{};
        } else {
          return ck_tile::HstuAttentionFwdTileSettingClass<
              HstuAttentionFwdBlockTile_Hdim128_M128_N0_64_Sub16_K1_32_silu,
              HstuAttentionFwdGemm0Warps,
              WarpTile_16x16x16,
              HstuAttentionFwdGemm1Warps,
              WarpTile_16x16x16>{};
        }
      }
    }
  } else if constexpr (MaxK == 256) {
    return ck_tile::HstuAttentionFwdTileSettingClass<
        HstuAttentionFwdBlockTile_Hdim256_M128_N0_32_Sub16_K1_16,
        HstuAttentionFwdGemm0Warps,
        WarpTile_16x16x16,
        HstuAttentionFwdGemm1Warps,
        WarpTile_16x16x16>{};
  } else {
    static_assert(false, "MaxK size not supported!");
  }
}
#endif
