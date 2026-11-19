// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "hstu_attention_fwd_tile_setting_define.hpp"
#include "hstu_attention_fwd_type_config.hpp"

template <ck_tile::index_t MaxK>
static constexpr auto GetHstuAttentionFwdSplitKVCombineTileSetting() {
  if constexpr (MaxK == 64) {
    return ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<32, 4, 64>{};
  } else if constexpr (MaxK == 96) {
    return ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<16, 4, 96>{};
  } else if constexpr (MaxK == 128) {
    return ck_tile::
        HstuAttentionFwdSplitKVCombineTileSettingClass<16, 4, 128>{};
  } else if constexpr (MaxK == 256) {
    return ck_tile::HstuAttentionFwdSplitKVCombineTileSettingClass<8, 4, 256>{};
  } else {
    static_assert(false, "MaxK size not supported!");
  }
}
