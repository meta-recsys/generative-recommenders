// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

static CK_TILE_HOST_DEVICE constexpr index_t ceil_to_qualified_tile_length(
    index_t len) {
  if (len == 96)
    return 128;
  if (len == 160)
    return 256;
  if (len == 192)
    return 192;

  // only length of 96, 160, 192 and power-of-two is supported
  if (!(len & (len - 1)))
    return len;

  return 0;
};

template <
    typename BlockTile_, // sequence<...
    typename Gemm0BlockWarps_,
    typename Gemm0WarpTile_,
    typename Gemm1BlockWarps_,
    typename Gemm1WarpTile_>
struct HstuAttentionFwdTileSettingClass {
  using BlockTile = remove_cvref_t<BlockTile_>;
  using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
  using Gemm0WarpTile = remove_cvref_t<Gemm0WarpTile_>;
  using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
  using Gemm1WarpTile = remove_cvref_t<Gemm1WarpTile_>;

  static_assert(BlockTile::size() == 6, "Check failed!");
  static_assert(Gemm0BlockWarps::size() == 3, "Check failed!");
  static_assert(Gemm0WarpTile::size() == 3, "Check failed!");
  static_assert(Gemm1BlockWarps::size() == 3, "Check failed!");
  static_assert(Gemm1WarpTile::size() == 3, "Check failed!");

  static constexpr index_t NumGemm0Warps =
      reduce_on_sequence(Gemm0BlockWarps{}, multiplies<>{}, number<1>{});
  static constexpr index_t NumGemm1Warps =
      reduce_on_sequence(Gemm1BlockWarps{}, multiplies<>{}, number<1>{});
  static_assert(NumGemm1Warps % NumGemm0Warps == 0);

  static constexpr index_t NumWarps = max(NumGemm0Warps, NumGemm1Warps);

  static constexpr index_t kM0 =
      BlockTile::at(number<0>{}); // tile size along q seqlen
  static constexpr index_t kN0 =
      BlockTile::at(number<1>{}); // tile size along k seqlen
  static constexpr index_t kN0Sub =
      BlockTile::at(number<2>{}); // tile size for dividing kN0
  static constexpr index_t kN1 =
      BlockTile::at(number<3>{}); // tile size along v head_dim
  static constexpr index_t kK1 =
      BlockTile::at(number<4>{}); // tile size along kv gemm unroll
  static constexpr index_t kQKHeaddim = BlockTile::at(
      number<5>{}); // total length of K0, used for pipeline that need load Q at
                    // once (or repeately load Q as a whole tile)
  static_assert((kN0 % kN0Sub == 0) && (kN0 % kK1 == 0), "Check failed!");
  static_assert((kK1 % kN0Sub == 0) || (kN0Sub % kK1 == 0), "Check failed!");
};

template <
    index_t kM_, // tile size in seqlen_q dimension
    index_t NumWarps_, // assume all warps are assigned to seqlen_q dimension
    index_t kOHeaddim_>
struct HstuAttentionFwdSplitKVCombineTileSettingClass {
  static constexpr index_t kM = kM_;
  static constexpr index_t NumWarps = NumWarps_;

  static_assert(kM % NumWarps == 0, "Check failed!");

  static constexpr index_t kOHeaddim = kOHeaddim_;

  static_assert(
      (kM * kOHeaddim) % (NumWarps * get_warp_size()) == 0,
      "Check failed!");

  static constexpr index_t kSubOHeaddim =
      ceil_to_qualified_tile_length(kOHeaddim);
};

} // namespace ck_tile
