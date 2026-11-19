// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

struct HstuAttentionFwdSplitKVCombinePipelinePolicy {
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeOaccDramTileDistribution() {
    constexpr index_t kMPerBlock = Problem::kM;
    constexpr index_t kKPerBlock = Problem::kOHeaddim;
    constexpr index_t NumWarps = Problem::NumWarps;

    constexpr index_t KVector = GetAlignmentOacc<Problem>();
    constexpr index_t OtherK = kKPerBlock / KVector;

    if constexpr ((OtherK & (OtherK - 1)) == 0) { // for
                                                  // kKPerBlock=32,64,128,256 or
                                                  // kKPerBlock=96 && kKVector=6
      constexpr index_t KPerThread = KVector;

      if constexpr (OtherK < get_warp_size()) {
        // try to assign more consecutive threads on dim-K
        constexpr index_t KThreads = OtherK;

        constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t MPerThread = kMPerBlock / (MThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<
                    sequence<MPerThread, NumWarps, MThreadPerWarp>,
                    sequence<KThreads, KPerThread>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});
      } else {
        // all threads in the warp are assigned on dim-K
        constexpr index_t KThreads = get_warp_size();
        constexpr index_t KRepPerThread = OtherK / KThreads;

        constexpr index_t MPerThread = kMPerBlock / NumWarps;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<
                    sequence<MPerThread, NumWarps>,
                    sequence<KRepPerThread, KThreads, KPerThread>>,
                tuple<sequence<1>, sequence<2>>,
                tuple<sequence<1>, sequence<1>>,
                sequence<1, 2, 2>,
                sequence<0, 0, 2>>{});
      };
    } else { // for kKPerBlock=160 or kKPerBlock=96 && kKVector=1/2/4/8
      constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
      constexpr index_t KThreads = OtherK / KRepPerThread;

      constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t MPerThread = kMPerBlock / (MThreadPerWarp * NumWarps);

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<MPerThread, NumWarps, MThreadPerWarp>,
                  sequence<KRepPerThread, KThreads, KVector>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 1>>,
              sequence<1, 2, 2>,
              sequence<0, 0, 2>>{});
    };
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccDramTileDistribution() {
    constexpr index_t kMPerBlock = Problem::kM;
    constexpr index_t kKPerBlock = Problem::kMaxSplits;
    constexpr index_t NumWarps = Problem::NumWarps;

    constexpr index_t KVector = GetAlignmentLSEacc<Problem>();
    constexpr index_t OtherK = kKPerBlock / KVector;

    static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");

    static_assert(kMPerBlock <= NumWarps * get_warp_size(), "Check failed!");

    constexpr index_t MThreadPerWarp = kMPerBlock / NumWarps;
    constexpr index_t MPerThread = 1;
    constexpr index_t KThreadPerWarp = get_warp_size() / MThreadPerWarp;
    constexpr index_t KRepPerThread = OtherK / KThreadPerWarp;

    // 32/64 Threads should be in lay-out [kThreads, MThreadPerWarp] since the
    // tile distribution will be used by block_tile_reduce_sync(...,
    // bool_constant<0>{})
    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,
            tuple<
                sequence<MThreadPerWarp, NumWarps, MPerThread>,
                sequence<KRepPerThread, KThreadPerWarp, KVector>>,
            tuple<sequence<1>, sequence<2, 1>>,
            tuple<sequence<1>, sequence<1, 0>>,
            sequence<1, 2, 2>,
            sequence<2, 0, 2>>{});
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeLSEscaleLdsBlockDescriptor() {
    constexpr index_t kMPerBlock = Problem::kM;
    constexpr index_t kKPerBlock = Problem::kMaxSplits;
    constexpr index_t kKVector = GetAlignmentLSEacc<Problem>();

    constexpr auto lse_lds_block_desc = make_naive_tensor_descriptor(
        make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
        make_tuple(number<kKPerBlock>{}, number<1>{}),
        number<kKVector>{},
        number<1>{});

    return lse_lds_block_desc;
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc() {
    return Problem::GetOaccDramTileAccessMaxVectorSize();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentLSEacc() {
    return Problem::GetLSEaccDramTileAccessMaxVectorSize();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO() {
    // should be same as GetAlignmentOacc() since o_tile will use the same
    // encoding as o_acc_tile
    return GetAlignmentOacc<Problem>();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    if constexpr (Problem::kUseSoftmax) {
      return sizeof(typename Problem::LSEDataType) *
          MakeLSEscaleLdsBlockDescriptor<Problem>().get_element_space_size();
    } else {
      // smem_ptr[] should not be zero bytes
      return 128;
    }
  }
};

} // namespace ck_tile
