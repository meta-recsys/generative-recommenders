// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>

#include "block_gemm_areg_bsmem_creg_v2_hack_0.hpp"
#include "block_gemm_areg_bsmem_creg_v2_hack_1.hpp"
#include "block_gemm_areg_bsmem_trload_creg_v2_hack_1.hpp"
#include "block_gemm_tdm_group_prefetch.hpp"

#include "hstu_attention_config.hpp"
#include "hstu_attention_kernel_util.hpp"
#include "hstu_attention_pipeline_policy_helper.hpp"

namespace ck_tile {

// Pipeline policy for Kernel 1 backward (dQ computation).
//
// Tile shapes in Kernel 1:
//   Q, dO, O : [kM0, kQKHeaddim]   -- A operands for Gemm0/Gemm2,
//   register-resident K, V     : [kN0Sub, kQKHeaddim] -- B operands for
//   Gemm0/Gemm2, staged through LDS dQ       : [kM0, kQKHeaddim]   -- C output
//   of Gemm4
//
// Gemm0: S    [kM0, kN0Sub] = alpha * Q  [kM0, kQKHeaddim] @ K [kN0Sub,
// kQKHeaddim] Gemm2: dP   [kM0, kN0Sub] = dO [kM0, kQKHeaddim] @ V [kN0Sub,
// kQKHeaddim] Gemm4: dQ   [kM0, kQKHeaddim] += alpha * dS [kM0, kN0Sub] @ K^T
// [kQKHeaddim, kN0Sub]
//
// K and V occupy separate LDS regions (Gemm0 and Gemm2 run in separate
// static_for loops), laid out consecutively:
//   [k_lds region | v_lds region | kt_lds region]
// GetSmemSize returns GetSmemSizeK + GetSmemSizeV + GetSmemSizeKT.
struct HstuAttentionBwdKernel1PipelinePolicy {
  // Gemm0, Gemm2 all use n0_loop, which unrolls the Gemm along kN0
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetNumN0Loops() {
    constexpr index_t n0_loops = Problem::HstuAttentionTileSetting::kN0 /
        Problem::HstuAttentionTileSetting::kN0Sub;

    return n0_loops;
  }

  // Gemm4 uses k1_loop, which unrolls the Gemm along kM0, kK1 reuse kM0Sub at
  // present
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetNumK1Loops() {
    constexpr index_t k1_loops = Problem::HstuAttentionTileSetting::kN0 /
        Problem::HstuAttentionTileSetting::kK1;

    return k1_loops;
  }

  // -------------------------------------------------------------------------
  // K, V each need two buffers to support pre-write of the prefecthed data to
  // LDS
  // -------------------------------------------------------------------------
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetNumKVLdsBuffers() {
    constexpr index_t n0_loops = Problem::HstuAttentionTileSetting::kN0 /
        Problem::HstuAttentionTileSetting::kN0Sub;

    return min(n0_loops, 2);
  }

  // -------------------------------------------------------------------------
  // Alignment helpers (vector load widths)
  // -------------------------------------------------------------------------

  // Q alignment  -- based on [kM0, kQKHeaddim] tile
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentQ() {
    constexpr index_t MaxVectorSize =
        16 / sizeof(typename Problem::QKVDataType);
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
  }

  // K alignment  -- based on [kN0Sub, kQKHeaddim] tile
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK() {
    return Problem::GetKDramTileAccessMaxVectorSize();
  }

  // V alignment  -- same tile shape as K, so same alignment
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV() {
    return GetAlignmentK<Problem>();
  }

  // dO alignment -- same tile shape as Q, so same alignment
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentOGrad() {
    return GetAlignmentQ<Problem>();
  }

  // dQ alignment -- same tile shape as Q (output [kM0, kQKHeaddim])
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentQGrad() {
    return GetAlignmentQ<Problem>();
  }

  // -------------------------------------------------------------------------
  // DRAM tile distributions (govern how data is partitioned across threads)
  // -------------------------------------------------------------------------

  // Q (and dO, O) : [kM0, kQKHeaddim], register-resident A operand
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeQRegTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    return BlockGemm::template MakeABlockTileDistribution<
        Problem::HstuAttentionTileSetting::kM,
        Problem::HstuAttentionTileSetting::kQKHeaddim>();
  }

  // dO -- identical to Q
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeOGradRegTileDistribution() {
    return MakeQRegTileDistribution<Problem>();
  }

  // O (softmax path) -- identical to Q
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeORegTileDistribution() {
    return MakeQRegTileDistribution<Problem>();
  }

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // K : [kN0Sub, kQKHeaddim], B operand loaded from DRAM into registers then
  // stored to LDS
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;
    constexpr index_t kKVector = GetAlignmentK<Problem>();
    constexpr index_t OtherK = kKPerBlock / kKVector;

    if constexpr (detail::IsPerfectHeaddimSize(kKPerBlock)) {
      static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");
      constexpr index_t KPerThread = kKVector;
      constexpr index_t KThreads = OtherK;
      constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t NPerThread = kNPerBlock / (NThreadPerWarp * NumWarps);

      static_assert(NPerThread > 0, "Check failed!");

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<NPerThread, NumWarps, NThreadPerWarp>,
                  sequence<KThreads, KPerThread>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 0>>,
              sequence<1, 2>,
              sequence<0, 1>>{});
    } else {
      static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");
      constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
      constexpr index_t KThreads = OtherK / KRepPerThread;
      constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t NPerThread = kNPerBlock / (NThreadPerWarp * NumWarps);

      static_assert(NPerThread > 0, "Check failed!");

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<NPerThread, NumWarps, NThreadPerWarp>,
                  sequence<KRepPerThread, KThreads, kKVector>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 1>>,
              sequence<1, 2, 2>,
              sequence<0, 0, 2>>{});
    }
  }
#else
  // K : [kN0Sub, kQKHeaddim], B operand loaded from DRAM into LDS through TDM
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeKDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    constexpr index_t NumWarps = kBlockSize / get_warp_size();

    static_assert(
        kNPerBlock % NumWarps == 0,
        "kNPerBlock must be divisible by NumWarps for trivial tile-major distribution");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>, // R: empty
            tuple<
                sequence<NumWarps, kNPerBlock / NumWarps>, // X[0]: N-axis, warp
                                                           // split
                sequence<kKPerBlock>>, // X[1]: K-axis, single full vector per
                                       // thread
            tuple<sequence<1>>, // PsToRH (warp dim mapping)
            tuple<sequence<0>>, // PsToRH_lid
            sequence<1, 2>, // YsToD Major
            sequence<1, 0>>{}, // YsToD Minor
        bool_constant<true>{}); // IsWarpLevelParallelOnly
  }
#endif

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // V : [kN0Sub, kVHeaddim], B operand loaded from DRAM into registers then
  // stored to LDS
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
    constexpr index_t kKVector = GetAlignmentV<Problem>();
    constexpr index_t OtherK = kKPerBlock / kKVector;

    if constexpr (detail::IsPerfectHeaddimSize(kKPerBlock)) {
      static_assert((OtherK & (OtherK - 1)) == 0, "Check failed!");
      constexpr index_t KPerThread = kKVector;
      constexpr index_t KThreads = OtherK;
      constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t NPerThread = kNPerBlock / (NThreadPerWarp * NumWarps);

      static_assert(NPerThread > 0, "Check failed!");

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<NPerThread, NumWarps, NThreadPerWarp>,
                  sequence<KThreads, KPerThread>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 0>>,
              sequence<1, 2>,
              sequence<0, 1>>{});
    } else {
      static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");
      constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
      constexpr index_t KThreads = OtherK / KRepPerThread;
      constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t NPerThread = kNPerBlock / (NThreadPerWarp * NumWarps);

      static_assert(NPerThread > 0, "Check failed!");

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<NPerThread, NumWarps, NThreadPerWarp>,
                  sequence<KRepPerThread, KThreads, kKVector>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 1>>,
              sequence<1, 2, 2>,
              sequence<0, 0, 2>>{});
    }
  }
#else
  // V : [kN0Sub, kVHeaddim], B operand loaded from DRAM into LDS through TDM
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    constexpr index_t NumWarps = kBlockSize / get_warp_size();

    static_assert(
        kNPerBlock % NumWarps == 0,
        "kNPerBlock must be divisible by NumWarps for trivial tile-major distribution");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>, // R: empty
            tuple<
                sequence<NumWarps, kNPerBlock / NumWarps>, // X[0]: N-axis, warp
                                                           // split
                sequence<kKPerBlock>>, // X[1]: K-axis, single full vector per
                                       // thread
            tuple<sequence<1>>, // PsToRH (warp dim mapping)
            tuple<sequence<0>>, // PsToRH_lid
            sequence<1, 2>, // YsToD Major
            sequence<1, 0>>{}, // YsToD Minor
        bool_constant<true>{}); // IsWarpLevelParallelOnly
  }
#endif

  // Bias : [kM0, kN0] -- use Gemm0 C-tile distribution
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeBiasDramTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    constexpr auto bias_block_dstr_encode =
        BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM,
            Problem::HstuAttentionTileSetting::kN0>();
    return make_static_tile_distribution(bias_block_dstr_encode);
  }

  // LSE : [kM0] -- row-wise scalar derived from Gemm0/Gemm2 output row
  // distribution
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeLSETileDistribution() {
    // The LSE/delta tile covers the same rows as the Gemm0 C tile.
    // We derive a 1-D distribution by reducing Gemm0's C [kM0, kN0] along N.
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    using CombinedTile = decltype(BlockGemm::template MakeCBlockTile<
                                  Problem::HstuAttentionTileSetting::kM,
                                  Problem::HstuAttentionTileSetting::kN0>());
    const auto f_sum = [](auto a, auto b) { return a + b; };
    using reduced_tile_type =
        decltype(block_tile_reduce<typename Problem::CompDataType>(
            CombinedTile{},
            sequence<1>{},
            f_sum,
            typename Problem::CompDataType{0}));
    return reduced_tile_type::get_tile_distribution();
  }

  // ML (delta) tile -- same 1-D distribution as LSE
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeMLTileDistribution() {
    return MakeLSETileDistribution<Problem>();
  }

  // Delta LDS descriptor: flat 1-D buffer of length kM0 for LDS shuffle of
  // delta tile
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeDeltaLdsBlockDescriptor() {
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM;

    constexpr auto desc = make_naive_tensor_descriptor(
        make_tuple(number<kMPerBlock>{}),
        make_tuple(number<1>{}),
        number<1>{},
        number<1>{});
    return desc;
  }

  // -------------------------------------------------------------------------
  // LDS block descriptors
  // -------------------------------------------------------------------------

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQKWarpGemmBScalarPerVector() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    using BEncoding = typename WG::BWarpDstrEncoding;
    return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto
  GetOGradVWarpGemmBScalarPerVector() {
    using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    using BEncoding = typename WG::BWarpDstrEncoding;
    return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
  }

  template <typename Problem, bool kUseTrLoad = false>
  CK_TILE_HOST_DEVICE static constexpr auto
  GetSGradKTWarpGemmBScalarPerVector() {
    using BlockGemm =
        remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem, kUseTrLoad>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    using BEncoding = typename WG::BWarpDstrEncoding;

    if constexpr (kUseTrLoad) {
      using BEncodingForTrLoad = typename InputTileDistributionTraits<
          BEncoding,
          typename BlockGemm::BDataType>::TransposedDstrEncode;

      return BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];
    } else {
      return BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];
    }
  };

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetKDirectLdsSwizzleUnit() {
    // On gfx950, ScalarPerVector for trload read could be smaller than
    // ScalarPerVector for normal read, and AlignmentK could also be smaller
    // since the tile divided by whole work-group when loading from Dram. So
    // here we use the lcm of the three values as swizzle unit
    return lcm(
        GetAlignmentK<Problem>(),
        lcm(GetQKWarpGemmBScalarPerVector<Problem>(),
            GetSGradKTWarpGemmBScalarPerVector<
                Problem,
                true /*kUseTrLoad */>()));
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetVDirectLdsSwizzleUnit() {
    // AlignmentK could also be smaller than ScalarPerVector of normal read
    // since the tile divided by whole work-group when loading from Dram. So
    // here we use the lcm of the three values as swizzle unit
    return lcm(
        GetAlignmentV<Problem>(), GetOGradVWarpGemmBScalarPerVector<Problem>());
  }

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // K LDS descriptor: NumKVLdsBuffers * [kN0Sub, kQKHeaddim]
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false,
      bool kUseXor = false>
  CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor() {
    constexpr index_t NumBuffers =
        kUseTrLoad ? GetNumN0Loops<Problem>() : GetNumKVLdsBuffers<Problem>();
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;
    constexpr index_t kKVector = GetAlignmentK<Problem>();

    if constexpr (kUseDirectLds) {
      static_assert(kUseTrLoad, "Check failed!");

      if constexpr (kUseXor) // true, used for reading from Lds
      {
        constexpr index_t kSwizzleUnit = GetKDirectLdsSwizzleUnit<Problem>();

        constexpr auto desc_native = detail::MakeSimplySwizzledNativeDesc<
            Problem,
            NumBuffers,
            kNPerBlock,
            kKPerBlock,
            kSwizzleUnit>();

        // merge: NumBuffers * [kN0Sub, kQKHeaddim] -> [kN0, kQKHeaddim]
        return transform_tensor_descriptor(
            desc_native,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                make_pass_through_transform(number<kKPerBlock>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
      } else // false, used for direct-lds writing to Lds
      {
        return detail::MakeRowMajorLdsPlainBlockDescriptor<
            NumBuffers,
            kNPerBlock,
            kKPerBlock>();
      }
    } else if constexpr (!detail::IsPerfectHeaddimSize(kKPerBlock)) {
      constexpr index_t SingleBufferSize = kNPerBlock * kKPerBlock;

      constexpr auto desc_0 = make_naive_tensor_descriptor(
          make_tuple(
              number<NumBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
          make_tuple(
              number<SingleBufferSize>{}, number<kKPerBlock>{}, number<1>{}),
          number<kKVector>{},
          number<1>{});
      return transform_tensor_descriptor(
          desc_0,
          make_tuple(
              make_merge_transform(
                  make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
              make_pass_through_transform(number<kKPerBlock>{})),
          make_tuple(sequence<0, 1>{}, sequence<2>{}),
          make_tuple(sequence<0>{}, sequence<1>{}));
    } else if constexpr (
        GetQKWarpGemmBScalarPerVector<Problem>() == GetAlignmentK<Problem>()) {
      // In the trload pipeline this k_lds is read BOTH normally (Gemm0 A
      // operand) and transposed (Gemm4 via ds_read_b64_tr). Profiling shows the
      // transpose read is the dominant LDS-conflict source (it is 2x the
      // normal-read count in kernel 1) and it prefers a plain (contiguous)
      // layout. Use a plain physical layout for the shared trload buffer (same
      // element space -> GetSmemSize/byte offsets unchanged), and keep the XOR
      // swizzle for the non-trload buffer whose read is normal-only.
      if constexpr (kUseTrLoad) {
        // k_trload_lds reuse k_lds, but we take k_trload_lds as higher priority
        using BlockGemm =
            remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto PaddingCfg = detail::GetLdsPaddingConfigForTrLoadRead<
            WG,
            false /*inputB */,
            kKPerBlock>();

        constexpr auto PadInterval = PaddingCfg[number<0>{}];
        constexpr auto PadLength = PaddingCfg[number<1>{}];

        return detail::MakeRowMajorLdsPaddedBlockDescriptor<
            NumBuffers,
            kNPerBlock,
            kKPerBlock,
            PadInterval,
            PadLength>();
      } else {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto PaddingCfg = detail::GetLdsPaddingConfigForNormalRead<
            WG,
            false /*inputB */,
            kKPerBlock>();

        constexpr auto PadInterval = PaddingCfg[number<0>{}];
        constexpr auto PadLength = PaddingCfg[number<1>{}];

        return detail::MakeRowMajorLdsPaddedBlockDescriptor<
            NumBuffers,
            kNPerBlock,
            kKPerBlock,
            PadInterval,
            PadLength>();
      }
    } else {
      constexpr index_t kDsReadVector =
          GetQKWarpGemmBScalarPerVector<Problem>();

      constexpr index_t SingleBufferSize =
          kKPerBlock * kNPerBlock + kKPerBlock * kDsReadVector / kKVector;

      constexpr auto desc_0 = make_naive_tensor_descriptor(
          make_tuple(
              number<NumBuffers>{},
              number<kKPerBlock / kKVector>{},
              number<kKVector / kDsReadVector>{},
              number<kNPerBlock>{},
              number<kDsReadVector>{}),
          make_tuple(
              number<SingleBufferSize>{},
              number<kNPerBlock * kKVector + kDsReadVector>{},
              number<kNPerBlock * kDsReadVector>{},
              number<kDsReadVector>{},
              number<1>{}),
          number<kDsReadVector>{},
          number<1>{});

      return transform_tensor_descriptor(
          desc_0,
          make_tuple(
              make_merge_transform(
                  make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
              make_merge_transform(make_tuple(
                  number<kKPerBlock / kKVector>{},
                  number<kKVector / kDsReadVector>{},
                  number<kDsReadVector>{}))),
          make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
          make_tuple(sequence<0>{}, sequence<1>{}));
    }
  }
#else
  // K LDS descriptor: NumKVLdsBuffers * [kN0Sub, kQKHeaddim]
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor() {
    constexpr index_t NumBuffers = GetNumN0Loops<Problem>();
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    // K-Lds is used for both normal read and trload read, we take trload read
    // as higher priority when considering bank-conflicts
    using BlockGemm = remove_cvref_t<
        decltype(GetSGradKTBlockGemm<Problem, true /*kUseTrLoad */>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr auto TdmPaddingCfg = detail::GetTdmLdsPaddingConfigForTrLoadRead<
        WG,
        false /*inputB */,
        kKPerBlock>();

    constexpr auto PadIntervalBytes = TdmPaddingCfg[number<0>{}];
    constexpr auto PadLengthBytes = TdmPaddingCfg[number<1>{}];

    static_assert(
        detail::IsTdmPaddingValid<PadIntervalBytes, PadLengthBytes>(),
        "Check Failed!");

    return detail::MakeRowMajorLdsPaddedBlockDescriptor<
        typename WG::BDataType,
        NumBuffers,
        kNPerBlock,
        kKPerBlock,
        PadIntervalBytes,
        PadLengthBytes>();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetKLdsPaddingConfig() {
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    // K-Lds is used for both normal read and trload read, we take trload read
    // as higher priority when considering bank-conflicts
    using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr auto TdmPaddingCfg = detail::GetTdmLdsPaddingConfigForTrLoadRead<
        WG,
        false /* inputB */,
        kKPerBlock>();

    constexpr auto PadIntervalBytes = TdmPaddingCfg[number<0>{}];
    constexpr auto PadLengthBytes = TdmPaddingCfg[number<1>{}];

    static_assert(
        detail::IsTdmPaddingValid<PadIntervalBytes, PadLengthBytes>(),
        "Check Failed!");

    constexpr auto TdmRawPaddingCfg =
        detail::GetTdmRawPaddingConfig<PadIntervalBytes, PadLengthBytes>();

    return make_tuple(
        number<true>{},
        TdmRawPaddingCfg[number<1>{}],
        TdmRawPaddingCfg[number<0>{}]);
  }
#endif

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // V LDS descriptor: NumKVLdsBuffers * [kN0Sub, kVHeaddim]
  template <typename Problem, bool kUseDirectLds = false, bool kUseXor = false>
  CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor() {
    constexpr index_t NumBuffers = GetNumKVLdsBuffers<Problem>();
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
    constexpr index_t kKVector = GetAlignmentV<Problem>();

    if constexpr (kUseDirectLds) {
      if constexpr (kUseXor) // true, used for reading from Lds
      {
        constexpr index_t kSwizzleUnit = GetVDirectLdsSwizzleUnit<Problem>();

        constexpr auto desc_native = detail::MakeSimplySwizzledNativeDesc<
            Problem,
            NumBuffers,
            kNPerBlock,
            kKPerBlock,
            kSwizzleUnit>();

        // merge: NumBuffers * [kN0Sub, kVHeaddim] -> [kN0, kVHeaddim]
        return transform_tensor_descriptor(
            desc_native,
            make_tuple(
                make_merge_transform(
                    make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
                make_pass_through_transform(number<kKPerBlock>{})),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
      } else // false, used for direct-lds writing to Lds
      {
        return detail::MakeRowMajorLdsPlainBlockDescriptor<
            NumBuffers,
            kNPerBlock,
            kKPerBlock>();
      }
    } else if constexpr (!detail::IsPerfectHeaddimSize(kKPerBlock)) {
      constexpr index_t SingleBufferSize = kNPerBlock * kKPerBlock;

      constexpr auto desc_0 = make_naive_tensor_descriptor(
          make_tuple(
              number<NumBuffers>{}, number<kNPerBlock>{}, number<kKPerBlock>{}),
          make_tuple(
              number<SingleBufferSize>{}, number<kKPerBlock>{}, number<1>{}),
          number<kKVector>{},
          number<1>{});
      return transform_tensor_descriptor(
          desc_0,
          make_tuple(
              make_merge_transform(
                  make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
              make_pass_through_transform(number<kKPerBlock>{})),
          make_tuple(sequence<0, 1>{}, sequence<2>{}),
          make_tuple(sequence<0>{}, sequence<1>{}));
    } else if constexpr (
        GetOGradVWarpGemmBScalarPerVector<Problem>() >=
        GetAlignmentV<Problem>()) {
      constexpr index_t kSwizzleUnit =
          GetOGradVWarpGemmBScalarPerVector<Problem>();

      // XOR-swizzled physical layout [NumBuffers, kNPerBlock, kKPerBlock] --
      // shared with the transposed staging buffers (see
      // MakeSwizzledNativeDesc).
      constexpr auto desc_native = detail::MakeSwizzledNativeDesc<
          Problem,
          NumBuffers,
          kNPerBlock,
          kKPerBlock,
          kSwizzleUnit>();

      // Logical view: [NumBuffers * kNPerBlock, kKPerBlock] -- buffers stacked
      // along dim0, matching the other branches and the per-buffer caller
      // slicing.
      return transform_tensor_descriptor(
          desc_native,
          make_tuple(
              make_merge_transform(
                  make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
              make_pass_through_transform(number<kKPerBlock>{})),
          make_tuple(sequence<0, 1>{}, sequence<2>{}),
          make_tuple(sequence<0>{}, sequence<1>{}));
    } else {
      constexpr index_t kDsReadVector =
          GetOGradVWarpGemmBScalarPerVector<Problem>();

      constexpr index_t SingleBufferSize =
          kKPerBlock * kNPerBlock + kKPerBlock * kDsReadVector / kKVector;

      constexpr auto desc_0 = make_naive_tensor_descriptor(
          make_tuple(
              number<NumBuffers>{},
              number<kKPerBlock / kKVector>{},
              number<kKVector / kDsReadVector>{},
              number<kNPerBlock>{},
              number<kDsReadVector>{}),
          make_tuple(
              number<SingleBufferSize>{},
              number<kNPerBlock * kKVector + kDsReadVector>{},
              number<kNPerBlock * kDsReadVector>{},
              number<kDsReadVector>{},
              number<1>{}),
          number<kDsReadVector>{},
          number<1>{});

      return transform_tensor_descriptor(
          desc_0,
          make_tuple(
              make_merge_transform(
                  make_tuple(number<NumBuffers>{}, number<kNPerBlock>{})),
              make_merge_transform(make_tuple(
                  number<kKPerBlock / kKVector>{},
                  number<kKVector / kDsReadVector>{},
                  number<kDsReadVector>{}))),
          make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
          make_tuple(sequence<0>{}, sequence<1>{}));
    }
  }
#else
  // V LDS descriptor: NumKVLdsBuffers * [kN0Sub, kVHeaddim]
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor() {
    constexpr index_t NumBuffers = GetNumKVLdsBuffers<Problem>();
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kN0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr auto TdmPaddingCfg = detail::GetTdmLdsPaddingConfigForNormalRead<
        WG,
        false /*inputB */,
        kKPerBlock>();

    constexpr auto PadIntervalBytes = TdmPaddingCfg[number<0>{}];
    constexpr auto PadLengthBytes = TdmPaddingCfg[number<1>{}];

    static_assert(
        detail::IsTdmPaddingValid<PadIntervalBytes, PadLengthBytes>(),
        "Check Failed!");

    return detail::MakeRowMajorLdsPaddedBlockDescriptor<
        typename WG::BDataType,
        NumBuffers,
        kNPerBlock,
        kKPerBlock,
        PadIntervalBytes,
        PadLengthBytes>();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetVLdsPaddingConfig() {
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr auto TdmPaddingCfg = detail::GetTdmLdsPaddingConfigForNormalRead<
        WG,
        false /* inputB */,
        kKPerBlock>();

    constexpr auto PadIntervalBytes = TdmPaddingCfg[number<0>{}];
    constexpr auto PadLengthBytes = TdmPaddingCfg[number<1>{}];

    static_assert(
        detail::IsTdmPaddingValid<PadIntervalBytes, PadLengthBytes>(),
        "Check Failed!");

    constexpr auto TdmRawPaddingCfg =
        detail::GetTdmRawPaddingConfig<PadIntervalBytes, PadLengthBytes>();

    return make_tuple(
        number<true>{},
        TdmRawPaddingCfg[number<1>{}],
        TdmRawPaddingCfg[number<0>{}]);
  }
#endif

#if !HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeKTLdsReadBlockDescriptor() {
    using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
    constexpr index_t kNPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

    constexpr auto kSwizzleUnit =
        detail::GetSwizzleUnitForNormalRead<WG, false /*inputB*/>();

    constexpr auto desc_native = detail::MakeSwizzledNativeDesc<
        Problem,
        NumBuffers,
        kNPerBlock,
        kKPerBlock,
        kSwizzleUnit>();

    // merge: NumK1Loops * [kQKHeaddim, kK1] -> [kQKHeaddim, kN0]
    return transform_tensor_descriptor(
        desc_native,
        make_tuple(
            make_pass_through_transform(number<kNPerBlock>{}),
            make_merge_transform(
                make_tuple(number<NumBuffers>{}, number<kKPerBlock>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeKTLdsWriteBlockDescriptor() {
    using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
    constexpr index_t kNPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

    constexpr auto kSwizzleUnit =
        detail::GetSwizzleUnitForNormalRead<WG, false /*inputB*/>();

    constexpr auto desc_native = detail::MakeSwizzledNativeDesc<
        Problem,
        NumBuffers,
        kNPerBlock,
        kKPerBlock,
        kSwizzleUnit>();

    // the same native tensor desc as the ReadBlockDescriptor, but transposed
    // tensor view
    return transform_tensor_descriptor(
        desc_native,
        make_tuple(
            make_merge_transform(
                make_tuple(number<NumBuffers>{}, number<kKPerBlock>{})),
            make_pass_through_transform(number<kNPerBlock>{})),
        make_tuple(sequence<0, 2>{}, sequence<1>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
  }
#endif

  // -------------------------------------------------------------------------
  // Shared memory sizing
  // K and V use separate LDS regions (Gemm0 and Gemm2 run in separate loops).
  // -------------------------------------------------------------------------
#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK() {
    // for kUseDirectLds = true,  kUseXor = true/false does not affect the k-lds
    // space size
    return MakeKLdsBlockDescriptor<Problem, kUseTrLoad, kUseDirectLds>()
               .get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

  template <typename Problem, bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV() {
    if constexpr (kUseDirectLds) {
      // for kUseDirectLds = true,  kUseXor = true/false does not affect the
      // k-lds space size
      return MakeVLdsBlockDescriptor<Problem, true /*kUseDirectLds*/>()
                 .get_element_space_size() *
          sizeof(typename Problem::QKVDataType);
    } else {
      return MakeVLdsBlockDescriptor<Problem, false /*kUseDirectLds*/>()
                 .get_element_space_size() *
          sizeof(typename Problem::QKVDataType);
    }
  }
#else
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK() {
    return MakeKLdsBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV() {
    return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }
#endif

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeKT() {
    return MakeKTLdsReadBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeDropout() {
    if constexpr (Problem::kHasDropout) {
      using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
      constexpr auto config =
          BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
      using WG = remove_cvref_t<decltype(config.template at<0>())>;
      constexpr bool IsWG32 = (WG::kM == 32);
      constexpr index_t MWarps = config.template at<1>();
      using BlockGemmShape = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
      constexpr index_t kMPerBlock = BlockGemmShape::kM;
      constexpr index_t MIterPerWarp =
          (!IsWG32 && kMPerBlock > MWarps * WG::kM) ? 2 : 1;
      constexpr index_t kMPerStep = MIterPerWarp * MWarps * WG::kM;
      // assume the all warps are assigned on dim-M
      constexpr index_t kNPerStep = WG::kN;

      return (kMPerStep + 1) * kNPerStep * sizeof(uint8_t);
    } else {
      return 0;
    }
  };

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // Total smem: k_lds + v_lds + kt_lds
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() {
    if constexpr (!kUseTrLoad) {
      // K region + V region, laid out consecutively.
      // The delta LDS shuffle reuses the K region (delta is consumed before the
      // main loop).
      return GetSmemSizeK<Problem>() + GetSmemSizeV<Problem>() +
          GetSmemSizeKT<Problem>() + GetSmemSizeDropout<Problem>();
    } else {
      if constexpr (kUseDirectLds) {
        // k_lds is ping-pong double-buffered (two regions); v_lds is a single
        // circular buffer (NumKVLdsBuffers slots inside one region), so V is
        // counted once.
        return GetSmemSizeK<Problem, kUseTrLoad, true /*kUseDirectLds*/>() * 2 +
            GetSmemSizeV<Problem, true /*kUseDirectLds*/>() +
            GetSmemSizeDropout<Problem>();
      } else {
        return GetSmemSizeK<Problem, kUseTrLoad>() + GetSmemSizeV<Problem>() +
            GetSmemSizeDropout<Problem>();
      }
    }
  }
#else
  // Total smem: k_lds * 2 + v_lds
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() {
    return GetSmemSizeK<Problem>() * 2 + GetSmemSizeV<Problem>() * 2 +
        GetSmemSizeDropout<Problem>();
  }
#endif

  // -------------------------------------------------------------------------
  // Block GEMM objects
  // -------------------------------------------------------------------------

  // Gemm0: S = alpha * Q @ K   [kM0, kN0Sub] = [kM0, kQKHeaddim] x [kN0Sub,
  // kQKHeaddim]
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm0Gemm2Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kM,
                Problem::HstuAttentionTileSetting::kN0Sub,
                Problem::HstuAttentionTileSetting::kQKHeaddim>,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile>>;

    auto warp_gemm = [&]() {
      if constexpr (
          (std::is_same_v<typename Problem::QKVDataType, half_t> ||
           std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
          std::is_same_v<typename Problem::GemmAccDataType, float>) {
        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<2>{});

#if defined(__hstu_gfx125__)
        static_assert(
            WarpGemmM == 16 && WarpGemmK == 32,
            "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
        static_assert(
            (WarpGemmM == 16 && WarpGemmK == 32) ||
                (WarpGemmM == 32 && WarpGemmK == 16),
            "Not supported WarpGemm sizes!");
#else
        static_assert(
            (WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
            "Not supported WarpGemm sizes!");
#endif
        return WarpGemmDispatcher<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<0>{}),
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<1>{}),
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<2>{}),
            true,
            false,
            false,
            WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
        decltype(warp_gemm)>;

#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
    constexpr index_t NIterPerWarp =
        Problem::HstuAttentionTileSetting::kN0Sub / decltype(warp_gemm)::kN;
    constexpr index_t KIterPerWarp =
        Problem::HstuAttentionTileSetting::kQKHeaddim / decltype(warp_gemm)::kK;
    constexpr index_t kTdmGemmPrefetchDepth = NIterPerWarp * KIterPerWarp / 2;
    constexpr index_t kTdmGemmStepsPerGroup = 1;

    return BlockGemmARegBSmemCRegTdmGroupPrefetch_0<
        GemmProblem,
        BlockGemmPolicy,
        kTdmGemmPrefetchDepth,
        kTdmGemmStepsPerGroup>{};
#else
    return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
#endif
  }

  // Same as GetQKBlockGemm but with kN0 (instead of kN0Sub) as the N tile
  // dimension. This is used as the BlockGemm template argument to
  // BlockDropout::Run() so that kNPerBlock = kN0, ensuring dropout is applied
  // to the full pcomp_tile [kM0, kN0] rather than only the first kN0Sub
  // columns.
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQKCombinedBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm0Gemm2Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kM,
                Problem::HstuAttentionTileSetting::kN0,
                Problem::HstuAttentionTileSetting::kQKHeaddim>,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile>>;

    auto warp_gemm = [&]() {
      if constexpr (
          (std::is_same_v<typename Problem::QKVDataType, half_t> ||
           std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
          std::is_same_v<typename Problem::GemmAccDataType, float>) {
        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<2>{});

#if defined(__hstu_gfx125__)
        static_assert(
            WarpGemmM == 16 && WarpGemmK == 32,
            "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
        static_assert(
            (WarpGemmM == 16 && WarpGemmK == 32) ||
                (WarpGemmM == 32 && WarpGemmK == 16),
            "Not supported WarpGemm sizes!");
#else
        static_assert(
            (WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
            "Not supported WarpGemm sizes!");
#endif
        return WarpGemmDispatcher<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<0>{}),
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<1>{}),
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<2>{}),
            true,
            false,
            false,
            WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
        decltype(warp_gemm)>;

    return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
  }

  // Gemm2: dP = dO @ V   [kM0, kN0Sub] = [kM0, kVHeaddim] x [kN0Sub, kVHeaddim]
  // Uses kVHeaddim as the reduction dimension (V head dim, not QK head dim).
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetOGradVBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm0Gemm2Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kM,
                Problem::HstuAttentionTileSetting::kN0Sub,
                Problem::HstuAttentionTileSetting::kVHeaddim>,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
            typename Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile>>;

    auto warp_gemm = [&]() {
      if constexpr (
          (std::is_same_v<typename Problem::QKVDataType, half_t> ||
           std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
          std::is_same_v<typename Problem::GemmAccDataType, float>) {
        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<2>{});

#if defined(__hstu_gfx125__)
        static_assert(
            WarpGemmM == 16 && WarpGemmK == 32,
            "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
        static_assert(
            (WarpGemmM == 16 && WarpGemmK == 32) ||
                (WarpGemmM == 32 && WarpGemmK == 16),
            "Not supported WarpGemm sizes!");
#else
        static_assert(
            (WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
            "Not supported WarpGemm sizes!");
#endif
        return WarpGemmDispatcher<
            typename Problem::QKVDataType,
            typename Problem::QKVDataType,
            typename Problem::GemmAccDataType,
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<0>{}),
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<1>{}),
            Problem::HstuAttentionTileSetting::Gemm0Gemm2WarpTile::at(
                number<2>{}),
            true,
            false,
            false,
            WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
        decltype(warp_gemm)>;

#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
    constexpr index_t NIterPerWarp =
        Problem::HstuAttentionTileSetting::kN0Sub / decltype(warp_gemm)::kN;
    constexpr index_t KIterPerWarp =
        Problem::HstuAttentionTileSetting::kQKHeaddim / decltype(warp_gemm)::kK;
    constexpr index_t kTdmGemmPrefetchDepth = NIterPerWarp * KIterPerWarp / 2;
    constexpr index_t kTdmGemmStepsPerGroup = 1;

    return BlockGemmARegBSmemCRegTdmGroupPrefetch_0<
        GemmProblem,
        BlockGemmPolicy,
        kTdmGemmPrefetchDepth,
        kTdmGemmStepsPerGroup>{};
#else
    return BlockGemmARegBSmemCRegV2Hack_0<GemmProblem, BlockGemmPolicy>{};
#endif
  }

  // Gemm4 single-rep N (used by the epilogue to stride over dQ output)
  template <typename Problem>
  CK_TILE_DEVICE static constexpr index_t GetSGradKTBlockGemmSingleRepN() {
    return Problem::HstuAttentionTileSetting::Gemm4BlockWarps::at(number<1>{}) *
        Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<1>{});
  }

  // Gemm4: dQ += alpha * dS @ K^T   [kM0, kQKHeaddim] = [kM0, kN0Sub] x
  // [kQKHeaddim, kN0Sub] A = dS cast to QKVDataType, B = K^T (QKVDataType LDS),
  // C = dQ accumulator
  template <typename Problem, bool kUseTrLoad = false>
  CK_TILE_HOST_DEVICE static constexpr auto GetSGradKTBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm4Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kM,
                Problem::HstuAttentionTileSetting::kQKHeaddim,
                Problem::HstuAttentionTileSetting::kK1>,
            typename Problem::HstuAttentionTileSetting::Gemm4BlockWarps,
            typename Problem::HstuAttentionTileSetting::Gemm4WarpTile>>;

    auto warp_gemm = [&]() {
      if constexpr (
          (std::is_same_v<typename Problem::QKVDataType, half_t> ||
           std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
          std::is_same_v<typename Problem::GemmAccDataType, float>) {
        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<2>{});

#if defined(__hstu_gfx125__)
        static_assert(
            WarpGemmM == 16 && WarpGemmK == 32,
            "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
        static_assert(
            (WarpGemmM == 16 && WarpGemmK == 32) ||
                (WarpGemmM == 32 && WarpGemmK == 16),
            "Not supported WarpGemm sizes!");
#else
        static_assert(
            (WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
                (WarpGemmM == 32 && (WarpGemmK == 8 || WarpGemmK == 16)),
            "Not supported WarpGemm sizes!");
#endif

        if constexpr (
            (WarpGemmM == 16 && WarpGemmK == 32) ||
            (WarpGemmM == 32 && WarpGemmK == 16))
          return WarpGemmDispatcher<
              typename Problem::QKVDataType,
              typename Problem::QKVDataType,
              typename Problem::GemmAccDataType,
              Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<0>{}),
              Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<1>{}),
              Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<2>{}),
              true,
              false,
              false,
#if defined(__hstu_gfx95__)
              WGAttrNumAccessEnum::Double
#else
              WGAttrNumAccessEnum::Default
#endif
              >{};
        else
          return WarpGemmDispatcher<
              typename Problem::QKVDataType,
              typename Problem::QKVDataType,
              typename Problem::GemmAccDataType,
              Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<0>{}),
              Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<1>{}),
              Problem::HstuAttentionTileSetting::Gemm4WarpTile::at(number<2>{}),
              true,
              false,
              false,
              WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm4BlockWarps,
        decltype(warp_gemm)>;

    if constexpr (!kUseTrLoad) {
      return BlockGemmARegBSmemCRegV2Hack_1<GemmProblem, BlockGemmPolicy>{};
    } else {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
      constexpr index_t NIterPerWarp =
          Problem::HstuAttentionTileSetting::kQKHeaddim /
          decltype(warp_gemm)::kN;
      constexpr index_t KIterPerWarp =
          Problem::HstuAttentionTileSetting::kK1 / decltype(warp_gemm)::kK;
      constexpr index_t kTdmGemmPrefetchDepth = NIterPerWarp * KIterPerWarp / 2;
      constexpr index_t kTdmGemmStepsPerGroup = 1;

      return BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetch_1<
          GemmProblem,
          BlockGemmPolicy,
          kTdmGemmPrefetchDepth,
          kTdmGemmStepsPerGroup>{};
#else
      return BlockGemmARegBSmemTrLoadCRegV2Hack_1<
          GemmProblem,
          BlockGemmPolicy>{};
#endif
    }
  }
};

} // namespace ck_tile
