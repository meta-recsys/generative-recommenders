// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>

#include "block_gemm_areg_bsmem_creg_slice_gemmk_full_gemmn.hpp"
#include "block_gemm_areg_bsmem_trload_creg_slice_gemmk_full_gemmn.hpp"
#include "block_gemm_asmem_breg_creg_full_gemmk_slice_gemmm.hpp"
#include "block_gemm_tdm_group_prefetch.hpp"
#include "block_gemm_tdm_group_prefetch_for_bwd.hpp"

#include "hstu_attention_config.hpp"
#include "hstu_attention_kernel_util.hpp"
#include "hstu_attention_pipeline_policy_helper.hpp"

namespace ck_tile {

struct HstuAttentionBwdKernel2PipelinePolicy {
  // Gemm0, Gemm2 use m0_loop, which unrolls the Gemm along kM0
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetNumM0Loops() {
    constexpr index_t m0_loops = Problem::HstuAttentionTileSetting::kM0 /
        Problem::HstuAttentionTileSetting::kM0Sub;

    return m0_loops;
  }

  // Gemm1/Gemm3 all use k1_loop, which unrolls the Gemm along kM0, kK1 reuse
  // kM0Sub at present
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetNumK1Loops() {
    constexpr index_t k1_loops = Problem::HstuAttentionTileSetting::kM0 /
        Problem::HstuAttentionTileSetting::kK1;

    return k1_loops;
  }

  // Number of Lds slots for q_lds, do_lds
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetNumQOGradLdsBuffers() {
    constexpr index_t m0_loops = Problem::HstuAttentionTileSetting::kM0 /
        Problem::HstuAttentionTileSetting::kM0Sub;

    return min(m0_loops, 2);
  }

  // -------------------------------------------------------------------------
  // Alignment helpers (vector load widths)
  // -------------------------------------------------------------------------

  // K alignment -- based on [kN0, kQKHeaddim] tile
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentK() {
    constexpr index_t MaxVectorSize =
        16 / sizeof(typename Problem::QKVDataType);
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;
    return min(MaxVectorSize, WG::kK / WG::WarpGemmAttribute::Impl::kABKLane);
  }

  // Q alignment -- based on [kM0Sub, kQKHeaddim] tile
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ() {
    return Problem::GetQDramTileAccessMaxVectorSize();
  }

  // V alignment -- same tile shape as K
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentV() {
    return GetAlignmentK<Problem>();
  }

  // dO alignment -- based ib [kM0Sub, kVHeaddim] tile
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOGrad() {
    return Problem::GetOGradDramTileAccessMaxVectorSize();
  }

  // dK alignment -- same tile shape as K (output [kN0, kQKHeaddim])
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentKGrad() {
    return GetAlignmentK<Problem>();
  }

  // dV alignment -- same tile shape as V (output [kN0, kVHeaddim])
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto GetAlignmentVGrad() {
    return GetAlignmentV<Problem>();
  }

  // -------------------------------------------------------------------------
  // DRAM tile distributions
  // -------------------------------------------------------------------------

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // Q DRAM distribution -- [kM0Sub, kQKHeaddim], loaded sub-tile by sub-tile
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeQDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    constexpr index_t kKVector = GetAlignmentQ<Problem>();
    constexpr index_t OtherK = kKPerBlock / kKVector;

    if constexpr (detail::IsPerfectHeaddimSize(kKPerBlock)) {
      constexpr index_t KPerThread = kKVector;
      constexpr index_t KThreads = OtherK;

      constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t MPerThread = kMPerBlock / (MThreadPerWarp * NumWarps);

      static_assert(MPerThread > 0, "Check failed!");

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
      static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");
      constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
      constexpr index_t KThreads = OtherK / KRepPerThread;
      constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t MPerThread = kMPerBlock / (MThreadPerWarp * NumWarps);

      static_assert(MPerThread > 0, "Check failed!");

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<MPerThread, NumWarps, MThreadPerWarp>,
                  sequence<KRepPerThread, KThreads, kKVector>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 1>>,
              sequence<1, 2, 2>,
              sequence<0, 0, 2>>{});
    }
  }
#else
  // Q DRAM distribution -- [kM0Sub, kQKHeaddim], used by load_tile_tdm()
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeQDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    constexpr index_t NumWarps = kBlockSize / get_warp_size();

    static_assert(
        kMPerBlock % NumWarps == 0,
        "kMPerBlock must be divisible by NumWarps for trivial tile-major distribution");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>, // R: empty
            tuple<
                sequence<NumWarps, kMPerBlock / NumWarps>, // X[0]: N-axis, warp
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
  // dO DRAM distribution -- [kM0Sub, kVHeaddim], loaded sub-tile by sub-tile
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeOGradDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    constexpr index_t kKVector = GetAlignmentOGrad<Problem>();
    constexpr index_t OtherK = kKPerBlock / kKVector;

    if constexpr (detail::IsPerfectHeaddimSize(kKPerBlock)) {
      constexpr index_t KPerThread = kKVector;
      constexpr index_t KThreads = OtherK;

      constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t MPerThread = kMPerBlock / (MThreadPerWarp * NumWarps);

      static_assert(MPerThread > 0, "Check failed!");

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
      static_assert((OtherK & (OtherK - 1)) != 0, "Check failed!");
      constexpr index_t KRepPerThread = (OtherK % 3 == 0) ? 3 : 5;
      constexpr index_t KThreads = OtherK / KRepPerThread;
      constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
      constexpr index_t NumWarps = kBlockSize / get_warp_size();
      constexpr index_t MPerThread = kMPerBlock / (MThreadPerWarp * NumWarps);

      static_assert(MPerThread > 0, "Check failed!");

      return make_static_tile_distribution(
          tile_distribution_encoding<
              sequence<1>,
              tuple<
                  sequence<MPerThread, NumWarps, MThreadPerWarp>,
                  sequence<KRepPerThread, KThreads, kKVector>>,
              tuple<sequence<1>, sequence<1, 2>>,
              tuple<sequence<1>, sequence<2, 1>>,
              sequence<1, 2, 2>,
              sequence<0, 0, 2>>{});
    }
  }
#else
  // dO DRAM distribution -- [kM0Sub, kVHeaddim], used by load_tile_tdm()
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeOGradDramTileDistribution() {
    constexpr index_t kBlockSize = Problem::kBlockSize;
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    constexpr index_t NumWarps = kBlockSize / get_warp_size();

    static_assert(
        kMPerBlock % NumWarps == 0,
        "kMPerBlock must be divisible by NumWarps for trivial tile-major distribution");

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<>, // R: empty
            tuple<
                sequence<NumWarps, kMPerBlock / NumWarps>, // X[0]: N-axis, warp
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

  // K : [kN0, kQKHeaddim], register-resident B operand
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeKRegTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    return BlockGemm::template MakeBBlockTileDistribution<
        Problem::HstuAttentionTileSetting::kN,
        Problem::HstuAttentionTileSetting::kQKHeaddim>();
  }

  // V : [kN0, kVHeaddim], register-resident B operand
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeVRegTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
    return BlockGemm::template MakeBBlockTileDistribution<
        Problem::HstuAttentionTileSetting::kN,
        Problem::HstuAttentionTileSetting::kVHeaddim>();
  }

  // PT register tile distribution
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakePTRegTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
    return BlockGemm::template MakeABlockTileDistribution<
        Problem::HstuAttentionTileSetting::kN,
        Problem::HstuAttentionTileSetting::kM0>();
  }

  // SGradT register tile distribution
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeSGradTRegTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
    return BlockGemm::template MakeABlockTileDistribution<
        Problem::HstuAttentionTileSetting::kN,
        Problem::HstuAttentionTileSetting::kM0>();
  }

  // Bias -- [kM0Sub, kN0], use the C-tile distribution of Gemm0
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeBiasDramTileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    constexpr auto bias_block_dstr_encode =
        BlockGemm::template MakeCBlockDistributionEncode<
            Problem::HstuAttentionTileSetting::kM0,
            Problem::HstuAttentionTileSetting::kN>();
    return make_static_tile_distribution(bias_block_dstr_encode);
  }

  // LSE -- [kM0], 1-D row scalar derived by reducing Gemm0 C-tile along N
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeLSETileDistribution() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    auto sacc_tile = BlockGemm::template MakeCBlockTile<
        Problem::HstuAttentionTileSetting::kM0,
        Problem::HstuAttentionTileSetting::kN>();
    const auto f_sum = [](auto a, auto b) { return a + b; };
    using reduced_tile_type =
        decltype(block_tile_reduce<typename Problem::CompDataType>(
            sacc_tile,
            sequence<1>{},
            f_sum,
            typename Problem::CompDataType{0}));
    return reduced_tile_type::get_tile_distribution();
  }

  // Delta (D[sq]) -- same 1-D distribution as LSE
  template <typename Problem>
  CK_TILE_DEVICE static constexpr auto MakeDeltaTileDistribution() {
    return MakeLSETileDistribution<Problem>();
  }

  // -------------------------------------------------------------------------
  // LDS smem sizing helpers
  // -------------------------------------------------------------------------

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ() {
    // for kUseDirectLds = true,  kUseXor = true/false does not affect the q_lds
    // space size
    return MakeQLdsBlockDescriptor<Problem, kUseTrLoad, kUseDirectLds>()
               .get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad() {
    // for kUseDirectLds = true,  kUseXor = true/false does not affect the
    // do_lds space size
    return MakeOGradLdsBlockDescriptor<Problem, kUseTrLoad, kUseDirectLds>()
               .get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }
#else
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ() {
    return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad() {
    return MakeOGradLdsBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }
#endif

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQT() {
    return MakeQTLdsReadBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGradT() {
    return MakeOGradTLdsReadBlockDescriptor<Problem>()
               .get_element_space_size() *
        sizeof(typename Problem::QKVDataType);
  }

#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // Total smem: q_lds + do_lds + qt_lds + dot_lds
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() {
    if constexpr (!kUseTrLoad) {
      return GetSmemSizeQ<Problem>() + GetSmemSizeOGrad<Problem>() +
          GetSmemSizeQT<Problem>() + GetSmemSizeOGradT<Problem>();
    } else {
      if constexpr (kUseDirectLds) {
        return GetSmemSizeQ<Problem, kUseTrLoad, true /*kUseDirectLds*/>() * 2 +
            GetSmemSizeOGrad<Problem, kUseTrLoad, true /*kUseDirectLds*/>() * 2;
      } else {
        return GetSmemSizeQ<Problem, kUseTrLoad>() +
            GetSmemSizeOGrad<Problem, kUseTrLoad>();
      }
    }
  }
#else
  // Total smem: q_lds * 2 + do_lds * 2
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() {
    return GetSmemSizeQ<Problem>() * 2 + GetSmemSizeOGrad<Problem>() * 2;
  }
#endif

  // -------------------------------------------------------------------------
  // WarpGemm K-per-thread helpers (used for LDS bank-conflict-free layouts)
  // -------------------------------------------------------------------------

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQKWarpGemmAScalarPerVector() {
    using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    using AEncoding = typename WG::AWarpDstrEncoding;
    return AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];
  }

  template <typename Problem, bool kUseTrLoad = false>
  CK_TILE_HOST_DEVICE static constexpr auto
  GetSGradTQTWarpGemmBScalarPerVector() {
    using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
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
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQDirectLdsSwizzleUnit() {
    // On gfx950, ScalarPerVector for trload read could be smaller than
    // ScalarPerVector for normal read, and AlignmentK could also be smaller
    // since the tile divided by whole work-group when loading from Dram. So
    // here we use the lcm of the three values as swizzle unit
    return lcm(
        GetAlignmentQ<Problem>(),
        lcm(GetQKWarpGemmAScalarPerVector<Problem>(),
            GetSGradTQTWarpGemmBScalarPerVector<
                Problem,
                true /*kUseTrLoad */>()));
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto
  GetOGradVWarpGemmAScalarPerVector() {
    using BlockGemm = remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    using AEncoding = typename WG::AWarpDstrEncoding;
    return AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];
  }

  template <typename Problem, bool kUseTrLoad = false>
  CK_TILE_HOST_DEVICE static constexpr auto
  GetPTOGradTWarpGemmBScalarPerVector() {
    using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
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
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetOGradDirectLdsSwizzleUnit() {
    // On gfx950, ScalarPerVector for trload read could be smaller than
    // ScalarPerVector for normal read, and AlignmentK could also be smaller
    // since the tile divided by whole work-group when loading from Dram. So
    // here we use the lcm of the three values as swizzle unit
    return lcm(
        GetAlignmentOGrad<Problem>(),
        lcm(GetOGradVWarpGemmAScalarPerVector<Problem>(),
            GetPTOGradTWarpGemmBScalarPerVector<
                Problem,
                true /*kUseTrLoad */>()));
  }

  // -------------------------------------------------------------------------
  // LDS block descriptors
  // -------------------------------------------------------------------------
#if !HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
  // q_lds write/read descriptor
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false,
      bool kUseXor = false>
  CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor() {
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;
    constexpr index_t kKVector = GetAlignmentQ<Problem>();

    constexpr index_t NumBuffers = kUseTrLoad
        ? GetNumM0Loops<Problem>()
        : GetNumQOGradLdsBuffers<Problem>();

    if constexpr (kUseDirectLds) {
      static_assert(kUseTrLoad, "Check failed!");

      if constexpr (kUseXor) // true, used for reading from Lds
      {
        constexpr index_t kSwizzleUnit = GetQDirectLdsSwizzleUnit<Problem>();

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
    } else if constexpr (GetQKWarpGemmAScalarPerVector<Problem>() >= kKVector) {
      if constexpr (kUseTrLoad) {
        // q_trload_lds reuse q_lds, but we take q_trload_lds as higher priority
        using BlockGemm =
            remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
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
            true /*inputA */,
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
      constexpr auto desc_native = detail::MakeSwizzledNativeDesc<
          Problem,
          NumBuffers,
          kNPerBlock,
          kKPerBlock,
          kKVector>();

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
    }
  }
#else
  // q_lds write/read descriptor
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor() {
    constexpr index_t NumBuffers = GetNumM0Loops<Problem>();
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    // Q-Lds is used for both normal read and trload read, we take trload read
    // as higher priority when considering bank-conflicts
    using BlockGemm = remove_cvref_t<
        decltype(GetSGradTQTBlockGemm<Problem, true /*kUseTrLoad */>())>;
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
        kMPerBlock,
        kKPerBlock,
        PadIntervalBytes,
        PadLengthBytes>();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQLdsPaddingConfig() {
    constexpr index_t kKPerBlock =
        Problem::HstuAttentionTileSetting::kQKHeaddim;

    // Q-Lds is used for both normal read and trload read, we take trload read
    // as higher priority when considering bank-conflicts
    using BlockGemm = remove_cvref_t<
        decltype(GetSGradTQTBlockGemm<Problem, true /*kUseTrLoad */>())>;
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
  // do_lds write/read descriptor
  template <
      typename Problem,
      bool kUseTrLoad = false,
      bool kUseDirectLds = false,
      bool kUseXor = false>
  CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor() {
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
    constexpr index_t kKVector = GetAlignmentOGrad<Problem>();

    constexpr index_t NumBuffers = kUseTrLoad
        ? GetNumM0Loops<Problem>()
        : GetNumQOGradLdsBuffers<Problem>();

    if constexpr (kUseDirectLds) {
      static_assert(kUseTrLoad, "Check failed!");

      if constexpr (kUseXor) // true, used for reading from Lds
      {
        constexpr index_t kSwizzleUnit =
            GetOGradDirectLdsSwizzleUnit<Problem>();

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
        GetOGradVWarpGemmAScalarPerVector<Problem>() >= kKVector) {
      if constexpr (kUseTrLoad) {
        // q_trload_lds reuse q_lds, but we take q_trload_lds as higher priority
        using BlockGemm =
            remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
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
        using BlockGemm =
            remove_cvref_t<decltype(GetOGradVBlockGemm<Problem>())>;
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr auto PaddingCfg = detail::GetLdsPaddingConfigForNormalRead<
            WG,
            true /*inputA */,
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
      constexpr auto desc_native = detail::MakeSwizzledNativeDesc<
          Problem,
          NumBuffers,
          kNPerBlock,
          kKPerBlock,
          kKVector>();

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
    }
  }
#else
  // do_lds write/read descriptor
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsBlockDescriptor() {
    constexpr index_t NumBuffers = GetNumM0Loops<Problem>();
    constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kM0Sub;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    // OGrad-Lds is used for both normal read and trload read, we take trload
    // read as higher priority when considering bank-conflicts
    using BlockGemm = remove_cvref_t<
        decltype(GetPTOGradTBlockGemm<Problem, true /*kUseTrLoad */>())>;
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
        kMPerBlock,
        kKPerBlock,
        PadIntervalBytes,
        PadLengthBytes>();
  }

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetOGradLdsPaddingConfig() {
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;

    // OGrad-Lds is used for both normal read and trload read, we take trload
    // read as higher
    using BlockGemm = remove_cvref_t<
        decltype(GetPTOGradTBlockGemm<Problem, true /*kUseTrLoad */>())>;
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

#if !HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeQTLdsWriteBlockDescriptor() {
    using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
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

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto
  MakeOGradTLdsWriteBlockDescriptor() {
    using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;

    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
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

  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto MakeQTLdsReadBlockDescriptor() {
    using BlockGemm = remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
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

    // merge: NumK1Loops * [kQKHeaddim, kK1] -> [kQKHeaddim, kM0]
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
  CK_TILE_HOST_DEVICE static constexpr auto MakeOGradTLdsReadBlockDescriptor() {
    using BlockGemm = remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;

    constexpr auto config =
        BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t NumBuffers = GetNumK1Loops<Problem>();
    constexpr index_t kNPerBlock = Problem::HstuAttentionTileSetting::kVHeaddim;
    constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kK1;

    constexpr auto kSwizzleUnit =
        detail::GetSwizzleUnitForNormalRead<WG, false /*inputB*/>();

    constexpr auto desc_native = detail::MakeSwizzledNativeDesc<
        Problem,
        NumBuffers,
        kNPerBlock,
        kKPerBlock,
        kSwizzleUnit>();

    // merge: NumK1Loops * [kQKHeaddim, kK1] -> [kQKHeaddim, kM0]
    return transform_tensor_descriptor(
        desc_native,
        make_tuple(
            make_pass_through_transform(number<kNPerBlock>{}),
            make_merge_transform(
                make_tuple(number<NumBuffers>{}, number<kKPerBlock>{}))),
        make_tuple(sequence<1>{}, sequence<0, 2>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
  }
#endif

  // -------------------------------------------------------------------------
  // Block GEMM objects
  // -------------------------------------------------------------------------

  // Gemm0: S = Q_lds @ K_reg    [kM0Sub, kN0] = [kM0Sub, kQKHeaddim] x [kN0,
  // kQKHeaddim] Gemm2: dP = dO_lds @ V_reg  [kM0Sub, kN0] = [kM0Sub, kVHeaddim]
  // x [kN0, kVHeaddim] A = Q/dO from LDS (A-smem), B = K/V register-resident
  // (B-reg)
  // -> BlockGemmASmemBRegCRegV1
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm0Gemm2Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kM0Sub,
                Problem::HstuAttentionTileSetting::kN,
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
            false, // not CTransposed
            false,
            false,
            WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
        decltype(warp_gemm)>;

#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
    constexpr index_t MIterPerWarp =
        Problem::HstuAttentionTileSetting::kM0Sub / decltype(warp_gemm)::kM;
    constexpr index_t KIterPerWarp =
        Problem::HstuAttentionTileSetting::kQKHeaddim / decltype(warp_gemm)::kK;
    constexpr index_t kTdmGemmPrefetchDepth = MIterPerWarp * KIterPerWarp / 2;
    constexpr index_t kTdmGemmStepsPerGroup = MIterPerWarp;

    return BlockGemmASmemBRegCRegTdmGroupPrefetchFullGemmKSliceGemmMWithPrologueHook<
        GemmProblem,
        BlockGemmPolicy,
        kTdmGemmPrefetchDepth,
        kTdmGemmStepsPerGroup>{};
#else
    return BlockGemmASmemBRegCRegFullGemmKSliceGemmM<
        GemmProblem,
        BlockGemmPolicy>{};
#endif
  }

  // Same as GetQKBlockGemm but with kM0 (instead of kM0Sub) as the M tile
  // dimension. This is used as the BlockGemm template argument to
  // BlockDropout::Run() so that kMPerBlock = kM0, ensuring dropout is applied
  // to the full pcomp_tile [kM0, kN0] rather than only the first kM0Sub rows.
  template <typename Problem>
  CK_TILE_HOST_DEVICE static constexpr auto GetQKCombinedBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm0Gemm2Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kM0,
                Problem::HstuAttentionTileSetting::kN,
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
            false, // not CTransposed
            false,
            false,
            WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
        decltype(warp_gemm)>;

    return BlockGemmASmemBRegCRegFullGemmKSliceGemmM<
        GemmProblem,
        BlockGemmPolicy>{};
  }

  // Gemm2: dP = dO @ V   [kM0Sub, kN0] = [kM0Sub, kVHeaddim] x [kN0, kVHeaddim]
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
                Problem::HstuAttentionTileSetting::kM0Sub,
                Problem::HstuAttentionTileSetting::kN,
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
            false, // not CTransposed
            false,
            false,
            WGAttrNumAccessEnum::Default>{};
      } else {
        static_assert(false, "Not supported data types!");
      }
    }();

    using BlockGemmPolicy = BlockGemmASmemBRegCRegV1CustomPolicy<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        typename Problem::HstuAttentionTileSetting::Gemm0Gemm2BlockWarps,
        decltype(warp_gemm)>;

#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
    constexpr index_t MIterPerWarp =
        Problem::HstuAttentionTileSetting::kM0Sub / decltype(warp_gemm)::kM;
    constexpr index_t KIterPerWarp =
        Problem::HstuAttentionTileSetting::kQKHeaddim / decltype(warp_gemm)::kK;
    constexpr index_t kTdmGemmPrefetchDepth = MIterPerWarp * KIterPerWarp / 2;
    constexpr index_t kTdmGemmStepsPerGroup = MIterPerWarp;

    return BlockGemmASmemBRegCRegTdmGroupPrefetchFullGemmKSliceGemmMWithPrologueHook<
        GemmProblem,
        BlockGemmPolicy,
        kTdmGemmPrefetchDepth,
        kTdmGemmStepsPerGroup>{};
#else
    return BlockGemmASmemBRegCRegFullGemmKSliceGemmM<
        GemmProblem,
        BlockGemmPolicy>{};
#endif
  }

  // is_target_warptile_16_32 == true selects the 16x16x32 (native mfma,
  // WGAttrNumAccessEnum:: Double) A operand for Gemm1. Its per-lane A register
  // count is twice that of the incoming 16x16 C fragment, so each K=32 A tile
  // is assembled from two consecutive 16x16 C sub-tiles.
  // is_target_warptile_16_32 == false keeps the original 16x16x16 (1:1)
  // transpose-free reuse.
  template <
      typename Problem,
      bool is_target_warptile_16_32,
      typename PTOutTensor,
      typename PInTensor>
  CK_TILE_DEVICE static constexpr void PTFromGemm0CToGemm1A(
      PTOutTensor& pt_out,
      const PInTensor& p_in) {
    if constexpr (
        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}) ==
        16) {
      using BlockGemm =
          remove_cvref_t<decltype(GetPTOGradTBlockGemm<Problem>())>;
      constexpr auto config =
          BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
      using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;

      constexpr index_t MWarp =
          Problem::HstuAttentionTileSetting::Gemm1BlockWarps::at(number<0>{});

      constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kN;
      constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kM0;

      constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
      constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

      using AWarpDstr = typename WarpGemm::AWarpDstr;
      using CWarpDstr = typename WarpGemm::CWarpDstr;
      auto p_warp_tensor =
          make_static_distributed_tensor<typename Problem::QKVDataType>(
              CWarpDstr{});
      auto pt_warp_tensor =
          make_static_distributed_tensor<typename Problem::QKVDataType>(
              AWarpDstr{});

      constexpr auto a_warp_y_lengths =
          to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
      constexpr auto c_warp_y_lengths =
          to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

      constexpr auto a_warp_y_index_zeros =
          uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
      constexpr auto c_warp_y_index_zeros =
          uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

      if constexpr (is_target_warptile_16_32) {
        // Number of 16x16 C sub-tiles packed along K into one 16x16x32 A tile
        // (== 2), and the per-lane register count of a single 16x16 fragment
        // (== C fragment size).
        constexpr index_t NumKSub = WarpGemm::kK / 16;
        constexpr index_t kSubPerThread =
            CWarpDstr{}.get_ys_to_d_descriptor().get_element_space_size();

        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
          constexpr auto kIter = number<km[number<0>{}]>{};
          constexpr auto mIter = number<km[number<1>{}]>{};

          static_for<0, NumKSub, 1>{}([&](auto kSub) {
            p_warp_tensor.get_thread_buffer() = p_in.get_y_sliced_thread_data(
                merge_sequences(
                    sequence<kIter * NumKSub + kSub, mIter>{},
                    c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            // C->A transpose is the register identity for one 16x16 fragment;
            // place it into sub-access kSub of the K=32 A tile (Double packs
            // the accesses as [sub0(kSubPerThread), sub1(kSubPerThread)] in the
            // thread buffer).
            static_for<0, kSubPerThread, 1>{}([&](auto i) {
              pt_warp_tensor.get_thread_buffer()(
                  number<kSub * kSubPerThread + i>{}) =
                  p_warp_tensor.get_thread_buffer()(number<i>{});
            });
          });

          pt_out.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
              pt_warp_tensor.get_thread_buffer());
        });
      } else {
        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
          constexpr auto kIter = number<km[number<0>{}]>{};
          constexpr auto mIter = number<km[number<1>{}]>{};
          p_warp_tensor.get_thread_buffer() = p_in.get_y_sliced_thread_data(
              merge_sequences(sequence<kIter, mIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

          pt_warp_tensor.get_thread_buffer() =
              p_warp_tensor.get_thread_buffer();

          pt_out.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
              pt_warp_tensor.get_thread_buffer());
        });
      }
    } else {
      pt_out.get_thread_buffer() = p_in.get_thread_buffer();
    }
  }

  // is_target_warptile_16_32 has the same meaning as in PTFromGemm0CToGemm1A,
  // but for Gemm3.
  template <
      typename Problem,
      bool is_target_warptile_16_32,
      typename SGradTOutTensor,
      typename SGradInTensor>
  CK_TILE_DEVICE static constexpr void SGradTFromGemm2CToGemm3A(
      SGradTOutTensor& dst_out,
      const SGradInTensor& ds_in) {
    if constexpr (
        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{}) ==
        16) {
      using BlockGemm =
          remove_cvref_t<decltype(GetSGradTQTBlockGemm<Problem>())>;
      constexpr auto config =
          BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
      using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;

      constexpr index_t MWarp =
          Problem::HstuAttentionTileSetting::Gemm3BlockWarps::at(number<0>{});

      constexpr index_t kMPerBlock = Problem::HstuAttentionTileSetting::kN;
      constexpr index_t kKPerBlock = Problem::HstuAttentionTileSetting::kM0;

      constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
      constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

      using AWarpDstr = typename WarpGemm::AWarpDstr;
      using CWarpDstr = typename WarpGemm::CWarpDstr;
      auto ds_warp_tensor =
          make_static_distributed_tensor<typename Problem::QKVDataType>(
              CWarpDstr{});
      auto dst_warp_tensor =
          make_static_distributed_tensor<typename Problem::QKVDataType>(
              AWarpDstr{});

      constexpr auto a_warp_y_lengths =
          to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
      constexpr auto c_warp_y_lengths =
          to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

      constexpr auto a_warp_y_index_zeros =
          uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
      constexpr auto c_warp_y_index_zeros =
          uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

      if constexpr (is_target_warptile_16_32) {
        constexpr index_t NumKSub = WarpGemm::kK / 16;
        constexpr index_t kSubPerThread =
            CWarpDstr{}.get_ys_to_d_descriptor().get_element_space_size();

        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
          constexpr auto kIter = number<km[number<0>{}]>{};
          constexpr auto mIter = number<km[number<1>{}]>{};

          static_for<0, NumKSub, 1>{}([&](auto kSub) {
            ds_warp_tensor.get_thread_buffer() = ds_in.get_y_sliced_thread_data(
                merge_sequences(
                    sequence<kIter * NumKSub + kSub, mIter>{},
                    c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            // C->A transpose is the register identity for one 16x16 fragment;
            // place it into sub-access kSub of the K=32 A tile (Double packs
            // the accesses as [sub0(kSubPerThread), sub1(kSubPerThread)] in the
            // thread buffer).
            static_for<0, kSubPerThread, 1>{}([&](auto i) {
              dst_warp_tensor.get_thread_buffer()(
                  number<kSub * kSubPerThread + i>{}) =
                  ds_warp_tensor.get_thread_buffer()(number<i>{});
            });
          });

          dst_out.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
              dst_warp_tensor.get_thread_buffer());
        });
      } else {
        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
          constexpr auto kIter = number<km[number<0>{}]>{};
          constexpr auto mIter = number<km[number<1>{}]>{};
          ds_warp_tensor.get_thread_buffer() = ds_in.get_y_sliced_thread_data(
              merge_sequences(sequence<kIter, mIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

          dst_warp_tensor.get_thread_buffer() =
              ds_warp_tensor.get_thread_buffer();
          dst_out.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, a_warp_y_lengths),
              dst_warp_tensor.get_thread_buffer());
        });
      }
    } else {
      dst_out.get_thread_buffer() = ds_in.get_thread_buffer();
    }
  }

  // -------------------------------------------------------------------------
  // Gemm1 single-rep N (used by the epilogue to stride over dV output)
  // -------------------------------------------------------------------------
  template <typename Problem>
  CK_TILE_DEVICE static constexpr index_t GetPTOGradTBlockGemmSingleRepN() {
    return Problem::HstuAttentionTileSetting::Gemm1BlockWarps::at(number<1>{}) *
        Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{});
  }

  // Gemm1: dV += P^T @ dO^T   [kN0, kVHeaddim] = [kN0, kK1] x [kVHeaddim, kK1]
  template <typename Problem, bool kUseTrLoad = false>
  CK_TILE_HOST_DEVICE static constexpr auto GetPTOGradTBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm1Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kN,
                Problem::HstuAttentionTileSetting::kVHeaddim,
                Problem::HstuAttentionTileSetting::kK1>,
            typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
            typename Problem::HstuAttentionTileSetting::Gemm1WarpTile>>;

    auto warp_gemm = [&]() {
      if constexpr (
          (std::is_same_v<typename Problem::QKVDataType, half_t> ||
           std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
          std::is_same_v<typename Problem::GemmAccDataType, float>) {
        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{});
#if defined(__hstu_gfx125__)
        static_assert(
            WarpGemmM == 16 && WarpGemmK == 32,
            "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
        // Gemm1 (dV = P^T @ dO^T) reuses Gemm0's C output as its A input via
        // the transpose-free register copy in PTFromGemm0CToGemm1A, which
        // requires the mfma A- and C-operand per-lane sizes to coincide (i.e.
        // WarpGemmK == 16). gfx950 still provides the 16x16x16 fp16 mfma, so
        // allow it here in addition to 16x16x32.
        static_assert(
            (WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
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
              Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
              Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
              Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
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
              Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<0>{}),
              Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<1>{}),
              Problem::HstuAttentionTileSetting::Gemm1WarpTile::at(number<2>{}),
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
        typename Problem::HstuAttentionTileSetting::Gemm1BlockWarps,
        decltype(warp_gemm)>;

    if constexpr (!kUseTrLoad) {
      return BlockGemmARegBSmemCRegSliceGemmKFullGemmN<
          GemmProblem,
          BlockGemmPolicy>{};
    } else {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
      constexpr index_t NIterPerWarp =
          Problem::HstuAttentionTileSetting::kVHeaddim /
          decltype(warp_gemm)::kN;
      constexpr index_t KIterPerWarp =
          Problem::HstuAttentionTileSetting::kK1 / decltype(warp_gemm)::kK;
      constexpr index_t kTdmGemmPrefetchDepth = NIterPerWarp * KIterPerWarp / 2;
      constexpr index_t kTdmGemmStepsPerGroup = 1;

      return BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetchSliceGemmKFullGemmN<
          GemmProblem,
          BlockGemmPolicy,
          kTdmGemmPrefetchDepth,
          kTdmGemmStepsPerGroup>{};
#else
      return BlockGemmARegBSmemTrLoadCRegSliceGemmKFullGemmN<
          GemmProblem,
          BlockGemmPolicy>{};
#endif
    }
  }

  // -------------------------------------------------------------------------
  // Gemm3 single-rep N (used by the epilogue to stride over dK output)
  // -------------------------------------------------------------------------
  template <typename Problem>
  CK_TILE_DEVICE static constexpr index_t GetSGradTQTBlockGemmSingleRepN() {
    return Problem::HstuAttentionTileSetting::Gemm3BlockWarps::at(number<1>{}) *
        Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<1>{});
  }

  // Gemm3: dK += dS^T @ Q^T   [kN0, kQKHeaddim] = [kN0, kK1] x [kQKHeaddim,
  // kK1] Uses Gemm3BlockWarps/Gemm3WarpTile which may differ from Gemm1's
  // configuration.
  template <typename Problem, bool kUseTrLoad = false>
  CK_TILE_HOST_DEVICE static constexpr auto GetSGradTQTBlockGemm() {
    using GemmProblem = BlockGemmProblem<
        typename Problem::QKVDataType,
        typename Problem::QKVDataType,
        typename Problem::GemmAccDataType,
        Problem::kNumGemm3Warps * get_warp_size(),
        TileGemmShape<
            sequence<
                Problem::HstuAttentionTileSetting::kN,
                Problem::HstuAttentionTileSetting::kQKHeaddim,
                Problem::HstuAttentionTileSetting::kK1>,
            typename Problem::HstuAttentionTileSetting::Gemm3BlockWarps,
            typename Problem::HstuAttentionTileSetting::Gemm3WarpTile>>;

    auto warp_gemm = [&]() {
      if constexpr (
          (std::is_same_v<typename Problem::QKVDataType, half_t> ||
           std::is_same_v<typename Problem::QKVDataType, bf16_t>) &&
          std::is_same_v<typename Problem::GemmAccDataType, float>) {
        constexpr index_t WarpGemmM =
            Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{});
        constexpr index_t WarpGemmK =
            Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{});

#if defined(__hstu_gfx125__)
        static_assert(
            WarpGemmM == 16 && WarpGemmK == 32,
            "Not supported WarpGemm sizes!");
#elif defined(__hstu_gfx95__)
        // Gemm3 (dK = dS^T @ Q^T) reuses Gemm2's C output as its A input via
        // the transpose-free register copy in SGradTFromGemm2CToGemm3A, which
        // requires the mfma A- and C-operand per-lane sizes to coincide (i.e.
        // WarpGemmK == 16). gfx950 still provides the 16x16x16 fp16 mfma, so
        // allow it here in addition to 16x16x32.
        static_assert(
            (WarpGemmM == 16 && (WarpGemmK == 16 || WarpGemmK == 32)) ||
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
              Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{}),
              Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<1>{}),
              Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{}),
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
              Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<0>{}),
              Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<1>{}),
              Problem::HstuAttentionTileSetting::Gemm3WarpTile::at(number<2>{}),
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
        typename Problem::HstuAttentionTileSetting::Gemm3BlockWarps,
        decltype(warp_gemm)>;

    if constexpr (!kUseTrLoad) {
      return BlockGemmARegBSmemCRegSliceGemmKFullGemmN<
          GemmProblem,
          BlockGemmPolicy>{};
    } else {
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
      constexpr index_t NIterPerWarp =
          Problem::HstuAttentionTileSetting::kQKHeaddim /
          decltype(warp_gemm)::kN;
      constexpr index_t KIterPerWarp =
          Problem::HstuAttentionTileSetting::kK1 / decltype(warp_gemm)::kK;
      constexpr index_t kTdmGemmPrefetchDepth = NIterPerWarp * KIterPerWarp / 2;
      constexpr index_t kTdmGemmStepsPerGroup = 1;

      return BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetchSliceGemmKFullGemmN<
          GemmProblem,
          BlockGemmPolicy,
          kTdmGemmPrefetchDepth,
          kTdmGemmStepsPerGroup>{};
#else
      return BlockGemmARegBSmemTrLoadCRegSliceGemmKFullGemmN<
          GemmProblem,
          BlockGemmPolicy>{};
#endif
    }
  }
};

} // namespace ck_tile
