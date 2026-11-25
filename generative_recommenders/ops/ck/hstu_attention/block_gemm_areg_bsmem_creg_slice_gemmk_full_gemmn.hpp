// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_default_policy.hpp>

namespace ck_tile {

// A is block distributed tensor
// B is block window on shared memory
// C is block distributed tensor
template <
    typename Problem_,
    typename Policy_ = BlockGemmARegBSmemCRegV2DefaultPolicy>
struct BlockGemmARegBSmemCRegSliceGemmKFullGemmN {
  using Problem = remove_cvref_t<Problem_>;
  using Policy = remove_cvref_t<Policy_>;
  using ADataType = remove_cvref_t<typename Problem::ADataType>;
  using BDataType = remove_cvref_t<typename Problem::BDataType>;
  using CDataType = remove_cvref_t<typename Problem::CDataType>;
  using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

  static constexpr index_t kBlockSize = Problem::kBlockSize;

  // C += A * B
  template <
      typename CBlockTensor,
      typename ABlockTensorTmp,
      typename BBlockWindowTmp>
  CK_TILE_DEVICE void operator()(
      CBlockTensor& c_block_tensor,
      const ABlockTensorTmp& a_block_tensor_tmp,
      const BBlockWindowTmp& b_block_window_tmp) const {
    static_assert(
        std::is_same_v<
            ADataType,
            remove_cv_t<typename ABlockTensorTmp::DataType>> &&
            std::is_same_v<
                BDataType,
                remove_cv_t<typename BBlockWindowTmp::DataType>> &&
            std::is_same_v<
                CDataType,
                remove_cv_t<typename CBlockTensor::DataType>>,
        "wrong!");

    constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
    constexpr index_t NPerBlock =
        BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
    constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

    static_assert(
        MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
            KPerBlock == BlockGemmShape::kK,
        "wrong!");

    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
    constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
    constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

    const index_t iNWarp = get_warp_id<false>() % NWarp;

    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

    // constrcut from A-block-tensor from A-Block-tensor-tmp
    // FIXME: need method to check a_block_tensor and a_block_tensor_tmp have
    // equivalent distribution
    auto a_block_tensor =
        make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

    a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

    // construct B-warp-window
    auto b_warp_window_tmp = make_tile_window(
        b_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<WG::kN>{}, number<WG::kK>{}),
        b_block_window_tmp.get_window_origin() +
            multi_index<2>{iNWarp * WG::kN, 0},
        make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

    statically_indexed_array<
        statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
        NIterPerWarp>
        b_warp_windows;

    // check C-block-distribution
    static_assert(
        std::is_same_v<
            remove_cvref_t<decltype(c_block_dstr_encode)>,
            remove_cvref_t<
                decltype(CBlockTensor::get_tile_distribution()
                             .get_static_tile_distribution_encoding())>>,
        "wrong!");

    using AWarpDstr = typename WG::AWarpDstr;
    using CWarpDstr = typename WG::CWarpDstr;

    using AWarpTensor = typename WG::AWarpTensor;
    using CWarpTensor = typename WG::CWarpTensor;

    constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    constexpr auto a_warp_y_index_zeros =
        uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros =
        uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    constexpr auto I0 = number<0>{};

    using b_warp_tensor_type = decltype(load_tile(b_warp_windows(I0)(I0)));

    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
      statically_indexed_array<b_warp_tensor_type, NIterPerWarp> b_warp_tensors;

      // read B warp tensor from B Block window
      b_warp_windows(I0)(kIter) = b_warp_window_tmp;
      move_tile_window(
          b_warp_windows(I0)(kIter),
          {0 * NPerBlockPerIter, kIter * KPerBlockPerIter});
      b_warp_tensors(I0) = load_tile(b_warp_windows(I0)(kIter));

      __builtin_amdgcn_sched_barrier(0);

      static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
        if constexpr (nIter < NIterPerWarp - 1) {
          // read B warp tensor from B Block window
          b_warp_windows(number<nIter + 1>{})(kIter) = b_warp_window_tmp;
          move_tile_window(
              b_warp_windows(number<nIter + 1>{})(kIter),
              {(nIter + 1) * NPerBlockPerIter, kIter * KPerBlockPerIter});
          b_warp_tensors(number<nIter + 1>{}) =
              load_tile(b_warp_windows(number<nIter + 1>{})(kIter));
        };

        __builtin_amdgcn_sched_barrier(0);

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
          // read A warp tensor from A block tensor
          AWarpTensor a_warp_tensor;

          a_warp_tensor.get_thread_buffer() =
              a_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

          // read C warp tensor from C block tensor
          CWarpTensor c_warp_tensor;

          c_warp_tensor.get_thread_buffer() =
              c_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

          // warp GEMM
          WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensors[nIter]);

          // write C warp tensor into C block tensor
          c_block_tensor.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
              c_warp_tensor.get_thread_buffer());
        });
      });
    });
  }

  template <
      index_t MPerBlock = BlockGemmShape::kM,
      index_t KPerBlock = BlockGemmShape::kK>
  CK_TILE_DEVICE static constexpr auto MakeABlockTileDistribution() {
    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr auto a_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<NWarp>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
        tuple<sequence<1, 0>>,
        tuple<sequence<1, 0>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto a_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

    return make_static_tile_distribution(a_block_dstr_encode);
  }

  CK_TILE_DEVICE static constexpr auto MakeCBlockTile() {
    constexpr index_t MPerBlock = BlockGemmShape::kM;
    constexpr index_t NPerBlock = BlockGemmShape::kN;

    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
    // constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
    constexpr auto c_block_dstr =
        make_static_tile_distribution(c_block_dstr_encode);
    auto c_block_tensor =
        make_static_distributed_tensor<CDataType>(c_block_dstr);
    return c_block_tensor;
  }

  // C = A * B
  template <typename ABlockTensorTmp, typename BBlockWindowTmp>
  CK_TILE_DEVICE auto operator()(
      const ABlockTensorTmp& a_block_tensor_tmp,
      const BBlockWindowTmp& b_block_window_tmp) const {
    auto c_block_tensor = MakeCBlockTile();
    operator()(c_block_tensor, a_block_tensor_tmp, b_block_window_tmp);
    return c_block_tensor;
  }
};

} // namespace ck_tile
