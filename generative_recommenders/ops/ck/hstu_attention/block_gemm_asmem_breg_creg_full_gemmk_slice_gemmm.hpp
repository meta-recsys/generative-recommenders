// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_default_policy.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block distributed tensor
// C is block distributed tensor
template <
    typename Problem_,
    typename Policy_ = BlockGemmASmemBRegCRegV1DefaultPolicy>
struct BlockGemmASmemBRegCRegFullGemmKSliceGemmM {
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
      typename ABlockWindowTmp,
      typename BBlockTensorTmp>
  CK_TILE_DEVICE void operator()(
      CBlockTensor& c_block_tensor,
      const ABlockWindowTmp& a_block_window_tmp,
      const BBlockTensorTmp& b_block_tensor_tmp) const {
    static_assert(
        std::is_same_v<
            ADataType,
            remove_cv_t<typename ABlockWindowTmp::DataType>> &&
            std::is_same_v<
                BDataType,
                remove_cv_t<typename BBlockTensorTmp::DataType>> &&
            std::is_same_v<
                CDataType,
                remove_cv_t<typename CBlockTensor::DataType>>,
        "wrong!");

    constexpr index_t MPerBlock =
        ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
    constexpr index_t NPerBlock = BBlockTensorTmp{}.get_lengths()[number<0>{}];
    constexpr index_t KPerBlock =
        ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

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

    constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
    constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

    const index_t iMWarp = 0;

    static_assert(MWarp == 1, "Check failed!");

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

    // constrcut from B-block-tensor from B-Block-tensor-tmp
    // FIXME: need method to check b_block_tensor and b_block_tensor_tmp have
    // equivalent distribution
    auto b_block_tensor =
        make_static_distributed_tensor<typename BBlockTensorTmp::DataType>(
            MakeBBlockTileDistribution());

    b_block_tensor.get_thread_buffer() = b_block_tensor_tmp.get_thread_buffer();

    // construct A-warp-window
    auto a_warp_window_tmp = make_tile_window(
        a_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<WG::kM>{}, number<WG::kK>{}),
        a_block_window_tmp.get_window_origin() +
            multi_index<2>{iMWarp * WG::kM, 0},
        make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

    statically_indexed_array<
        statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
        MIterPerWarp>
        a_warp_windows;

    // check C-block-distribution
    static_assert(
        std::is_same_v<
            remove_cvref_t<decltype(c_block_dstr_encode)>,
            remove_cvref_t<
                decltype(CBlockTensor::get_tile_distribution()
                             .get_static_tile_distribution_encoding())>>,
        "wrong!");

    using BWarpDstr = typename WG::BWarpDstr;
    using CWarpDstr = typename WG::CWarpDstr;

    using BWarpTensor = typename WG::BWarpTensor;
    using CWarpTensor = typename WG::CWarpTensor;

    constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    constexpr auto b_warp_y_index_zeros =
        uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros =
        uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    constexpr auto I0 = number<0>{};

    // hot loop:
    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
      using a_warp_tensor_type = decltype(load_tile(a_warp_windows(I0)(I0)));

      statically_indexed_array<a_warp_tensor_type, KIterPerWarp> a_warp_tensors;

      // read A warp tensor from A Block window
      a_warp_windows(mIter)(I0) = a_warp_window_tmp;
      move_tile_window(
          a_warp_windows(mIter)(I0),
          {mIter * MPerBlockPerIter, 0 * KPerBlockPerIter});
      a_warp_tensors[I0] = load_tile(a_warp_windows(mIter)(I0));

      __builtin_amdgcn_sched_barrier(0);
      static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        if constexpr (kIter < KIterPerWarp - 1) {
          // read A warp tensor from A Block window
          a_warp_windows(mIter)(number<kIter + 1>{}) = a_warp_window_tmp;
          move_tile_window(
              a_warp_windows(mIter)(number<kIter + 1>{}),
              {mIter * MPerBlockPerIter, (kIter + 1) * KPerBlockPerIter});
          a_warp_tensors[number<kIter + 1>{}] =
              load_tile(a_warp_windows(mIter)(number<kIter + 1>{}));
        };

        __builtin_amdgcn_sched_barrier(0);

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
          // read A warp tensor from A block tensor
          BWarpTensor b_warp_tensor;

          b_warp_tensor.get_thread_buffer() =
              b_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

          // read C warp tensor from C block tensor
          CWarpTensor c_warp_tensor;

          if constexpr (kIter == 0) {
            // warp GEMM
            c_warp_tensor = WG{}(a_warp_tensors[kIter], b_warp_tensor);
          } else {
            c_warp_tensor.get_thread_buffer() =
                c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            // warp GEMM
            WG{}(c_warp_tensor, a_warp_tensors[kIter], b_warp_tensor);
          }

          // write C warp tensor into C block tensor
          c_block_tensor.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
              c_warp_tensor.get_thread_buffer());
        });
      });
    });
  }

  // B operand (register-resident) distribution encoding -- [NPerBlock,
  // KPerBlock]
  template <
      index_t NPerBlock = BlockGemmShape::kN,
      index_t KPerBlock = BlockGemmShape::kK>
  CK_TILE_DEVICE static constexpr auto MakeBBlockDistributionEncode() {
    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
    constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr auto b_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<MWarp>,
        tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
        tuple<sequence<0, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto b_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            b_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

    return b_block_dstr_encode;
  }

  template <
      index_t NPerBlock = BlockGemmShape::kN,
      index_t KPerBlock = BlockGemmShape::kK>
  CK_TILE_DEVICE static constexpr auto MakeBBlockTileDistribution() {
    constexpr auto b_block_dstr_encode =
        MakeBBlockDistributionEncode<NPerBlock, KPerBlock>();

    return make_static_tile_distribution(b_block_dstr_encode);
  }

  template <
      index_t MPerBlock = BlockGemmShape::kM,
      index_t NPerBlock = BlockGemmShape::kN>
  CK_TILE_DEVICE static constexpr auto MakeCBlockDistributionEncode() {
    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

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

    return c_block_dstr_encode;
  }

  template <
      index_t MPerBlock = BlockGemmShape::kM,
      index_t NPerBlock = BlockGemmShape::kN>
  CK_TILE_DEVICE static constexpr auto MakeCBlockTile() {
    constexpr auto c_block_dstr_encode =
        MakeCBlockDistributionEncode<MPerBlock, NPerBlock>();

    constexpr auto c_block_dstr =
        make_static_tile_distribution(c_block_dstr_encode);
    auto c_block_tensor =
        make_static_distributed_tensor<CDataType>(c_block_dstr);
    return c_block_tensor;
  }

  // C = A * B
  template <typename ABlockWindowTmp, typename BBlockTensorTmp>
  CK_TILE_DEVICE auto operator()(
      const ABlockWindowTmp& a_block_window_tmp,
      const BBlockTensorTmp& b_block_tensor_tmp) const {
    auto c_block_tensor = MakeCBlockTile();
    operator()(c_block_tensor, a_block_window_tmp, b_block_tensor_tmp);
    return c_block_tensor;
  }
};

} // namespace ck_tile
