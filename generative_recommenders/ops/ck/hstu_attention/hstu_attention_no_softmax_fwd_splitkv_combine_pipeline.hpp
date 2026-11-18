// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_fwd_splitkv_combine_pipeline_policy.hpp"

namespace ck_tile {

template <
    typename Problem_,
    typename Traits_,
    typename Policy_ = HstuAttentionFwdSplitKVCombinePipelinePolicy>
struct HstuAttentionNoSoftmaxFwdSplitKVCombinePipeline {
  using Problem = remove_cvref_t<Problem_>;
  using Traits = remove_cvref_t<Traits_>;
  using Policy = remove_cvref_t<Policy_>;
  using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;
  using ODataType = remove_cvref_t<typename Problem::ODataType>;

  static constexpr index_t kBlockSize = Problem::kBlockSize;

  static constexpr index_t kM = Problem::kM;
  static constexpr index_t kOHeaddim = Problem::kOHeaddim;

  static_assert(
      kOHeaddim <= 256,
      "hdim bigger than 256 is not suitable for this pipeline!");

  static_assert(
      Problem::kUseSoftmax == false,
      "This pipeline only works with not-using softmax");

  static constexpr bool kIsJagged = Problem::kIsJagged;

  static constexpr bool kPadSeqLenQ = Traits::kPadSeqLenQ;
  static constexpr bool kPadHeadDimO = Traits::kPadHeadDimO;

  // last dimension vector length used to create tensor view(and decide
  // buffer_load vector length)
  // ... together with tensor distribution. tensor dist should able to overwrite
  // this
  static constexpr index_t kAlignmentO =
      kPadHeadDimO ? 1 : Policy::template GetAlignmentO<Problem>();

  static constexpr index_t kBlockPerCu = []() {
    if constexpr (Traits::kBlockPerCu != -1)
      return Traits::kBlockPerCu;
    else {
      return 2;
    }
  }();

  CK_TILE_DEVICE static constexpr ck_tile::index_t GetSmemSize() {
    return Policy::template GetSmemSize<Problem>();
  }

  template <typename OAccDramBlockWindowTmp>
  CK_TILE_DEVICE auto operator()(
      const OAccDramBlockWindowTmp&
          o_acc_dram_block_window_tmp, // M0*kOHeaddim tile
      ck_tile::index_t o_acc_split_stride,
      ck_tile::index_t num_splits) const {
    static_assert(
        std::is_same_v<
            OaccDataType,
            remove_cvref_t<typename OAccDramBlockWindowTmp::DataType>>,
        "wrong!");

    static_assert(
        kM == OAccDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
            kOHeaddim ==
                OAccDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
        "wrong!");

    auto o_acc_dram_window = make_tile_window(
        o_acc_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM>{}, number<kOHeaddim>{}),
        o_acc_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeOaccDramTileDistribution<Problem>());

    auto o_acc_ptr =
        o_acc_dram_window.get_bottom_tensor_view().get_buffer_view().p_data_;

    auto o_acc = load_tile(o_acc_dram_window);

    for (int i = 1; i < num_splits; i++) {
      o_acc_dram_window.set_bottom_tensor_view_data_ptr(
          o_acc_ptr + o_acc_split_stride * i);
      auto o_acc_tile = load_tile(o_acc_dram_window);

      tile_elementwise_inout(
          [](auto& x, const auto& y) { x = x + y; }, o_acc, o_acc_tile);
    };

    return o_acc;
  }
};

} // namespace ck_tile
