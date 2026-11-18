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
struct HstuAttentionWithSoftmaxFwdSplitKVCombinePipeline {
  using Problem = remove_cvref_t<Problem_>;
  using Traits = remove_cvref_t<Traits_>;
  using Policy = remove_cvref_t<Policy_>;
  using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;
  using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;
  using ODataType = remove_cvref_t<typename Problem::ODataType>;

  static constexpr index_t kBlockSize = Problem::kBlockSize;

  static constexpr index_t kM = Problem::kM;
  static constexpr index_t kOHeaddim = Problem::kOHeaddim;
  static constexpr index_t kMaxSplits = Problem::kMaxSplits;

  static_assert(
      kOHeaddim <= 256,
      "hdim bigger than 256 is not suitable for this pipeline!");

  static_assert(
      Problem::kUseSoftmax == true,
      "This pipeline only works with using softmax");

  static constexpr bool kIsJagged = Problem::kIsJagged;

  static constexpr bool kPadSeqLenQ = Traits::kPadSeqLenQ;
  static constexpr bool kPadNumSplits = Traits::kPadNumSplits;
  static constexpr bool kPadHeadDimO = Traits::kPadHeadDimO;

  // last dimension vector length used to create tensor view(and decide
  // buffer_load vector length)
  // ... together with tensor distribution. tensor dist should able to overwrite
  // this
  static constexpr index_t kAlignmentO =
      kPadHeadDimO ? 1 : Policy::template GetAlignmentO<Problem>();

  static constexpr index_t kAlignmentLSEacc =
      kPadNumSplits ? 1 : Policy::template GetAlignmentLSEacc<Problem>();

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

  template <
      typename LSEaccDramBlockWindowTmp,
      typename OAccDramBlockWindowTmp,
      typename LSEDramBlockWindow>
  CK_TILE_DEVICE auto operator()(
      const LSEaccDramBlockWindowTmp& lse_acc_dram_block_window_tmp, // kM tile
      const OAccDramBlockWindowTmp&
          o_acc_dram_block_window_tmp, // kM*kOHeaddim tile
      LSEDramBlockWindow& lse_dram_block_window, // kM tile
      index_t o_acc_split_stride,
      index_t num_splits,
      void* smem_ptr) const {
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

    // lse_scale tile in LDS, layout [kM, kMaxSplits], accessed randomly using
    // [row, col] coordinates
    LSEDataType* lse_scale_lds_ptr = static_cast<LSEDataType*>(
        static_cast<void*>(static_cast<char*>(smem_ptr)));
    auto lse_scale_lds =
        [=,
         lds_desc = Policy::template MakeLSEscaleLdsBlockDescriptor<Problem>()](
            index_t row, index_t col) -> LSEDataType& {
      return lse_scale_lds_ptr[lds_desc.calculate_offset(make_tuple(row, col))];
    };

    auto lse_scale_lds_write_window = [&]() {
      auto view = make_tensor_view<address_space_enum::lds>(
          lse_scale_lds_ptr,
          Policy::template MakeLSEscaleLdsBlockDescriptor<Problem>());
      return make_tile_window(
          view, make_tuple(number<kM>{}, number<kMaxSplits>{}), {0, 0});
    }();

    auto lse_acc_dram_window = make_tile_window(
        lse_acc_dram_block_window_tmp.get_bottom_tensor_view(),
        lse_acc_dram_block_window_tmp.get_window_lengths(),
        lse_acc_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeLSEaccDramTileDistribution<Problem>());

    auto lse_acc = load_tile(lse_acc_dram_window);

    using lse_acc_type = decltype(lse_acc);
    constexpr auto lse_spans = lse_acc_type::get_distributed_spans();

    const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    // provide partition_index for LDS tile window so that warp_id is in vgpr
    array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

    // calculate max of lse_acc[] across all splits for all rows in the tile,
    // lse_max is only used for stablizing the exp()
    auto lse_max = block_tile_reduce<LSEDataType>(
        lse_acc, sequence<1>{}, f_max, -numeric<LSEDataType>::infinity());
    block_tile_reduce_sync(lse_max, f_max, bool_constant<false>{});

    using lse_max_type = decltype(lse_max);

    // calculate exp(x-m) for all elements in the tile
    lse_acc_type lse_exp;
    sweep_tile_span(lse_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      if (lse_max[i_idx] == -numeric<LSEDataType>::infinity()) {
        sweep_tile_span(lse_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          lse_exp(i_j_idx) = ck_tile::type_convert<LSEDataType>(0.0f);
        });
      } else {
        sweep_tile_span(lse_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          lse_exp(i_j_idx) = ck_tile::exp(lse_acc[i_j_idx] - lse_max[i_idx]);
        });
      }
    });

    // calculate sum of exp(x-m)... across all splits for all rows in the tile
    auto lse_sum = block_tile_reduce<LSEDataType>(
        lse_exp, sequence<1>{}, f_sum, type_convert<LSEDataType>(0));
    block_tile_reduce_sync(lse_sum, f_sum, bool_constant<false>{});

    // calculate log(sum of exp(x)...) across all splits for all rows in the
    // tile
    lse_max_type lse_logsum;
    {
      constexpr auto logsum_spans = lse_max_type::get_distributed_spans();
      sweep_tile_span(logsum_spans[number<0>{}], [&](auto idx0) {
        constexpr auto i_idx = make_tuple(idx0);

        if (lse_sum[i_idx] == ck_tile::type_convert<LSEDataType>(0.0f))
          lse_logsum(i_idx) = -numeric<LSEDataType>::infinity();
        else
          lse_logsum(i_idx) = ck_tile::log(lse_sum[i_idx]) + lse_max[i_idx];
      });
    }

    // in case kStoreLSE is false, LSEDramBlockWindow is null
    if constexpr (!is_null_tile_window_v<LSEDramBlockWindow>) {
      store_tile(lse_dram_block_window, lse_logsum);
    }

    // calculate scale value (used for adjusting the o_acc) for all splits for
    // all rows in the tile
    lse_acc_type& lse_scale = lse_acc;
    sweep_tile_span(lse_spans[number<0>{}], [&](auto idx0) {
      constexpr auto i_idx = make_tuple(idx0);
      if (lse_logsum[i_idx] == -numeric<LSEDataType>::infinity()) {
        sweep_tile_span(lse_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          lse_scale(i_j_idx) = ck_tile::type_convert<LSEDataType>(0.0f);
        });
      } else {
        sweep_tile_span(lse_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          lse_scale(i_j_idx) =
              ck_tile::exp(lse_acc[i_j_idx] - lse_logsum[i_idx]);
        });
      }
    });

    store_tile(lse_scale_lds_write_window, lse_scale, partition_index);
    block_sync_lds();

    auto o_acc_dram_window = make_tile_window(
        o_acc_dram_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM>{}, number<kOHeaddim>{}),
        o_acc_dram_block_window_tmp.get_window_origin(),
        Policy::template MakeOaccDramTileDistribution<Problem>());

    auto o_acc_ptr =
        o_acc_dram_window.get_bottom_tensor_view().get_buffer_view().p_data_;

    using o_acc_type = decltype(load_tile(o_acc_dram_window));

    o_acc_type o_acc;

    clear_tile(o_acc);

    for (int i_split = 0; i_split < num_splits; i_split++) {
      o_acc_dram_window.set_bottom_tensor_view_data_ptr(
          o_acc_ptr + o_acc_split_stride * i_split);
      auto o_acc_tile = load_tile(o_acc_dram_window);

      constexpr auto o_acc_spans = o_acc_type::get_distributed_spans();
      sweep_tile_span(o_acc_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(o_acc_spans[number<1>{}], [&](auto idx1) {
          constexpr auto i_j_idx = make_tuple(idx0, idx1);

          const auto x_indices = get_x_indices_from_distributed_indices(
              o_acc_tile.get_tile_distribution(), i_j_idx, partition_index);

          const auto row = x_indices.at(number<0>{});

          const LSEDataType lse_scale_val = lse_scale_lds(row, i_split);

          o_acc(i_j_idx) +=
              o_acc_tile[i_j_idx] * type_convert<OaccDataType>(lse_scale_val);
        });
      });
    };

    return o_acc;
  }
};

} // namespace ck_tile
