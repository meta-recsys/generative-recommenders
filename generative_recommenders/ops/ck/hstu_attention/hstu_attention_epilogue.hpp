// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/common/tensor_layout.hpp>
#include <ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp>

namespace ck_tile {

template <typename Problem_, typename Policy_ = void>
struct NRepetitions2DEpilogue {
  using Problem = remove_cvref_t<Problem_>;
  using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
  using ODataType = remove_cvref_t<typename Problem::ODataType>;
  static constexpr bool kPadM = Problem::kPadM;
  static constexpr bool kPadN = Problem::kPadN;

  CK_TILE_DEVICE static constexpr index_t GetSmemSize() {
    return 0;
  }

  template <
      typename ODramWindowTmp,
      typename OAccTile,
      index_t NumNRepetition,
      memory_operation_enum out_memory_data_op = memory_operation_enum::set>
  CK_TILE_DEVICE auto operator()(
      ODramWindowTmp& o_dram_window_tmp,
      const OAccTile& o_acc_tile,
      number<NumNRepetition>) {
    constexpr index_t kM = ODramWindowTmp{}.get_window_lengths()[number<0>{}];
    constexpr index_t kN = ODramWindowTmp{}.get_window_lengths()[number<1>{}];

    static_assert(kN % NumNRepetition == 0, "Check failed!");

    constexpr index_t kSingleRepN = kN / NumNRepetition;

    auto o_nrep_dram_window = make_tile_window(
        o_dram_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kM>{}, number<kSingleRepN>{}),
        o_dram_window_tmp.get_window_origin());

    static_for<0, NumNRepetition, 1>{}([&](auto i_rep) {
      if constexpr (out_memory_data_op == memory_operation_enum::set) {
        auto tile_for_store = cast_tile<ODataType>(get_slice_tile(
            o_acc_tile,
            sequence<0, i_rep * kSingleRepN>{},
            sequence<kM, (i_rep + 1) * kSingleRepN>{}));
        store_tile(o_nrep_dram_window, tile_for_store);
      } else {
        auto tile_for_store = cast_tile<ODataType>(get_slice_tile(
            o_acc_tile,
            sequence<0, i_rep * kSingleRepN>{},
            sequence<kM, (i_rep + 1) * kSingleRepN>{}));
        update_tile(o_nrep_dram_window, tile_for_store);
      }

      move_tile_window(o_nrep_dram_window, {0, kSingleRepN});
    });
  }
};

template <typename Problem_, typename Policy_ = void>
struct MRepetitions2DEpilogue {
  using Problem = remove_cvref_t<Problem_>;
  using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
  using ODataType = remove_cvref_t<typename Problem::ODataType>;
  static constexpr bool kPadM = Problem::kPadM;
  static constexpr bool kPadN = Problem::kPadN;

  CK_TILE_DEVICE static constexpr index_t GetSmemSize() {
    return 0;
  }

  template <
      typename ODramWindowTmp,
      typename OAccTile,
      index_t NumMRepetition,
      memory_operation_enum out_memory_data_op = memory_operation_enum::set>
  CK_TILE_DEVICE auto operator()(
      ODramWindowTmp& o_dram_window_tmp,
      const OAccTile& o_acc_tile,
      number<NumMRepetition>) {
    constexpr index_t kM = ODramWindowTmp{}.get_window_lengths()[number<0>{}];
    constexpr index_t kN = ODramWindowTmp{}.get_window_lengths()[number<1>{}];

    static_assert(kM % NumMRepetition == 0, "Check failed!");

    constexpr index_t kSingleRepM = kM / NumMRepetition;

    auto o_mrep_dram_window = make_tile_window(
        o_dram_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<kSingleRepM>{}, number<kN>{}),
        o_dram_window_tmp.get_window_origin());

    static_for<0, NumMRepetition, 1>{}([&](auto i_rep) {
      if constexpr (out_memory_data_op == memory_operation_enum::set) {
        auto tile_for_store = cast_tile<ODataType>(get_slice_tile(
            o_acc_tile,
            sequence<i_rep * kSingleRepM, 0>{},
            sequence<(i_rep + 1) * kSingleRepM, kN>{}));
        store_tile(o_mrep_dram_window, tile_for_store);
      } else {
        auto tile_for_store = cast_tile<ODataType>(get_slice_tile(
            o_acc_tile,
            sequence<i_rep * kSingleRepM, 0>{},
            sequence<(i_rep + 1) * kSingleRepM, kN>{}));
        store_tile(o_mrep_dram_window, tile_for_store);
      }

      move_tile_window(o_mrep_dram_window, {kSingleRepM, 0});
    });
  }
};

} // namespace ck_tile
