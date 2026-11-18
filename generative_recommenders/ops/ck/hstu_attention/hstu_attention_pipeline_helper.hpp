// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

namespace detail {

// Per-issue state of a raw direct-to-LDS DMA into lds_window (a monolithic 2D
// [Rows, RowLength] LDS window spanning all NumLdsBuffers buffers), computed
// once: each issue's M0 (the wave's LDS destination) and its per-lane dram
// element offset. The kernel's K/V dram views are XOR-swizzled
// (make_direct_lds_xor_dram_view), so these offsets already write the swizzled
// LDS layout.
template <
    index_t NumLdsBuffers,
    index_t NumWarps,
    typename LdsMonolithicTileWindow,
    index_t LdsBufIdx,
    typename DramTileWindow>
CK_TILE_DEVICE auto MakeDirectLdsIssues(
    const LdsMonolithicTileWindow& lds_window,
    [[maybe_unused]] const number<LdsBufIdx> i_buf,
    const DramTileWindow& dram_window) {
  using DataType = typename LdsMonolithicTileWindow::DataType;
  using WindowTraits = typename DramTileWindow::Traits;
  using SFC_Ys = typename WindowTraits::SFC_Ys;

  constexpr index_t NumAccess = DramTileWindow::NumAccessPerCoord;

  constexpr auto dram_win_lengths = typename DramTileWindow::WindowLengths{};
  constexpr auto lds_win_lengths =
      typename LdsMonolithicTileWindow::WindowLengths{};

  static_assert(
      dram_win_lengths[number<1>{}] == lds_win_lengths[number<1>{}],
      "Check failed!");
  static_assert(
      dram_win_lengths[number<0>{}] * NumLdsBuffers ==
          lds_win_lengths[number<0>{}],
      "Check failed!");

  constexpr index_t RowLength = dram_win_lengths[number<1>{}];
  constexpr index_t Rows = dram_win_lengths[number<0>{}];
  constexpr index_t kDmaVector =
      remove_cvref_t<decltype(dram_window)>::Traits::ScalarPerVector;
  constexpr index_t issues_per_tile =
      (Rows * RowLength) / (NumWarps * get_warp_size() * kDmaVector);

  // this assumption ensures some warp index will be assigned on the dimension
  // of Rows
  static_assert(get_warp_size() * kDmaVector >= RowLength, "Check failed!");

  constexpr index_t RowsPerWarp = get_warp_size() * kDmaVector / RowLength;
  constexpr index_t RowsPerIssue = NumWarps * RowsPerWarp;

  static_assert(issues_per_tile == NumAccess, "Check failed!");

  // Absolute LDS byte offset of this window's buffer: the direct-to-LDS DMA
  // ignores the smem pointer argument and writes purely to M0, so m0 must be
  // measured from the workgroup LDS base. Casting the (generic/flat) buffer
  // pointer to an address_space(3) pointer lowers to an addrspacecast that
  // yields exactly that 32-bit LDS byte offset (0 for the K region at smem
  // base, GetSmemSizeK for the V region). A C-style cast is required here:
  // clang rejects reinterpret_cast across address spaces.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
  const auto lds_as3_ptr = (DataType __attribute__((address_space(
      3)))*)(lds_window.get_bottom_tensor_view().get_buffer_view().p_data_);
#pragma clang diagnostic pop
  const index_t lds_base_byte_offset =
      static_cast<index_t>(reinterpret_cast<uintptr_t>(lds_as3_ptr));

  // The LDS block descriptor is statically known (all strides are number<>), so
  // a default- constructed instance is a constant expression and
  // calculate_offset() folds at compile time.
  constexpr auto lds_desc = remove_cvref_t<
      decltype(lds_window.get_bottom_tensor_view().get_tensor_descriptor())>{};

  constexpr index_t buff_offset =
      lds_desc.calculate_offset(make_tuple(number<Rows>{}, number<0>{})) *
      LdsBufIdx * sizeof(DataType);
  constexpr index_t size_per_issue = lds_desc.calculate_offset(make_tuple(
                                         number<RowsPerIssue>{}, number<0>{})) *
      sizeof(DataType);
  constexpr index_t size_per_warp = lds_desc.calculate_offset(make_tuple(
                                        number<RowsPerWarp>{}, number<0>{})) *
      sizeof(DataType);

  const index_t m0_base = amd_wave_read_first_lane(
      lds_base_byte_offset + buff_offset +
      size_per_warp * get_warp_id(bool_constant<false>{}));

  const auto tile_dstr = dram_window.get_tile_distribution();
  const auto coords = dram_window.prepare_coords(
      dram_window.get_bottom_tensor_view(),
      dram_window.get_window_origin(),
      tile_dstr,
      get_partition_index(tile_dstr));

  auto window_adaptor_thread_coord = coords[number<0>{}][number<0>{}];
  auto bottom_tensor_thread_coord = coords[number<0>{}][number<1>{}];

  statically_indexed_array<index_t, NumAccess> m0;
  statically_indexed_array<index_t, NumAccess> voffset;

  static_for<0, NumAccess, 1>{}([&](auto i_access) {
    constexpr index_t i = decltype(i_access)::value;

    m0(i_access) = m0_base + i * size_per_issue;
    voffset(i_access) = bottom_tensor_thread_coord.get_offset();

    if constexpr (i != NumAccess - 1) {
      constexpr auto idx_diff_ps_ys = container_concat(
          generate_tuple(
              [&](auto) { return number<0>{}; },
              number<DramTileWindow::NDimP>{}),
          SFC_Ys::get_forward_step(i_access));

      dram_window.move_window_adaptor_and_bottom_tensor_thread_coordinate(
          window_adaptor_thread_coord,
          bottom_tensor_thread_coord,
          idx_diff_ps_ys);
    }
  });

  return make_tuple(m0, voffset);
}

// The raw (inline-asm, so not waited for by the compiler) DMA of one LDS
// buffer, issues from MakeDirectLdsIssues, dram rows starting row_offset
// elements into the tensor
template <typename DataType, typename DramTileWindow, typename Issues>
CK_TILE_DEVICE void IssueDirectLds(
    DataType* smem,
    const DramTileWindow& dram_window,
    const Issues& issues,
    index_t row_offset) {
  using vector_t = typename DramTileWindow::Traits::vector_t;
  constexpr index_t NumAccess = DramTileWindow::NumAccessPerCoord;

  static_for<0, NumAccess, 1>{}([&](auto i_access) {
    m0_set_with_memory(issues[number<0>{}][i_access]);
    dram_window.get_bottom_tensor_view()
        .get_buffer_view()
        .template async_get_raw<vector_t>(
            smem,
            issues[number<1>{}][i_access] + row_offset,
            0,
            true,
            bool_constant<true>{} /*pre_nop*/);
  });
}

}; // namespace detail
}; // namespace ck_tile
