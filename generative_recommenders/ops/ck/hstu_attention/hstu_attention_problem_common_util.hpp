// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <type_traits>

#include <ck_tile/core.hpp>

namespace ck_tile {

namespace detail {

// kUseDwordx3 also allows 6 x 16-bit / 3 or 6 x 32-bit accesses
// (buffer_load_dwordx3); the forward problem uses it, the backward problem
// keeps power-of-two vector sizes.
template <typename DataType, index_t ElemPerThread, bool kUseDwordx3 = false>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxVectorSize() {
  if constexpr (
      std::is_same_v<DataType, half_t> || std::is_same_v<DataType, bf16_t>) {
    if constexpr (ElemPerThread % 8 == 0)
      return 8;
    else if constexpr (kUseDwordx3 && ElemPerThread % 6 == 0)
      return 6;
    else if constexpr (ElemPerThread % 4 == 0)
      return 4;
    else if constexpr (ElemPerThread % 2 == 0)
      return 2;
    return 1;
  } else if constexpr (std::is_same_v<DataType, float>) {
    if constexpr (kUseDwordx3 && ElemPerThread % 6 == 0)
      return 6;
    else if constexpr (ElemPerThread % 4 == 0)
      return 4;
    else if constexpr (kUseDwordx3 && ElemPerThread % 3 == 0)
      return 3;
    else if constexpr (ElemPerThread % 2 == 0)
      return 2;
    return 1;
  } else
    static_assert(false, "The data type is not supported!");
};

template <typename DataType, index_t ElemPerThread>
CK_TILE_HOST_DEVICE static constexpr auto GetDirectLdsMaxVectorSize() {
  if constexpr (
      std::is_same_v<DataType, half_t> || std::is_same_v<DataType, bf16_t>) {
#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
    if constexpr (ElemPerThread % 8 == 0)
      return 8;
    else if constexpr (ElemPerThread % 2 == 0)
      return 2;
    return 1;
#else
    if constexpr (ElemPerThread % 2 == 0)
      return 2;
    return 1;
#endif
  } else if constexpr (std::is_same_v<DataType, float>) {
#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
    if constexpr (ElemPerThread % 4 == 0)
      return 4;
    else
      return 1;
#else
    return 1;
#endif
  } else
    static_assert(false, "The data type is not supported!");
};

template <
    typename DataType,
    index_t kThreadBlockSize,
    index_t kHigherDimSize,
    index_t kLowerDimSize,
    bool kUseDirectLds = false,
    bool kUseDwordx3 = false>
CK_TILE_HOST_DEVICE static constexpr auto GetDramTileAccessMaxVectorSize() {
  constexpr index_t ElemPerThread =
      (kHigherDimSize * kLowerDimSize) / kThreadBlockSize;

  if constexpr (kUseDirectLds)
    return GetDirectLdsMaxVectorSize<DataType, ElemPerThread>();
  else
    return GetMaxVectorSize<DataType, ElemPerThread, kUseDwordx3>();
}

}; // namespace detail

} // namespace ck_tile
