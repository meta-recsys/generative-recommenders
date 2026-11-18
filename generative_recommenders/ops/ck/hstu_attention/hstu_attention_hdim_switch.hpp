// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <stdexcept>

#define HDIM_SWITCH(HDIM_1, HDIM_2, CONST_NAME, ...)             \
  [&] {                                                          \
    if (HDIM_1 <= 64 && HDIM_2 <= 64) {                          \
      constexpr ck_tile::index_t CONST_NAME = 64;                \
      __VA_ARGS__();                                             \
    } else if (HDIM_1 <= 96 && HDIM_2 <= 96) {                   \
      constexpr ck_tile::index_t CONST_NAME = 96;                \
      __VA_ARGS__();                                             \
    } else if (HDIM_1 <= 128 && HDIM_2 <= 128) {                 \
      constexpr ck_tile::index_t CONST_NAME = 128;               \
      __VA_ARGS__();                                             \
    } else if (HDIM_1 <= 256 && HDIM_2 <= 256) {                 \
      constexpr ck_tile::index_t CONST_NAME = 256;               \
      __VA_ARGS__();                                             \
    } else {                                                     \
      throw std::runtime_error("Head-dim sizes not supported!"); \
    }                                                            \
  }()
