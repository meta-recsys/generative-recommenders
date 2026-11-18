// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <stdexcept>

// num_splits should not be bigger than 64
#define MAX_SPLITS_SWITCH(num_splits, CONST_NAME, ...)            \
  [&] {                                                           \
    if (num_splits <= 16) {                                       \
      constexpr ck_tile::index_t CONST_NAME = 16;                 \
      __VA_ARGS__();                                              \
    } else if (num_splits <= 32) {                                \
      constexpr ck_tile::index_t CONST_NAME = 32;                 \
      __VA_ARGS__();                                              \
    } else if (num_splits <= 64) {                                \
      constexpr ck_tile::index_t CONST_NAME = 64;                 \
      __VA_ARGS__();                                              \
    } else {                                                      \
      throw std::runtime_error("num_splits size not supported!"); \
    }                                                             \
  }()
