// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#define BOOL_SWITCH_RETURN(COND1, CONST_NAME1, ...) \
  [&] {                                             \
    if (COND1) {                                    \
      constexpr bool CONST_NAME1 = true;            \
      return __VA_ARGS__();                         \
    } else {                                        \
      constexpr bool CONST_NAME1 = false;           \
      return __VA_ARGS__();                         \
    }                                               \
  }()

#define BOOL_SWITCH_RETURN_2(COND1, CONST_NAME1, COND2, CONST_NAME2, ...) \
  [&] {                                                                   \
    if (COND1) {                                                          \
      constexpr bool CONST_NAME1 = true;                                  \
      return BOOL_SWITCH_RETURN(COND2, CONST_NAME2, ##__VA_ARGS__);       \
    } else {                                                              \
      constexpr bool CONST_NAME1 = false;                                 \
      return BOOL_SWITCH_RETURN(COND2, CONST_NAME2, ##__VA_ARGS__);       \
    }                                                                     \
  }()

#define BOOL_SWITCH_RETURN_3(                                        \
    COND1, CONST_NAME1, COND2, CONST_NAME2, COND3, CONST_NAME3, ...) \
  [&] {                                                              \
    if (COND1) {                                                     \
      constexpr bool CONST_NAME1 = true;                             \
      return BOOL_SWITCH_RETURN_2(                                   \
          COND2, CONST_NAME2, COND3, CONST_NAME3, ##__VA_ARGS__);    \
    } else {                                                         \
      constexpr bool CONST_NAME1 = false;                            \
      return BOOL_SWITCH_RETURN_2(                                   \
          COND2, CONST_NAME2, COND3, CONST_NAME3, ##__VA_ARGS__);    \
    }                                                                \
  }()
