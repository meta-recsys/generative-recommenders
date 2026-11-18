// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/bfloat16.hpp>
#include <ck_tile/core/numeric/half.hpp>

template <typename InOutDataType>
struct HstuAttentionBwdTypeConfig;

template <>
struct HstuAttentionBwdTypeConfig<ck_tile::fp16_t> {
  using BiasDataType = ck_tile::fp16_t;
  using GemmAccDataType = float;
  using CompDataType = float; // data type for non-linear calculation
};

template <>
struct HstuAttentionBwdTypeConfig<ck_tile::bf16_t> {
  using BiasDataType = ck_tile::bf16_t;
  using GemmAccDataType = float;
  using CompDataType = float; // data type for non-linear calculation
};
