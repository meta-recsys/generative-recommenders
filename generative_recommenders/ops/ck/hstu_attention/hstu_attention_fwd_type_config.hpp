// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/bfloat16.hpp>
#include <ck_tile/core/numeric/half.hpp>

template <typename InOutDataType>
struct HstuAttentionFwdTypeConfig;

template <>
struct HstuAttentionFwdTypeConfig<ck_tile::fp16_t> {
  using BiasDataType = ck_tile::fp16_t;
  using GemmAccDataType = float;
  using CompDataType = float; // data type for non-linear calculation
  using OaccDataType = GemmAccDataType;
  using ODataType = ck_tile::fp16_t;
};

template <>
struct HstuAttentionFwdTypeConfig<ck_tile::bf16_t> {
  using BiasDataType = ck_tile::bf16_t;
  using GemmAccDataType = float;
  using CompDataType = float; // data type for non-linear calculation
  using OaccDataType = GemmAccDataType;
  using ODataType = ck_tile::bf16_t;
};
