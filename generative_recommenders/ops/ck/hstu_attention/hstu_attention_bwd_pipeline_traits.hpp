// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>

namespace ck_tile {

template <
    bool kPadSeqLenQ_,
    bool kPadSeqLenK_,
    bool kPadHeadDimQK_,
    bool kPadHeadDimV_,
    index_t kBlockPerCuForKernel1_,
    index_t kBlockPerCuForKernel2_>
struct HstuAttentionBwdTraits {
  static constexpr bool kPadSeqLenQ = kPadSeqLenQ_;
  static constexpr bool kPadSeqLenK = kPadSeqLenK_;
  static constexpr bool kPadHeadDimQK = kPadHeadDimQK_;
  static constexpr bool kPadHeadDimV = kPadHeadDimV_;

  static constexpr index_t kBlockPerCuForKernel1 = kBlockPerCuForKernel1_;
  static constexpr index_t kBlockPerCuForKernel2 = kBlockPerCuForKernel2_;
};

} // namespace ck_tile
