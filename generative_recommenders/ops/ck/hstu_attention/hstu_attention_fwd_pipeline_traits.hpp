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
    index_t kBlockPerCu_>
struct HstuAttentionFwdTraits {
  static constexpr bool kPadSeqLenQ = kPadSeqLenQ_;
  static constexpr bool kPadSeqLenK = kPadSeqLenK_;
  static constexpr bool kPadHeadDimQK = kPadHeadDimQK_;
  static constexpr bool kPadHeadDimV = kPadHeadDimV_;

  static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

template <
    bool kPadSeqLenQ_ /* padding for seqlen_q */,
    bool kPadHeadDimO_ /* padding for hdim_o */,
    bool kPadNumSplits_ = false, /* padding for num_splits */
    index_t kBlockPerCu_ = -1 /* overwrite occupancy if not -1 */>
struct HstuAttentionFwdSplitKVCombineTraits {
  static constexpr bool kPadSeqLenQ = kPadSeqLenQ_;
  static constexpr bool kPadHeadDimO = kPadHeadDimO_;
  static constexpr bool kPadNumSplits = kPadNumSplits_;

  static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace ck_tile
