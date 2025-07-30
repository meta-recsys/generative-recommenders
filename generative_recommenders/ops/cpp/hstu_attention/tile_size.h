/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
 *Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

namespace hstu {

constexpr int kBlockM_bwd(
    const int arch,
    const int headdim,
    const bool causal,
    const bool is_local) {
  int const kBlockM_sm90 = headdim <= 64
      ? 64
      : (headdim <= 96
             ? 64
             : (headdim <= 128 ? (causal || is_local ? 64 : 80) : 64));
  int const kBlockM_sm80 = headdim <= 64 ? 128 : 64;
  int const kBlockM = arch >= 90 ? kBlockM_sm90 : kBlockM_sm80;
  return kBlockM;
}

constexpr int kBlockN_bwd(const int arch, const int headdim) {
  int const kBlockN_sm90 = headdim <= 128 ? 128 : (headdim <= 192 ? 96 : 80);
  int const kBlockN_sm80 = headdim <= 128 ? 128 : (headdim <= 192 ? 80 : 64);
  int const kBlockN = arch >= 90 ? kBlockN_sm90 : kBlockN_sm80;
  return kBlockN;
}

constexpr int NumMmaWarpGroups_bwd(const int arch, const int headdim) {
  if (headdim <= 128) {
    return 2;
  } else if (headdim == 192) {
    return arch >= 90 ? 3 : 2;
  } else {
    return 2;
  }
}

constexpr bool V_in_regs_bwd(const int arch, const int headdim) {
  if (arch >= 90 && headdim == 96) {
    return true;
  }
  return false;
}

// Stages_dO, Stages_dS_or_QSm80
constexpr std::tuple<int, int> Stages_bwd(const int arch, const int headdim) {
  if (headdim <= 128) {
    return {2, 2};
  }
  if (headdim == 192) {
    if (arch >= 90) {
      return {1, 1};
    } else {
      return {1, 2};
    }
  } else {
    return {1, 1};
  }
}

// AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ
constexpr std::tuple<int, int, int> AtomLayout_bwd(
    const int arch,
    const int headdim) {
  if (headdim <= 64) {
    if (arch >= 90) {
      return {1, 2, 1};
    } else {
      return {4, 4, 4};
    }
  } else if (headdim <= 96) {
    if (arch >= 90) {
      return {1, 2, 1};
    } else {
      return {2, 4, 2};
    }
  } else if (headdim <= 128) {
    if (arch >= 90) {
      return {1, 2, 1};
    } else {
      return {2, 2, 2};
    }
  } else {
    if (arch >= 90) {
      return {1, 1, 1};
    } else {
      return {4, 2, 2};
    }
  }
}

// SdP_swapAB, dKV_swapAB, dQ_swapAB
constexpr std::tuple<bool, bool, bool> swapAB_bwd(
    const int arch,
    const int headdim,
    const bool causal,
    const bool local) {
  if (headdim <= 96) {
    return {arch >= 90 ? true : false, false, false};
  } else if (headdim == 128) {
    bool SdP_swapAB = arch >= 90 ? true : false;
    bool dKV_swapAB = false;
    bool dQ_swapAB = arch >= 90 ? ((causal || local) ? false : true) : false;
    return {SdP_swapAB, dKV_swapAB, dQ_swapAB};
  } else if (headdim == 192) {
    return {false, true, false};
  } else {
    return {false, arch >= 90 ? true : false, arch >= 90 ? true : false};
  }
}

// Return {kBlockM, kBlockN, Mma1_is_RS}
constexpr std::tuple<int, int, bool> tile_size_fwd_sm90(
    int headdim,
    bool is_causal,
    bool is_local,
    int element_size = 2,
    bool v_colmajor = false,
    bool Cross = false,
    bool Training = true) {
  // for cross attention, q is usually much smaller than k/v, so we reduce the
  // BlockM size to increase parallelism
  bool small_blockm = Cross && (!Training);
  if (element_size == 2) {
    if (headdim <= 64) {
      return {small_blockm ? 64 : 192, 128, true};
      // Good for long seqlen (>= 4k) but suffers from tile quantization at
      // short seqlen return {192, is_causal || is_local ? 192 : 176, true,
      // false};
    } else if (headdim <= 96) {
      return {small_blockm ? 64 : 192, is_local ? 128 : 144, false};
    } else if (headdim <= 128) {
      return {small_blockm ? 64 : 128, is_causal || is_local ? 128 : 176, true};
      // {128, 192, false, false} and {192, 128, false, true} are quite good too
      // 128 x 192 hits the limit of smem if Mma1_is_RS, 128 x 144 hits the
      // limit if !Mma1_is_RS
    } else if (headdim <= 192) {
      return {
          small_blockm ? 64 : 128,
          is_local ? 96 : 112,
          true}; // 128 x 112 hits the limit of smem
    } else {
      return {
          small_blockm ? 64 : 128,
          is_local ? 64 : 80,
          true}; // 128 x 80 hits the limit of smem
    }
  } else {
    if (headdim <= 64) {
      return {192, 160, true};
    } else if (headdim <= 96) {
      return {192, 128, true};
    } else if (headdim <= 128) {
      return {128, (v_colmajor ? 192 : 224), true};
    } else if (headdim <= 192) {
      return {128, 160, true};
    } else {
      return {128, is_local ? 64 : 128, true};
    }
  }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
    bool sm86_or_89,
    int headdim,
    bool is_causal,
    bool is_local,
    int element_size = 2) {
  if (element_size == 2) {
    if (headdim <= 64) {
      return {128, (is_local ? 96 : 112), 4, 1, false};
    } else if (headdim <= 96) {
      return {128, is_local ? 48 : 64, 4, 1, false};
    } else if (headdim <= 128) {
      bool const use_8_warps = sm86_or_89;
      return {
          128,
          use_8_warps ? (is_local ? 96 : 128) : (is_local ? 48 : 64),
          use_8_warps ? 8 : 4,
          1,
          use_8_warps};
    } else if (headdim <= 192) {
      bool const kBlockN_64 = is_local;
      return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
    } else {
      return {
          128,
          sm86_or_89 ? (is_local ? 48 : 64) : (is_local ? 64 : 96),
          8,
          1,
          false};
    }
  } else {
    // Placeholder for now
    return {128, 64, 8, 2, false};
  }
}
} // namespace hstu
