// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstring>

#include <hip/hip_runtime.h>

#include "hstu_attention_fwd_strategy_core.hpp"
#include "hstu_attention_host_util.hpp"

// Jagged forward tile / split-kv / grid-order selection on the current device:
// the first rule of hstu_fwd_strategy_for_arch() whose match accepts the call
// picks the config, else the max/min seqlen heuristics do.

static inline const char* get_hstu_fwd_arch() {
  static thread_local int cached_device = -1;
  static thread_local char arch[64] = {0};
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    return "";
  if (device != cached_device) {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return "";
    // gcnArchName carries feature suffixes, e.g. "gfx950:sramecc+:xnack-"
    const size_t n =
        std::min(std::strcspn(props.gcnArchName, ":"), sizeof(arch) - 1);
    std::memcpy(arch, props.gcnArchName, n);
    arch[n] = '\0';
    cached_device = device;
  }
  return arch;
}

static inline int select_hstu_fwd_config_on_device(
    const HstuFwdStrategy& strategy,
    const HstuFwdCallFeatures& f,
    HstuFwdConfig& cfg) {
  return select_hstu_fwd_config(
      strategy, f, get_hstu_fwd_arch(), get_number_of_cu(), cfg);
}
