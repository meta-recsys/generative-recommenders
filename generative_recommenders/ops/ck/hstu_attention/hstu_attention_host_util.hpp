// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <ck_tile/host/hip_check_error.hpp>
#include "hstu_attention_params.hpp"

#define HSTU_CHECK(COND, ERR)              \
  if (!(COND)) {                           \
    std::ostringstream ostr;               \
    ostr << "'" #COND "' failed: " << ERR; \
    throw std::runtime_error(ostr.str());  \
  }

// Cached per thread and device: the dispatch heuristics call this several times
// per launch.
static inline int get_number_of_cu() {
  static thread_local int cached_device = -1;
  static thread_local int cached_cus = 0;

  int device;

  HIP_CHECK_ERROR(hipGetDevice(&device));

  if (device != cached_device) {
    hipDeviceProp_t props;

    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, device));

    cached_cus = props.multiProcessorCount;
    cached_device = device;
  }
  return cached_cus;
}

static inline bool is_almost_invariant_seqlen(
    HstuAttentionNoGroupFwdParams& param) {
  float threshold = 0.7f;

  if (param.is_jagged) {
    bool res = (static_cast<float>(param.min_seqlen_q) / param.max_seqlen_q) >
        threshold;
    if (param.is_cross_attention) {
      res = res &&
          ((static_cast<float>(param.min_seqlen_kv) / param.max_seqlen_kv) >
           threshold);
    }

    return res;
  } else {
    return true;
  }
};

static inline bool is_almost_invariant_seqlen(
    HstuAttentionGroupFwdParams& param) {
  float threshold = 0.7f;

  bool res =
      (static_cast<float>(param.min_seqlen_q) / param.max_seqlen_q) > threshold;
  if (param.is_cross_attention) {
    res = res &&
        ((static_cast<float>(param.min_seqlen_kv) / param.max_seqlen_kv) >
         threshold);
  }

  return res;
};

static inline bool is_almost_invariant_seqlen_q(
    HstuAttentionNoGroupFwdParams& param) {
  float threshold = 0.7f;

  if (param.is_jagged) {
    bool res = (static_cast<float>(param.min_seqlen_q) / param.max_seqlen_q) >
        threshold;

    return res;
  } else {
    return true;
  }
};

static inline bool is_almost_invariant_seqlen_q(
    HstuAttentionGroupFwdParams& param) {
  float threshold = 0.7f;

  bool res =
      (static_cast<float>(param.min_seqlen_q) / param.max_seqlen_q) > threshold;

  return res;
};

static inline bool is_almost_invariant_seqlen(
    HstuAttentionNoGroupBwdParams& param) {
  float threshold = 0.7f;

  if (param.is_jagged) {
    bool res = (static_cast<float>(param.min_seqlen_q) / param.max_seqlen_q) >
        threshold;
    if (param.is_cross_attention) {
      res = res &&
          ((static_cast<float>(param.min_seqlen_kv) / param.max_seqlen_kv) >
           threshold);
    }

    return res;
  } else {
    return true;
  }
};

static inline bool is_almost_invariant_seqlen(
    HstuAttentionGroupBwdParams& param) {
  float threshold = 0.7f;

  bool res =
      (static_cast<float>(param.min_seqlen_q) / param.max_seqlen_q) > threshold;
  if (param.is_cross_attention) {
    res = res &&
        ((static_cast<float>(param.min_seqlen_kv) / param.max_seqlen_kv) >
         threshold);
  }

  return res;
};
