// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"

#include "instances/hstu_attention_group_forward_fp16_instances_ref.hpp"

void hstu_attention_group_forward_fp16(
    HstuAttentionGroupFwdParams& param,
    hipStream_t stream) {
  const bool use_causal = param.use_causal;
  const bool store_lse = (param.use_softmax && param.is_training);

  constexpr bool kHasBias = false;
  constexpr bool kHasDropout = false;
  BOOL_SWITCH_2(use_causal, kUseCausal, param.use_softmax, kUseSoftmax, [&] {
    HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
      BOOL_SWITCH(store_lse, kStoreLSE, [&] {
        if constexpr (kUseSoftmax || !kStoreLSE) {
          run_group_forward_dispatch<
              ck_tile::fp16_t,
              kUseCausal,
              kUseSoftmax,
              kStoreLSE,
              kHasBias,
              kHasDropout,
              MaxK>(param, stream);
        }
      });
    });
  });
};
