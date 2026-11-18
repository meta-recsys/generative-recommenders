// @noautodeps
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_hdim_switch.hpp"

#include "instances/hstu_attention_group_backward_fp16_instances_ref.hpp"

void hstu_attention_group_backward_fp16(
    HstuAttentionGroupBwdParams& param,
    hipStream_t stream) {
  constexpr bool kHasBias = false;
  constexpr bool kHasDropout = false;
  BOOL_SWITCH_2(
      param.use_causal, kUseCausal, param.use_softmax, kUseSoftmax, [&] {
        HDIM_SWITCH(param.hdim_qk, param.hdim_v, MaxK, [&] {
          run_group_backward_dispatch<
              ck_tile::fp16_t,
              kUseCausal,
              kUseSoftmax,
              kHasBias,
              kHasDropout,
              MaxK>(param, stream);
        });
      });
}
