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

#pragma once

#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
  using index_t = int64_t;
  // The QKV matrices.
  void* __restrict__ q_ptr;
  void* __restrict__ k_ptr;
  void* __restrict__ v_ptr;

  // The stride between rows of the Q, K and V matrices.
  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t v_dim_stride;

  // The number of heads.
  int h;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
  using index_t = int64_t;

  // The O matrix (output).
  void* __restrict__ o_ptr;

  // The stride between rows of O.
  index_t o_batch_stride;
  index_t o_row_stride;
  index_t o_head_stride;

  // For FP8 scaling
  float* __restrict__ q_descale_ptr;
  float* __restrict__ k_descale_ptr;
  float* __restrict__ v_descale_ptr;
  index_t q_descale_batch_stride;
  index_t q_descale_head_stride;
  index_t k_descale_batch_stride;
  index_t k_descale_head_stride;
  index_t v_descale_batch_stride;
  index_t v_descale_head_stride;

  // The dimensions.
  int b, max_seq_len, qk_d, v_d, total_seq_len;

  // The scaling factors for the kernel.
  float alpha;

  int* __restrict__ seq_offsets;
  int* __restrict__ num_targets;
  float* __restrict__ attn_scale;

  // Local window size
  int max_attn_len, contextual_seq_len, min_full_attn_seq_len;

  // Pointer to the RNG seed (idx 0) and offset (idx 1).
  uint64_t* rng_state;

  bool is_bf16;
  bool is_fp32;
  bool is_e4m3;
  bool is_causal;
  bool is_local;
  bool has_contexual_mask;

  int* __restrict__ tile_count_semaphore;

  int arch;
  int num_sm;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {
  using index_t = int64_t;

  // The dO and dQKV matrices.
  void* __restrict__ do_ptr;
  void* __restrict__ dq_ptr;
  void* __restrict__ dk_ptr;
  void* __restrict__ dv_ptr;

  // To accumulate dQ
  void* __restrict__ dq_accum_ptr;
  int* __restrict__ dq_semaphore;

  // The stride between rows of the dO, dQ, dK and dV matrices.
  index_t do_batch_stride;
  index_t do_row_stride;
  index_t do_head_stride;
  index_t dq_batch_stride;
  index_t dk_batch_stride;
  index_t dv_batch_stride;
  index_t dq_row_stride;
  index_t dk_row_stride;
  index_t dv_row_stride;
  index_t dq_head_stride;
  index_t dk_head_stride;
  index_t dv_head_stride;

  int* __restrict__ sort_by_length_indices;

  int max_seq_len_rounded, qk_d_rounded, v_d_rounded;

  bool deterministic;
  index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Arch, typename T, int Headdim>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);
template <int Arch, typename T, int Headdim>
void run_mha_bwd_(Flash_bwd_params& params, cudaStream_t stream);
