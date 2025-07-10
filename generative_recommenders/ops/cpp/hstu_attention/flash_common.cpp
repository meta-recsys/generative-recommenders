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

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/nn/functional.h>
#include <torch/version.h> // For TORCH_VERSION* macros

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "flash_common.h"
#include "static_switch.h"
#include "tile_size.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                           \
  TORCH_CHECK(                                        \
      x.sizes() == torch::IntArrayRef({__VA_ARGS__}), \
      #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor switch_to_contiguous_if_needed(const at::Tensor& x) {
  if (x.stride(x.dim() - 1) == 1) {
    return x;
  }
  return x.contiguous();
}

void set_params_fprop(
    Flash_fwd_params& params,
    // sizes
    const size_t b,
    const size_t total_seq_len,
    const size_t max_seq_len,
    const size_t h,
    const size_t qk_d,
    const size_t v_d,
    // device pointers
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    void* seq_offsets,
    void* num_targets,
    void* attn_scale,
    bool causal,
    float alpha,
    const bool scalar_scale,
    const int max_attn_len,
    const int min_full_attn_seq_len,
    const int contextual_seq_len,
    const int sm_margin = 0) {
  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;
  params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.v_dim_stride = v.stride(-1);

  if (seq_offsets == nullptr) {
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
  }

  params.seq_offsets = static_cast<int*>(seq_offsets);
  params.num_targets = static_cast<int*>(num_targets);
  params.attn_scale = static_cast<float*>(attn_scale);

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.total_seq_len = total_seq_len;
  params.max_seq_len = max_seq_len;
  params.qk_d = qk_d;
  params.v_d = v_d;

  params.alpha = alpha;

  params.is_local = max_attn_len > 0;
  params.is_causal = causal && (!params.is_local);
  params.has_contexual_mask = contextual_seq_len > 0;
  params.scalar_scale = scalar_scale;

  params.max_attn_len = max_attn_len;
  params.min_full_attn_seq_len = min_full_attn_seq_len;
  params.contextual_seq_len = contextual_seq_len;

  params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 +
      at::cuda::getCurrentDeviceProperties()->minor;
  params.num_sm =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(
      !params.is_local,
      "This flash attention build does not support local attention.");
#endif
}

void set_params_dgrad(
    Flash_bwd_params& params,
    // sizes
    const size_t b,
    const size_t total_seq_len,
    const size_t max_seq_len,
    const size_t max_seq_len_q_rounded,
    const size_t h,
    const size_t qk_d,
    const size_t v_d,
    const size_t qk_d_rounded,
    const size_t v_d_rounded,
    // device pointers
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& dout,
    const at::Tensor& dq,
    const at::Tensor& dk,
    const at::Tensor& dv,
    void* dq_accum_d,
    void* seq_offsets,
    void* num_targets,
    void* attn_scale,
    void* sort_by_length_indices,
    const bool scalar_scale,
    const bool causal,
    const float alpha,
    const int max_attn_len,
    const int min_full_attn_seq_len,
    const int contextual_seq_len,
    bool deterministic = false,
    int const sm_margin = 0) {
  set_params_fprop(
      params,
      b,
      total_seq_len,
      max_seq_len,
      h,
      qk_d,
      v_d,
      q,
      k,
      v,
      seq_offsets,
      num_targets,
      attn_scale,
      causal,
      alpha,
      scalar_scale,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      sm_margin);

  // Set the pointers and strides.
  params.do_ptr = dout.data_ptr();
  params.do_row_stride = dout.stride(-3);
  params.do_head_stride = dout.stride(-2);
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.dq_row_stride = dq.stride(-3);
  params.dk_row_stride = dk.stride(-3);
  params.dv_row_stride = dv.stride(-3);
  params.dq_head_stride = dq.stride(-2);
  params.dk_head_stride = dk.stride(-2);
  params.dv_head_stride = dv.stride(-2);

  params.qk_d_rounded = qk_d_rounded;
  params.v_d_rounded = v_d_rounded;
  params.max_seq_len_rounded = max_seq_len_q_rounded;

  params.sort_by_length_indices = static_cast<int*>(sort_by_length_indices);

  if (seq_offsets == nullptr) {
    params.do_batch_stride = dout.stride(0);
    params.dq_batch_stride = dq.stride(0);
    params.dk_batch_stride = dk.stride(0);
    params.dv_batch_stride = dv.stride(0);
  }
  params.dq_accum_ptr = dq_accum_d;
  params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  // HEADDIM_SWITCH(params.d, [&] {
  //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
  // });
  ARCH_SWITCH(params.arch, Arch, [&] {
    if (!params.is_e4m3) {
      if (params.is_bf16) {
#ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.qk_d <= 64) {
          return run_mha_fwd_<Arch, cutlass::bfloat16_t, 64>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.qk_d <= 96) {
          return run_mha_fwd_<Arch, cutlass::bfloat16_t, 96>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.qk_d <= 128) {
          return run_mha_fwd_<Arch, cutlass::bfloat16_t, 128>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.qk_d <= 192) {
          return run_mha_fwd_<Arch, cutlass::bfloat16_t, 192>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.qk_d <= 256) {
          return run_mha_fwd_<Arch, cutlass::bfloat16_t, 256>(params, stream);
        }
#endif
      } else {
#ifndef FLASHATTENTION_DISABLE_FP16
#ifndef FLASHATTENTION_DISABLE_HDIM64
        if (params.qk_d <= 64) {
          return run_mha_fwd_<Arch, cutlass::half_t, 64>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
        if (params.qk_d <= 96) {
          return run_mha_fwd_<Arch, cutlass::half_t, 96>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
        if (params.qk_d <= 128) {
          return run_mha_fwd_<Arch, cutlass::half_t, 128>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
        if (params.qk_d <= 192) {
          return run_mha_fwd_<Arch, cutlass::half_t, 192>(params, stream);
        }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
        if (params.qk_d <= 256) {
          return run_mha_fwd_<Arch, cutlass::half_t, 256>(params, stream);
        }
#endif
#else
                                TORCH_CHECK(false, "This flash attention build does not support FP16.");
#endif
      }
    } else {
#ifndef FLASHATTENTION_DISABLE_FP8
#ifndef FLASHATTENTION_DISABLE_HDIM64
      if (params.qk_d <= 64) {
        return run_mha_fwd_<90, cutlass::float_e4m3_t, 64>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
      if (params.qk_d <= 96) {
        return run_mha_fwd_<90, cutlass::float_e4m3_t, 96>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
      if (params.qk_d <= 128) {
        return run_mha_fwd_<90, cutlass::float_e4m3_t, 128>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
      if (params.qk_d <= 192) {
        return run_mha_fwd_<90, cutlass::float_e4m3_t, 192>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
      if (params.qk_d <= 256) {
        return run_mha_fwd_<90, cutlass::float_e4m3_t, 256>(params, stream);
      }
#endif
#else
                            TORCH_CHECK(false, "This flash attention build does not support FP8.");
#endif
    }
  });
}

namespace hstu {

at::Tensor hstu_mha_fwd(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
    const std::optional<at::Tensor>& k_descale, // (b, h_k)
    const std::optional<at::Tensor>& v_descale, // (b, h_k)
    int64_t const sm_margin = 0) {
  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major >= 8;
  TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

  q = switch_to_contiguous_if_needed(q);
  k = switch_to_contiguous_if_needed(k);
  v = switch_to_contiguous_if_needed(v);

  auto q_type = q.scalar_type();
  TORCH_CHECK(
      q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 ||
          q_type == at::ScalarType::Float8_e4m3fn,
      "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
  if (dprops->major < 9) {
    TORCH_CHECK(
        q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
        "FlashAttention on Ampere/Ada cards only supports fp16 and bf16 data type");
  }
  TORCH_CHECK(
      k.scalar_type() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(
      v.scalar_type() == q_type, "query and value must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);

  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

  at::Tensor seq_offsets_;
  bool const is_jagged = seq_offsets.has_value();
  if (is_jagged) {
    seq_offsets_ = seq_offsets.value();
    CHECK_DEVICE(seq_offsets_);
    CHECK_CONTIGUOUS(seq_offsets_);
    TORCH_CHECK(
        seq_offsets_.dtype() == torch::kInt32,
        "seq_offsets_ must have dtype torch.int32");
  }
  at::Tensor num_targets_;
  bool const has_multiple_targets = num_targets.has_value();
  if (has_multiple_targets) {
    num_targets_ = num_targets.value();
    CHECK_DEVICE(num_targets_);
    CHECK_CONTIGUOUS(num_targets_);
    TORCH_CHECK(
        num_targets_.dtype() == torch::kInt32,
        "num_targets_ must have dtype torch.int32");
  }
  at::Tensor attn_scale_;
  bool scalar_scale = true;
  bool const has_attn_scale = attn_scale.has_value();
  if (has_attn_scale) {
    attn_scale_ = attn_scale.value();
    scalar_scale = attn_scale_.numel() == 1;
    CHECK_DEVICE(attn_scale_);
    TORCH_CHECK(
        attn_scale_.dtype() == torch::kFloat32,
        "attn_scale_ must have dtype torch.float32");
  }
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
  if (is_jagged && has_multiple_targets) {
    auto uih_lengths = seq_offsets_.slice(0, 1)
                           .sub(seq_offsets_.slice(0, 0, -1))
                           .sub(num_targets_);
    TORCH_CHECK(
        (uih_lengths.gt(0)).sum().item<int64_t>() == num_targets_.size(0),
        "some uih seqlen is 0");
    TORCH_CHECK(
        (uih_lengths.greater_equal(contextual_seq_len)).sum().item<int64_t>() ==
            num_targets_.size(0),
        "some uih seqlen is less than contextual_seq_len");
  }
#endif
  TORCH_CHECK(
      q.size(-1) == k.size(-1) && k.size(-1) == v.size(-1),
      "only attndim == hidden_dim is supported");

  auto const sizes = q.sizes();
  const int batch_size = !is_jagged ? sizes[0] : seq_offsets_.size(0) - 1;
  int total_seq_len = !is_jagged ? batch_size * max_seq_len : sizes[0];
  int num_heads = q.size(-2);
  int const qk_head_size = q.size(-1);
  int const v_head_size = v.size(-1);
  int const max_headdim = get_max_headdim();
  TORCH_CHECK(
      qk_head_size <= max_headdim && v_head_size <= max_headdim,
      "FlashAttention forward only supports head dimension at most " +
          std::to_string(max_headdim));
  TORCH_CHECK(max_attn_len >= 0, "max_attn_len must be at least 0");
  TORCH_CHECK(
      min_full_attn_seq_len >= 0, "min_full_attn_seq_len must be at least 0");
  TORCH_CHECK(contextual_seq_len >= 0, "contextual_seq_len must be at least 0");
  if (max_attn_len > 0) {
    TORCH_CHECK(
        min_full_attn_seq_len > 0,
        "min_full_attn_seq_len=0 not supported when max_attn_len > 0");
  }
  if (!is_jagged) {
    CHECK_SHAPE(q, batch_size, max_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(k, batch_size, max_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(v, batch_size, max_seq_len, num_heads, v_head_size);
  } else {
    CHECK_SHAPE(q, total_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(k, total_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(v, total_seq_len, num_heads, v_head_size);
    CHECK_SHAPE(seq_offsets_, batch_size + 1);
  }
  if (has_multiple_targets) {
    CHECK_SHAPE(num_targets_, batch_size);
  }

  int const alignment = q_type == torch::kFloat8_e4m3fn ? 16 : 8;
  TORCH_CHECK(
      qk_head_size % alignment == 0 && v_head_size % alignment == 0,
      "head_size should be a multiple of " + std::to_string(alignment));

  auto opts = q.options();
  auto out_type = q_type == at::ScalarType::Float8_e4m3fn
      ? at::ScalarType::BFloat16
      : q_type;
  at::Tensor out;
  if (!is_jagged) {
    out = torch::empty(
        {batch_size, max_seq_len, num_heads, v_head_size},
        opts.dtype(out_type));
  } else {
    out = torch::empty(
        {total_seq_len, num_heads, v_head_size}, opts.dtype(out_type));
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  Flash_fwd_params params;
  set_params_fprop(
      params,
      batch_size,
      total_seq_len,
      max_seq_len,
      num_heads,
      qk_head_size,
      v_head_size,
      q,
      k,
      v,
      !is_jagged ? nullptr : seq_offsets_.data_ptr(),
      !has_multiple_targets ? nullptr : num_targets_.data_ptr(),
      !has_attn_scale ? nullptr : attn_scale_.data_ptr(),
      causal,
      alpha,
      scalar_scale,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      sm_margin);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);
  if (!is_jagged) {
    params.o_batch_stride = out.stride(0);
  }
  at::Tensor tile_count_semaphore;
  // We don't use the persistent scheduler if not jagged
  bool const persistent_scheduler = params.arch >= 90
      ? (params.is_causal || params.is_local || is_jagged)
      : (params.is_causal || is_jagged);
  if (persistent_scheduler) {
    tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
  } else {
    params.tile_count_semaphore = nullptr;
  }

  if (q_type == at::ScalarType::Float8_e4m3fn) {
    if (q_descale.has_value()) {
      auto q_descale_ = q_descale.value();
      CHECK_DEVICE(q_descale_);
      CHECK_SHAPE(q_descale_, batch_size, num_heads);
      params.q_descale_ptr = q_descale_.data_ptr<float>();
      params.q_descale_batch_stride = q_descale_.stride(0);
      params.q_descale_head_stride = q_descale_.stride(1);
    } else {
      params.q_descale_ptr = nullptr;
    }
    if (k_descale.has_value()) {
      auto k_descale_ = k_descale.value();
      CHECK_DEVICE(k_descale_);
      CHECK_SHAPE(k_descale_, batch_size, num_heads);
      params.k_descale_ptr = k_descale_.data_ptr<float>();
      params.k_descale_batch_stride = k_descale_.stride(0);
      params.k_descale_head_stride = k_descale_.stride(1);
    } else {
      params.k_descale_ptr = nullptr;
    }
    if (v_descale.has_value()) {
      auto v_descale_ = v_descale.value();
      CHECK_DEVICE(v_descale_);
      CHECK_SHAPE(v_descale_, batch_size, num_heads);
      params.v_descale_ptr = v_descale_.data_ptr<float>();
      params.v_descale_batch_stride = v_descale_.stride(0);
      params.v_descale_head_stride = v_descale_.stride(1);
    } else {
      params.v_descale_ptr = nullptr;
    }
  }

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(
      !params.is_local,
      "This flash attention build does not support local attention.");
#endif

  if (total_seq_len > 0 && num_heads > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);
  }
  return out;
}

void run_mha_bwd(Flash_bwd_params& params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
  // FP16_SWITCH(!params.is_bf16, [&] {
  //     HEADDIM_SWITCH(params.d, [&] {
  //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
  //     });
  // });
  ARCH_SWITCH(params.arch, Arch, [&] {
    if (!params.is_bf16) {
#ifndef FLASHATTENTION_DISABLE_FP16
#ifndef FLASHATTENTION_DISABLE_HDIM64
      if (params.qk_d <= 64) {
        return run_mha_bwd_<Arch, cutlass::half_t, 64>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
      if (params.qk_d <= 96) {
        return run_mha_bwd_<Arch, cutlass::half_t, 96>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
      if (params.qk_d <= 128) {
        return run_mha_bwd_<Arch, cutlass::half_t, 128>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
      if (params.qk_d <= 192) {
        return run_mha_bwd_<Arch, cutlass::half_t, 192>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
      if (params.qk_d <= 256) {
        return run_mha_bwd_<Arch, cutlass::half_t, 256>(params, stream);
      }
#endif
#else
                TORCH_CHECK(false, "This flash attention build does not support FP16.");
#endif
    } else {
#ifndef FLASHATTENTION_DISABLE_HDIM64
      if (params.qk_d <= 64) {
        return run_mha_bwd_<Arch, cutlass::bfloat16_t, 64>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM96
      if (params.qk_d <= 96) {
        return run_mha_bwd_<Arch, cutlass::bfloat16_t, 96>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM128
      if (params.qk_d <= 128) {
        return run_mha_bwd_<Arch, cutlass::bfloat16_t, 128>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM192
      if (params.qk_d <= 192) {
        return run_mha_bwd_<Arch, cutlass::bfloat16_t, 192>(params, stream);
      }
#endif
#ifndef FLASHATTENTION_DISABLE_HDIM256
      if (params.qk_d <= 256) {
        return run_mha_bwd_<Arch, cutlass::bfloat16_t, 256>(params, stream);
      }
#endif
    }
  });
#endif
}

std::vector<at::Tensor> hstu_mha_bwd(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& dout,
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    bool sort_by_length,
    bool const deterministic,
    int64_t const sm_margin = 0) {
#ifdef FLASHATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif

  auto dprops = at::cuda::getCurrentDeviceProperties();
  bool is_sm8x = dprops->major >= 8;
  TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

  q = switch_to_contiguous_if_needed(q);
  k = switch_to_contiguous_if_needed(k);
  v = switch_to_contiguous_if_needed(v);
  dout = switch_to_contiguous_if_needed(dout);

  auto q_type = q.dtype();
  TORCH_CHECK(
      q_type == torch::kFloat16 || q_type == torch::kBFloat16,
      "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_type, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_type, "query and value must have the same dtype");
  TORCH_CHECK(
      dout.dtype() == q_type, "query and dout must have the same dtype");

  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_DEVICE(dout);

  TORCH_CHECK(
      q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
  TORCH_CHECK(
      dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

  at::Tensor seq_offsets_;
  bool const is_jagged = seq_offsets.has_value();
  if (is_jagged) {
    seq_offsets_ = seq_offsets.value();
    CHECK_DEVICE(seq_offsets_);
    CHECK_CONTIGUOUS(seq_offsets_);
    TORCH_CHECK(
        seq_offsets_.dtype() == torch::kInt32,
        "seq_offsets_ must have dtype torch.int32");
  }
  at::Tensor sort_by_length_indices_;
  if (sort_by_length && is_jagged) {
    auto seq_lengths =
        seq_offsets_.slice(0, 1).sub(seq_offsets_.slice(0, 0, -1));
    std::tuple<torch::Tensor, torch::Tensor> sort_result = torch::sort(
        seq_lengths, false /*stable*/, 0 /*dim*/, true /*descending*/);
    sort_by_length_indices_ = std::get<1>(sort_result).to(torch::kInt32);
    CHECK_DEVICE(sort_by_length_indices_);
    CHECK_CONTIGUOUS(sort_by_length_indices_);
    TORCH_CHECK(
        sort_by_length_indices_.dtype() == torch::kInt32,
        "sort_by_length_indices_ must have dtype torch.int32");
  }
  at::Tensor num_targets_;
  bool const has_multiple_targets = num_targets.has_value();
  if (has_multiple_targets) {
    num_targets_ = num_targets.value();
    CHECK_DEVICE(num_targets_);
    CHECK_CONTIGUOUS(num_targets_);
    TORCH_CHECK(
        num_targets_.dtype() == torch::kInt32,
        "num_targets_ must have dtype torch.int32");
  }
  at::Tensor attn_scale_;
  bool scalar_scale = true;
  bool const has_attn_scale = attn_scale.has_value();
  if (has_attn_scale) {
    attn_scale_ = attn_scale.value();
    scalar_scale = attn_scale_.numel() == 1;
    CHECK_DEVICE(attn_scale_);
    TORCH_CHECK(
        attn_scale_.dtype() == torch::kFloat32,
        "attn_scale_ must have dtype torch.float32");
  }
  auto const sizes = q.sizes();
  int const batch_size = !is_jagged ? sizes[0] : seq_offsets_.size(0) - 1;
  if (!is_jagged) {
    max_seq_len = sizes[1];
  }
  int const total_seq_len = !is_jagged ? batch_size * sizes[1] : sizes[0];
  int const num_heads = q.size(-2);
  int const qk_head_size = q.size(-1);
  int const v_head_size = v.size(-1);
  TORCH_CHECK(
      qk_head_size % 8 == 0 && v_head_size % 8 == 0,
      "head_size should be a multiple of 8");
  int const max_headdim = get_max_headdim();
  TORCH_CHECK(
      qk_head_size <= max_headdim && v_head_size <= max_headdim,
      "FlashAttention backward only supports head dimension at most " +
          std::to_string(max_headdim));
  TORCH_CHECK(max_attn_len >= 0, "max_attn_len must be at least 0");
  TORCH_CHECK(
      min_full_attn_seq_len >= 0, "min_full_attn_seq_len must be at least 0");
  TORCH_CHECK(contextual_seq_len >= 0, "contextual_seq_len must be at least 0");
  if (!is_jagged) {
    CHECK_SHAPE(q, batch_size, max_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(k, batch_size, max_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(v, batch_size, max_seq_len, num_heads, v_head_size);
    CHECK_SHAPE(dout, batch_size, max_seq_len, num_heads, v_head_size);
    CHECK_SHAPE(dq, batch_size, max_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(dk, batch_size, max_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(dv, batch_size, max_seq_len, num_heads, v_head_size);
  } else {
    CHECK_SHAPE(q, total_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(k, total_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(v, total_seq_len, num_heads, v_head_size);
    CHECK_SHAPE(dout, total_seq_len, num_heads, v_head_size);
    CHECK_SHAPE(dq, total_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(dk, total_seq_len, num_heads, qk_head_size);
    CHECK_SHAPE(dv, total_seq_len, num_heads, v_head_size);
    CHECK_SHAPE(seq_offsets_, batch_size + 1);
  }
  if (has_multiple_targets) {
    CHECK_SHAPE(num_targets_, batch_size);
  }
  int const arch = at::cuda::getCurrentDeviceProperties()->major * 10 +
      at::cuda::getCurrentDeviceProperties()->minor;
  int const qk_head_size_rounded = round_up_headdim(qk_head_size);
  int const v_head_size_rounded = round_up_headdim(v_head_size);
  // Very important that these match the kernel configs
  bool const is_local = max_attn_len > 0;
  int const kBlockM = kBlockM_bwd(arch, qk_head_size_rounded, causal, is_local);
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  int const max_seq_len_q_rounded = round_multiple(max_seq_len, kBlockM);
  int const total_seq_len_q_padded_rounded =
      round_multiple(total_seq_len + batch_size * kBlockM, kBlockM);

  TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
  CHECK_DEVICE(dq);
  TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
  if (!is_jagged) {
    CHECK_SHAPE(dq, batch_size, max_seq_len, num_heads, qk_head_size);
  } else {
    CHECK_SHAPE(dq, total_seq_len, num_heads, qk_head_size);
  }
  TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
  CHECK_DEVICE(dk);
  TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
  if (!is_jagged) {
    CHECK_SHAPE(dk, batch_size, max_seq_len, num_heads, qk_head_size);
  } else {
    CHECK_SHAPE(dk, total_seq_len, num_heads, qk_head_size);
  }
  TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
  CHECK_DEVICE(dv);
  TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
  if (!is_jagged) {
    CHECK_SHAPE(dv, batch_size, max_seq_len, num_heads, v_head_size);
  } else {
    CHECK_SHAPE(dv, total_seq_len, num_heads, v_head_size);
  }

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  auto opts = q.options();

  at::Tensor dq_accum;
  if (!is_jagged) {
    dq_accum = torch::empty(
        {batch_size, num_heads, max_seq_len_q_rounded * qk_head_size_rounded},
        opts.dtype(at::kFloat));
  } else {
    dq_accum = torch::empty(
        {num_heads, total_seq_len_q_padded_rounded * qk_head_size_rounded},
        opts.dtype(at::kFloat));
  }

  Flash_bwd_params params;
  set_params_dgrad(
      params,
      batch_size,
      total_seq_len,
      max_seq_len,
      max_seq_len_q_rounded,
      num_heads,
      qk_head_size,
      v_head_size,
      qk_head_size_rounded,
      v_head_size_rounded,
      q,
      k,
      v,
      dout,
      dq,
      dk,
      dv,
      dq_accum.data_ptr(),
      !is_jagged ? nullptr : seq_offsets_.data_ptr(),
      !has_multiple_targets ? nullptr : num_targets_.data_ptr(),
      !has_attn_scale ? nullptr : attn_scale_.data_ptr(),
      !(sort_by_length && is_jagged) ? nullptr
                                     : sort_by_length_indices_.data_ptr(),
      scalar_scale,
      causal,
      alpha,
      max_attn_len,
      min_full_attn_seq_len,
      contextual_seq_len,
      deterministic,
      sm_margin);

  // auto tile_count_semaphore = (params.is_causal || params.is_local) ?
  // torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1},
  // opts.dtype(torch::kInt32)); params.tile_count_semaphore =
  // tile_count_semaphore.data_ptr<int>(); Will be zero'ed out in the
  // backward preprocess kernel
  at::Tensor dq_semaphore = torch::empty(
      {(max_seq_len + kBlockM - 1) / kBlockM, batch_size, num_heads},
      opts.dtype(torch::kInt32));
  params.dq_semaphore = dq_semaphore.data_ptr<int>();

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(
      !params.is_local,
      "This flash attention build does not support local attention.");
#endif

  if (total_seq_len > 0 && num_heads > 0) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_bwd(params, stream);
  }
  return {dq, dk, dv};
}

} // namespace hstu
