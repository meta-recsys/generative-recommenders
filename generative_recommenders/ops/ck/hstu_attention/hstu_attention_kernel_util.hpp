// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

namespace detail {

// A helper struct for detecting kUseTrLoad
// T is the pipeline class used by the kernel instance
template <typename T, typename = void>
struct has_use_trload_flag : std::false_type {};

template <typename T>
struct has_use_trload_flag<
    T,
    std::enable_if_t<
        std::is_convertible_v<decltype(T::kUseTrLoad), bool> && T::kUseTrLoad>>
    : std::true_type {};

template <typename T>
static inline constexpr bool is_using_trload_v = has_use_trload_flag<T>::value;

// A helper struct for detecting kUseTdm. The Tdm pipelines also set kUseTrLoad,
// because gemm_1 still reads V with ds_read_tr, so is_using_trload_v cannot
// tell them apart from the trload pipelines and they need a marker of their
// own. T is the pipeline class used by the kernel instance
template <typename T, typename = void>
struct has_use_tdm_flag : std::false_type {};

template <typename T>
struct has_use_tdm_flag<
    T,
    std::enable_if_t<
        std::is_convertible_v<decltype(T::kUseTdm), bool> && T::kUseTdm>>
    : std::true_type {};

template <typename T>
static inline constexpr bool is_using_tdm_v = has_use_tdm_flag<T>::value;

// A helper struct for detecting kUseDirectLds (the direct-to-LDS pipelines).
// T is the pipeline class used by the kernel instance
template <typename T, typename = void>
struct has_direct_lds_pipeline_flag : std::false_type {};

template <typename T>
struct has_direct_lds_pipeline_flag<
    T,
    std::enable_if_t<
        std::is_convertible_v<decltype(T::kUseDirectLds), bool> &&
        T::kUseDirectLds>> : std::true_type {};

template <typename T>
static inline constexpr bool is_direct_lds_pipeline_v =
    has_direct_lds_pipeline_flag<T>::value;

// [rows, kRowLength] view with the kSwizzleUnit-element groups of row r XORed
// by r % (kRowLength / kSwizzleUnit). The direct-to-LDS pipelines write LDS
// linearly and read it through the same XOR swizzle
// (MakeSimplySwizzledNativeDesc), so their K/V dram views apply the XOR on the
// way in.
template <index_t kRowLength, index_t kSwizzleUnit, typename TensorView>
CK_TILE_DEVICE auto make_direct_lds_xor_dram_view(const TensorView& view) {
  constexpr index_t kGroups = kRowLength / kSwizzleUnit;

  const auto rows = view.get_tensor_descriptor().get_lengths()[number<0>{}];

  const auto unmerged = transform_tensor_view(
      view,
      make_tuple(
          make_pass_through_transform(rows),
          make_unmerge_transform(
              make_tuple(number<kGroups>{}, number<kSwizzleUnit>{}))),
      make_tuple(sequence<0>{}, sequence<1>{}),
      make_tuple(sequence<0>{}, sequence<1, 2>{}));

  const auto permuted = transform_tensor_view(
      unmerged,
      make_tuple(
          make_xor_transform(make_tuple(rows, number<kGroups>{})),
          make_pass_through_transform(number<kSwizzleUnit>{})),
      make_tuple(sequence<0, 1>{}, sequence<2>{}),
      make_tuple(sequence<0, 1>{}, sequence<2>{}));

  return transform_tensor_view(
      permuted,
      make_tuple(
          make_pass_through_transform(rows),
          make_merge_transform_v3_division_mod(
              make_tuple(number<kGroups>{}, number<kSwizzleUnit>{}))),
      make_tuple(sequence<0>{}, sequence<1, 2>{}),
      make_tuple(sequence<0>{}, sequence<1>{}));
}

// scale is uniform (scalar register), c is per-lane (vector register)
// GFX9 VOP2: V_MUL_F32 VDST, SRC0, SRC1 - SRC0 can be SGPR, SRC1 must be VGPR

// scale is uniform (scalar register), c is per-lane (vector register)
// GFX9 VOP2: V_MUL_F32 VDST, SRC0, SRC1 - SRC0 can be SGPR, SRC1 must be VGPR
CK_TILE_DEVICE static void v_mul_f32_two(float& c0, float& c1, float scale) {
  asm volatile(
      "v_mul_f32 %[v_c0], %[s_scale], %[v_c0] \n\
                  v_mul_f32 %[v_c1], %[s_scale], %[v_c1]"
      : [v_c0] "+v"(c0), [v_c1] "+v"(c1)
      : [s_scale] "s"(scale)
      :);
}

CK_TILE_DEVICE static void v_mul_f32(float& c, float scale) {
  asm volatile("v_mul_f32 %[v_c], %[s_scale], %[v_c]"
               : [v_c] "+v"(c)
               : [s_scale] "s"(scale)
               :);
}

CK_TILE_DEVICE fp32x2_t pk_mul_f32(fp32x2_t lhs, fp32x2_t rhs) {
  fp32x2_t result;
  asm volatile("v_pk_mul_f32 %[result], %[lhs], %[rhs]"
               : [result] "=v"(result)
               : [lhs] "v"(lhs), [rhs] "v"(rhs));
  return result;
}

template <typename InOutDstrTensor>
CK_TILE_DEVICE static void scale_tile_in_scalar(
    InOutDstrTensor& in_out_dstr_tensor,
    float scale) {
  using DataType = typename InOutDstrTensor::DataType;

  if constexpr (std::is_same_v<std::remove_cv_t<DataType>, float>) {
    constexpr index_t thread_buffer_size =
        InOutDstrTensor::get_thread_buffer_size();

    static_for<0, thread_buffer_size, 2>{}([&](auto idx) {
      v_mul_f32_two(
          in_out_dstr_tensor.thread_buf_[idx],
          in_out_dstr_tensor.thread_buf_[idx + 1],
          scale);
    });
  } else {
    auto tmp_scale = type_convert<DataType>(scale);
    tile_elementwise_inout(
        [&tmp_scale](auto& x) { x = x * tmp_scale; }, in_out_dstr_tensor);
  };
};

template <typename InOutDstrTensor>
CK_TILE_DEVICE static void scale_tile_in_pack(
    InOutDstrTensor& in_out_dstr_tensor,
    float scale) {
  using DataType = typename InOutDstrTensor::DataType;

  if constexpr (std::is_same_v<std::remove_cv_t<DataType>, float>) {
    fp32x2_t pk_scale;

    pk_scale.x = scale;
    pk_scale.y = scale;

    constexpr index_t thread_buffer_size =
        InOutDstrTensor::get_thread_buffer_size();

    static_for<0, thread_buffer_size, 2>{}([&](auto idx) {
      fp32x2_t input = {
          in_out_dstr_tensor.thread_buf_[idx],
          in_out_dstr_tensor.thread_buf_[idx + 1]};
      auto output = pk_mul_f32(input, pk_scale);
      in_out_dstr_tensor.thread_buf_[idx] = output.x;
      in_out_dstr_tensor.thread_buf_[idx + 1] = output.y;
    });
  } else {
    auto tmp_scale = type_convert<DataType>(scale);
    tile_elementwise_inout(
        [&tmp_scale](auto& x) { x = x * tmp_scale; }, in_out_dstr_tensor);
  };
};

CK_TILE_DEVICE static constexpr bool IsPerfectHeaddimSize(index_t kHeaddim) {
  return kHeaddim > 0 && ((kHeaddim & (kHeaddim - 1)) == 0);
};

// --- attn_scale resolver -----------------------------------------------------
// Resolve the per-invocation SiLU scale from an optional device tensor.
//
// Deliberately marked noinline and kept in a separate function so the uniform
// global load + null/zero checks do NOT participate in the register allocation
// / instruction scheduling of the hot attention kernel body. When inlined, this
// logic perturbs the whole kernel's SGPR/VGPR coloring and desynchronizes waves
// at the main-loop barriers (~+6% regression on gfx942 for the jagged bf16
// case), even though the device branch is not taken.
//
// Semantics (identical to the previously inlined version):
//   * attn_scale_ptr == nullptr             -> keep host_default
//   * *attn_scale_ptr is +/-0 (classf 0x60) -> keep host_default
//   * otherwise                             -> use *attn_scale_ptr
__device__ __attribute__((noinline)) inline float hstu_resolve_scale_p(
    const float* attn_scale_ptr,
    float host_default) {
  if (attn_scale_ptr == nullptr)
    return host_default;
  const float s = *attn_scale_ptr;
  return __builtin_amdgcn_classf(s, 0x60) ? host_default : s;
}

// Lane mask of `pred` over the whole wave, on wave32 (gfx12/gfx125) and wave64
// (gfx9) targets.
CK_TILE_DEVICE static uint64_t wave_ballot(bool pred) {
  if constexpr (get_warp_size() == 32)
    return __builtin_amdgcn_ballot_w32(pred);
  else
    return __builtin_amdgcn_ballot_w64(pred);
}

// True if `pred` holds on any lane of the wave (a wave-uniform result).
CK_TILE_DEVICE static bool wave_any(bool pred) {
  return wave_ballot(pred) != 0;
}

} // namespace detail

} // namespace ck_tile
