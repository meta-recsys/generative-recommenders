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

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/numeric_types.h>

namespace cute {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_BULK_REDUCE_ADD {
  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr, float* gmem_ptr, int32_t store_bytes) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
        :
        : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  CUTE_HOST_DEVICE static void copy(
      float const* smem_ptr,
      float* gmem_ptr,
      int32_t store_bytes,
      uint64_t cache_hint) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [%0], [%1], %2, %3;\n"
        :
        : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  CUTE_HOST_DEVICE static void copy(
      [[maybe_unused]] cutlass::bfloat16_t const* smem_ptr,
      [[maybe_unused]] cutlass::bfloat16_t* gmem_ptr,
      [[maybe_unused]] int32_t store_bytes) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;\n"
        :
        : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  CUTE_HOST_DEVICE static void copy(
      [[maybe_unused]] cutlass::bfloat16_t const* smem_ptr,
      [[maybe_unused]] cutlass::bfloat16_t* gmem_ptr,
      [[maybe_unused]] int32_t store_bytes,
      [[maybe_unused]] uint64_t cache_hint) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.noftz.bf16 [%0], [%1], %2, %3;\n"
        :
        : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  // fp16 variant. As with bf16, `.add` for f16 is only available as `.noftz`.
  CUTE_HOST_DEVICE static void copy(
      [[maybe_unused]] cutlass::half_t const* smem_ptr,
      [[maybe_unused]] cutlass::half_t* gmem_ptr,
      [[maybe_unused]] int32_t store_bytes) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;\n"
        :
        : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  CUTE_HOST_DEVICE static void copy(
      [[maybe_unused]] cutlass::half_t const* smem_ptr,
      [[maybe_unused]] cutlass::half_t* gmem_ptr,
      [[maybe_unused]] int32_t store_bytes,
      [[maybe_unused]] uint64_t cache_hint) {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile(
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.noftz.f16 [%0], [%1], %2, %3;\n"
        :
        : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
        : "memory");
#else
    CUTE_INVALID_CONTROL_PATH(
        "Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
