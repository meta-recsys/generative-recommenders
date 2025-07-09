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
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "mask.h"
#include "seqlen.h"
#include "utils.h"

// #ifndef __CUDACC__
// #define __CUDACC__
// #include <device_functions.h>
// #endif

namespace flash {

using namespace cute;

template <
    int kNWarps,
    int Stages,
    bool Q_in_regs,
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    bool Causal,
    bool Local,
    bool Contexual_mask,
    bool Jagged,
    bool Has_targets>
struct CollectiveMainloopFwdSm80 {
  static constexpr int kStages = Stages;
  static_assert(kStages > 0, "kStages must be greater than 0");
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  static constexpr bool Is_FP8 =
      cute::is_same_v<Element, cutlass::float_e4m3_t> ||
      cute::is_same_v<Element, cutlass::float_e5m2_t>;
  static constexpr bool Transpose_V = Is_FP8;
  using SeqlenInfo_t = flash::SeqlenInfoQKFwd<Jagged, Has_targets>;

  static_assert(ArchTag::kMinComputeCapability >= 80);

  static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  using MMA_Atom_Arch = std::conditional_t<
      ArchTag::kMinComputeCapability >= 80,
      std::conditional_t<
          std::is_same_v<Element, cutlass::half_t>,
          MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
          MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>,
      MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>>;
  using TiledMma = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>, // 4x1x1 or 8x1x1 thread group
      Tile<Int<16 * kNWarps>, _16, _16>>;

  static constexpr int NumMmaThreads = size(TiledMma{});
  static constexpr int NumProducerThreads =
      NumMmaThreads; // For compatibility with TileScheduler

  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      kHeadDim % kGmemElemsPerLoad == 0,
      "Headdim must be a multiple of kGmemElemsPerLoad");
  // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g.
  // if hdim=128, we want each thread to have 4 loads in the M direction and 2
  // vectorized load in the K direction.
  static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) /
      sizeof(Element);

  static constexpr int kSwizzle = kBlockKGmem == 128
      ? 4
      : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
  static constexpr int kSwizzleBase =
      sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
  using SmemLayoutAtomQKV = decltype(composition(
      Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
      Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      select<0, 2>(TileShape_MNK{})));

  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));

  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomQKV{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{},
      make_ordered_layout(
          make_shape(
              shape<2>(TileShape_MNK{}),
              shape<1>(TileShape_MNK{}),
              Int<kStages>{}),
          Step<_2, _1, _3>{})));

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

  // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we
  // won't be reading from the same address by the same threadblock. This is
  // slightly faster.
  using GmemCopyAtom = Copy_Atom<
      std::conditional_t<
          Has_cp_async,
          SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
          AutoVectorizingCopyWithAssumedAlignment<128>>,
      Element>;

  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(
      NumMmaThreads % kGmemThreadsPerRow == 0,
      "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom = Layout<
      Shape<Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(
      GmemCopyAtom{},
      GmemLayoutAtom{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals
                                                     // per read
  // So that we don't have to check if we overshot kBlockM when we load Q
  static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);

  // For AppendKV, We want each thread to have at least 2 loads in the K
  // direction since in the case of non-interleaved rotary (combining elements
  // at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc), each thread will
  // load twice from the same row.
  static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
  static constexpr int kBlockKGmemAppend =
      (kBytePerHalfRow % 128 == 0 ? 128
                                  : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) /
      sizeof(Element);
  static constexpr int kGmemThreadsPerRowAppend =
      kBlockKGmemAppend / kGmemElemsPerLoad;
  static_assert(
      NumMmaThreads % kGmemThreadsPerRowAppend == 0,
      "NumMmaThreads must be a multiple of kGmemThreadsPerRowAppend");
  // We assume threads loading the same row are in the same warp. This is for an
  // optimization in PagedKV where these threads share the same page table entry
  // and share the work of computing pointers to paged K and paged V.
  static_assert(
      cutlass::NumThreadsPerWarp % kGmemThreadsPerRowAppend == 0,
      "kGmemThreadsPerRowAppend must divide NumThreadsPerWarp");
  using GmemLayoutAtomAppend = Layout<
      Shape<
          Int<NumMmaThreads / kGmemThreadsPerRowAppend>,
          Int<kGmemThreadsPerRowAppend>>,
      Stride<Int<kGmemThreadsPerRowAppend>, _1>>;
  // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to
  // avoid predication
  using GmemTiledCopyAppendKV = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
      GmemLayoutAtomAppend{},
      Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{})); // Val layout, 8 or 16 vals
                                                     // per store

  using ShapeQKV =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen, d, head,
                                                       // batch)
  using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using StrideV = StrideQK;
  // ((qhead_per_khead, seqlen), d, nheads_kv, batch, num_splits)
  using ShapeQPacked = std::conditional_t<
      true,
      ShapeQKV,
      cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
  using StrideQPacked = std::conditional_t<
      true,
      StrideQK,
      cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
  using ShapePageTable =
      cute::Shape<int32_t, int32_t>; // (batch, max_num_pages_per_seq)
  using StridePageTable = cute::Stride<int64_t, _1>;
  using ShapeRotary =
      cute::Shape<int32_t, int32_t>; // (seqlen_ro, rotary_dim // 2)
  using StrideRotary = cute::Stride<int64_t, _1>;
  using StrideDescale = cute::Stride<int64_t, int64_t>;

  static constexpr bool Share_QV_Smem = Q_in_regs;

  struct TensorStorageSharedQV : cute::aligned_struct<128> {
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
  };

  struct TensorStorageSeparateQV : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
  };

  using TensorStorage = std::conditional_t<
      Share_QV_Smem,
      TensorStorageSharedQV,
      TensorStorageSeparateQV>;

  // Host side kernel arguments
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    Element* const
        ptr_K; // Not Element const* since we might append to KV cache in-place
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    StrideV const stride_V;
    float const *ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    float const max_seq_len_inv;
    float const alpha;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    int const* const seq_offsets = nullptr;
    int const* const num_targets = nullptr;
    float const* const attn_scale = nullptr;
    bool const scalar_scale = true;
  };

  // Device side kernel params
  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQK const stride_Q;
    ShapeQPacked const shape_Q_packed;
    StrideQPacked const stride_Q_packed;
    Element* const ptr_K;
    ShapeQKV const shape_K;
    StrideQK const stride_K;
    Element* const ptr_V;
    StrideV const stride_V;
    float const *ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    float const max_seq_len_inv;
    float const alpha;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    int const* const seq_offsets = nullptr;
    int const* const num_targets = nullptr;
    float const* const attn_scale = nullptr;
    bool const scalar_scale = true;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // If PackGQA, reshape Q to be ((qhead_per_khead, seqlen), head_size,
    // nhead_k, batch_size)
    int const qhead_per_khead = 1;
    auto const shape_Q_packed = cute::conditional_return<true>(
        args.shape_Q,
        make_shape(
            make_shape(qhead_per_khead, get<0>(args.shape_Q)),
            get<1>(args.shape_Q),
            get<2>(args.shape_K),
            get<3>(args.shape_Q)));
    auto const stride_Q_packed = cute::conditional_return<true>(
        args.stride_Q,
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)),
            get<1>(args.stride_Q),
            get<2>(args.stride_Q) * qhead_per_khead,
            get<3>(args.stride_Q)));
    return {
        args.ptr_Q,
        args.shape_Q,
        args.stride_Q,
        shape_Q_packed,
        stride_Q_packed,
        args.ptr_K,
        args.shape_K,
        args.stride_K,
        args.ptr_V,
        args.stride_V,
        args.ptr_q_descale,
        args.ptr_k_descale,
        args.ptr_v_descale,
        args.stride_q_descale,
        args.stride_k_descale,
        args.stride_v_descale,
        args.max_seq_len_inv,
        args.alpha,
        args.max_attn_len,
        args.min_full_attn_seq_len,
        args.contextual_seq_len,
        args.seq_offsets,
        args.num_targets,
        args.attn_scale,
        args.scalar_scale};
  }

  CUTLASS_DEVICE
  cute::tuple<int, int>
  get_n_block_min_max(int max_attn_len, int uihlen, int m_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int n_block_max;
    int n_block_min;
    // Non-target part, n_block_max
    if constexpr (Causal || Local) {
      int m_idx_max = (m_block + 1) * kBlockM;
      n_block_max = cute::ceil_div(std::min(m_idx_max, uihlen), kBlockN);
    } else {
      n_block_max = cute::ceil_div(uihlen, kBlockN);
    }
    // Non-target part, n_block_min
    if constexpr (Local) {
      int m_idx_min = m_block * kBlockM;
      n_block_min = std::max(int(0), (m_idx_min - max_attn_len) / kBlockN);
    } else {
      n_block_min = 0;
    }
    // Target part
    if constexpr (Has_targets) {
      int m_idx_max = (m_block + 1) * kBlockM;
      if (m_idx_max > uihlen) {
        n_block_min = 0;
        n_block_max = cute::ceil_div(uihlen, kBlockN);
      }
    }
    return {n_block_min, n_block_max};
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_target_n_block_min_max(
      int n_block_max,
      int uihlen,
      int seqlen,
      int m_block) {
    if constexpr (!Has_targets) {
      return {n_block_max, n_block_max};
    }
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int m_idx_max = (m_block + 1) * kBlockM;
    if (m_idx_max <= uihlen) { // Non-target part
      return {n_block_max, n_block_max};
    } else { // Target part
      int m_idx_min = m_block * kBlockM;
      return {
          std::max(n_block_max, m_idx_min / kBlockN),
          cute::ceil_div(std::min(m_idx_max, seqlen), kBlockN)};
    }
  }

  CUTLASS_DEVICE
  int get_pipeline_n_block(
      int const n_block,
      int const stage,
      int const n_block_min,
      int const n_block_max,
      int const target_n_block_min,
      int const target_n_block_max) {
    int const out_n_block = n_block - stage;
    if constexpr (!Has_targets) {
      if (out_n_block >= n_block_min && out_n_block < n_block_max) {
        return out_n_block;
      } else {
        return -1;
      }
    } else { // Has_targets
      if (n_block < n_block_max) {
        if (out_n_block >= n_block_min) {
          return out_n_block;
        }
        int n_block_diff = n_block_min - out_n_block;
        if (target_n_block_max - n_block_diff >= target_n_block_min) {
          return target_n_block_max - n_block_diff;
        } else {
          return -1;
        }
      } else { // n_block >= n_block_max
        if (out_n_block >= target_n_block_min) {
          return out_n_block;
        } else {
          return -1;
        }
      }
    }
  }

  template <typename SharedStorage, typename FrgTensorO>
  CUTLASS_DEVICE bool mma(
      Params const& params,
      FrgTensorO& tOrO,
      int const thread_idx,
      SeqlenInfo_t const& seqlen_info,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage) {
    static_assert(
        is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});

    // can't use auto [m_block, ...] = block_coord since structured binding
    // cannot be captured in lambda
    int const m_block = get<0>(block_coord);
    int const bidh = get<1>(block_coord);
    int const bidb = get<2>(block_coord);
    int const split_idx = get<3>(block_coord);
    if constexpr (Jagged) {
      if (m_block * kBlockM >= seqlen_info.seqlen) {
        return false;
      }
    }
    auto n_block_min_max =
        get_n_block_min_max(params.max_attn_len, seqlen_info.uihlen, m_block);
    int const n_block_min = get<0>(n_block_min_max);
    int const n_block_max = get<1>(n_block_min_max);
    auto target_n_block_min_max = get_target_n_block_min_max(
        n_block_max, seqlen_info.uihlen, seqlen_info.seqlen, m_block);
    int const target_n_block_min = get<0>(target_n_block_min_max);
    int const target_n_block_max = get<1>(target_n_block_min_max);

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutV{});
    Tensor sVt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutVt{});

    int const bidb_kv = bidb;
    Tensor mQ = make_tensor(
        make_gmem_ptr(
            params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)),
        params.shape_Q_packed,
        params.stride_Q_packed)(_, _, bidh, !Jagged ? bidb : 0);
    Tensor gQ = local_tile(
        mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{})); // (M, K)
    Tensor mK = make_tensor(
        make_gmem_ptr(
            params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K)),
        params.shape_K,
        params.stride_K)(_, _, bidh, !Jagged ? bidb_kv : 0);
    Tensor gK = local_tile(
        mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)
    Tensor mV = make_tensor(
        make_gmem_ptr(
            params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V)),
        params.shape_K,
        params.stride_V)(_, _, bidh, !Jagged ? bidb_kv : 0);
    Tensor gV = local_tile(
        mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{})); // (N, K, _)

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
    auto gmem_thr0_copy_QKV =
        gmem_tiled_copy_QKV.get_thread_slice(_0{}); // For index calculation

    Tensor tKgK =
        gmem_thr_copy_QKV.partition_S(gK); // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV =
        gmem_thr_copy_QKV.partition_S(gV); // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    // Allocate "fragments/descriptors"
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    // Copy Atom retiling
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    auto smem_tiled_copy_V =
        make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // Predicates
    Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
#pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K);
    }

    int n_block = n_block_max - 1;

    // Prologue: load Q, K, V
    // If persistent, we don't need to wait for the previous work_idx to finish
    // since we assume that all MMA threads sync in the epilogue before writing
    // to smem_o. So any thread gets there, all threads must have finished the
    // previous MMA and at least started writing to smem_o. If persistent, need
    // to sync to make sure all threads have finished with smem_o before writing
    // to smem_v
    if constexpr (Share_QV_Smem) {
      __syncthreads();
    }
    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q);
    }
    // Instead of passing in tQcQ, we pass in t0QcQ and subtract the offset
    // from the limit (seqlen - m_block * kBlockM). This is because the
    // entries of t0QcQ are known at compile time. We don't need to clear the
    // sQ smem tiles since we'll only write out the valid outputs
    flash::copy<
        /*Is_even_MN=*/false,
        /*Is_even_K=*/false,
        /*Clear_OOB_MN=*/false,
        /*Clear_OOB_K=*/true>(
        gmem_tiled_copy_QKV,
        tQgQ,
        tQsQ,
        t0QcQ,
        tQpQ,
        seqlen_info.seqlen - m_block * kBlockM -
            get<0>(tQcQ(_0{}, _0{}, _0{})));
    cute::cp_async_fence();

    auto load_K = [&](int const n_block,
                      int const smem_pipe_write,
                      auto need_seqlenk_masking_type) {
      static constexpr bool Seqlenk_mask =
          decltype(need_seqlenk_masking_type)::value;
      // Do we need bound check to make sure the row doesn't go above kBlockN
      static constexpr bool EvenN =
          kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
      Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);
      // Instead of passing in tKVcKV, we pass in t0KVcKV and subtract the
      // offset from the limit (seqlen - n_block * kBlockN). This is because
      // the entries of t0KVcKV are known at compile time.
      int const seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) +
          (EvenN ? seqlen_info.seqlen - n_block * kBlockN
                 : (!Seqlenk_mask ? kBlockN
                                  : std::min(
                                        seqlen_info.seqlen - n_block * kBlockN,
                                        kBlockN)));
      // We don't need to clear the sK smem tiles since we'll mask out the
      // scores anyway.
      flash::copy<
          /*Is_even_MN=*/!Seqlenk_mask && EvenN,
          /*Is_even_K=*/false,
          /*Clear_OOB_MN=*/false,
          /*Clear_OOB_K=*/true>(
          gmem_tiled_copy_QKV,
          tKgK(_, _, _, n_block),
          tKsK_cur,
          t0KVcKV,
          tKVpKV,
          seqlenk_row_limit);
    };

    auto load_V = [&](int const n_block,
                      int const smem_pipe_write,
                      auto need_seqlenk_masking_type) {
      static constexpr bool Seqlenk_mask =
          decltype(need_seqlenk_masking_type)::value;
      // Do we need bound check to make sure the row doesn't go above
      // kBlockN
      static constexpr bool EvenN =
          kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
      Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);
      // We don't call flash::copy since it doesn't support bound checking
      // to not overshot kBlockN when writing to smem.
      Tensor tVgV_cur = tVgV(_, _, _, n_block);
      int const seqlenk_row_limit = seqlen_info.seqlen - n_block * kBlockN -
          get<0>(tKVcKV(_0{}, _0{}, _0{}));
#pragma unroll
      for (int m = 0; m < size<1>(tVsV); ++m) {
        // If kBlockN doesn't evenly divide the tiled copy, only the last
        // `m` needs to be checked
        if (EvenN || m < size<1>(tVsV) - 1 ||
            get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
          bool const predicate_n = !Seqlenk_mask ||
              get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
#pragma unroll
          for (int k = 0; k < size<2>(tVsV); ++k) {
            cute::copy(
                gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n),
                tVgV_cur(_, m, k),
                tVsV_cur(_, m, k));
          }
        }
      }
    };

    auto preprocess_Q = [&] {
      flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();

      if constexpr (Q_in_regs) {
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
        cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
      }
    };

    // If Share_QV_Smem, we load Q, then load 1 stage of K, then
    // (optionally) rotate Q and read from smem_q to registers, then load
    // V. If !Share_QV, Smem, we load Q, load all stages of K & V, then
    // (optionally) rotate Q.

    if constexpr (Share_QV_Smem) {
      load_K(n_block, 0, cute::true_type{} /*Seqlenk_mask*/);
      cute::cp_async_fence();
      preprocess_Q();
      __syncthreads(); // Make sure all threads have read smem_q before
                       // loading
                       // V
    }

    // For persistent, make sure all threads have finished reading smem_o
    if constexpr (!Share_QV_Smem) {
      __syncthreads();
    }
    // Note, using the for_each() function here to ensure `stage` is of
    // type Int<x>.
    for_each(make_int_sequence<kStages>{}, [&](auto stage) {
      int const prefetch_n_block = get_pipeline_n_block(
          n_block,
          stage,
          n_block_min,
          n_block_max,
          target_n_block_min,
          target_n_block_max);
      static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
      static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
      if constexpr (!Share_QV_Smem || !Is_first_stage) {
        if (Is_first_stage || prefetch_n_block != -1) {
          load_K(
              prefetch_n_block,
              stage,
              cute::bool_constant < Is_first_stage ||
                  Has_targets > {} /*Seqlenk_mask*/);
        }
        // We want the fence outside the if statement to have a fixed
        // number of cp.async commits. so that we can wait with the
        // correct number of outstanding commits.
        cute::cp_async_fence();
      }
      if constexpr (!Is_last_stage) {
        if (Is_first_stage || prefetch_n_block != -1) {
          load_V(
              prefetch_n_block,
              stage,
              cute::bool_constant < Is_first_stage ||
                  Has_targets > {} /*Seqlenk_mask*/);
        }
        cute::cp_async_fence();
      }
    });

    if constexpr (!Share_QV_Smem) {
      preprocess_Q();
    }

    flash::Mask<kBlockM, kBlockN, TiledMma> mask(
        thread_idx,
        seqlen_info.seqlen,
        params.max_attn_len,
        params.min_full_attn_seq_len,
        params.contextual_seq_len,
        seqlen_info.uihlen);
    int smem_pipe_read = 0, smem_pipe_write = kStages - 1;

    // attention scale
    float scalar_scale_val = params.scalar_scale
        ? (params.attn_scale == nullptr ? params.max_seq_len_inv
                                        : params.attn_scale[0])
        : 0;
    static constexpr int Qdim = 0;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto thread0_mma = tiled_mma.get_thread_slice(_0{});
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tScS_rowcol = make_tensor(
        tScS.data(),
        flash::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(
        t0ScS.data(),
        flash::convert_layout_acc_rowcol</*Transposed=*/false>(t0ScS.layout()));
    int const thread_qdim_offset = get<Qdim>(tScS_rowcol(_0{}, _0{}));
    SiluScaleOp<float> silu_scale_op;
    int qdim_offset = params.scalar_scale
        ? 0
        : m_block * kBlockM + thread_qdim_offset + seqlen_info.offset_q;

    auto load_K_next = [&] {
      int const prefetch_n_block = get_pipeline_n_block(
          n_block,
          kStages,
          n_block_min,
          n_block_max,
          target_n_block_min,
          target_n_block_max);
      if (prefetch_n_block != -1) {
        load_K(
            prefetch_n_block,
            kStages > 1 ? smem_pipe_write : 0,
            cute::bool_constant<Has_targets>{} /*Seqlenk_mask*/);
      }
      cute::cp_async_fence();
    };

    auto sync = [&] {
      flash::cp_async_wait<kStages * 2 - 2>();
      __syncthreads();
    };

    clear(tOrO);

    auto fwd_step = [&](int const n_block,
                        auto mask_fn,
                        auto is_boundary_iter_type) {
      static constexpr bool Is_boundary_iter =
          decltype(is_boundary_iter_type)::value;
      Tensor tSrS =
          partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
      Tensor tSrS_rowcol = make_tensor(
          tSrS.data(),
          flash::convert_layout_acc_rowcol</*Transposed=*/false>(
              tSrS.layout()));
      clear(tSrS);
      sync();
      auto load_V_next = [&] {
        int const prefetch_n_block = get_pipeline_n_block(
            n_block,
            kStages - 1,
            n_block_min,
            n_block_max,
            target_n_block_min,
            target_n_block_max);
        if (prefetch_n_block != -1) {
          load_V(
              prefetch_n_block,
              kStages > 1 ? smem_pipe_write : 0,
              cute::bool_constant < (Is_boundary_iter && kStages == 1) ||
                  Has_targets >
                      {} /*Seqlenk_mask*/); // Always do Seqlenk_mask when
                                            // Has_targets as the logic is
                                            // complicated with kStages != 1
        }
        cute::cp_async_fence();
      };
      Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(
          tSrQ, thr_mma.partition_fragment_A(sQ));
      Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));
      flash::gemm_sm80<Q_in_regs>(
          tSrS,
          tSrQ_cur,
          tSrK,
          tSsQ,
          tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0),
          tiled_mma,
          smem_tiled_copy_Q,
          smem_tiled_copy_K,
          smem_thr_copy_Q,
          smem_thr_copy_K,
          load_V_next);
      smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
#pragma unroll
      for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
        float scale = scalar_scale_val;
        if (!params.scalar_scale) {
          int q_index = qdim_offset + int(get<Qdim>(t0ScS_rowcol(mi, _0{})));
          if (q_index < seqlen_info.seqlen) {
            scale = params.attn_scale[q_index];
          }
        }
#pragma unroll
        for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
          tSrS_rowcol(mi, ni) =
              silu_scale_op(tSrS_rowcol(mi, ni) * params.alpha, scale);
        }
      }
      // Faster to load_K before gemm if we only have 1 stage
      if constexpr (kStages == 1) {
        sync();
        load_K_next();
      }
      mask_fn(tSrS, n_block);
      if constexpr (Is_FP8) {
        flash::permute_Cregs_fp8(tSrS);
      }
      Tensor tOrP_acc = make_tensor(
          tSrS.data(),
          flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
      Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
      convert_type_out(tOrP_acc, tOrP);
      if constexpr (kStages > 1) {
        sync();
      }
      Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
      flash::gemm_rs_sm80(
          tOrO,
          tOrP,
          tOrV,
          tOsVt(_, _, _, kStages > 1 ? smem_pipe_read : 0),
          tiled_mma,
          smem_tiled_copy_V,
          smem_thr_copy_V);
      if constexpr (kStages > 1) {
        load_K_next();
      }
      smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
    };

    int const m_idx_max = (m_block + 1) * kBlockM;
    if (m_idx_max <= seqlen_info.uihlen) {
      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            false /*Seqlenk_mask*/,
            Causal,
            Local,
            false /*Contexual_mask*/,
            false /*Target_mask*/>(tSrS, m_block, n_block);
      };
      fwd_step(
          n_block, first_iter_mask_fn, cute::true_type{} /*is_boundary_iter*/);
    } else if (
        m_idx_max <= cute::ceil_div(seqlen_info.uihlen, kBlockM) * kBlockM) {
      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            Local,
            false /*Contexual_mask*/,
            Has_targets>(tSrS, m_block, n_block);
      };
      fwd_step(
          n_block, first_iter_mask_fn, cute::true_type{} /*is_boundary_iter*/);
    } else {
      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal*/,
            false /*Local*/,
            false /*Contexual_mask*/,
            Has_targets>(tSrS, m_block, n_block);
      };
      fwd_step(
          n_block, first_iter_mask_fn, cute::true_type{} /*is_boundary_iter*/);
    }
    --n_block;
    if constexpr (Causal || Local) { // Separate iterations with
                                     // causal or local masking
      if (m_idx_max <= cute::ceil_div(seqlen_info.uihlen, kBlockM) * kBlockM) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              false /*Seqlenk_mask*/,
              Causal,
              Local,
              false /*Contexual_mask*/,
              false /*Target_mask*/>(tSrS, m_block, n_block);
        };
        int const m_idx_min = m_block * kBlockM;
        int const n_block_min_causal_local_mask =
            std::max(n_block_min, m_idx_min / kBlockN);
#pragma unroll 1
        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, cute::false_type{} /*is_boundary_iter*/);
        }
      }
    }
    int const n_block_min_before_local_mask = !Local
        ? n_block_min
        : std::max(
              n_block_min,
              cute::ceil_div(m_idx_max - params.max_attn_len, kBlockN));
    auto no_mask_fn = [](auto& tSrS, int n_block) {};
#pragma unroll 1
    for (; n_block >= n_block_min_before_local_mask; --n_block) {
      fwd_step(n_block, no_mask_fn, cute::false_type{} /*is_boundary_iter*/);
    }
    // Separate masking iterations on the left for local attention
    if constexpr (Local) {
      auto local_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            false /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            Local,
            false /*Contexual_mask*/,
            false /*Target_mask*/>(tSrS, m_block, n_block);
      };
#pragma unroll 1
      for (; n_block >= n_block_min; --n_block) {
        fwd_step(
            n_block, local_mask_fn, cute::false_type{} /*is_boundary_iter*/);
      }
    }
    // Target part GEMM
    if constexpr (Has_targets) {
      auto target_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            false /*Local*/,
            false /*Contexual_mask*/,
            Has_targets>(tSrS, m_block, n_block);
      };
#pragma unroll 1
      for (n_block = target_n_block_max - 1; n_block >= target_n_block_min;
           --n_block) {
        fwd_step(
            n_block, target_mask_fn, cute::false_type{} /*is_boundary_iter*/);
      }
    }
    float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr
        ? 1.0f
        : params.ptr_v_descale
              [bidb * get<0>(params.stride_v_descale) +
               bidh * get<1>(params.stride_v_descale)];
    if constexpr (Is_FP8) {
      flash::permute_output_fp8(tOrO);
    }
    return true;
  }
};

} // namespace flash
