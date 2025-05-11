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

#include "cute/tensor.hpp"

#include "mask.h"
#include "seqlen.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <
    int Stages,
    int Stages_dO,
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    bool Causal,
    bool Local,
    bool Contexual_mask,
    bool Jagged,
    bool Has_targets,
    bool Deterministic,
    bool SdP_swapAB_,
    bool dKV_swapAB_,
    bool dQ_swapAB_,
    int NumMmaWarpGroups = 2,
    int AtomLayoutMSdP = 1,
    int AtomLayoutNdKV = 8,
    int AtomLayoutMdQ = 1,
    bool V_in_regs = false>
struct CollectiveMainloopBwdSm80 {
  static constexpr int kStages = Stages;
  static constexpr int kStages_dO = Stages_dO;
  static_assert(kStages >= kStages_dO);
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  using SeqlenInfo_t = flash::SeqlenInfoQKBwd<
      Jagged,
      Has_targets,
      CUTE_STATIC_V(get<0>(TileShape_MNK{}))>;
  static constexpr int NumMmaWarps =
      NumMmaWarpGroups * cutlass::NumWarpsPerWarpGroup;

  static constexpr bool SdP_swapAB = SdP_swapAB_;
  static constexpr bool dKV_swapAB = dKV_swapAB_;
  static constexpr bool dQ_swapAB = dQ_swapAB_;

  static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  static_assert(ArchTag::kMinComputeCapability >= 80);

  static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;

  static constexpr int NumMmaThreads = NumMmaWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads =
      NumMmaThreads; // For compatibility with TileScheduler

  using MMA_Atom_Arch = std::conditional_t<
      ArchTag::kMinComputeCapability >= 80,
      std::conditional_t<
          std::is_same_v<Element, cutlass::half_t>,
          MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
          MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>,
      MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>>;

  static_assert(NumMmaWarps % AtomLayoutMSdP == 0);
  static_assert(NumMmaWarps % AtomLayoutNdKV == 0);
  static_assert(NumMmaWarps % AtomLayoutMdQ == 0);
  static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 &&
      AtomLayoutNdKV == NumMmaWarps && SdP_swapAB && !dKV_swapAB;
  static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarps &&
      AtomLayoutMdQ == NumMmaWarps && !SdP_swapAB &&
      !dQ_swapAB; // If dQ_swapAB we can't use RS

  using AtomLayoutSdP = std::conditional_t<
      !SdP_swapAB,
      Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarps / AtomLayoutMSdP>, _1>>,
      Layout<
          Shape<Int<NumMmaWarps / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>>;
  static constexpr bool MmaSdPEvenN =
      ((!SdP_swapAB ? kBlockN : kBlockM) / size<1>(AtomLayoutSdP{})) % 16 == 0;
  using TiledMmaSdP = TiledMMA<
      MMA_Atom_Arch,
      AtomLayoutSdP,
      Tile<
          Int<16 * CUTE_STATIC_V(size<0>(AtomLayoutSdP{}))>,
          Int<(MmaSdPEvenN ? 16 : 8) * CUTE_STATIC_V(size<1>(AtomLayoutSdP{}))>,
          _16>>;

  using AtomLayoutdKV = std::conditional_t<
      !dKV_swapAB,
      Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarps / AtomLayoutNdKV>, _1>>,
      Layout<
          Shape<Int<NumMmaWarps / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>>;
  static constexpr bool MmadKVEvenN =
      ((!dKV_swapAB ? kHeadDim : kBlockN) / size<1>(AtomLayoutdKV{})) % 16 == 0;
  using TiledMmadKV = TiledMMA<
      MMA_Atom_Arch,
      AtomLayoutdKV,
      Tile<
          Int<16 * CUTE_STATIC_V(size<0>(AtomLayoutdKV{}))>,
          Int<(MmadKVEvenN ? 16 : 8) * CUTE_STATIC_V(size<1>(AtomLayoutdKV{}))>,
          _16>>;

  using AtomLayoutdQ = std::conditional_t<
      !dQ_swapAB,
      Layout<Shape<Int<AtomLayoutMdQ>, Int<NumMmaWarps / AtomLayoutMdQ>, _1>>,
      Layout<Shape<Int<NumMmaWarps / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>>;
  static constexpr bool MmadQEvenN =
      ((!dQ_swapAB ? kHeadDim : kBlockM) / size<1>(AtomLayoutdQ{})) % 16 == 0;
  using TiledMmadQ = TiledMMA<
      MMA_Atom_Arch,
      AtomLayoutdQ,
      Tile<
          Int<16 * CUTE_STATIC_V(size<0>(AtomLayoutdQ{}))>,
          Int<(MmadQEvenN ? 16 : 8) * CUTE_STATIC_V(size<1>(AtomLayoutdQ{}))>,
          _16>>;

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

  // We need to accommodate both Q and Q^T (and dO and dO^T) in shared memory.
  // Q & dO are used in the SdP Mma and Q^T and dO^T are used in the dKV Mma.
  // Since this is GMMA::Major::K, the M dimension (kBlockM) doesn't matter for
  // the layout, only the K dimension changes the layout.
  using SmemLayoutAtomQdO = decltype(composition(
      Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
      Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));
  using SmemLayoutdO = decltype(tile_to_shape(
      SmemLayoutAtomQdO{},
      make_shape(
          shape<0>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages_dO>{})));

  using SmemLayoutAtomKV = decltype(composition(
      Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
      // TODO: FA2 has a slightly different layout, does it matter?
      Layout<Shape<_8, Int<kBlockKGmem>>, Stride<Int<kBlockKGmem>, _1>>{}));
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomKV{},
      select<1, 2>(TileShape_MNK{})));

  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomKV{},
      select<1, 2>(TileShape_MNK{})));

  // TD [2023-03-19]: Idk why kPBlockN = 16 and kSwizzlePdS=3 is the fastest.
  static constexpr int kPBlockN =
      kBlockN % 64 == 0 ? 64 : (kBlockN % 32 == 0 ? 32 : 16);
  static_assert(kPBlockN == 16 || kPBlockN == 32 || kPBlockN == 64);
  // static constexpr int kSwizzlePdS = kPBlockN == 16 ? 1 : (kPBlockN == 32 ? 2
  // : 3);
  static constexpr int kSwizzlePdS = 3;
  using SmemLayoutAtomPdS = decltype(composition(
      Swizzle<kSwizzlePdS, kSwizzleBase, kSwizzleBase>{},
      Layout<Shape<Int<kBlockM>, Int<kPBlockN>>, Stride<Int<kPBlockN>, _1>>{}));
  using SmemLayoutPdS = decltype(tile_to_shape(
      SmemLayoutAtomPdS{},
      make_shape(Int<kBlockM>{}, Int<kBlockN>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutQt = decltype(cute::composition(
      SmemLayoutQ{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<0>(TileShape_MNK{}),
              Int<kStages>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
  using SmemLayoutdOt = decltype(cute::composition(
      SmemLayoutdO{},
      make_layout(
          make_shape(
              get<2>(TileShape_MNK{}),
              get<0>(TileShape_MNK{}),
              Int<kStages_dO>{}),
          make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
  using SmemLayoutKt = decltype(cute::composition(
      SmemLayoutK{},
      make_layout(
          make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
          make_stride(Int<kBlockN>{}, _1{}))));
  using SmemLayoutPdSt = decltype(cute::composition(
      SmemLayoutPdS{},
      make_layout(
          make_shape(Int<kBlockN>{}, Int<kBlockM>{}),
          make_stride(Int<kBlockM>{}, _1{}))));

  // Thread layout, 256 or 384 threads per row
  using R2SLayoutAtomdQaccum = Layout<Shape<Int<NumMmaThreads>>>;
  using R2STiledCopydQaccum = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      R2SLayoutAtomdQaccum{},
      Layout<Shape<_1>>{})); // Val layout, 1 vals per store

  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
  // For the case where the N dimension of MmaSdP is divisible by 8 but not by
  // 16
  using SmemCopyAtomHalf = Copy_Atom<SM75_U32x2_LDSM_N, Element>;
  // For the case where the N dimension of MmadQ is divisible by 8 but not by 16
  using SmemCopyAtomTransposedHalf = Copy_Atom<SM75_U16x4_LDSM_T, Element>;
  // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt /
  // dSt. If PdS_major is MN, then we need to "transpose" the write.
  // TODO: check this write
  using R2SCopyAtomPdS =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

  // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we
  // won't be reading from the same address by the same threadblock. This is
  // slightly faster.
  using GmemCopyStruct = std::conditional_t<
      Has_cp_async,
      SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
      AutoVectorizingCopyWithAssumedAlignment<128>>;
  using GmemCopyAtom = Copy_Atom<GmemCopyStruct, Element>;

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
  // static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);

  using ShapeQKV =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen, d, head,
                                                       // batch)
  using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using ShapedQaccum =
      cute::Shape<int32_t, int32_t, int32_t>; // (seqlen * d, head, batch)
  using StridedQaccum = cute::Stride<_1, int64_t, int64_t>;

  // These are tuned for speed. They don't affect correctness.
  // We have separate iterations with causal masking. Not necessary for hdim 128
  // but for hdim 64 this helps quite a bit to not have to do causal masking for
  // most of the iterations. For hdim 192, separating masking iterations results
  // in register spills. static constexpr bool SeparateMaskingIterations =
  // kHeadDim <= 64;
  static constexpr bool SeparateMaskingIterations = false;

  static constexpr bool Share_QV_Smem = V_in_regs;
  using SmemP_t = std::conditional_t<
      Mma_dKV_is_RS,
      cute::array<Element, 0>,
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>>>;

  struct TensorStorageSharedQV : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    union {
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
      cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
    SmemP_t smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>> smem_ds;
  };

  struct TensorStorageSeparateQV : cute::aligned_struct<128> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
    SmemP_t smem_p;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>> smem_ds;
  };

  using TensorStorage = std::conditional_t<
      Share_QV_Smem,
      TensorStorageSharedQV,
      TensorStorageSeparateQV>;

  // Host side kernel arguments
  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQKV const stride_Q;
    Element const* const ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    Element const* const ptr_V;
    StrideQKV const stride_V;
    Element const* const ptr_dO;
    StrideQKV const stride_dO;
    ElementAccum* const ptr_dQaccum;
    ShapedQaccum const shape_dQaccum;
    StridedQaccum const stride_dQaccum;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    float const max_seq_len_inv;
    float const alpha;
    int const num_batch;
    int* const dq_semaphore;
    int const* const seq_offsets = nullptr;
    int const* const num_targets = nullptr;
    float const* const attn_scale = nullptr;
  };

  // Device side kernel params
  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const shape_Q;
    StrideQKV const stride_Q;
    Element const* const ptr_K;
    ShapeQKV const shape_K;
    StrideQKV const stride_K;
    Element const* const ptr_V;
    StrideQKV const stride_V;
    Element const* const ptr_dO;
    StrideQKV const stride_dO;
    ElementAccum* const ptr_dQaccum;
    ShapedQaccum const shape_dQaccum;
    StridedQaccum stride_dQaccum;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    float const max_seq_len_inv;
    float const alpha;
    int const num_batch;
    int* const dq_semaphore;
    int const* const seq_offsets;
    int const* const num_targets;
    float const* const attn_scale;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    if constexpr (Deterministic) {
      assert(args.dq_semaphore != nullptr);
    }
    return {
        args.ptr_Q,
        args.shape_Q,
        args.stride_Q,
        args.ptr_K,
        args.shape_K,
        args.stride_K,
        args.ptr_V,
        args.stride_V,
        args.ptr_dO,
        args.stride_dO,
        args.ptr_dQaccum,
        args.shape_dQaccum,
        args.stride_dQaccum,
        args.max_attn_len,
        args.min_full_attn_seq_len,
        args.contextual_seq_len,
        args.max_seq_len_inv,
        args.alpha,
        args.num_batch,
        args.dq_semaphore,
        args.seq_offsets,
        args.num_targets,
        args.attn_scale};
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_m_block_min_max(
      int const max_attn_len,
      int const uihlen,
      int const seqlen,
      int const n_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (Has_targets) {
      int n_idx_min = n_block * kBlockN;
      if (n_idx_min >= uihlen) {
        int n_idx_max = (n_block + 1) * kBlockN;
        return {
            std::max(0, n_idx_min / kBlockM),
            cute::ceil_div(std::min(n_idx_max, seqlen), kBlockM)};
      }
    }
    // uih part
    int m_block_max = cute::ceil_div(seqlen, kBlockM);
    if constexpr (Local) {
      if (n_block >= cute::ceil_div(0, kBlockN)) {
        m_block_max = std::min(
            m_block_max,
            cute::ceil_div((n_block + 1) * kBlockN + max_attn_len, kBlockM));
      }
    }
    int m_block_min = 0;
    if constexpr (Causal || Local) {
      m_block_min = std::max(m_block_min, (n_block * kBlockN) / kBlockM);
    }
    return {m_block_min, m_block_max};
  }

  template <typename SharedStorage, typename FrgTensordKV>
  CUTLASS_DEVICE bool mma(
      Params const& params,
      FrgTensordKV& tdKrdK,
      FrgTensordKV& tdVrdV,
      int thread_idx,
      cute::tuple<int32_t, int32_t, int32_t> block_coord,
      SharedStorage& shared_storage) {
    static_assert(
        is_rmem<FrgTensordKV>::value,
        "dK and dV tensor must be rmem resident.");

    int n_block = get<0>(block_coord);
    int bidh = get<1>(block_coord);
    int bidb = get<2>(block_coord);
    SeqlenInfo_t seqlen_info{
        bidb, get<0>(params.shape_Q), params.seq_offsets, params.num_targets};
    if constexpr (Jagged) {
      static constexpr int kBlockN = get<1>(TileShape_MNK{});
      if (n_block * kBlockN >= seqlen_info.seqlen) {
        return false;
      }
    }
    auto m_block_min_max = get_m_block_min_max(
        params.max_attn_len, seqlen_info.uihlen, seqlen_info.seqlen, n_block);
    int const m_block_min = get<0>(m_block_min_max);
    int const m_block_max = get<1>(m_block_min_max);

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sdO = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdO{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutV{});
    Tensor sQt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQt{});
    Tensor sdOt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_do.data()),
        SmemLayoutdOt{});
    Tensor sKt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutKt{});
    Tensor sP = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
        SmemLayoutPdS{});
    Tensor sPt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
        SmemLayoutPdSt{});
    Tensor sdS = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
        SmemLayoutPdS{});
    Tensor sdSt = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_ds.data()),
        SmemLayoutPdSt{});

    Tensor mQ = make_tensor(
        make_gmem_ptr(params.ptr_Q), params.shape_Q, params.stride_Q)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mdO = make_tensor(
        make_gmem_ptr(params.ptr_dO), params.shape_Q, params.stride_dO)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mK = make_tensor(
        make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mV = make_tensor(
        make_gmem_ptr(params.ptr_V), params.shape_K, params.stride_V)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mdQaccum = make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
        params.shape_dQaccum,
        params.stride_dQaccum)(_, bidh, !Jagged ? bidb : 0);

    Tensor gQ = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (M, K, _)
    Tensor gdO = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q, _0{}), mdO),
        select<0, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (M, K, _)
    Tensor gK = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (N, K)
    Tensor gV = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV),
        select<1, 2>(TileShape_MNK{}),
        make_coord(n_block, _0{})); // (N, K)
    Tensor gdQaccum = local_tile(
        domain_offset(
            make_coord(seqlen_info.offset_q_padded * kHeadDim), mdQaccum),
        Shape<Int<kBlockM * kHeadDim>>{},
        make_coord(_)); // (M * K, _)

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
    auto gmem_thr0_copy_QKV =
        gmem_tiled_copy_QKV.get_thread_slice(_0{}); // For index calculation
    R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
    auto r2s_thr_copy_dQaccum =
        r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_QKV.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_QKV.partition_D(sdO);
    // We can reuse r2s_thr_copy_dQaccum for this partitioning
    Tensor tdQgdQaccum = r2s_thr_copy_dQaccum.partition_D(gdQaccum);
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(mdQaccum);
    // printf("\n"); print(gdQaccum_); printf("\n"); print(gdQaccum);
    // printf("\n"); print(tdQgdQaccum); printf("\n"); }

    TiledMmaSdP tiled_mma_SdP;
    TiledMmadKV tiled_mma_dKV;
    TiledMmadQ tiled_mma_dQ;

    auto thr_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
    auto thr_mma_dKV = tiled_mma_dKV.get_thread_slice(thread_idx);
    auto thr_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors"
    // We have to use the templated mma_partition_fragment_AB instead of
    // cute::conditional_return or lambda, because some partition_fragment_A/B
    // don't compile.
    // https://stackoverflow.com/questions/50051473/if-constexpr-in-c17-does-not-work-in-a-non-templated-function
    Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV);

    // Copy Atom retiling
    auto smem_copy_atom_SdP_B = cute::conditional_return<MmaSdPEvenN>(
        SmemCopyAtom{}, SmemCopyAtomHalf{});
    auto smem_tiled_copy_QdO = cute::conditional_return<!SdP_swapAB>(
        make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_SdP),
        make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP));
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(thread_idx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = cute::conditional_return<!SdP_swapAB>(
        make_tiled_copy_B(smem_copy_atom_SdP_B, tiled_mma_SdP),
        make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_SdP));
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(thread_idx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    auto r2s_tiled_copy_PdS =
        make_tiled_copy_C(R2SCopyAtomPdS{}, tiled_mma_SdP);
    auto r2s_thr_copy_PdS = r2s_tiled_copy_PdS.get_thread_slice(thread_idx);
    Tensor tPsP =
        r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(
            sP, sPt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS =
        r2s_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(
            sdS, sdSt)); // ((Atom,AtomNum),PIPE_M,PIPE_N)
    // if (blockIdx.x == 0 && threadIdx.x == 128) { print(r2s_thr_copy_PdS);
    // print(sP); printf("\n"); print(sPt); printf("\n"); print(tPsP);
    // printf("\n"); print(tdSsdS); printf("\n"); }

    auto smem_copy_atom_dKV_B = cute::conditional_return<MmadKVEvenN>(
        SmemCopyAtomTransposed{}, SmemCopyAtomTransposedHalf{});
    auto smem_tiled_copy_PdSt = cute::conditional_return<!dKV_swapAB>(
        make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dKV),
        make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV));
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(thread_idx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = cute::conditional_return<!dKV_swapAB>(
        make_tiled_copy_B(smem_copy_atom_dKV_B, tiled_mma_dKV),
        make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dKV));
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(thread_idx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = cute::conditional_return<!dQ_swapAB>(
        make_tiled_copy_A(SmemCopyAtom{}, tiled_mma_dQ),
        make_tiled_copy_B(
            cute::conditional_return<MmadQEvenN>(
                SmemCopyAtom{}, SmemCopyAtomHalf{}),
            tiled_mma_dQ));
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(thread_idx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = cute::conditional_return<!dQ_swapAB>(
        make_tiled_copy_B(
            cute::conditional_return<MmadQEvenN>(
                SmemCopyAtomTransposed{}, SmemCopyAtomTransposedHalf{}),
            tiled_mma_dQ),
        make_tiled_copy_A(SmemCopyAtomTransposed{}, tiled_mma_dQ));
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(thread_idx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    // Predicates
    Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q);
    }

    flash::Mask<kBlockM, kBlockN, TiledMmaSdP, SdP_swapAB> mask(
        thread_idx,
        seqlen_info.seqlen,
        params.max_attn_len,
        params.min_full_attn_seq_len,
        params.contextual_seq_len,
        seqlen_info.uihlen);

    {
      Tensor tKgK =
          gmem_thr_copy_QKV.partition_S(gK); // (KCPY, KCPY_N, KCPY_K, nblocksN)
      Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
      Tensor tVgV =
          gmem_thr_copy_QKV.partition_S(gV); // (VCPY, VCPY_N, VCPY_K, nblocksN)
      Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
      // Predicates
      Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
      Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
      Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
      Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
#pragma unroll
      for (int k = 0; k < size(tKVpKV); ++k) {
        tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K);
      }
      // Do we need bound check to make sure the row doesn't go above kBlockN
      static constexpr bool EvenN =
          kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
      // static_assert(EvenN);  // It simplifies the loading of K and V
      // Instead of passing in tKVcKV, we pass in t0KVcKV and subtract the
      // offset from the limit (seqlen - n_block * kBlockN). This is because
      // the entries of t0KVcKV are known at compile time. int const
      // seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) + (EvenN
      //     ? seqlen_info.seqlen - n_block * kBlockN
      //     : std::min(seqlen_info.seqlen - n_block * kBlockN, kBlockN));
      // // Need Clear_OOB_MN to be true here since the gemm will sum over the
      // kBlockN dimension flash::copy</*Is_even_MN=*/false,
      // /*Is_even_K=*/false, /*Clear_OOB_MN=*/true, /*Clear_OOB_K=*/true>(
      //     gmem_tiled_copy_QKV, tVgV, tVsV, t0KVcKV, tKVpKV,
      //     seqlenk_row_limit);
      int const seqlenk_row_limit = seqlen_info.seqlen - n_block * kBlockN -
          get<0>(tKVcKV(_0{}, _0{}, _0{}));
#pragma unroll
      for (int m = 0; m < size<1>(tVsV); ++m) {
        // If kBlockN doesn't evenly divide the tiled copy, only the last `m`
        // needs to be checked
        if (EvenN || m < size<1>(tVsV) - 1 ||
            get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
          bool const predicate_n =
              get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
#pragma unroll
          for (int k = 0; k < size<2>(tVsV); ++k) {
            cute::copy(
                gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n),
                tVgV(_, m, k),
                tVsV(_, m, k));
          }
        }
      }
      if constexpr (V_in_regs) {
        flash::cp_async_fence();
      }
// flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true,
// /*Clear_OOB_K=*/true>(
//     gmem_tiled_copy_QKV, tKgK, tKsK, t0KVcKV, tKVpKV, seqlenk_row_limit);
#pragma unroll
      for (int m = 0; m < size<1>(tKsK); ++m) {
        if (EvenN || m < size<1>(tKsK) - 1 ||
            get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
          bool const predicate_n =
              get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
#pragma unroll
          for (int k = 0; k < size<2>(tKsK); ++k) {
            cute::copy(
                gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n),
                tKgK(_, m, k),
                tKsK(_, m, k));
          }
        }
      }
      flash::cp_async_fence();
    }

    if constexpr (V_in_regs) {
      flash::cp_async_wait<1>();
      __syncthreads();
      Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
      Tensor tdPsV_copy_view = smem_thr_copy_KV.partition_S(sV);
      cute::copy(smem_tiled_copy_KV, tdPsV_copy_view, tdPrV_copy_view);
      __syncthreads(); // Sync to avoid loading Q to smem_q, which overlaps with
                       // smem_v
    }

    // Do we need bound check to make sure the row doesn't go above kBlockM
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr bool EvenM =
        kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;

    auto load_Q = [&](int const m_block, int const smem_pipe_write) {
      Tensor tQsQ_cur = tQsQ(_, _, _, smem_pipe_write);
      Tensor tQgQ_cur = tQgQ(_, _, _, m_block);
      int const seqlenq_row_limit = seqlen_info.seqlen - m_block * kBlockM -
          get<0>(tQcQ(_0{}, _0{}, _0{}));
#pragma unroll
      for (int m = 0; m < size<1>(tQsQ); ++m) {
        // If kBlockM doesn't evenly divide the tiled copy, only the last `m`
        // needs to be checked
        if (EvenM || m < size<1>(tQsQ) - 1 ||
            get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
          bool const predicate_m =
              get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
#pragma unroll
          for (int k = 0; k < size<2>(tQsQ); ++k) {
            cute::copy(
                gmem_tiled_copy_QKV.with(tQpQ(k) && predicate_m),
                tQgQ_cur(_, m, k),
                tQsQ_cur(_, m, k));
          }
        }
      }
    };

    auto load_dO = [&](int const m_block, int const smem_pipe_write) {
      Tensor tdOsdO_cur = tdOsdO(_, _, _, smem_pipe_write);
      Tensor tdOgdO_cur = tdOgdO(_, _, _, m_block);
      int const seqlenq_row_limit = seqlen_info.seqlen - m_block * kBlockM -
          get<0>(tQcQ(_0{}, _0{}, _0{}));
#pragma unroll
      for (int m = 0; m < size<1>(tdOsdO); ++m) {
        // If kBlockM doesn't evenly divide the tiled copy, only the last `m`
        // needs to be checked
        if (EvenM || m < size<1>(tdOsdO) - 1 ||
            get<0>(tQcQ(_0{}, m, _0{})) < kBlockM) {
          bool const predicate_m =
              get<0>(t0QcQ(_0{}, m, _0{})) < seqlenq_row_limit;
#pragma unroll
          for (int k = 0; k < size<2>(tdOsdO); ++k) {
            cute::copy(
                gmem_tiled_copy_QKV.with(tQpQ(k) && predicate_m),
                tdOgdO_cur(_, m, k),
                tdOsdO_cur(_, m, k));
          }
        }
      }
    };

    int m_block = m_block_min;

    // Note, using the for_each() function here to ensure `stage` is of type
    // Int<x>.
    for_each(make_int_sequence<kStages>{}, [&](auto stage) {
      static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
      static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
      if constexpr (!Is_last_stage || kStages == 1) {
        if (Is_first_stage || m_block + stage < m_block_max) {
          load_Q(m_block + stage, stage);
        }
      }
      // We want the fence outside the if statement to have a fixed number of
      // cp.async commits. so that we can wait with the correct number of
      // outstanding commits.
      cute::cp_async_fence();
      if constexpr (stage < kStages_dO) {
        if (Is_first_stage || m_block + stage < m_block_max) {
          load_dO(m_block + stage, stage);
        }
        cute::cp_async_fence();
      }
    });

    int smem_pipe_read = 0, smem_pipe_read_do = 0,
        smem_pipe_write = kStages - 1, smem_pipe_write_do = 0;

    auto load_Q_next = [&] {
      if (m_block + (kStages > 1 ? kStages - 1 : 1) < m_block_max) {
        load_Q(
            m_block + (kStages > 1 ? kStages - 1 : 1),
            kStages > 1 ? smem_pipe_write : 0);
      }
      cute::cp_async_fence();
    };

    auto load_dO_next = [&] {
      if (m_block + kStages_dO < m_block_max) {
        load_dO(m_block + kStages_dO, kStages_dO > 1 ? smem_pipe_write_do : 0);
      }
      cute::cp_async_fence();
    };

    clear(tdKrdK);
    clear(tdVrdV);

    auto bwd_step = [&](int m_block, auto mask_fn) {
      Tensor tSrS = partition_fragment_C(
          tiled_mma_SdP,
          select < !SdP_swapAB ? 0 : 1,
          !SdP_swapAB ? 1 : 0 > (TileShape_MNK{}));
      clear(tSrS);
      flash::cp_async_wait<(kStages > 1) ? 1 : 0>();
      __syncthreads();
      Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(
          thr_mma_SdP, sQ(_, _, _0{}));
      Tensor tSrK =
          mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sK);
      // if (cute::thread0()) { print(tiled_mma_SdP); print(tSrS); printf("\n");
      // print(tSrQ); printf("\n"); print(tSrK); printf("\n"); print(tSsQ);
      // printf("\n"); print(tSsK); printf("\n"); }
      flash::gemm_sm80<false /*A_in_regs*/, false /*B_in_regs*/, SdP_swapAB>(
          tSrS,
          tSrQ,
          tSrK,
          tSsQ(_, _, _, kStages > 1 ? smem_pipe_read : 0),
          tSsK,
          tiled_mma_SdP,
          smem_tiled_copy_QdO,
          smem_tiled_copy_KV,
          smem_thr_copy_QdO,
          smem_thr_copy_KV,
          nullptr /*hook*/);

      // Reshape tSrS from (4, MMA_N, MMA_M) to (nrow=(2, MMA_M), ncol=(2,
      // MMA_N))
      Tensor scores = make_tensor(
          tSrS.data(),
          flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
              tSrS.layout()));
      Tensor tSrS_sigmoid = make_tensor_like(tSrS);
      Tensor sigmoid = make_tensor(
          tSrS_sigmoid.data(),
          flash::convert_layout_acc_rowcol</*Transposed=*/SdP_swapAB>(
              tSrS_sigmoid.layout()));

      float scale = params.attn_scale == nullptr
          ? params.max_seq_len_inv
          : params.attn_scale[0]; // TODO: generalize
      mask_fn(tSrS, m_block);
#pragma unroll
      for (int mi = 0; mi < size<0>(scores); ++mi) {
#pragma unroll
        for (int ni = 0; ni < size<1>(scores); ++ni) {
          scores(mi, ni) = scores(mi, ni) * params.alpha;
          sigmoid(mi, ni) =
              __fdividef(1., 1.0f + cutlass::fast_exp(-scores(mi, ni)));
          scores(mi, ni) = sigmoid(mi, ni) * scores(mi, ni) * scale;
        }
      }
      mask_fn(tSrS_sigmoid, m_block);

      Tensor tdPrdP = partition_fragment_C(
          tiled_mma_SdP,
          select < !SdP_swapAB ? 0 : 1,
          !SdP_swapAB ? 1 : 0 > (TileShape_MNK{}));
      clear(tdPrdP);
      int smem_pipe_read_do_cur =
          Q_dO_same_stages ? smem_pipe_read : smem_pipe_read_do;
      flash::cp_async_wait<(kStages_dO > 1) ? 1 : 0>();
      __syncthreads();
      auto hook = cute::conditional_return<(kStages > 1)>(load_Q_next, nullptr);
      Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(
          thr_mma_SdP, sdO(_, _, _0{}));
      Tensor tdPrV_cur = cute::conditional_return<V_in_regs>(
          tdPrV, mma_partition_fragment_AB</*A=*/SdP_swapAB>(thr_mma_SdP, sV));
      flash::gemm_sm80<false /*A_in_regs*/, V_in_regs, SdP_swapAB>(
          tdPrdP,
          tdPrdO,
          tdPrV_cur,
          tdPsdO(_, _, _, kStages_dO > 1 ? smem_pipe_read_do_cur : 0),
          tdPsV,
          tiled_mma_SdP,
          smem_tiled_copy_QdO,
          smem_tiled_copy_KV,
          smem_thr_copy_QdO,
          smem_thr_copy_KV,
          hook);

      // Reshape tdPrdP from (4, MMA_N, MMA_M) to (nrow=(2, MMA_M), ncol=(2,
      // MMA_N))
      Tensor dS = make_tensor(tdPrdP.data(), scores.layout());
#pragma unroll
      for (int mi = 0; mi < size<0>(dS); ++mi) {
#pragma unroll
        for (int ni = 0; ni < size<1>(dS); ++ni) {
          dS(mi, ni) = dS(mi, ni) * sigmoid(mi, ni) * scale +
              dS(mi, ni) * scores(mi, ni) * (1.f - sigmoid(mi, ni));
          dS(mi, ni) = dS(mi, ni) * params.alpha;
        }
      }
      // if (cute::thread0()) { print_tensor(dS); }

      // Convert scores from fp32 to fp16/bf16
      Tensor rP = make_tensor_like<Element>(tSrS);
      flash::convert_type_out(tSrS, rP);
      if constexpr (!Mma_dKV_is_RS) {
        Tensor tPaP =
            r2s_thr_copy_PdS.retile_S(rP); // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(r2s_tiled_copy_PdS, tPaP, tPsP);
      }
      Tensor rdS = make_tensor_like<Element>(tdPrdP);
      flash::convert_type_out(tdPrdP, rdS);
      if constexpr (!Mma_dKV_is_RS) {
        __syncthreads();
      } // Make sure P is written
      // For hdim 64, It's faster to write to smem_dS first before the dV gemm
      Tensor tdSadS =
          r2s_thr_copy_PdS.retile_S(rdS); // ((Atom,AtomNum), MMA_N, MMA_N)
      cute::copy(r2s_tiled_copy_PdS, tdSadS, tdSsdS);

      Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(
          thr_mma_dKV, sdOt(_, _, _0{}));
      Tensor tdVsdO_cur =
          tdVsdOt(_, _, _, kStages_dO > 1 ? smem_pipe_read_do_cur : 0);
      if constexpr (Mma_dKV_is_RS) {
        Tensor tdVrP = make_tensor(
            rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
        flash::gemm_rs_sm80(
            tdVrdV,
            tdVrP,
            tdVrdO,
            tdVsdO_cur,
            tiled_mma_dKV,
            smem_tiled_copy_QdOt,
            smem_thr_copy_QdOt);
      } else {
        Tensor tdVrP =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sPt);
        flash::gemm_sm80<
            false /*A_in_regs*/,
            false /*B_in_regs*/,
            /*SwapAB=*/dKV_swapAB>(
            tdVrdV,
            tdVrP,
            tdVrdO,
            tdVsPt,
            tdVsdO_cur,
            tiled_mma_dKV,
            smem_tiled_copy_PdSt,
            smem_tiled_copy_QdOt,
            smem_thr_copy_PdSt,
            smem_thr_copy_QdOt,
            nullptr);
      }
      // if (cute::thread0()) { print_tensor(tdVrdV); }
      __syncthreads(); // make sure sdS is written
      auto do_mma_dQ = [&](auto hook) {
        Tensor tdQrdQ = partition_fragment_C(
            tiled_mma_dQ,
            select < !dQ_swapAB ? 0 : 2,
            !dQ_swapAB ? 2 : 0 > (TileShape_MNK{}));
        clear(tdQrdQ);
        Tensor tdQrdS =
            mma_partition_fragment_AB</*A=*/!dQ_swapAB>(thr_mma_dQ, sdS);
        Tensor tdQrK =
            mma_partition_fragment_AB</*A=*/dQ_swapAB>(thr_mma_dQ, sKt);
        flash::gemm_sm80<
            false /*A_in_regs*/,
            false /*B_in_regs*/,
            /*SwapAB=*/dQ_swapAB>(
            tdQrdQ,
            tdQrdS,
            tdQrK,
            tdQsdS,
            tdQsKt,
            tiled_mma_dQ,
            smem_tiled_copy_dS,
            smem_tiled_copy_Kt,
            smem_thr_copy_dS,
            smem_thr_copy_Kt,
            hook);
        // if (cute::thread0()) { print_tensor(tdQrdQ); }
        // We can reuse r2s_thr_copy_dQaccum for this partitioning
        Tensor tdQrdQ_atomic = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);
        Tensor tdQgdQaccum_atomic = tdQgdQaccum(_, _, m_block);
        static_assert(
            CUTE_STATIC_V(size(tdQrdQ_atomic)) ==
            CUTE_STATIC_V(size(tdQgdQaccum_atomic)));
#pragma unroll
        for (int i = 0; i < size(tdQrdQ_atomic); ++i) {
          atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i));
        }
      };
      // If kStages == 1, we want to do Mma_dK first so we can start loading Q
      // for the next iteration
      if constexpr (kStages > 1) {
        do_mma_dQ(load_dO_next);
      }
      Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(
          thr_mma_dKV, sQt(_, _, _0{}));
      Tensor tdKsQ_cur = tdKsQt(_, _, _, kStages > 1 ? smem_pipe_read : 0);
      if constexpr (Mma_dKV_is_RS) {
        Tensor tdKrdS = make_tensor(
            rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
        flash::gemm_rs_sm80(
            tdKrdK,
            tdKrdS,
            tdKrQ,
            tdKsQ_cur,
            tiled_mma_dKV,
            smem_tiled_copy_QdOt,
            smem_thr_copy_QdOt);
      } else {
        Tensor tdKrdS =
            mma_partition_fragment_AB</*A=*/!dKV_swapAB>(thr_mma_dKV, sdSt);
        flash::gemm_sm80<
            false /*A_in_regs*/,
            false /*B_in_regs*/,
            /*SwapAB=*/dKV_swapAB>(
            tdKrdK,
            tdKrdS,
            tdKrQ,
            tdKsdSt,
            tdKsQ_cur,
            tiled_mma_dKV,
            smem_tiled_copy_PdSt,
            smem_tiled_copy_QdOt,
            smem_thr_copy_PdSt,
            smem_thr_copy_QdOt,
            cute::conditional_return<(kStages > 1)>(nullptr, load_dO_next));
      }
      if constexpr (kStages == 1) {
        __syncthreads();
        do_mma_dQ(load_Q_next);
      }
      // if (cute::thread0()) { print_tensor(tdKrdK); }

      smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
      smem_pipe_read_do =
          smem_pipe_read_do < kStages_dO - 1 ? smem_pipe_read_do + 1 : 0;
      smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
      smem_pipe_write_do =
          smem_pipe_write_do < kStages_dO - 1 ? smem_pipe_write_do + 1 : 0;
    };

    if constexpr (Has_targets) {
      if (n_block * kBlockN >= seqlen_info.uihlen) {
        auto mask_fn = [&](auto& tSrS, int m_block) {
          mask.template apply<
              true /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal*/,
              false /*Local*/,
              false /*Contexual_mask*/,
              Has_targets>(tSrS, m_block, n_block);
        };
        for (; m_block < m_block_max; ++m_block) {
          bwd_step(m_block, mask_fn);
        }
        return true;
      } else if ((n_block + 1) * kBlockN >= seqlen_info.uihlen) {
        if constexpr ((Causal || Local) && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                Causal,
                Local,
                false /*Contexual_mask*/,
                Has_targets>(tSrS, m_block, n_block);
          };
          int const m_block_masking_max =
              ((n_block + 1) * kBlockN - 1) / kBlockM + 1;
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < std::min(m_block_max, m_block_masking_max);
               ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }

        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const m_block_max_before_local_mask =
            !Local || !SeparateMaskingIterations
            ? m_block_max
            : std::min(
                  m_block_max,
                  (n_block * kBlockN + params.max_attn_len) / kBlockM);

        auto mask_fn = [&](auto& tSrS, int m_block) {
          mask.template apply<
              true /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              Causal && !SeparateMaskingIterations,
              Local && !SeparateMaskingIterations,
              false /*Contexual_mask*/,
              Has_targets>(tSrS, m_block, n_block);
        };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max_before_local_mask; ++m_block) {
          bwd_step(m_block, mask_fn);
        }

        if constexpr (Local && SeparateMaskingIterations) {
          auto mask_fn = [&](auto& tSrS, int m_block) {
            mask.template apply<
                true /*Seqlenq_mask*/,
                true /*Seqlenk_mask*/,
                false /*Causal_mask*/,
                Local,
                false /*Contexual_mask*/,
                Has_targets>(tSrS, m_block, n_block);
          };
          CUTLASS_PRAGMA_NO_UNROLL
          for (; m_block < m_block_max; ++m_block) {
            bwd_step(m_block, mask_fn);
          }
        }
        return true;
      }
    }

    // We have separate iterations with causal masking. Not necessary for hdim
    // 128 but for hdim 64 this helps quite a bit to not have to do causal
    // masking for most of the iterations.
    if constexpr ((Causal || Local) && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            Local,
            false /*Contexual_mask*/,
            false /*Target_mask*/>(tSrS, m_block, n_block);
      };
      int const m_block_masking_max =
          ((n_block + 1) * kBlockN - 1) / kBlockM + 1;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < std::min(m_block_max, m_block_masking_max); ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }

    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    int const m_block_max_before_local_mask =
        !Local || !SeparateMaskingIterations
        ? m_block_max
        : std::min(
              m_block_max, (n_block * kBlockN + params.max_attn_len) / kBlockM);

    auto mask_fn = [&](auto& tSrS, int m_block) {
      mask.template apply<
          true /*Seqlenq_mask*/,
          true /*Seqlenk_mask*/,
          Causal && !SeparateMaskingIterations,
          Local && !SeparateMaskingIterations,
          false /*Contexual_mask*/,
          false /*Target_mask*/>(tSrS, m_block, n_block);
    };
    CUTLASS_PRAGMA_NO_UNROLL
    for (; m_block < m_block_max_before_local_mask; ++m_block) {
      bwd_step(m_block, mask_fn);
    }

    if constexpr (Local && SeparateMaskingIterations) {
      auto mask_fn = [&](auto& tSrS, int m_block) {
        mask.template apply<
            true /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal_mask*/,
            Local,
            false /*Contexual_mask*/,
            false /*Target_mask*/>(tSrS, m_block, n_block);
      };
      CUTLASS_PRAGMA_NO_UNROLL
      for (; m_block < m_block_max; ++m_block) {
        bwd_step(m_block, mask_fn);
      }
    }
    return true;
  }
};

} // namespace flash
