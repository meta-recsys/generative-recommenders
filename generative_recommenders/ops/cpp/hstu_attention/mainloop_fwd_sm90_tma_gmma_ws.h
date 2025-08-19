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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "mask.h"
#include "named_barrier.h"
#include "seqlen.h"
#include "sm90_pipeline_no_cluster.h"
#include "utils.h"

namespace hstu {

using namespace cute;

template <
    int Stages,
    class ClusterShape_,
    class TileShape_MNK_,
    class Element_,
    class ElementAccum_,
    class ArchTag_,
    bool Causal,
    bool Local,
    bool Contexual_mask,
    bool Jagged,
    bool Has_targets,
    bool Mma1_is_RS,
    bool V_colmajor_,
    bool Cross>
struct CollectiveMainloopFwdSm90 {
  static constexpr int kStages = Stages;
  using ClusterShape = ClusterShape_;
  using TileShape_MNK = TileShape_MNK_;
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  using ArchTag = ArchTag_;
  static constexpr bool Is_FP8 =
      cute::is_same_v<Element, cutlass::float_e4m3_t> ||
      cute::is_same_v<Element, cutlass::float_e5m2_t>;
  ;
  static constexpr bool V_colmajor = V_colmajor_;
  static constexpr bool Transpose_V = Is_FP8 && !V_colmajor;
  using SeqlenInfo_t = hstu::SeqlenInfoQKFwd<Jagged, Cross, Has_targets>;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr cute::GMMA::Major MmaMajorV =
      !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
  static constexpr cute::GMMA::Major TmaMajorV =
      !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  // Register bandwidth is actually a bottleneck so we don't want Q to be in
  // registers. Leaving this option here for reference.
  static constexpr bool Mma0_is_RS = false;
  // We can have Mma1 (P @ V) with P in smem in rmem to reduce register pressure
  // at the cost of more smem.
  static_assert(!(!Mma1_is_RS && Is_FP8), "Mma1 must be RS if FP8");
  static_assert(
      !(!Mma1_is_RS && Transpose_V),
      "Mma1 must be RS if Transpose_V");

  using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using TiledMma0 = decltype(cute::make_tiled_mma(
      std::conditional_t<
          !Mma0_is_RS,
          decltype(cute::GMMA::ss_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShape_MNK>()),
          decltype(cute::GMMA::rs_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   TileShape_MNK>())>{},
      AtomLayoutMNK{}));
  using TiledMma1 = decltype(cute::make_tiled_mma(
      std::conditional_t<
          !Mma1_is_RS,
          decltype(cute::GMMA::ss_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   decltype(select<0, 2, 1>(TileShape_MNK{})),
                   GMMA::Major::K,
                   MmaMajorV>()),
          decltype(cute::GMMA::rs_op_selector<
                   Element,
                   Element,
                   ElementAccum,
                   decltype(select<0, 2, 1>(TileShape_MNK{})),
                   GMMA::Major::K,
                   MmaMajorV>())>{},
      AtomLayoutMNK{}));

  static constexpr int NumMmaThreads = size(TiledMma0{});
  static constexpr int NumProducerThreads = !Transpose_V
      ? cutlass::NumThreadsPerWarp
      : cutlass::NumThreadsPerWarpGroup;
  static_assert(NumMmaThreads % cutlass::NumThreadsPerWarpGroup == 0);
  static constexpr int NumMmaWarpGroups =
      NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
  static_assert(
      NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

  using SmemLayoutAtomQ =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));

  using SmemLayoutAtomVt =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               TmaMajorV,
               Element,
               decltype(cute::get<2>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(
          shape<2>(TileShape_MNK{}),
          shape<1>(TileShape_MNK{}),
          Int<kStages>{}),
      std::conditional_t<
          TmaMajorV == GMMA::Major::K,
          cute::Step<_1, _2, _3>,
          cute::Step<_2, _1, _3>>{}));

  using SmemLayoutAtomVtMma =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               MmaMajorV,
               Element,
               decltype(cute::get<2>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{},
      make_shape(
          shape<2>(TileShape_MNK{}),
          shape<1>(TileShape_MNK{}),
          Int<kStages>{}),
      std::conditional_t<
          MmaMajorV == GMMA::Major::K,
          cute::Step<_1, _2, _3>,
          cute::Step<_2, _1, _3>>{}));

  // Only used if we're using cp.async to load V
  using SmemLayoutAtomVCpAsync =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutVCpAsync = decltype(tile_to_shape(
      SmemLayoutAtomVCpAsync{},
      make_shape(
          shape<1>(TileShape_MNK{}),
          shape<2>(TileShape_MNK{}),
          Int<kStages>{})));

  using SmemLayoutAtomP =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutP =
      decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_MNK{})));

  using SmemCopyAtomP = Copy_Atom<cute::SM90_U32x4_STSM_N, Element>;

  // Use LDSM.T and STSM to transpose V in the case of FP8 and V being
  // row-major. For FP16/BF16 we don't do any transposing.
  static_assert(!Transpose_V || (kHeadDim % 32 == 0 && kBlockN % 32 == 0));
  static constexpr bool kHeadDim_multiple_64 = kHeadDim % 64 == 0;
  // Either kHeadDim is a multiple of 64 (in which case we use a block size of
  // 64 x 32 for the transpose), or we need kBlockN to be a multiple of 64 (in
  // which case we use a block size of 32 x 64 for the transpose).
  static_assert(!Transpose_V || (kHeadDim_multiple_64 || kBlockN % 64 == 0));
  using LDSM_thread_shape = std::conditional_t<
      kHeadDim_multiple_64,
      Shape<_32, _4, _1, _1>,
      Shape<_16, _4, _1, _2>>;
  using LDSM_thread_stride = std::conditional_t<
      kHeadDim_multiple_64,
      Stride<_4, _1, _0, _0>,
      Stride<_4, _1, _0, _64>>;
  using LDSM_value_shape = Shape<_2, _2, _1, _4>;
  using LDSM_value_stride = Stride<_1, _2, _16, _4>;
  using LDSM_divide_shape =
      std::conditional_t<kHeadDim_multiple_64, Shape<_64, _8>, Shape<_32, _8>>;
  using S2RTiledCopyVt = decltype(make_tiled_copy(
      Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
      Layout<LDSM_thread_shape, LDSM_thread_stride>{},
      Layout<LDSM_value_shape, LDSM_value_stride>{}));

  using STSM_thread_shape = std::conditional_t<
      kHeadDim_multiple_64,
      Shape<_8, _4, _4, _1>,
      Shape<_8, _4, _2, _2>>;
  using STSM_thread_stride = std::conditional_t<
      kHeadDim_multiple_64,
      Stride<_4, _1, _32, _0>,
      Stride<_4, _1, _32, _64>>;
  using STSM_value_shape = Shape<_1, _4, _2, _2>;
  using STSM_value_stride = Stride<_0, _1, _4, _8>;
  using STSM_divide_shape = Shape<_8, _16>;
  // These will not permute the columns of V (the kHeadDim dimension) but incur
  // bank conflicts so a little slower (e.g. 1150 TFLOPS for hdim 256 instead of
  // 1200 TFLOPS). Instead we will permute the cols of V, and un-permute the
  // cols of O in the epilogue. using STSM_value_shape = Shape<_2, _4, _1, _2>;
  // using STSM_value_stride = Stride<_4, _1, _0, _8>;
  // using STSM_divide_shape = Shape<_16, _16>;
  using R2STiledCopyV = decltype(make_tiled_copy(
      Copy_Atom<SM90_U32x4_STSM_N, Element>{},
      Layout<STSM_thread_shape, STSM_thread_stride>{},
      Layout<STSM_value_shape, STSM_value_stride>{}));

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV =
      decltype(cutlass::gemm::collective::detail::
                   sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));

  // We use CpAsync for K and V if PagedKV and AppendKV, since TMA doesn't work
  // there
  static constexpr int kGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(
      kHeadDim % kGmemElemsPerLoad == 0,
      "Headdim must be a multiple of kGmemElemsPerLoad");
  // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g.
  // if hdim=128, we want each thread to have 4 loads in the M direction and 2
  // vectorized load in the K direction. We want each thread to have at least 2
  // loads in the K direction since in the case of non-interleaved rotary
  // (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1,
  // etc), each thread will load twice from the same row.
  static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
  static constexpr int kBlockKGmem =
      (kBytePerHalfRow % 128 == 0 ? 128
                                  : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) /
      sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
  static_assert(
      NumMmaThreads % kGmemThreadsPerRow == 0,
      "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
  // We assume threads loading the same row are in the same warp. This is for an
  // optimization in PagedKV where these threads share the same page table entry
  // and share the work of computing pointers to paged K and paged V.
  static_assert(
      cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0,
      "kGmemThreadsPerRow must divide NumThreadsPerWarp");
  using GmemLayoutAtom = Layout<
      Shape<Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to
  // avoid predication
  static_assert(
      kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0,
      "kBlockM must be a multiple of NumMmaThreads / kGmemThreadsPerRow");

  using ShapeQKV =
      cute::Shape<int32_t, int32_t, int32_t, int32_t>; // (seqlen, d, head,
                                                       // batch)
  using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using StrideV = std::conditional_t<
      !V_colmajor,
      StrideQK,
      cute::Stride<_1, int64_t, int64_t, int64_t>>;
  // ((qhead_per_khead, seqlen), d, nheads_kv, batch, num_splits)
  using ShapeQPacked = ShapeQKV;
  using StrideQPacked = StrideQK;
  using StrideDescale = cute::Stride<int64_t, int64_t>;

  using TMA_Q = decltype(make_tma_copy_A_sm90(
      GmemTiledCopyQ{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideQK{}),
      SmemLayoutQ{},
      TileShape_MNK{},
      ClusterShape{}));

  using TMA_K = decltype(make_tma_copy_B_sm90(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          StrideQK{}),
      take<0, 2>(SmemLayoutK{}),
      TileShape_MNK{},
      ClusterShape{})); // mcast along M mode for this N load, if any

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)),
          ShapeQKV{},
          select<1, 0, 2, 3>(StrideV{})),
      take<0, 2>(SmemLayoutVt{}),
      select<2, 1>(TileShape_MNK{}),
      size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

  // Set the bytes transferred in this TMA transaction (may involve multiple
  // issues)
  static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(
      size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutVt{})) * cutlass::sizeof_bits_v<Element> / 8);
  static_assert(TmaTransactionBytesK == TmaTransactionBytesV);

  using PipelineTmaAsync = std::conditional_t<
      CUTE_STATIC_V(size(ClusterShape{})) == 1,
      typename cutlass::PipelineTmaAsyncNoCluster<kStages>,
      typename cutlass::PipelineTmaAsync<kStages>>;
  using MainloopPipelineK = PipelineTmaAsync;
  using MainloopPipelineV = std::conditional_t<
      !Transpose_V,
      PipelineTmaAsync,
      typename cutlass::PipelineAsync<kStages>>;
  using MainloopPipelineVt = PipelineTmaAsync;
  // We always use TMA for K_new and V_new
  using MainloopPipelineKVNew = PipelineTmaAsync;
  using PipelineState = cutlass::PipelineState<kStages>;

  // If PackGQA, we use cp.async (instead of TMA) to load Q, so we want smem_q
  // to be aligned and have sQ being position_independent_swizzle_tensor. If
  // !Use_TMA_KV, we use cp.async (instead of TMA) to load K & V, so we want
  // smem_k and smem_v to be aligned.
  static constexpr size_t SmemAlignmentQ =
      !Mma0_is_RS ? 128 : cutlass::detail::alignment_for_swizzle(SmemLayoutQ{});
  static constexpr size_t SmemAlignmentK = 128;
  static constexpr size_t SmemAlignmentVtNoTranspose =
      cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
  static_assert(
      SmemAlignmentQ >= 128 and SmemAlignmentK >= 128 &&
          SmemAlignmentVtNoTranspose >= 128,
      "Require at least 128B alignment");
  static constexpr size_t SmemAlignmentP =
      cutlass::detail::alignment_for_swizzle(SmemLayoutP{});
  static_assert(SmemAlignmentP >= 128, "Require at least 128B alignment");

  using SmemP_t = std::conditional_t<
      Mma1_is_RS,
      cute::array<Element, 0>,
      cute::
          array_aligned<Element, cute::cosize_v<SmemLayoutP>, SmemAlignmentP>>;
  // Sometimes even with SmemP_t = cute::array<Element, 0>, putting it in the
  // TensorStorage struct causes smem size to go from 227KB to 228KB and we get
  // "invalid argument".

  struct TensorStorageWithoutPNoTranspose : cute::aligned_struct<cute::max(
                                                SmemAlignmentQ,
                                                SmemAlignmentK,
                                                SmemAlignmentVtNoTranspose)> {
    cute::array_aligned<
        Element,
        cute::cosize_v<SmemLayoutVt>,
        SmemAlignmentVtNoTranspose>
        smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ>
        smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK>
        smem_k;
  };

  struct TensorStorageWithPNoTranspose : cute::aligned_struct<cute::max(
                                             SmemAlignmentQ,
                                             SmemAlignmentK,
                                             SmemAlignmentVtNoTranspose,
                                             SmemAlignmentP)> {
    cute::array_aligned<
        Element,
        cute::cosize_v<SmemLayoutVt>,
        SmemAlignmentVtNoTranspose>
        smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ>
        smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK>
        smem_k;
    SmemP_t smem_p;
  };

  using TensorStorageNoTranspose = std::conditional_t<
      Mma1_is_RS,
      TensorStorageWithoutPNoTranspose,
      TensorStorageWithPNoTranspose>;

  static constexpr size_t SmemAlignmentVt =
      cutlass::detail::alignment_for_swizzle(SmemLayoutVt{});
  static constexpr size_t SmemAlignmentV =
      cutlass::detail::alignment_for_swizzle(SmemLayoutVtMma{});
  static_assert(
      SmemAlignmentVt >= 128 and SmemAlignmentV >= 128,
      "Require at least 128B alignment");
  struct TensorStorageTransposeV
      : cute::aligned_struct<
            cute::max(SmemAlignmentQ, SmemAlignmentK, SmemAlignmentV)> {
    cute::
        array_aligned<Element, cute::cosize_v<SmemLayoutVtMma>, SmemAlignmentV>
            smem_v;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutVt>, SmemAlignmentVt>
        smem_vt;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQ>
        smem_q;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentK>
        smem_k;
  };

  using TensorStorage = std::conditional_t<
      !Transpose_V,
      TensorStorageNoTranspose,
      TensorStorageTransposeV>;

  // These are tuned for speed. They don't affect correctness.
  static constexpr bool UseSchedulerBarrier =
      (NumMmaWarpGroups >= 2) && (!Is_FP8 ? kHeadDim <= 128 : kHeadDim >= 128);
  static constexpr bool RescaleOBeforeGemm = kHeadDim > 128 &&
      (!Is_FP8 || V_colmajor);

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
    int const num_softmax_heads;
    int const num_groups;
    int const batch_size_per_group;
    int const* const seq_offsets = nullptr;
    int const* const seq_offsets_q = nullptr;
    int const* const num_targets = nullptr;
    int const* const max_seq_len_tensor = nullptr;
    int const* const contextual_seq_len_tensor = nullptr;
    int const* const max_attn_len_tensor = nullptr;
    int const* const min_full_attn_seq_len_tensor = nullptr;
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
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    float const *ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
    StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
    float const max_seq_len_inv;
    float const alpha;
    float const alpha_log2;
    int const max_attn_len;
    int const min_full_attn_seq_len;
    int const contextual_seq_len;
    int const num_softmax_heads;
    int const num_groups;
    int const batch_size_per_group;
    int const* const seq_offsets = nullptr;
    int const* const seq_offsets_q = nullptr;
    int const* const num_targets = nullptr;
    int const* const max_seq_len_tensor = nullptr;
    int const* const contextual_seq_len_tensor = nullptr;
    int const* const max_attn_len_tensor = nullptr;
    int const* const min_full_attn_seq_len_tensor = nullptr;
    float const* const attn_scale = nullptr;
    bool const scalar_scale = true;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ =
        make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
    TMA_Q tma_load_Q = make_tma_copy_A_sm90(
        GmemTiledCopyQ{},
        mQ,
        SmemLayoutQ{},
        TileShape_MNK{},
        ClusterShape{}); // no mcast for Q
    Tensor mK =
        make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TMA_K tma_load_K = make_tma_copy_B_sm90(
        GmemTiledCopyKV{},
        mK,
        take<0, 2>(SmemLayoutK{}),
        TileShape_MNK{},
        ClusterShape{}); // mcast along M mode for this N load, if any
    Tensor mV = make_tensor(
        make_gmem_ptr(args.ptr_V),
        select<1, 0, 2, 3>(args.shape_K),
        select<1, 0, 2, 3>(args.stride_V));
    TMA_V tma_load_V = make_tma_copy(
        GmemTiledCopyKV{},
        mV,
        take<0, 2>(SmemLayoutVt{}),
        select<2, 1>(TileShape_MNK{}),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any
    auto const shape_Q_packed = cute::conditional_return<true>(
        args.shape_Q,
        make_shape(
            make_shape(1, get<0>(args.shape_Q)),
            get<1>(args.shape_Q),
            get<2>(args.shape_K),
            get<3>(args.shape_Q)));
    auto const stride_Q_packed = cute::conditional_return<true>(
        args.stride_Q,
        make_stride(
            make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)),
            get<1>(args.stride_Q),
            get<2>(args.stride_Q),
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
        tma_load_Q,
        tma_load_K,
        tma_load_V,
        args.ptr_q_descale,
        args.ptr_k_descale,
        args.ptr_v_descale,
        args.stride_q_descale,
        args.stride_k_descale,
        args.stride_v_descale,
        args.max_seq_len_inv,
        args.alpha,
        float(args.alpha * M_LOG2E),
        args.max_attn_len,
        args.min_full_attn_seq_len,
        args.contextual_seq_len,
        args.num_softmax_heads,
        args.num_groups,
        args.batch_size_per_group,
        args.seq_offsets,
        args.seq_offsets_q,
        args.num_targets,
        args.max_seq_len_tensor,
        args.contextual_seq_len_tensor,
        args.max_attn_len_tensor,
        args.min_full_attn_seq_len_tensor,
        args.attn_scale,
        args.scalar_scale};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best
  /// performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_n_block_min_max(
      int max_attn_len,
      int min_full_attn_seq_len,
      int contextual_seq_len,
      int uihlen,
      int m_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (Contexual_mask) {
      if (m_block * kBlockM < contextual_seq_len) {
        return {0, cute::ceil_div(uihlen, kBlockN)};
      }
    }
    if constexpr (Has_targets) {
      int m_idx_max = (m_block + 1) * kBlockM;
      if (m_idx_max > uihlen) {
        return {0, cute::ceil_div(uihlen, kBlockN)};
      }
    }
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
      int m_idx_max = (m_block + 1) * kBlockM;
      if (min_full_attn_seq_len == 0 ||
          m_idx_max <= uihlen - min_full_attn_seq_len) {
        n_block_min = std::max(int(0), (m_idx_min - max_attn_len) / kBlockN);
        if constexpr (Contexual_mask) {
          // row contexual without sink
          if (n_block_min * kBlockN < contextual_seq_len) {
            n_block_min = 0;
          }
        }
      } else {
        n_block_min = 0;
      }
    } else {
      n_block_min = 0;
    }
    return {n_block_min, n_block_max};
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_target_n_block_min_max(
      int n_block_max,
      int uihlen,
      int seqlen,
      int m_block) {
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
  int get_contexual_n_block_max(
      int n_block_min,
      int min_full_attn_seq_len,
      int contextual_seq_len,
      int uihlen,
      int m_block) {
    return 0;
    // TODO: reenable below once contexual + semi local implementation is
    // finalized
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (!Local) {
      return 0;
    }
    if (m_block * kBlockM < contextual_seq_len) {
      return 0;
    }
    int m_idx_max = (m_block + 1) * kBlockM;
    if constexpr (Has_targets) {
      if (m_idx_max > uihlen) {
        return 0;
      }
    }
    if (min_full_attn_seq_len == 0 ||
        m_idx_max <= uihlen - min_full_attn_seq_len) {
      return std::min(n_block_min, cute::ceil_div(contextual_seq_len, kBlockN));
    }
    return 0;
  }

  CUTLASS_DEVICE
  cute::tuple<int, int> get_cross_n_block_min_max(
      int const uihlen_q,
      int const seqlen_q,
      int const seqlen_kv,
      int const m_block) {
    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    if constexpr (!Causal) {
      return {0, cute::ceil_div(seqlen_kv, kBlockN)};
    }
    int n_block_max =
        std::min(seqlen_kv, (m_block + 1) * kBlockM + seqlen_kv - uihlen_q);
    return {0, cute::ceil_div(n_block_max, kBlockN)};
  }

  template <typename SchedulerPrefetch, typename SharedStorage>
  CUTLASS_DEVICE void load(
      Params const& params,
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      MainloopPipelineVt pipeline_vt,
      PipelineState& smem_pipe_write,
      SharedStorage& shared_storage,
      SchedulerPrefetch const& scheduler_prefetch,
      SeqlenInfo_t const& seqlen_info,
      cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
      int& work_idx) {
    auto [m_block, bidh, bidb, split_idx] = block_coord;
    if constexpr (Jagged) {
      static constexpr int kBlockM = get<0>(TileShape_MNK{});
      if (m_block * kBlockM >= seqlen_info.seqlen_q) {
        scheduler_prefetch();
        return;
      }
    }
    int min_full_attn_seq_len_, max_attn_len_, contextual_seq_len_;
    if (params.num_groups > 1) {
      int group_id = bidb / params.batch_size_per_group;
      min_full_attn_seq_len_ = params.min_full_attn_seq_len_tensor[group_id];
      max_attn_len_ = params.max_attn_len_tensor[group_id];
      contextual_seq_len_ = params.contextual_seq_len_tensor[group_id];
    } else {
      min_full_attn_seq_len_ = params.min_full_attn_seq_len;
      max_attn_len_ = params.max_attn_len;
      contextual_seq_len_ = params.contextual_seq_len;
    }
    int n_block_min, n_block_max;
    if constexpr (Cross) {
      auto n_block_min_max = get_cross_n_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          m_block);
      n_block_min = get<0>(n_block_min_max);
      n_block_max = get<1>(n_block_min_max);
    } else {
      auto n_block_min_max = get_n_block_min_max(
          max_attn_len_,
          min_full_attn_seq_len_,
          contextual_seq_len_,
          seqlen_info.uihlen_q,
          m_block);
      n_block_min = get<0>(n_block_min_max);
      n_block_max = get<1>(n_block_min_max);
    }
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    if (n_block_max <= n_block_min) {
      std::printf(
          "mainloop_fwd_sm90: n_block_max <= n_block_min not expected.");
      scheduler_prefetch();
      return;
    }
#endif

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sK_pi = as_position_independent_swizzle_tensor(sK);
    // as_position_independent_swizzle_tensor makes address calculation easier
    // when we do LDSM & STSM to transpose. But it requires smem_vt and smem_v
    // to be aligned to e.g 512 bytes.
    Tensor sVt = [&] {
      if constexpr (!Transpose_V) {
        return make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
            SmemLayoutVt{});
      } else {
        return cute::as_position_independent_swizzle_tensor(make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_vt.data()),
            SmemLayoutVt{}));
      }
    }();
    // Only used if Transpose_V
    Tensor sV = cute::as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutVtMma{}));

    int const thread_idx = threadIdx.x % NumProducerThreads;

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {
        block_rank_in_cluster % cluster_shape_x,
        block_rank_in_cluster / cluster_shape_x};

    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mK_TMA = params.tma_load_K.get_tma_tensor(params.shape_K)(
        _, _, bidh, !Jagged ? bidb : 0);
    Tensor mVt_TMA = params.tma_load_V.get_tma_tensor(
        select<1, 0, 2, 3>(params.shape_K))(_, _, bidh, !Jagged ? bidb : 0);

    Tensor gQ = local_tile(
        domain_offset(make_coord(seqlen_info.offset_q, _0{}), mQ),
        select<0, 2>(TileShape_MNK{}),
        make_coord(m_block, _0{})); // (M, K)
    // if (cute::thread0()) { printf("Jagged = %d, params.leftpad_k = %p,
    // leftpad_k = %d\n", Jagged, params.leftpad_k, leftpad_k); }
    Tensor gK_TMA = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK_TMA),
        select<1, 2>(TileShape_MNK{}),
        make_coord(_, _0{})); // (N, K, _)
    Tensor gVt_TMA = local_tile(
        domain_offset(make_coord(_0{}, seqlen_info.offset_k), mVt_TMA),
        select<2, 1>(TileShape_MNK{}),
        make_coord(_0{}, _)); // (K, N, _)

    auto block_tma_Q = params.tma_load_Q.get_slice(_0{});
    Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ)); // (TMA)
    Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ)); // (TMA)
    // tma_partition doesn't handle position_independent_swizzle_tensor
    // correctly, so we need to do it manually
    auto block_tma_K = params.tma_load_K.get_slice(cluster_local_block_id.x);
    Tensor tKgK_TMA =
        group_modes<0, 3>(block_tma_K.partition_S(gK_TMA)); // (TMA, k)
    Tensor tKsK_TMA =
        group_modes<0, 3>(block_tma_K.partition_D(sK)); // (TMA, PIPE)
    auto block_tma_V = params.tma_load_V.get_slice(cluster_local_block_id.x);
    Tensor tVgVt_TMA =
        group_modes<0, 3>(block_tma_V.partition_S(gVt_TMA)); // (TMA, k)
    Tensor tVsVt_TMA =
        group_modes<0, 3>(block_tma_V.partition_D(sVt)); // (TMA, PIPE)

    // Set up for transposing V, only used if Transpose_V
    S2RTiledCopyVt s2r_tiled_copy_vt;
    R2STiledCopyV r2s_tiled_copy_v;
    auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(thread_idx);
    auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(thread_idx);
    // flat_divide(sVt, LDSM_divide_shape{}):  (64, 8, kHeadDim / 64, kBlockN /
    // 8, kStages)
    Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(
        flat_divide(sVt, LDSM_divide_shape{})); // ((16, 1), 1, 1, kHeadDim /
                                                // 64, kBlockN / 32, kStages)
    // flat_divide(sV, STSM_divide_shape{}):  (8, 16, kHeadDim / 8, (4, kBlockN
    // / 64), kStages)
    Tensor tTranssV_ = r2s_thr_copy_v.partition_D(
        flat_divide(sV, STSM_divide_shape{})); // ((16, 1), 1, 1, kHeadDim / 64,
                                               // (2, kBlockN / 64), kStages)
    CUTE_STATIC_ASSERT_V(rank(tTranssVt_) == rank(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<0>(tTranssVt_) == size<0>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<1>(tTranssVt_) == size<1>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<2>(tTranssVt_) == size<2>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<3>(tTranssVt_) == size<3>(tTranssV_));
    CUTE_STATIC_ASSERT_V(size<4>(tTranssVt_) == size<4>(tTranssV_));
    // Faster to have 2 LDSM.T, byte permute, STSM for better ILP
    static constexpr int Transpose_ILP =
        (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
    Tensor tTranssVt = logical_divide(
        group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
        Shape<Underscore, Int<Transpose_ILP>>{}); // ((16, 1), (2, kHeadDim / 64
                                                  // * kBlockN / 32 / 2),
                                                  // kStages)
    Tensor tTranssV = logical_divide(
        group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
        Shape<Underscore, Int<Transpose_ILP>>{}); // ((16, 1), (2, kHeadDim / 64
                                                  // * kBlockN / 32 / 2),
                                                  // kStages)
    auto transpose_V = [&](int stage) {
      if constexpr (Transpose_V) {
#pragma unroll
        for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
          Tensor tTransrV =
              make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
          static_assert(size<0>(tTransrV) == 16);
          Tensor tTransrV_64 = recast<uint2>(tTransrV);
          cute::copy(
              s2r_tiled_copy_vt,
              tTranssVt(_, make_coord(_, i), stage),
              tTransrV);
#pragma unroll
          for (int j = 0; j < size(tTransrV_64); ++j) {
            uint32_t upper = tTransrV_64[j].x;
            uint32_t lower = tTransrV_64[j].y;
            tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
            tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
          }
          cute::copy(
              r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage));
        }
      }
    };

    uint16_t mcast_mask_kv = 0;
    if constexpr (cute::is_same_v<GmemTiledCopyKV, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_kv |=
            (uint16_t(1) << block_layout(m, cluster_local_block_id.y, _0{}));
      }
    }

    auto load_K = [&](int const n_block, auto const& smem_pipe_write) {
      pipeline_k.producer_acquire(smem_pipe_write);
      copy(
          params.tma_load_K.with(
              *pipeline_k.producer_get_barrier(smem_pipe_write),
              mcast_mask_kv,
              TMA::CacheHintSm90::EVICT_LAST),
          tKgK_TMA(_, n_block),
          tKsK_TMA(_, smem_pipe_write.index()));
    };

    auto load_V = [&](int const n_block, auto const& smem_pipe_write) {
      auto pipeline_v_load =
          cute::conditional_return<!Transpose_V>(pipeline_v, pipeline_vt);
      pipeline_v_load.producer_acquire(smem_pipe_write);
      copy(
          params.tma_load_V.with(
              *pipeline_v_load.producer_get_barrier(smem_pipe_write),
              mcast_mask_kv,
              TMA::CacheHintSm90::EVICT_LAST),
          tVgVt_TMA(_, n_block),
          tVsVt_TMA(_, smem_pipe_write.index()));
    };

    auto copy_Vt_to_V = [&](auto const& smem_pipe_write) {
      // Instead of maintaining smem_pipe_read as a separate variable, we can
      // just use smem_pipe_write, and exploit the invariance that
      // smem_pipe_write.phase() == smem_pipe_read.phase() ^ 1. This saves 1 or
      // 2 registers.
      PipelineState smem_pipe_read{
          smem_pipe_write.index(),
          smem_pipe_write.phase() ^ 1,
          smem_pipe_write.count()};
      pipeline_vt.consumer_wait(smem_pipe_read);
      pipeline_v.producer_acquire(smem_pipe_write);
      transpose_V(smem_pipe_write.index());
      // SMEM fence to make sure V is transposed before math
      cutlass::arch::fence_view_async_shared();
      pipeline_v.producer_commit(smem_pipe_write);
      // Very important: PipelineTmaAsync::consumer_release assumes that the
      // warpgroup is synchronized before calling. Without this we get race
      // conditions.
      cutlass::arch::NamedBarrier::sync(
          cutlass::NumThreadsPerWarpGroup,
          static_cast<uint32_t>(FwdNamedBarriers::ProducerWG) /*id*/);
      pipeline_vt.consumer_release(smem_pipe_read);
    };

    int n_block = n_block_max - 1;

    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // If this is true, we're guaranteed that only the first warp will execute
    // this function
    static constexpr bool SingleProducerWarp =
        NumProducerThreads == cutlass::NumThreadsPerWarp;
    bool should_load_KV =
        ((SingleProducerWarp || warp_idx_in_warpgroup == 0) &&
         cute::elect_one_sync());

    if (should_load_KV) {
      if constexpr (Transpose_V) {
        load_V(n_block, smem_pipe_write);
      }
      // if (thread_idx == 0) { printf("Producer: main load, before load_K,
      // index = %d\n", smem_pipe_write.index());}
      load_K(n_block, smem_pipe_write);
      // if (thread_idx == 0) { printf("Producer: main load, after load K, index
      // = %d\n", smem_pipe_write.index());}
    }

    // TMA_Q, Wait for the MMA warpgroups to signal that smem_q is ready
    if (SingleProducerWarp || warp_idx_in_warpgroup == 0) {
      cutlass::arch::NamedBarrier::sync(
          NumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
    }
    if ((SingleProducerWarp || warp_idx_in_warpgroup == 0) &&
        cute::elect_one_sync()) {
      shared_storage.pipelines.barrier_Q.arrive_and_expect_tx(
          TmaTransactionBytesQ);
      copy(
          params.tma_load_Q.with(
              reinterpret_cast<typename cutlass::arch::
                                   ClusterTransactionBarrier::ValueType&>(
                  shared_storage.pipelines.barrier_Q),
              0 /*mcast_mask*/,
              TMA::CacheHintSm90::EVICT_FIRST),
          tQgQ,
          tQsQ);
    }

    // Wait for the MMA WGs to signal that smem_v are ready and V can be copied
    // from gmem Need ClusterBarrier, not just NamedBarrier. Otherwise we might
    // have CTA 0 finishing the TMA store on O first, call TMA multicast load on
    // V, before CTA 1 can finishing TMA store on O. if (thread_idx == 0) {
    // printf("Producer: main load, before barrier_O, work_idx = %d\n",
    // work_idx);}
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    // if (thread_idx == 0) { printf("Producer: main load, after barrier_O\n");}

    int n_block_prev = n_block;
    --n_block;
#pragma unroll(!Transpose_V ? 2 : 1)
    for (; n_block >= n_block_min; --n_block) {
      PipelineState smem_pipe_write_v =
          smem_pipe_write; // copy the state, write_v is always 1 step behind
      ++smem_pipe_write;
      if (should_load_KV) {
        if constexpr (Transpose_V) {
          load_V(n_block, smem_pipe_write);
        } else {
          load_V(n_block_prev, smem_pipe_write_v);
        }
        load_K(n_block, smem_pipe_write);
      }
      n_block_prev = n_block;
      if constexpr (Transpose_V) {
        copy_Vt_to_V(smem_pipe_write_v);
      }
    }
    scheduler_prefetch();
    if constexpr (!Transpose_V) {
      if (should_load_KV) {
        load_V(n_block_prev, smem_pipe_write);
      }
    }
    if constexpr (Transpose_V) {
      copy_Vt_to_V(smem_pipe_write);
    }
    ++smem_pipe_write;
    if constexpr (!Cross) {
      if constexpr (Has_targets) {
        auto [target_n_block_min, target_n_block_max] =
            get_target_n_block_min_max(
                n_block_max,
                seqlen_info.uihlen_q,
                seqlen_info.seqlen_kv,
                m_block);
#pragma unroll 1
        for (n_block = target_n_block_max - 1; n_block >= target_n_block_min;
             --n_block) {
          if (should_load_KV) {
            load_V(n_block, smem_pipe_write);
            load_K(n_block, smem_pipe_write);
          }
          if constexpr (Transpose_V) {
            copy_Vt_to_V(smem_pipe_write);
          }
          ++smem_pipe_write;
        }
      }
      if constexpr (Contexual_mask) {
        int contexual_n_block_max = get_contexual_n_block_max(
            n_block_min,
            min_full_attn_seq_len_,
            contextual_seq_len_,
            seqlen_info.uihlen_q,
            m_block);
#pragma unroll 1
        for (n_block = contexual_n_block_max - 1; n_block >= 0; --n_block) {
          if (should_load_KV) {
            load_V(n_block, smem_pipe_write);
            load_K(n_block, smem_pipe_write);
          }
          if constexpr (Transpose_V) {
            copy_Vt_to_V(smem_pipe_write);
          }
          ++smem_pipe_write;
        }
      }
    }
    // At the end, all threads have the correct smem_pipe_write.
    ++work_idx;
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load_tail(
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      MainloopPipelineVt pipeline_vt,
      PipelineState& smem_pipe_write,
      SharedStorage& shared_storage,
      int const work_idx) {
    // If we don't wait for barrier_O here, when using Cluster, CTA0 might exit
    // early and CTA1 will try to arrive on barrier_O of CTA0, causing
    // "unspecified launch failure".
    shared_storage.pipelines.barrier_O.wait((work_idx + 1) % 2);
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    // TODO: check if this should be called by 1 thread or more
    if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
      /* This helps avoid early exit of blocks in Cluster
       *  Waits for all stages to either be released (all Consumer UNLOCKs), or
       * if the stage was never used then would just be acquired since the phase
       * was still inverted from make_producer_start_state
       */
      pipeline_k.producer_tail(smem_pipe_write);
      pipeline_v.producer_tail(smem_pipe_write);
      if constexpr (Transpose_V) {
        pipeline_vt.producer_tail(smem_pipe_write);
      }
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(
          2 * cutlass::NumThreadsPerWarpGroup,
          static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) - 1 +
              hstu::canonical_warp_group_idx_nosync() /*id*/);
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    if constexpr (UseSchedulerBarrier) {
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
      int const cur_WG = hstu::canonical_warp_group_idx_nosync() - 1;
      int const next_WG = NumMmaWarpGroups == 2
          ? 1 - cur_WG
          : (cur_WG < NumMmaWarpGroups - 1 ? cur_WG + 1 : 0);
      cutlass::arch::NamedBarrier::arrive(
          2 * cutlass::NumThreadsPerWarpGroup,
          static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) +
              next_WG /*id*/);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // Tell producers that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
    if constexpr (UseSchedulerBarrier) {
      // We have NamedBarrier for up to 3 WGs
      static_assert(NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);
      // WG1 needs the very first signal to start
      if (hstu::canonical_warp_group_idx_nosync() == 1) {
        cutlass::arch::NamedBarrier::arrive(
            2 * cutlass::NumThreadsPerWarpGroup,
            static_cast<uint32_t>(FwdNamedBarriers::WarpSchedulerWG1) /*id*/);
      }
    }
  }

  template <typename SharedStorage, typename FrgTensorO>
  CUTLASS_DEVICE bool mma(
      Params const& params,
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      PipelineState& smem_pipe_read,
      FrgTensorO& tOrO,
      int const thread_idx,
      int& work_idx,
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
      static constexpr int kBlockM = get<0>(TileShape_MNK{});
      if (m_block * kBlockM >= seqlen_info.seqlen_q) {
        return false;
      }
    }
    int min_full_attn_seq_len_, max_attn_len_, contextual_seq_len_;
    float scalar_scale_val_;
    if (params.num_groups > 1) {
      int group_id = bidb / params.batch_size_per_group;
      min_full_attn_seq_len_ = params.min_full_attn_seq_len_tensor[group_id];
      max_attn_len_ = params.max_attn_len_tensor[group_id];
      contextual_seq_len_ = params.contextual_seq_len_tensor[group_id];
      int max_seq_len_per_group = params.max_seq_len_tensor[group_id];
      // attention scale
      scalar_scale_val_ = params.scalar_scale
          ? (params.attn_scale == nullptr ? 1.0f / max_seq_len_per_group
                                          : params.attn_scale[group_id])
          : 0;
    } else {
      min_full_attn_seq_len_ = params.min_full_attn_seq_len;
      max_attn_len_ = params.max_attn_len;
      contextual_seq_len_ = params.contextual_seq_len;
      // attention scale
      scalar_scale_val_ = params.scalar_scale
          ? (params.attn_scale == nullptr ? params.max_seq_len_inv
                                          : params.attn_scale[0])
          : 0;
    }
    int n_block_min, n_block_max;
    if constexpr (Cross) {
      auto n_block_min_max = get_cross_n_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          m_block);
      n_block_min = get<0>(n_block_min_max);
      n_block_max = get<1>(n_block_min_max);
    } else {
      auto n_block_min_max = get_n_block_min_max(
          max_attn_len_,
          min_full_attn_seq_len_,
          contextual_seq_len_,
          seqlen_info.uihlen_q,
          m_block);
      n_block_min = get<0>(n_block_min_max);
      n_block_max = get<1>(n_block_min_max);
    }

#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    if (n_block_max <= n_block_min) {
      std::printf(
          "mainloop_fwd_sm90: n_block_max <= n_block_min not expected.");
      return false;
    }
#endif

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutVtMma{});
    Tensor sP = [&] {
      if constexpr (Mma1_is_RS) {
        // We might not have smem_p if !Mma1_is_RS1, just use smem_q as a
        // placeholder since we don't use it
        return make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
            SmemLayoutP{});
      } else {
        return make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
            SmemLayoutP{});
      }
    }();

    if constexpr (!Mma0_is_RS) {
      static_assert(
          stride<0>(typename TiledMma0::ALayout{}) == 0 and
              stride<0>(typename TiledMma0::BLayout{}) == 0 and
              size<0>(typename TiledMma0::ALayout{}) ==
                  cutlass::NumThreadsPerWarpGroup and
              size<0>(typename TiledMma0::BLayout{}) ==
                  cutlass::NumThreadsPerWarpGroup,
          "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    }
    constexpr int MmaWarpGroups =
        size(TiledMma0{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        make_shape(Int<MmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMma0 tiled_mma0;
    TiledMma1 tiled_mma1;
    auto wg_mma0 =
        tiled_mma0.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma1 =
        tiled_mma1.get_slice(warp_group_thread_layout(warp_group_idx));

    auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma0);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors"
    Tensor tSrQ = wg_mma0.partition_fragment_A(sQ);
    Tensor tSrK = wg_mma0.partition_fragment_B(sK);
    Tensor tOrV = wg_mma1.partition_fragment_B(sV);
    Tensor tOsP = wg_mma1.partition_fragment_A(sP);
    Tensor tPsP = smem_thr_copy_P.partition_D(
        cute::as_position_independent_swizzle_tensor(sP));

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    clear(tOrO);

    int n_block = n_block_max - 1;

    hstu::Mask<kBlockM, kBlockN, TiledMma0> mask(
        thread_idx,
        seqlen_info.seqlen_q,
        seqlen_info.seqlen_kv,
        max_attn_len_,
        min_full_attn_seq_len_,
        contextual_seq_len_,
        seqlen_info.uihlen_q);

    auto& barrier_Q = shared_storage.pipelines.barrier_Q;
    barrier_Q.wait(work_idx % 2);

    static constexpr int Qdim = 0;
    auto thread_mma = tiled_mma0.get_thread_slice(thread_idx);
    auto thread0_mma = tiled_mma0.get_thread_slice(_0{});
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tScS_rowcol = make_tensor(
        tScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(
        t0ScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/false>(t0ScS.layout()));
    int const thread_qdim_offset = get<Qdim>(tScS_rowcol(_0{}, _0{}));
    SiluScaleOp<float> silu_scale_op;
    int qdim_offset = params.scalar_scale
        ? 0
        : m_block * kBlockM + thread_qdim_offset + seqlen_info.offset_q;

    if constexpr (Mma0_is_RS) {
      using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma0);
      auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
      Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
      Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(
          cute::as_position_independent_swizzle_tensor(sQ));
      cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
    }

    Tensor tSrS =
        partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
    Tensor tSrS_rowcol = make_tensor(
        tSrS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
    consumer_wait(pipeline_k, smem_pipe_read);
    hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
        tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
    warpgroup_wait<0>();
    pipeline_k.consumer_release(smem_pipe_read);
#pragma unroll
    for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
      float scale = scalar_scale_val_;
      if (!params.scalar_scale) {
        int q_index = qdim_offset + int(get<Qdim>(t0ScS_rowcol(mi, _0{})));
        // Convert global index to local sequence position for bounds checking
        int q_local = q_index - seqlen_info.offset_q;
        if (q_local < seqlen_info.seqlen_q) {
          scale = params.attn_scale[q_index];
        }
      }
#pragma unroll
      for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
        tSrS_rowcol(mi, ni) =
            silu_scale_op(tSrS_rowcol(mi, ni) * params.alpha, scale);
      }
    }
    int const m_idx_max = (m_block + 1) * kBlockM;
    if constexpr (Cross) {
      mask.template apply<
          false /*Seqlenq_mask*/,
          true /*Seqlenk_mask*/,
          Causal,
          false /*Local*/,
          false /*Contexual_mask*/,
          false /*Target_mask*/,
          Cross,
          false /*Softmax*/>(tSrS, m_block, n_block);
    } else {
      if (m_idx_max <= seqlen_info.uihlen_q) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            false /*Seqlenk_mask*/,
            Causal,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            Cross,
            false /*Softmax*/>(tSrS, m_block, n_block);
      } else if (
          m_idx_max <=
          cute::ceil_div(seqlen_info.uihlen_q, kBlockM) * kBlockM) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            Local,
            Contexual_mask,
            Has_targets,
            Cross,
            false /*Softmax*/>(tSrS, m_block, n_block);
      } else {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal*/,
            false,
            Contexual_mask,
            Has_targets,
            Cross,
            false /*Softmax*/>(tSrS, m_block, n_block);
      }
    }
    if constexpr (Is_FP8 && !V_colmajor) {
      hstu::permute_Cregs_fp8(tSrS);
    }
    Tensor tOrP_acc = make_tensor(
        tSrS.data(), hstu::convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
    Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
    convert_type_out(tOrP_acc, tOrP);
    if constexpr (Is_FP8 && V_colmajor) {
      hstu::permute_Aregs_fp8(tOrP);
    }
    if constexpr (!Mma1_is_RS) {
      cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
      cutlass::arch::fence_view_async_shared();
      __syncwarp(); // Only need syncwarp since each warp is using its own P
                    // values for Mma1
    }
    --n_block;

    // Each step does gemm0 and silu for iter n_block and gemm1 for prev iter.
    auto fwd_step_intra_warp_pipeline = [&](int const n_block, auto mask_fn) {
      PipelineState smem_pipe_read_v(
          smem_pipe_read.index(),
          smem_pipe_read.phase(),
          smem_pipe_read.count());
      ++smem_pipe_read;
      Tensor tSrS =
          partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
      Tensor tSrS_rowcol = make_tensor(
          tSrS.data(),
          hstu::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
      if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_k, smem_pipe_read);
      }
      warp_scheduler_barrier_sync();
      hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
          tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
      if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_v, smem_pipe_read_v);
      }
      hstu::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
          tiled_mma1,
          cute::conditional_return<Mma1_is_RS>(tOrP, tOsP),
          tOrV(_, _, _, smem_pipe_read_v.index()),
          tOrO);
      warp_scheduler_barrier_arrive();
      warpgroup_wait<1>();
      pipeline_k.consumer_release(smem_pipe_read); // release K
#pragma unroll
      for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
        float scale = scalar_scale_val_;
        if (!params.scalar_scale) {
          int q_index = qdim_offset + int(get<Qdim>(t0ScS_rowcol(mi, _0{})));
          // Convert global index to local sequence position for bounds checking
          int q_local = q_index - seqlen_info.offset_q;
          if (q_local < seqlen_info.seqlen_q) {
            scale = params.attn_scale[q_index];
          }
        }
#pragma unroll
        for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
          tSrS_rowcol(mi, ni) =
              silu_scale_op(tSrS_rowcol(mi, ni) * params.alpha, scale);
        }
      }
      mask_fn(tSrS, n_block);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v); // release V
      if constexpr (Is_FP8 && !V_colmajor) {
        hstu::permute_Cregs_fp8(tSrS);
      }
      convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
      if constexpr (Is_FP8 && V_colmajor) {
        hstu::permute_Aregs_fp8(tOrP);
      }
      if constexpr (!Mma1_is_RS) {
        cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
      }
      if constexpr (!Mma1_is_RS) {
        cutlass::arch::fence_view_async_shared();
        __syncwarp();
      }
    };

    if constexpr (Cross) {
      if constexpr (Causal) {
        if (m_idx_max <=
            cute::ceil_div(seqlen_info.uihlen_q, kBlockM) * kBlockM) {
          auto mask_fn = [&](auto& tSrS, int n_block) {
            mask.template apply<
                false /*Seqlenq_mask*/,
                false /*Seqlenk_mask*/,
                Causal,
                false /*Local*/,
                false /*Contexual_mask*/,
                false /*Target_mask*/,
                Cross,
                false /*Softmax*/>(tSrS, m_block, n_block);
          };
          int const m_idx_min = m_block * kBlockM;
          int const n_block_min_causal_mask = std::max(
              n_block_min,
              (m_idx_min + seqlen_info.seqlen_kv - seqlen_info.uihlen_q) /
                  kBlockN);
#pragma unroll 1
          for (; n_block >= n_block_min_causal_mask; --n_block) {
            fwd_step_intra_warp_pipeline(n_block, mask_fn);
          }
        }
      }
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
#pragma unroll 1
      for (; n_block >= n_block_min; --n_block) {
        fwd_step_intra_warp_pipeline(n_block, no_mask_fn);
      }
    } else {
      if constexpr (Causal || Local) { // Separate iterations with causal
                                       // or local masking
        if (m_idx_max <=
            cute::ceil_div(seqlen_info.uihlen_q, kBlockM) * kBlockM) {
          auto mask_fn = [&](auto& tSrS, int n_block) {
            mask.template apply<
                false /*Seqlenq_mask*/,
                false /*Seqlenk_mask*/,
                Causal,
                Local,
                Contexual_mask,
                false /*Has_targets*/,
                Cross,
                false /*Softmax*/>(tSrS, m_block, n_block);
          };
          int const m_idx_min = m_block * kBlockM;
          int const n_block_min_causal_local_mask =
              std::max(n_block_min, m_idx_min / kBlockN);
#pragma unroll 1
          for (; n_block >= n_block_min_causal_local_mask; --n_block) {
            fwd_step_intra_warp_pipeline(n_block, mask_fn);
          }
        }
      }
      int n_block_min_before_local_mask = n_block_min;
      if constexpr (Local) {
        if (m_idx_max <=
            cute::ceil_div(
                seqlen_info.uihlen_q - min_full_attn_seq_len_, kBlockM) *
                kBlockM) {
          n_block_min_before_local_mask = std::max(
              n_block_min, cute::ceil_div(m_idx_max - max_attn_len_, kBlockN));
        }
      }
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
#pragma unroll 1
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step_intra_warp_pipeline(n_block, no_mask_fn);
      }
      // Separate masking iterations on the left for local attention
      if constexpr (Local) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              false /*Seqlenk_mask*/,
              false /*Causal_mask*/,
              Local,
              Contexual_mask,
              false /*Has_targets*/,
              Cross,
              false /*Softmax*/>(tSrS, m_block, n_block);
        };
#pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
          fwd_step_intra_warp_pipeline(n_block, local_mask_fn);
        }
      }
      // Target part GEMM
      if constexpr (Has_targets) {
        auto [target_n_block_min, target_n_block_max] =
            get_target_n_block_min_max(
                n_block_max,
                seqlen_info.uihlen_q,
                seqlen_info.seqlen_kv,
                m_block);
        auto target_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal_mask*/,
              false /*Local*/,
              Contexual_mask,
              Has_targets,
              Cross,
              false /*Softmax*/>(tSrS, m_block, n_block);
        };
#pragma unroll 1
        for (n_block = target_n_block_max - 1; n_block >= target_n_block_min;
             --n_block) {
          fwd_step_intra_warp_pipeline(n_block, target_mask_fn);
        }
      }
      if constexpr (Contexual_mask) {
        int contexual_n_block_max = get_contexual_n_block_max(
            n_block_min,
            min_full_attn_seq_len_,
            contextual_seq_len_,
            seqlen_info.uihlen_q,
            m_block);
        auto contexual_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal_mask*/,
              Local,
              Contexual_mask,
              Has_targets,
              Cross,
              false /*Softmax*/>(tSrS, m_block, n_block);
        };
#pragma unroll 1
        for (n_block = contexual_n_block_max - 1; n_block >= 0; --n_block) {
          fwd_step_intra_warp_pipeline(n_block, contexual_mask_fn);
        }
      }
    }
    // Tell producers that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
    consumer_wait(pipeline_v, smem_pipe_read);
    hstu::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
        tiled_mma1,
        cute::conditional_return<Mma1_is_RS>(tOrP, tOsP),
        tOrV(_, _, _, smem_pipe_read.index()),
        tOrO);
    warpgroup_wait<0>();
    pipeline_v.consumer_release(
        smem_pipe_read); // release V, otherwise producers will hang
    if constexpr (Is_FP8 && !V_colmajor) {
      hstu::permute_output_fp8(tOrO);
    }
    ++smem_pipe_read;
    ++work_idx;
    return true;
  }

  template <typename SharedStorage, typename FrgTensorO, typename Softmax>
  CUTLASS_DEVICE bool mma_softmax(
      Params const& params,
      MainloopPipelineK pipeline_k,
      MainloopPipelineV pipeline_v,
      PipelineState& smem_pipe_read,
      FrgTensorO& tOrO,
      Softmax& softmax,
      int const thread_idx,
      int& work_idx,
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
      static constexpr int kBlockM = get<0>(TileShape_MNK{});
      if (m_block * kBlockM >= seqlen_info.seqlen_q) {
        return false;
      }
    }
    int min_full_attn_seq_len_, max_attn_len_, contextual_seq_len_;
    if (params.num_groups > 1) {
      int group_id = bidb / params.batch_size_per_group;
      min_full_attn_seq_len_ = params.min_full_attn_seq_len_tensor[group_id];
      max_attn_len_ = params.max_attn_len_tensor[group_id];
      contextual_seq_len_ = params.contextual_seq_len_tensor[group_id];
    } else {
      min_full_attn_seq_len_ = params.min_full_attn_seq_len;
      max_attn_len_ = params.max_attn_len;
      contextual_seq_len_ = params.contextual_seq_len;
    }
    int n_block_min, n_block_max;
    if constexpr (Cross) {
      auto n_block_min_max = get_cross_n_block_min_max(
          seqlen_info.uihlen_q,
          seqlen_info.seqlen_q,
          seqlen_info.seqlen_kv,
          m_block);
      n_block_min = get<0>(n_block_min_max);
      n_block_max = get<1>(n_block_min_max);
    } else {
      auto n_block_min_max = get_n_block_min_max(
          max_attn_len_,
          min_full_attn_seq_len_,
          contextual_seq_len_,
          seqlen_info.uihlen_q,
          m_block);
      n_block_min = get<0>(n_block_min_max);
      n_block_max = get<1>(n_block_min_max);
    }

#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    if (n_block_max <= n_block_min) {
      std::printf(
          "mainloop_fwd_sm90: n_block_max <= n_block_min not expected.");
      return false;
    }
#endif

    Tensor sQ = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
        SmemLayoutQ{});
    Tensor sK = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()),
        SmemLayoutK{});
    Tensor sV = make_tensor(
        make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()),
        SmemLayoutVtMma{});
    Tensor sP = [&] {
      if constexpr (Mma1_is_RS) {
        // We might not have smem_p if !Mma1_is_RS1, just use smem_q as a
        // placeholder since we don't use it
        return make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()),
            SmemLayoutP{});
      } else {
        return make_tensor(
            make_smem_ptr(shared_storage.tensors.mainloop.smem_p.data()),
            SmemLayoutP{});
      }
    }();

    if constexpr (!Mma0_is_RS) {
      static_assert(
          stride<0>(typename TiledMma0::ALayout{}) == 0 and
              stride<0>(typename TiledMma0::BLayout{}) == 0 and
              size<0>(typename TiledMma0::ALayout{}) ==
                  cutlass::NumThreadsPerWarpGroup and
              size<0>(typename TiledMma0::BLayout{}) ==
                  cutlass::NumThreadsPerWarpGroup,
          "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
    }
    constexpr int MmaWarpGroups =
        size(TiledMma0{}) / cutlass::NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(
        make_shape(Int<MmaWarpGroups>{}),
        make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

    int warp_group_idx = __shfl_sync(
        0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
    TiledMma0 tiled_mma0;
    TiledMma1 tiled_mma1;
    auto wg_mma0 =
        tiled_mma0.get_slice(warp_group_thread_layout(warp_group_idx));
    auto wg_mma1 =
        tiled_mma1.get_slice(warp_group_thread_layout(warp_group_idx));

    auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomP{}, tiled_mma0);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors"
    Tensor tSrQ = wg_mma0.partition_fragment_A(sQ);
    Tensor tSrK = wg_mma0.partition_fragment_B(sK);
    Tensor tOrV = wg_mma1.partition_fragment_B(sV);
    Tensor tOsP = wg_mma1.partition_fragment_A(sP);
    Tensor tPsP = smem_thr_copy_P.partition_D(
        cute::as_position_independent_swizzle_tensor(sP));

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    clear(tOrO);

    int n_block = n_block_max - 1;

    hstu::Mask<kBlockM, kBlockN, TiledMma0> mask(
        thread_idx,
        seqlen_info.seqlen_q,
        seqlen_info.seqlen_kv,
        max_attn_len_,
        min_full_attn_seq_len_,
        contextual_seq_len_,
        seqlen_info.uihlen_q);

    auto& barrier_Q = shared_storage.pipelines.barrier_Q;
    barrier_Q.wait(work_idx % 2);
    static constexpr int Qdim = 0;
    auto thread_mma = tiled_mma0.get_thread_slice(thread_idx);
    auto thread0_mma = tiled_mma0.get_thread_slice(_0{});
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tScS_rowcol = make_tensor(
        tScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(
        t0ScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/false>(t0ScS.layout()));
    int const thread_qdim_offset = get<Qdim>(tScS_rowcol(_0{}, _0{}));
    int qdim_offset = params.scalar_scale
        ? 0
        : m_block * kBlockM + thread_qdim_offset + seqlen_info.offset_q;

    if constexpr (Mma0_is_RS) {
      using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
      auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma0);
      auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
      Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
      Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(
          cute::as_position_independent_swizzle_tensor(sQ));
      cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
    }

    Tensor tSrS =
        partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
    Tensor tSrS_rowcol = make_tensor(
        tSrS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
    consumer_wait(pipeline_k, smem_pipe_read);
    hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
        tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
    warpgroup_wait<0>();
    pipeline_k.consumer_release(smem_pipe_read);
    int const m_idx_max = (m_block + 1) * kBlockM;
    if constexpr (Cross) {
      mask.template apply<
          false /*Seqlenq_mask*/,
          true /*Seqlenk_mask*/,
          Causal,
          false /*Local*/,
          false /*Contexual_mask*/,
          false /*Target_mask*/,
          Cross,
          true /*Softmax*/>(tSrS, m_block, n_block);
    } else {
      if (m_idx_max <= seqlen_info.uihlen_q) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            false /*Seqlenk_mask*/,
            Causal,
            Local,
            Contexual_mask,
            false /*Target_mask*/,
            Cross,
            true /*Softmax*/>(tSrS, m_block, n_block);
      } else if (
          m_idx_max <=
          cute::ceil_div(seqlen_info.uihlen_q, kBlockM) * kBlockM) {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            Causal,
            Local,
            Contexual_mask,
            Has_targets,
            Cross,
            true /*Softmax*/>(tSrS, m_block, n_block);
      } else {
        mask.template apply<
            false /*Seqlenq_mask*/,
            true /*Seqlenk_mask*/,
            false /*Causal*/,
            false,
            Contexual_mask,
            Has_targets,
            Cross,
            true /*Softmax*/>(tSrS, m_block, n_block);
      }
    }
    Tensor scores_scale = softmax.template max_get_scale<
        /*Is_first=*/true,
        /*Check_inf=*/true>(tSrS);
    softmax.template online_softmax</*Is_first=*/true, /*Check_inf=*/true>(
        tSrS);
    if constexpr (Is_FP8 && !V_colmajor) {
      hstu::permute_Cregs_fp8(tSrS);
    }
    Tensor tOrP_acc = make_tensor(
        tSrS.data(), hstu::convert_layout_acc_Aregs<TiledMma1>(tSrS.layout()));
    Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
    convert_type_out(tOrP_acc, tOrP);
    if constexpr (Is_FP8 && V_colmajor) {
      hstu::permute_Aregs_fp8(tOrP);
    }
    if constexpr (!Mma1_is_RS) {
      cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
      cutlass::arch::fence_view_async_shared();
      __syncwarp(); // Only need syncwarp since each warp is using its own P
                    // values for Mma1
    }
    --n_block;

    // Each step does gemm0 and softmax for iter n_block and gemm1 for prev
    auto fwd_step_intra_warp_pipeline = [&](int const n_block,
                                            auto mask_fn,
                                            auto check_inf_type) {
      static constexpr bool Check_inf = decltype(check_inf_type)::value;
      PipelineState smem_pipe_read_v(
          smem_pipe_read.index(),
          smem_pipe_read.phase(),
          smem_pipe_read.count());
      ++smem_pipe_read;
      Tensor tSrS =
          partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
      Tensor tSrS_rowcol = make_tensor(
          tSrS.data(),
          hstu::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
      if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_k, smem_pipe_read);
      }
      warp_scheduler_barrier_sync();
      hstu::gemm</*zero_init=*/true, /*wg_wait=*/-1>(
          tiled_mma0, tSrQ, tSrK(_, _, _, smem_pipe_read.index()), tSrS);
      if (!UseSchedulerBarrier || warp_group_idx == 0) {
        consumer_wait(pipeline_v, smem_pipe_read_v);
      }
      hstu::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
          tiled_mma1,
          cute::conditional_return<Mma1_is_RS>(tOrP, tOsP),
          tOrV(_, _, _, smem_pipe_read_v.index()),
          tOrO);
      warp_scheduler_barrier_arrive();
      warpgroup_wait<1>();
      pipeline_k.consumer_release(smem_pipe_read); // release K
      mask_fn(tSrS, n_block);
      cute::copy(
          softmax.template max_get_scale</*Is_first=*/false, Check_inf>(tSrS),
          scores_scale);
      softmax.template online_softmax</*Is_first=*/false, Check_inf>(tSrS);
      softmax.rescale_o(tOrO, scores_scale);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v); // release V
      if constexpr (Is_FP8 && !V_colmajor) {
        hstu::permute_Cregs_fp8(tSrS);
      }
      convert_type_out(make_tensor(tSrS.data(), tOrP.layout()), tOrP);
      if constexpr (Is_FP8 && V_colmajor) {
        hstu::permute_Aregs_fp8(tOrP);
      }
      if constexpr (!Mma1_is_RS) {
        cute::copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(tOrP), tPsP);
      }
      if constexpr (!Mma1_is_RS) {
        cutlass::arch::fence_view_async_shared();
        __syncwarp();
      }
    };

    if constexpr (Cross) {
      if constexpr (Causal) {
        if (m_idx_max <=
            cute::ceil_div(seqlen_info.uihlen_q, kBlockM) * kBlockM) {
          auto mask_fn = [&](auto& tSrS, int n_block) {
            mask.template apply<
                false /*Seqlenq_mask*/,
                false /*Seqlenk_mask*/,
                Causal,
                false /*Local*/,
                false /*Contexual_mask*/,
                false /*Target_mask*/,
                Cross,
                true /*Softmax*/>(tSrS, m_block, n_block);
          };
          int const m_idx_min = m_block * kBlockM;
          int const n_block_min_causal_mask = std::max(
              n_block_min,
              (m_idx_min + seqlen_info.seqlen_kv - seqlen_info.uihlen_q) /
                  kBlockN);
#pragma unroll 1
          for (; n_block >= n_block_min_causal_mask; --n_block) {
            fwd_step_intra_warp_pipeline(n_block, mask_fn, cute::true_type{});
          }
        }
      }
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
#pragma unroll 1
      for (; n_block >= n_block_min; --n_block) {
        fwd_step_intra_warp_pipeline(n_block, no_mask_fn, cute::false_type{});
      }
    } else {
      if constexpr (Causal || Local) { // Separate iterations with causal
                                       // or local masking
        if (m_idx_max <=
            cute::ceil_div(seqlen_info.uihlen_q, kBlockM) * kBlockM) {
          auto mask_fn = [&](auto& tSrS, int n_block) {
            mask.template apply<
                false /*Seqlenq_mask*/,
                false /*Seqlenk_mask*/,
                Causal,
                Local,
                Contexual_mask,
                false /*Has_targets*/,
                Cross,
                true /*Softmax*/>(tSrS, m_block, n_block);
          };
          int const m_idx_min = m_block * kBlockM;
          int const n_block_min_causal_local_mask =
              std::max(n_block_min, m_idx_min / kBlockN);
#pragma unroll 1
          for (; n_block >= n_block_min_causal_local_mask; --n_block) {
            fwd_step_intra_warp_pipeline(n_block, mask_fn, cute::true_type{});
          }
        }
      }
      int n_block_min_before_local_mask = n_block_min;
      if constexpr (Local) {
        if (m_idx_max <=
            cute::ceil_div(
                seqlen_info.uihlen_q - min_full_attn_seq_len_, kBlockM) *
                kBlockM) {
          n_block_min_before_local_mask = std::max(
              n_block_min, cute::ceil_div(m_idx_max - max_attn_len_, kBlockN));
        }
      }
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
#pragma unroll 1
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step_intra_warp_pipeline(n_block, no_mask_fn, cute::false_type{});
      }
      // Separate masking iterations on the left for local attention
      if constexpr (Local) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              false /*Seqlenk_mask*/,
              false /*Causal_mask*/,
              Local,
              Contexual_mask,
              false /*Has_targets*/,
              Cross,
              true /*Softmax*/>(tSrS, m_block, n_block);
        };
#pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
          fwd_step_intra_warp_pipeline(
              n_block, local_mask_fn, cute::true_type{});
        }
      }
      // Target part GEMM
      if constexpr (Has_targets) {
        auto [target_n_block_min, target_n_block_max] =
            get_target_n_block_min_max(
                n_block_max,
                seqlen_info.uihlen_q,
                seqlen_info.seqlen_kv,
                m_block);
        auto target_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal_mask*/,
              false /*Local*/,
              Contexual_mask,
              Has_targets,
              Cross,
              true /*Softmax*/>(tSrS, m_block, n_block);
        };
#pragma unroll 1
        for (n_block = target_n_block_max - 1; n_block >= target_n_block_min;
             --n_block) {
          fwd_step_intra_warp_pipeline(
              n_block, target_mask_fn, cute::true_type{});
        }
      }
      if constexpr (Contexual_mask) {
        int contexual_n_block_max = get_contexual_n_block_max(
            n_block_min,
            min_full_attn_seq_len_,
            contextual_seq_len_,
            seqlen_info.uihlen_q,
            m_block);
        auto contexual_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply<
              false /*Seqlenq_mask*/,
              true /*Seqlenk_mask*/,
              false /*Causal_mask*/,
              Local,
              Contexual_mask,
              Has_targets,
              Cross,
              true /*Softmax*/>(tSrS, m_block, n_block);
        };
#pragma unroll 1
        for (n_block = contexual_n_block_max - 1; n_block >= 0; --n_block) {
          fwd_step_intra_warp_pipeline(
              n_block, contexual_mask_fn, cute::true_type{});
        }
      }
    }
    // Tell producers that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(
        NumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<uint32_t>(FwdNamedBarriers::QueryEmpty) /*id*/);
    consumer_wait(pipeline_v, smem_pipe_read);
    hstu::gemm</*zero_init=*/false, /*wg_wait=*/-1>(
        tiled_mma1,
        cute::conditional_return<Mma1_is_RS>(tOrP, tOsP),
        tOrV(_, _, _, smem_pipe_read.index()),
        tOrO);
    cute::copy(softmax.finalize(1.0f), scores_scale);
    softmax.rescale_o(tOrO, scores_scale);
    warpgroup_wait<0>();
    pipeline_v.consumer_release(
        smem_pipe_read); // release V, otherwise producers will hang
    if constexpr (Is_FP8 && !V_colmajor) {
      hstu::permute_output_fp8(tOrO);
    }
    ++smem_pipe_read;
    ++work_idx;
    return true;
  }
};

} // namespace hstu
