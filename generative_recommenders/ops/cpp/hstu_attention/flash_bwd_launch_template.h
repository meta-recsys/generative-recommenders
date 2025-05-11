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

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp" // For ClusterLauncher
#include "cutlass/device_kernel.h" // For device_kernel
#include "cutlass/kernel_launch.h" // For kernel_launch

#include "epilogue_bwd.h"
#include "flash.h"
#include "flash_bwd_kernel_sm80.h"
#include "flash_bwd_kernel_sm90.h"
#include "flash_bwd_postprocess_kernel.h"
#include "flash_bwd_preprocess_kernel.h"
#include "mainloop_bwd_sm80.h"
#include "mainloop_bwd_sm90_tma_gmma_ws.h"
#include "static_switch.h"
#include "tile_scheduler.h"
#include "tile_size.h"

using namespace cute;

template <
    int Arch,
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    typename Element,
    bool Causal,
    bool Local,
    bool Contexual_mask,
    bool Jagged,
    bool Has_targets,
    bool Deterministic,
    int Stages_dO = 2,
    int Stages_dS_or_QSm80 = 2,
    bool SdP_swapAB = true,
    bool dKV_swapAB = false,
    bool dQ_swapAB = false,
    int NumMmaWarpGroups = 2,
    int AtomLayoutMSdP = 1,
    int AtomLayoutNdKV = 2,
    int AtomLayoutMdQ = 1,
    bool V_in_regs = false>
void run_flash_bwd(Flash_bwd_params& params, cudaStream_t stream) {
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
  std::printf(
      "[flash_bwd_launch_template] Local: (%d), Jagged: (%d), Has_targets: (%d), Causal: (%d), max_seq_len: (%d), kHeadDim: (%d), kBlockM: (%d), kBlockN: (%d)\n",
      Local,
      Jagged,
      Has_targets,
      Causal,
      params.max_seq_len,
      kHeadDim,
      kBlockM,
      kBlockN);
#endif
  static_assert(
      !(Causal && Local), "Causal and Local cannot be true at the same time.");
  using ElementAccum = float;
  using ArchTag =
      std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

  int const total_q_padded_rounded =
      cute::round_up(params.total_seq_len + params.b * kBlockM, kBlockM);
  int seqlen = !Jagged ? params.max_seq_len : params.total_seq_len;
  int seqlen_q_rounded =
      !Jagged ? params.max_seq_len_rounded : total_q_padded_rounded;
  int batch = !Jagged ? params.b : 1;

  using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
  using PreprocessKernel = flash::FlashAttnBwdPreprocess<
      TileShape_MK,
      Element,
      ElementAccum,
      ArchTag,
      /*Clear_dQaccum=*/true,
      Jagged>;
  typename PreprocessKernel::Arguments preprocess_args{
      static_cast<ElementAccum*>(params.dq_accum_ptr),
      {seqlen_q_rounded * params.qk_d_rounded,
       params.h,
       batch}, // shape_dQaccum
      {_1{},
       seqlen_q_rounded * params.qk_d_rounded,
       !Jagged ? params.qk_d_rounded * params.max_seq_len_rounded * params.h
               : 0}, // stride_dQaccum
      params.b,
      params.h,
      params.max_seq_len,
      params.dq_semaphore,
      params.seq_offsets};
  typename PreprocessKernel::Params preprocess_params =
      PreprocessKernel::to_underlying_arguments(preprocess_args);
  int num_m_block = cute::ceil_div(params.max_seq_len, kBlockM);
  dim3 grid_m(num_m_block, params.h, params.b);
  cutlass::kernel_launch<PreprocessKernel>(
      grid_m,
      PreprocessKernel::MaxThreadsPerBlock,
      PreprocessKernel::SharedStorageSize,
      stream,
      preprocess_params,
      false /*launch_with_pdl*/);
  CHECK_CUDA_KERNEL_LAUNCH();

  using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape =
      cute::Shape<_1, Int<1>, _1>; // Currently doesn't not support cluster
  // Stages_dS_or_QSm80 is Stages_dS if Sm90 and Stages if Sm80
  static constexpr int Stages = Arch >= 90 ? 2 : Stages_dS_or_QSm80;
  static constexpr int Stages_dS = Arch >= 90 ? Stages_dS_or_QSm80 : 1;
  using CollectiveMainloop = std::conditional_t<
      Arch >= 90,
      flash::CollectiveMainloopBwdSm90<
          Stages,
          Stages_dO,
          Stages_dS,
          ClusterShape,
          TileShape_MNK,
          Element,
          ElementAccum,
          cutlass::arch::Sm90,
          Causal,
          Local,
          Contexual_mask,
          Jagged,
          Has_targets,
          Deterministic,
          SdP_swapAB,
          dKV_swapAB,
          dQ_swapAB,
          NumMmaWarpGroups,
          AtomLayoutMSdP,
          AtomLayoutNdKV,
          AtomLayoutMdQ,
          V_in_regs>,
      flash::CollectiveMainloopBwdSm80<
          Stages,
          Stages_dO,
          TileShape_MNK,
          Element,
          ElementAccum,
          cutlass::arch::Sm80,
          Causal,
          Local,
          Contexual_mask,
          Jagged,
          Has_targets,
          Deterministic,
          SdP_swapAB,
          dKV_swapAB,
          dQ_swapAB,
          NumMmaWarpGroups,
          AtomLayoutMSdP,
          AtomLayoutNdKV,
          AtomLayoutMdQ,
          V_in_regs>>;
  using CollectiveEpilogue = flash::CollectiveEpilogueBwd<
      TileShape_MNK,
      Element,
      ArchTag,
      CollectiveMainloop::NumMmaThreads,
      Jagged,
      dKV_swapAB,
      NumMmaWarpGroups*(Arch >= 90 ? 1 : cutlass::NumWarpsPerWarpGroup) /
          AtomLayoutNdKV>;
  using Scheduler =
      flash::SingleTileScheduler<Jagged, kBlockN, false /*Sort_by_length*/>;
  using AttnKernel = std::conditional_t<
      Arch >= 90,
      flash::enable_sm90_or_later<flash::FlashAttnBwdSm90<
          CollectiveMainloop,
          CollectiveEpilogue,
          Scheduler>>,
      flash::enable_sm80_to_sm89<flash::FlashAttnBwdSm80<
          CollectiveMainloop,
          CollectiveEpilogue,
          Scheduler>>>;

  typename CollectiveMainloop::Arguments mainloop_args{
      static_cast<Element const*>(params.q_ptr),
      {seqlen, params.qk_d, params.h, batch}, // shape_Q
      {params.q_row_stride,
       _1{},
       params.q_head_stride,
       !Jagged ? params.q_batch_stride : 0}, // stride_Q
      static_cast<Element const*>(params.k_ptr),
      {seqlen, params.qk_d, params.h, batch}, // shape_K
      {params.k_row_stride,
       _1{},
       params.k_head_stride,
       !Jagged ? params.k_batch_stride : 0}, // stride_K
      static_cast<Element const*>(params.v_ptr),
      {params.v_row_stride,
       _1{},
       params.v_head_stride,
       !Jagged ? params.v_batch_stride : 0}, // stride_V
      static_cast<Element const*>(params.do_ptr),
      {params.do_row_stride,
       _1{},
       params.do_head_stride,
       !Jagged ? params.do_batch_stride : 0}, // stride_dO
      static_cast<ElementAccum*>(params.dq_accum_ptr),
      {seqlen_q_rounded * params.qk_d_rounded,
       params.h,
       batch}, // shape_dQaccum
      {_1{},
       seqlen_q_rounded * params.qk_d_rounded,
       !Jagged ? params.qk_d_rounded * params.max_seq_len_rounded * params.h
               : 0}, // stride_dQaccum
      params.max_attn_len,
      params.min_full_attn_seq_len,
      params.contextual_seq_len,
      1.0f / params.max_seq_len,
      params.alpha,
      params.b,
      params.dq_semaphore,
      params.seq_offsets,
      params.num_targets,
      params.attn_scale};
  typename CollectiveEpilogue::Arguments epilogue_args{
      static_cast<typename CollectiveEpilogue::Element*>(params.dk_ptr),
      [&] {
        return typename CollectiveEpilogue::ShapedKV{
            seqlen, params.qk_d, params.h, batch}; // shape_dK
      }(),
      [&] {
        return typename CollectiveEpilogue::StridedKV{
            params.dk_row_stride,
            _1{},
            params.dk_head_stride,
            !Jagged ? params.dk_batch_stride : 0}; // stride_dK
      }(),
      static_cast<typename CollectiveEpilogue::Element*>(params.dv_ptr),
      [&] {
        return typename CollectiveEpilogue::StridedKV{
            params.dv_row_stride,
            _1{},
            params.dv_head_stride,
            !Jagged ? params.dv_batch_stride : 0}; // stride_dV
      }(),
      params.h,
      params.seq_offsets};

  int num_blocks_n =
      cutlass::ceil_div(params.max_seq_len, get<1>(TileShape_MNK{}));
  num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
  typename flash::TileSchedulerArguments scheduler_args{
      num_blocks_n,
      params.h,
      params.b,
      params.max_seq_len,
      params.qk_d,
      sizeof(Element),
      params.tile_count_semaphore,
      params.seq_offsets,
      params.sort_by_length_indices};

  int device;
  cudaGetDevice(&device);
  typename AttnKernel::Params kernel_params =
      AttnKernel::to_underlying_arguments(
          {mainloop_args,
           epilogue_args,
           {device, params.num_sm},
           scheduler_args});

  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;
  if constexpr (size(ClusterShape{}) > 1) {
    void const* kernel = (void const*)cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 cluster_dims(
        size<0>(ClusterShape{}),
        size<1>(ClusterShape{}),
        size<2>(ClusterShape{}));
    cutlass::ClusterLauncher::launch(
        grid_dims,
        cluster_dims,
        block_dims,
        smem_size,
        stream,
        kernel,
        kernel_params,
        false /*launch_with_pdl*/);
  } else {
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(
          cutlass::device_kernel<AttnKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size));
    }
    cutlass::kernel_launch<AttnKernel>(
        grid_dims,
        block_dims,
        smem_size,
        stream,
        kernel_params,
        false /*launch_with_pdl*/);
  }
  CHECK_CUDA_KERNEL_LAUNCH();

  using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<
      TileShape_MK,
      Element,
      ElementAccum,
      ArchTag,
      AttnKernel::CollectiveMainloop::NumMmaThreads,
      typename AttnKernel::CollectiveMainloop::TiledMmadQ,
      AttnKernel::CollectiveMainloop::dQ_swapAB,
      Jagged>;
  typename PostprocessKernel::Arguments postprocess_args{
      static_cast<ElementAccum const*>(params.dq_accum_ptr),
      {seqlen_q_rounded * params.qk_d_rounded,
       params.h,
       batch}, // shape_dQaccum
      {_1{},
       seqlen_q_rounded * params.qk_d_rounded,
       !Jagged ? params.qk_d_rounded * params.max_seq_len_rounded * params.h
               : 0}, // stride_dQaccum
      static_cast<Element*>(params.dq_ptr),
      {seqlen, params.qk_d, params.h, batch}, // shape_dQ
      {params.dq_row_stride,
       _1{},
       params.dq_head_stride,
       params.dq_batch_stride}, // stride_dQ
      params.seq_offsets};
  typename PostprocessKernel::Params postprocess_params =
      PostprocessKernel::to_underlying_arguments(postprocess_args);
  int num_m_block_postprocess =
      cute::ceil_div(params.max_seq_len, get<0>(TileShape_MK{}));
  dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
  int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
  if (smem_size_postprocess >= 48 * 1024) {
    CHECK_CUDA(cudaFuncSetAttribute(
        cutlass::device_kernel<PostprocessKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_postprocess));
  }
  cutlass::kernel_launch<PostprocessKernel>(
      grid_m_postprocess,
      PostprocessKernel::MaxThreadsPerBlock,
      smem_size_postprocess,
      stream,
      postprocess_params,
      false /*launch_with_pdl*/);
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <
    int Arch,
    typename T,
    int kBlockM,
    int kBlockN,
    int kHeadDim,
    bool Causal,
    bool Local,
    int Stages_dO = 2,
    int Stages_dS_or_QSm80 = 2,
    bool SdP_swapAB = true,
    bool dKV_swapAB = false,
    bool dQ_swapAB = false,
    int NumMmaWarpGroups = 2,
    int AtomLayoutMSdP = 1,
    int AtomLayoutNdKV = 2,
    int AtomLayoutMdQ = 1,
    bool V_in_regs = false>
void run_mha_bwd_dispatch(Flash_bwd_params& params, cudaStream_t stream) {
  BOOL_SWITCH(params.seq_offsets != nullptr, Jagged, [&] {
    BOOL_SWITCH(params.num_targets != nullptr, Has_targets, [&] {
      BOOL_SWITCH(params.has_contexual_mask, Contexual_mask, [&] {
        run_flash_bwd<
            Arch,
            kHeadDim,
            kBlockM,
            kBlockN,
            T,
            Causal,
            Local,
            Contexual_mask,
            Jagged,
            Has_targets,
            false /*Deterministic*/,
            Stages_dO,
            Stages_dS_or_QSm80,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            NumMmaWarpGroups,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs>(params, stream);
      });
    });
  });
}

template <int Arch, typename T, int kHeadDim>
void run_mha_bwd_(Flash_bwd_params& params, cudaStream_t stream) {
  CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Causal, Local, [&] {
    int const kBlockM = kBlockM_bwd(Arch, kHeadDim, Causal, Local);
    int const kBlockN = kBlockN_bwd(Arch, kHeadDim);
    bool const V_in_regs = V_in_regs_bwd(Arch, kHeadDim);
    static constexpr std::tuple<int, int> Stages = Stages_bwd(Arch, kHeadDim);
    static constexpr std::tuple<bool, bool, bool> swapAB =
        swapAB_bwd(Arch, kHeadDim, Causal, Local);
    int const NumMmaWarpGroups = NumMmaWarpGroups_bwd(Arch, kHeadDim);
    static constexpr std::tuple<int, int, int> AtomLayout =
        AtomLayout_bwd(Arch, kHeadDim);
    run_mha_bwd_dispatch<
        Arch,
        T,
        kBlockM,
        kBlockN,
        kHeadDim,
        Causal,
        Local,
        std::get<0>(Stages), /*Stages_dO*/
        std::get<1>(Stages), /*Stages_dS_or_QSm80*/
        std::get<0>(swapAB), /*SdP_swapAB*/
        std::get<1>(swapAB), /*dKV_swapAB*/
        std::get<2>(swapAB), /*dQ_swapAB*/
        NumMmaWarpGroups,
        std::get<0>(AtomLayout), /*AtomLayoutMSdP*/
        std::get<1>(AtomLayout), /*AtomLayoutNdKV*/
        std::get<2>(AtomLayout), /*AtomLayoutMdQ*/
        V_in_regs>(params, stream);
  });
}
