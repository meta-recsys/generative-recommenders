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

// clang-format off
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "static_switch.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.h"
#include "flash_fwd_kernel_sm90.h"
#include "flash_fwd_kernel_sm80.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.h"
#include "mainloop_fwd_sm80.h"
#include "epilogue_fwd.h"
// clang-format on

using namespace cute;

template <
    int Arch,
    int kHeadDim,
    int ClusterM,
    typename Element,
    typename ElementOut,
    bool Causal,
    bool Local,
    bool Contexual_mask,
    bool Jagged,
    bool Has_targets,
    bool V_colmajor>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(
      !(Causal && Local),
      "Causal and Local cannot be enabled at the same time");
  static constexpr bool Is_FP8 =
      cute::is_same_v<Element, cutlass::float_e4m3_t> ||
      cute::is_same_v<Element, cutlass::float_e5m2_t>;
  static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
  using ArchTag =
      std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

  // Can't use structured binding since it's not compatible with constexpr
  static constexpr std::tuple<int, int, bool> kBlockMN_RS = tile_size_fwd_sm90(
      kHeadDim, Causal, Local, sizeof(Element) /*element_size*/, V_colmajor);
  static constexpr std::tuple<int, int, int, int, bool>
      kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(
          Arch == 86 || Arch == 89,
          kHeadDim,
          Causal,
          Local,
          sizeof(Element) /*element_size*/);
  static constexpr int kBlockM = Arch >= 90
      ? std::get<0>(kBlockMN_RS)
      : std::get<0>(kBlockMN_kNWarps_Stages_RS);
  static constexpr int kBlockN = Arch >= 90
      ? std::get<1>(kBlockMN_RS)
      : std::get<1>(kBlockMN_kNWarps_Stages_RS);
  static constexpr bool Mma1_is_RS = std::get<2>(kBlockMN_RS);
  static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
  static constexpr int kStages =
      Arch >= 90 ? 2 : std::get<3>(kBlockMN_kNWarps_Stages_RS);
  static constexpr bool Q_in_regs =
      Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);

#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
  std::printf(
      "kBlockM: (%d), kBlockN: (%d), Mma1_is_RS: (%d), kNWarps: (%d), kStages: (%d), Q_in_regs: (%d)\n",
      kBlockM,
      kBlockN,
      Mma1_is_RS,
      kNWarps,
      kStages,
      Q_in_regs);
#endif

  using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape = cute::Shape<Int<ClusterM>, _1, _1>;
  using CollectiveMainloop = std::conditional_t<
      Arch >= 90,
      flash::CollectiveMainloopFwdSm90<
          kStages,
          ClusterShape,
          TileShape_MNK,
          Element,
          float,
          cutlass::arch::Sm90,
          Causal,
          Local,
          Contexual_mask,
          Jagged,
          Has_targets,
          Mma1_is_RS,
          V_colmajor>,
      flash::CollectiveMainloopFwdSm80<
          kNWarps,
          kStages,
          Q_in_regs,
          TileShape_MNK,
          Element,
          float,
          cutlass::arch::Sm80,
          Causal,
          Local,
          Contexual_mask,
          Jagged,
          Has_targets>>;
  using CollectiveEpilogue = flash::CollectiveEpilogueFwd<
      TileShape_MNK,
      ClusterShape,
      ElementOut,
      ArchTag,
      CollectiveMainloop::NumMmaThreads,
      Jagged,
      FP8_TransposeV>;

  static constexpr int NumProducerThreads = Arch >= 90
      ? CollectiveMainloop::NumProducerThreads
      : CollectiveMainloop::NumMmaThreads;
  using SchedulerPersistent = std::conditional_t<
      Jagged,
      flash::VarlenDynamicPersistentTileScheduler<
          kBlockM,
          CollectiveMainloop::NumMmaThreads,
          NumProducerThreads,
          Arch >= 90 /*WarpSpecialized*/>,
      std::conditional_t<
          !Causal && !Local,
          flash::StaticPersistentTileScheduler,
          flash::DynamicPersistentTileScheduler<
              CollectiveMainloop::NumMmaThreads,
              NumProducerThreads,
              Arch >= 90 /*WarpSpecialized*/>>>;
  using SchedulerSingleTile = flash::
      SingleTileScheduler<Jagged, kBlockM, false /*Sort_by_length_indices*/>;
  // If Split then we probably don't have enough work for PersistentScheduler to
  // be useful. However, if Jagged (e.g., during decode where we have
  // max_seqlens), using PersistentScheduler is better since we'll avoid
  // launching a bunch of thread blocks that immediately exit. On Sm80,
  // noncausal persistent seems a bit slower.
  using Scheduler = std::conditional_t<
      Arch >= 90 ? false : !(Causal && !Jagged),
      SchedulerSingleTile,
      SchedulerPersistent>;
  using AttnKernel = std::conditional_t<
      Arch >= 90,
      flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<
          CollectiveMainloop,
          CollectiveEpilogue,
          Scheduler>>,
      flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<
          CollectiveMainloop,
          CollectiveEpilogue,
          Scheduler>>>;

  int seqlen = !Jagged ? params.max_seq_len : params.total_seq_len;
  int batch = !Jagged ? params.b : 1;
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
  std::printf("max/total seqlen: (%d), batch: (%d)\n", seqlen, batch);
#endif
  typename CollectiveMainloop::StrideV v_strides =
      cute::conditional_return<!V_colmajor>(
          make_stride(
              params.v_row_stride,
              _1{},
              params.v_head_stride,
              !Jagged ? params.v_batch_stride : 0),
          make_stride(
              _1{},
              params.v_dim_stride,
              params.v_head_stride,
              !Jagged ? params.v_batch_stride : 0));
  typename CollectiveMainloop::Arguments mainloop_args{
      static_cast<Element const*>(params.q_ptr),
      {seqlen, params.qk_d, params.h, batch}, // shape_Q
      {params.q_row_stride,
       _1{},
       params.q_head_stride,
       !Jagged ? params.q_batch_stride : 0}, // stride_Q
      static_cast<Element*>(params.k_ptr),
      {seqlen, params.qk_d, params.h, batch}, // shape_K
      {params.k_row_stride,
       _1{},
       params.k_head_stride,
       !Jagged ? params.k_batch_stride : 0}, // stride_K
      static_cast<Element*>(params.v_ptr),
      v_strides, // stride_V
      params.q_descale_ptr,
      params.k_descale_ptr,
      params.v_descale_ptr,
      {params.q_descale_batch_stride, params.q_descale_head_stride},
      {params.k_descale_batch_stride, params.k_descale_head_stride},
      {params.v_descale_batch_stride, params.v_descale_head_stride},
      1.0f / params.max_seq_len,
      params.alpha,
      params.max_attn_len,
      params.min_full_attn_seq_len,
      params.contextual_seq_len,
      params.seq_offsets,
      params.num_targets,
      params.attn_scale,
  };
  typename CollectiveEpilogue::Arguments epilogue_args{
      static_cast<ElementOut*>(params.o_ptr),
      {seqlen, params.v_d, params.h, batch, 1}, // shape_O
      {params.o_row_stride,
       _1{},
       params.o_head_stride,
       !Jagged ? params.o_batch_stride : 0,
       0}, // stride_O
      params.h,
      params.seq_offsets};

  int num_blocks_m =
      cutlass::ceil_div(params.max_seq_len, get<0>(TileShape_MNK{}));
  num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
  typename flash::TileSchedulerArguments scheduler_args{
      num_blocks_m,
      params.h,
      params.b,
      params.max_seq_len,
      params.qk_d,
      sizeof(Element),
      params.tile_count_semaphore,
      params.seq_offsets,
      nullptr /*sort_by_length_indices*/};

  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  typename AttnKernel::Params kernel_params =
      AttnKernel::to_underlying_arguments(
          {mainloop_args,
           epilogue_args,
           {device, params.num_sm},
           scheduler_args});

  dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
  dim3 block_dims = AttnKernel::get_block_shape();
  int smem_size = AttnKernel::SharedStorageSize;
  // int smem_size_q = sizeof(decltype((typename
  // CollectiveMainloop::TensorStorage{}).smem_q)); int smem_size_k =
  // sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
  // int smem_size_v = sizeof(decltype((typename
  // CollectiveMainloop::TensorStorage{}).smem_v)); printf("smem_size = %d, q =
  // %d, k = %d, v = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v);
  // Get the ptr to kernel function.
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
    cutlass::ClusterLaunchParams launch_params{
        grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
  } else {
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    std::cout << "ClusterShape = 1" << std::endl;
    std::cout << "grid_dims = " << grid_dims << std::endl;
    std::cout << "block_dims = " << block_dims << std::endl;
    std::cout << "smem_size = " << smem_size << std::endl;
#endif
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
      CHECK_CUDA(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
  }
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <int Arch, typename T, int kHeadDim>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(
      sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
  static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> ||
      cute::is_same_v<T, cutlass::float_e5m2_t>;
  using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
  CAUSAL_LOCAL_SWITCH(params.is_causal, params.is_local, Causal, Local, [&] {
    VCOLMAJOR_SWITCH(params.v_dim_stride != 1, V_colmajor_, [&] {
      static constexpr bool V_colmajor = V_colmajor_ && sizeof(T) == 1;
      BOOL_SWITCH(params.num_targets, Has_targets, [&] {
        BOOL_SWITCH(params.seq_offsets, Jagged, [&] {
          BOOL_SWITCH(params.has_contexual_mask, Contexual_mask, [&] {
            // Only needed here to decide if we should use cluster
            static constexpr int kBlockM = Arch >= 90
                ? std::get<0>(tile_size_fwd_sm90(
                      kHeadDim,
                      Causal,
                      Local,
                      sizeof(T) /*element_size*/,
                      V_colmajor))
                : 128;
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
            std::printf(
                "[flash_fwd_launch_template] Local: (%d), Jagged: (%d), Has_targets: (%d), Causal: (%d), max_seq_len: (%d), kHeadDim: (%d)\n",
                Local,
                Jagged,
                Has_targets,
                Causal,
                params.max_seq_len,
                kHeadDim);
#endif
            static constexpr bool Enable_cluster = Arch >= 90 &&
                (sizeof(T) == 2 ? (kHeadDim >= 128) : (kHeadDim == 192)) &&
                !Causal && !Local && !Jagged;
            CLUSTER_SWITCH(
                cutlass::ceil_div(params.max_seq_len, kBlockM) % 2 == 0,
                Use_cluster,
                [&] {
                  static constexpr int ClusterM =
                      Enable_cluster && Use_cluster ? 2 : 1;
                  run_flash_fwd<
                      Arch,
                      kHeadDim,
                      ClusterM,
                      T,
                      T_out,
                      Causal,
                      Local,
                      Contexual_mask,
                      Jagged,
                      Has_targets,
                      V_colmajor>(params, stream);
                });
          });
        });
      });
    });
  });
}
