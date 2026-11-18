// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/hip_check_error.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_config.hpp"
#include "hstu_attention_fwd_pipeline_problem.hpp"
#include "hstu_attention_fwd_pipeline_traits.hpp"
#include "hstu_attention_fwd_setting.hpp"
#include "hstu_attention_fwd_splitkv_combine_kernel.hpp"
#include "hstu_attention_fwd_splitkv_combine_setting.hpp"
#include "hstu_attention_fwd_splitkv_kernel.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_max_splits_switch.hpp"
#include "hstu_attention_no_softmax_fwd_direct_lds_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_splitkv_combine_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_tdm_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_splitkv_helper.hpp"
#include "hstu_attention_with_softmax_fwd_direct_lds_trload_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_splitkv_combine_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_tdm_trload_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_trload_pipeline.hpp"

template <
    typename InOutDataType,
    bool kUseCausal,
    bool kUseSoftmax,
    bool kStoreLSE,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK,
    bool kHdimPadded,
    ck_tile::index_t MTile>
struct batched_forward_splitkv_dispatch {
  // The pipeline kind is resolved first: the tile setting depends on it,
  // because the Tdm pipelines need a block tile the trload ones cannot use.
  // Dropout stays on the trload pipeline, whose dropout path has upstream's
  // 9a13bcc32 fix (not on hstu_attention_fwd yet)
  static constexpr HstuFwdPipelineKind kPipelineKind =
      (kHasDropout &&
       get_hstu_fwd_pipeline_kind<kUseSoftmax, MaxK, kHdimPadded>() ==
           HstuFwdPipelineKind::DirectLdsTrLoad)
      ? HstuFwdPipelineKind::TrLoad
      : get_hstu_fwd_pipeline_kind<kUseSoftmax, MaxK, kHdimPadded>();

  using HstuAttentionFwdTileSetting = decltype(GetHstuAttentionFwdTileSetting<
                                               MaxK,
                                               MTile,
                                               kUseSoftmax,
                                               kUseCausal,
                                               kPipelineKind>());

  using HstuAttentionCombineTileSetting =
      decltype(GetHstuAttentionFwdSplitKVCombineTileSetting<MaxK>());

  template <bool kIsCrossAttention>
  using HstuFwdPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
      InOutDataType,
      typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
      typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
      typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
      kIsCrossAttention,
      false, // kUseGroup
      false, // kIsJagged
      kHasBias,
      kHasDropout,
      kUseCausal,
      kUseSoftmax,
      kStoreLSE,
      HstuAttentionFwdTileSetting>;

  using OaccDataType = HstuAttentionFwdTypeConfig<InOutDataType>::OaccDataType;
  using ODataType = HstuAttentionFwdTypeConfig<InOutDataType>::ODataType;
  using LSEDataType = HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType;

  template <ck_tile::index_t kMaxSplits>
  using HstuCombinePipelineProblemTemp =
      ck_tile::HstuAttentionFwdSplitKVCombinePipelineProblem<
          OaccDataType,
          LSEDataType,
          ODataType,
          false /* kIsJagged */,
          kUseSoftmax,
          kStoreLSE,
          HstuAttentionCombineTileSetting,
          kMaxSplits>;

  static void Run(HstuAttentionNoGroupFwdParams& param, hipStream_t stream) {
    constexpr ck_tile::index_t occupancy = -1;

    SplitkvWorkspace ws;

    {
      const bool pad_seqlen_k =
          !(param.seqlen_kv % HstuAttentionFwdTileSetting::kN0 == 0);
      const bool pad_headdim_qk =
          !(param.hdim_qk % HstuAttentionFwdTileSetting::kQKHeaddim == 0);
      const bool pad_headdim_v =
          !(param.hdim_v % HstuAttentionFwdTileSetting::kN1 == 0);

      // no need to check seqlen_q since it is not used as fastest dim,
      // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
      constexpr bool kPadSeqLenQ = false;

      BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
        using HstuPipelineProblem =
            HstuFwdPipelineProblemTemp<kIsCrossAttention>;

        if constexpr (kPipelineKind == HstuFwdPipelineKind::DirectLdsTrLoad) {
          BOOL_SWITCH(pad_seqlen_k, kPadSeqLenK, [&] {
            using HstuTraits = ck_tile::HstuAttentionFwdTraits<
                kPadSeqLenQ,
                kPadSeqLenK,
                false /* kPadHeadDimQK */,
                false /* kPadHeadDimQK */,
                occupancy>;

            using HstuEpilogue =
                ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                    OaccDataType,
                    OaccDataType, // keep output as OaccDataType
                    kPadSeqLenQ,
                    false /* kPadHeadDimV */,
                    false>>;

            // we only use async_load_tile() for K/V when its maximum
            // vector_size is
            //  dwordx4 (ToDO: use dwordx3 when hdim is 96)
            constexpr ck_tile::index_t k_vector_bytes =
                HstuPipelineProblem::GetKDramTileAccessMaxVectorSize(
                    ck_tile::bool_constant<true>{}) *
                sizeof(InOutDataType);
            constexpr ck_tile::index_t v_vector_bytes =
                HstuPipelineProblem::GetVDramTileAccessMaxVectorSize(
                    ck_tile::bool_constant<true>{}) *
                sizeof(InOutDataType);

            static_assert(k_vector_bytes == 4 * sizeof(int), "Check failed!");
            static_assert(v_vector_bytes == 4 * sizeof(int), "Check failed!");
            static_assert(MaxK != 96, "Check failed!");

            using HstuPipeline = std::conditional_t<
                kUseSoftmax,
                ck_tile::
                    HstuAttentionWithSoftmaxFwdPipelineQRKSVSDirectLdsTrLoad<
                        HstuPipelineProblem,
                        HstuTraits>,
                ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSDirectLdsTrLoad<
                    HstuPipelineProblem,
                    HstuTraits>>;

            using HstuKernel = ck_tile::
                HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

            RunWithFwdSplitKVKernel<HstuKernel>(param, ws, stream);
          });
        } else {
          BOOL_SWITCH_3(
              pad_seqlen_k,
              kPadSeqLenK,
              pad_headdim_qk,
              kPadHeadDimQK,
              pad_headdim_v,
              kPadHeadDimV,
              [&] {
                using HstuTraits = ck_tile::HstuAttentionFwdTraits<
                    kPadSeqLenQ,
                    kPadSeqLenK,
                    kPadHeadDimQK,
                    kPadHeadDimV,
                    occupancy>;

                using HstuEpilogue = ck_tile::Default2DEpilogue<
                    ck_tile::Default2DEpilogueProblem<
                        OaccDataType,
                        OaccDataType, // keep output as OaccDataType
                        kPadSeqLenQ,
                        kPadHeadDimV,
                        false>>;

                if constexpr (kPipelineKind == HstuFwdPipelineKind::Default) {
                  using HstuPipeline = std::conditional_t<
                      kUseSoftmax,
                      ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVS<
                          HstuPipelineProblem,
                          HstuTraits>,
                      ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<
                          HstuPipelineProblem,
                          HstuTraits>>;

                  using HstuKernel = ck_tile::
                      HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

                  RunWithFwdSplitKVKernel<HstuKernel>(param, ws, stream);
                } else if constexpr (
                    kPipelineKind == HstuFwdPipelineKind::TdmTrLoad) {
                  using HstuPipeline = std::conditional_t<
                      kUseSoftmax,
                      ck_tile::
                          HstuAttentionWithSoftmaxFwdPipelineQRKSVSTdmTrLoad<
                              HstuPipelineProblem,
                              HstuTraits>,
                      ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTdmTrLoad<
                          HstuPipelineProblem,
                          HstuTraits>>;

                  using HstuKernel = ck_tile::
                      HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

                  RunWithFwdSplitKVKernel<HstuKernel>(param, ws, stream);
                } else {
                  using HstuPipeline = std::conditional_t<
                      kUseSoftmax,
                      ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad<
                          HstuPipelineProblem,
                          HstuTraits>,
                      ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad<
                          HstuPipelineProblem,
                          HstuTraits>>;

                  using HstuKernel = ck_tile::
                      HstuAttentionFwdSplitKVKernel<HstuPipeline, HstuEpilogue>;

                  RunWithFwdSplitKVKernel<HstuKernel>(param, ws, stream);
                }
              });
        }
      });
    };

    if constexpr (kUseSoftmax) {
      static constexpr auto kM = HstuAttentionCombineTileSetting::kM;
      // ToDo: be carefule to use get_warp_size() on host-layer on Wave32
      static constexpr auto kBlockSize =
          HstuAttentionCombineTileSetting::NumWarps * ck_tile::get_warp_size();

      const bool pad_headdim_o =
          !(param.hdim_v % HstuAttentionCombineTileSetting::kOHeaddim == 0);

      // no need to check seqlen_q since it is not used as fastest dim,
      // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
      constexpr bool kPadSeqLenQ = false;

      MAX_SPLITS_SWITCH(ws.num_splits, TMP_MAX_SPLITS, [&] {
        constexpr ck_tile::index_t kMaxSplits = [&]() {
          if constexpr (kM * TMP_MAX_SPLITS >= kBlockSize)
            return TMP_MAX_SPLITS;
          else if constexpr (kM * 2 * TMP_MAX_SPLITS >= kBlockSize)
            return 2 * TMP_MAX_SPLITS;
          else
            return 4 * TMP_MAX_SPLITS;
        }();

        const bool pad_num_splits = (ws.num_splits < kMaxSplits);

        using HstuCombinePipelineProblem =
            HstuCombinePipelineProblemTemp<kMaxSplits>;

        BOOL_SWITCH_2(
            pad_headdim_o, kPadHeadDimO, pad_num_splits, kPadNumSplits, [&] {
              using HstuTraits = ck_tile::HstuAttentionFwdSplitKVCombineTraits<
                  kPadSeqLenQ,
                  kPadHeadDimO,
                  kPadNumSplits,
                  occupancy>;

              using HstuEpilogue =
                  ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                      OaccDataType,
                      ODataType,
                      kPadSeqLenQ,
                      kPadHeadDimO,
                      false>>;

              using HstuPipeline =
                  ck_tile::HstuAttentionWithSoftmaxFwdSplitKVCombinePipeline<
                      HstuCombinePipelineProblem,
                      HstuTraits>;

              using HstuKernel = ck_tile::HstuAttentionFwdSplitKVCombineKernel<
                  HstuPipeline,
                  HstuEpilogue>;

              RunWithFwdSplitKVCombineKernel<HstuKernel>(param, ws, stream);
            });
      });
    } else {
      const bool pad_headdim_o =
          !(param.hdim_v % HstuAttentionCombineTileSetting::kOHeaddim == 0);

      // no need to check seqlen_q since it is not used as fastest dim,
      // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
      constexpr bool kPadSeqLenQ = false;

      using HstuCombinePipelineProblem = HstuCombinePipelineProblemTemp<0>;

      BOOL_SWITCH(pad_headdim_o, kPadHeadDimO, [&] {
        using HstuTraits = ck_tile::HstuAttentionFwdSplitKVCombineTraits<
            kPadSeqLenQ,
            kPadHeadDimO,
            false /*kPadNumSplits*/,
            occupancy>;

        using HstuEpilogue =
            ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<
                OaccDataType,
                ODataType,
                kPadSeqLenQ,
                kPadHeadDimO,
                false>>;

        using HstuPipeline =
            ck_tile::HstuAttentionNoSoftmaxFwdSplitKVCombinePipeline<
                HstuCombinePipelineProblem,
                HstuTraits>;

        using HstuKernel = ck_tile::
            HstuAttentionFwdSplitKVCombineKernel<HstuPipeline, HstuEpilogue>;

        RunWithFwdSplitKVCombineKernel<HstuKernel>(param, ws, stream);
      });
    }
  };

  template <typename HstuKernel>
  static void RunWithFwdSplitKVKernel(
      HstuAttentionNoGroupFwdParams& param,
      SplitkvWorkspace& ws,
      hipStream_t stream) {
    ws.num_splits = get_suggested_num_splits(
        true, /* almost_invariant_seqlen*/
        param.num_batch,
        param.num_head,
        param.seqlen_q,
        param.seqlen_kv,
        HstuAttentionFwdTileSetting::kN0);

    // assume the workspace for o_acc is in compact shape of [num_batch,
    // seqlen_q, num_head, num_splits, hdim]
    size_t workspace_bytes = static_cast<size_t>(param.num_batch) *
        param.seqlen_q * param.num_head * ws.num_splits * param.hdim_v *
        sizeof(OaccDataType);

    ws.o_acc_ptr = splitkv_workspace_alloc(
        param.workspace_allocator, workspace_bytes, stream);

    if constexpr (kUseSoftmax) {
      // assume the workspace for l_acc is in compact shape of [num_batch,
      // seqlen_q, num_head, num_splits]
      workspace_bytes = static_cast<size_t>(param.num_batch) * param.seqlen_q *
          param.num_head * ws.num_splits * sizeof(LSEDataType);

      ws.lse_acc_ptr = splitkv_workspace_alloc(
          param.workspace_allocator, workspace_bytes, stream);
    }

    const auto kargs = [&] {
      return HstuKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.bias_ptr,
          ws.o_acc_ptr,
          ws.lse_acc_ptr,
          ws.num_splits,
          param.seqlen_q,
          param.is_cross_attention ? param.seqlen_kv : param.seqlen_q,
          param.hdim_qk,
          param.hdim_v,
          param.num_head,
          param.scale_s,
          param.attn_scale,
          param.attn_scale_ptr,
          param.seq_stride_q,
          param.seq_stride_k,
          param.seq_stride_v,
          param.seq_stride_bias,
          param.nhead_stride_q,
          param.nhead_stride_k,
          param.nhead_stride_v,
          param.nhead_stride_bias,
          param.batch_stride_q,
          param.batch_stride_k,
          param.batch_stride_v,
          param.batch_stride_bias,
          param.num_targets_ptr,
          param.contextual_seqlen,
          param.window_size,
          param.min_full_attn_seqlen,
          param.p_drop,
          param.philox_seed,
          param.philox_offset);
    }();

    bool has_minfull_attn_seqlen = (param.min_full_attn_seqlen > 0);
    dim3 kGridSize = HstuKernel::GridSize(
        param.num_batch,
        param.num_head,
        param.seqlen_q,
        param.hdim_v,
        ws.num_splits,
        true, // almost_invariant_seqlen
        has_minfull_attn_seqlen);
    dim3 kBlockSize = HstuKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
  };

  template <typename HstuKernel>
  static void RunWithFwdSplitKVCombineKernel(
      HstuAttentionNoGroupFwdParams& param,
      SplitkvWorkspace& ws,
      hipStream_t stream) {
    const auto kargs = [&] {
      return HstuKernel::MakeKargs(
          ws.o_acc_ptr,
          ws.lse_acc_ptr,
          param.o_ptr,
          param.lse_ptr,
          param.batch_stride_o,
          param.batch_stride_lse,
          param.seq_stride_o,
          param.seq_stride_lse,
          param.nhead_stride_o,
          param.nhead_stride_lse,
          param.seqlen_q,
          param.num_head,
          ws.num_splits,
          param.hdim_v);
    }();

    dim3 kGridSize = HstuKernel::GridSize(
        param.num_batch,
        param.num_head,
        param.seqlen_q,
        true /* almost_invariant_seqlen */);
    dim3 kBlockSize = HstuKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            HstuKernel{}, kGridSize, kBlockSize, 0, kargs));

    splitkv_workspace_free(param.workspace_allocator, ws.o_acc_ptr, stream);
    if constexpr (kUseSoftmax) {
      splitkv_workspace_free(param.workspace_allocator, ws.lse_acc_ptr, stream);
    }
  };
};
