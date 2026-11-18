// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// Modifications Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_config.hpp"
#include "hstu_attention_epilogue.hpp"
#include "hstu_attention_fwd_kernel.hpp"
#include "hstu_attention_fwd_pipeline_problem.hpp"
#include "hstu_attention_fwd_pipeline_traits.hpp"
#include "hstu_attention_fwd_setting.hpp"
#include "hstu_attention_fwd_type_config.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_no_softmax_fwd_direct_lds_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_tdm_trload_pipeline.hpp"
#include "hstu_attention_no_softmax_fwd_trload_pipeline.hpp"
#include "hstu_attention_params.hpp"
#include "hstu_attention_with_softmax_fwd_direct_lds_trload_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_tdm_trload_pipeline.hpp"
#include "hstu_attention_with_softmax_fwd_trload_pipeline.hpp"

#include "hstu_attention_group_forward_splitkv_dispatch.hpp"
#include "hstu_attention_host_util.hpp"

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
struct group_forward_dispatch {
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

  using HstuAttentionTileSetting = decltype(GetHstuAttentionFwdTileSetting<
                                            MaxK,
                                            MTile,
                                            kUseSoftmax,
                                            kUseCausal,
                                            kPipelineKind>());

  template <bool kIsCrossAttention>
  using HstuPipelineProblemTemp = ck_tile::HstuAttentionFwdPipelineProblem<
      InOutDataType,
      typename HstuAttentionFwdTypeConfig<InOutDataType>::GemmAccDataType,
      typename HstuAttentionFwdTypeConfig<InOutDataType>::CompDataType,
      typename HstuAttentionFwdTypeConfig<InOutDataType>::BiasDataType,
      kIsCrossAttention,
      true, // kUseGroup
      true, // kIsJagged
      kHasBias,
      kHasDropout,
      kUseCausal,
      kUseSoftmax,
      kStoreLSE,
      HstuAttentionTileSetting>;

  static void Run(HstuAttentionGroupFwdParams& param, hipStream_t stream) {
    constexpr ck_tile::index_t occupancy = -1;

    const bool pad_headdim_qk =
        !(param.hdim_qk % HstuAttentionTileSetting::kQKHeaddim == 0);
    const bool pad_headdim_v =
        !(param.hdim_v % HstuAttentionTileSetting::kN1 == 0);

    // no need to check seqlen_q since it is not used as fastest dim,
    // buffer_load_dwordxx/buffer_store_dwordxx can handle oob access
    constexpr bool kPadSeqLenQ = false;

    constexpr bool kPadSeqLenK = true;

    BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
      using HstuPipelineProblem = HstuPipelineProblemTemp<kIsCrossAttention>;

      if constexpr (kPipelineKind == HstuFwdPipelineKind::DirectLdsTrLoad) {
        using HstuTraits = ck_tile::HstuAttentionFwdTraits<
            kPadSeqLenQ,
            kPadSeqLenK,
            false /*kPadHeadDimQK*/,
            false /*kPadHeadDimV*/,
            occupancy>;

        using HstuEpilogue =
            ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename HstuAttentionFwdTypeConfig<
                    InOutDataType>::OaccDataType,
                typename HstuAttentionFwdTypeConfig<InOutDataType>::ODataType,
                kPadSeqLenQ,
                false /*kPadHeadDimV*/>>;

        // we only use async_load_tile() for K/V when its maximum vector_size is
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
            ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSDirectLdsTrLoad<
                HstuPipelineProblem,
                HstuTraits>,
            ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSDirectLdsTrLoad<
                HstuPipelineProblem,
                HstuTraits>>;

        using HstuKernel =
            ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

        RunWithKernel<HstuKernel>(param, stream);
      } else {
        BOOL_SWITCH_2(
            pad_headdim_qk, kPadHeadDimQK, pad_headdim_v, kPadHeadDimV, [&] {
              using HstuTraits = ck_tile::HstuAttentionFwdTraits<
                  kPadSeqLenQ,
                  kPadSeqLenK,
                  kPadHeadDimQK,
                  kPadHeadDimV,
                  occupancy>;

              using HstuEpilogue = ck_tile::NRepetitions2DEpilogue<
                  ck_tile::Default2DEpilogueProblem<
                      typename HstuAttentionFwdTypeConfig<
                          InOutDataType>::OaccDataType,
                      typename HstuAttentionFwdTypeConfig<
                          InOutDataType>::ODataType,
                      kPadSeqLenQ,
                      kPadHeadDimV>>;

              if constexpr (kPipelineKind == HstuFwdPipelineKind::Default) {
                using HstuPipeline = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVS<
                        HstuPipelineProblem,
                        HstuTraits>,
                    ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVS<
                        HstuPipelineProblem,
                        HstuTraits>>;

                using HstuKernel =
                    ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                RunWithKernel<HstuKernel>(param, stream);
              } else if constexpr (
                  kPipelineKind == HstuFwdPipelineKind::TdmTrLoad) {
                using HstuPipeline = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSTdmTrLoad<
                        HstuPipelineProblem,
                        HstuTraits>,
                    ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTdmTrLoad<
                        HstuPipelineProblem,
                        HstuTraits>>;

                using HstuKernel =
                    ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                RunWithKernel<HstuKernel>(param, stream);
              } else {
                using HstuPipeline = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::HstuAttentionWithSoftmaxFwdPipelineQRKSVSTrLoad<
                        HstuPipelineProblem,
                        HstuTraits>,
                    ck_tile::HstuAttentionNoSoftmaxFwdPipelineQRKSVSTrLoad<
                        HstuPipelineProblem,
                        HstuTraits>>;

                using HstuKernel =
                    ck_tile::HstuAttentionFwdKernel<HstuPipeline, HstuEpilogue>;

                RunWithKernel<HstuKernel>(param, stream);
              }
            });
      }
    });
  };

  template <typename HstuKernel>
  static void RunWithKernel(
      HstuAttentionGroupFwdParams& param,
      hipStream_t stream) {
    bool almost_invariant_seqlen = is_almost_invariant_seqlen(param);

    const auto kargs = [&] {
      return HstuKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.bias_ptr,
          param.o_ptr,
          param.lse_ptr,
          param.num_batch / param.num_group,
          param.seq_q_offsets_ptr,
          param.is_cross_attention ? param.seq_kv_offsets_ptr
                                   : param.seq_q_offsets_ptr,
          param.group_max_seqlen_q_ptr,
          param.group_contextual_seqlen_ptr,
          param.group_window_size_ptr,
          param.group_min_full_attn_seqlen_ptr,
          param.group_attn_scale_ptr,
          param.hdim_qk,
          param.hdim_v,
          param.num_head,
          param.scale_s,
          almost_invariant_seqlen,
          param.seq_stride_q,
          param.seq_stride_k,
          param.seq_stride_v,
          param.seq_stride_bias,
          param.seq_stride_o,
          param.seq_stride_lse,
          param.nhead_stride_q,
          param.nhead_stride_k,
          param.nhead_stride_v,
          param.nhead_stride_bias,
          param.nhead_stride_o,
          param.nhead_stride_lse,
          param.num_targets_ptr,
          param.p_drop,
          param.philox_seed,
          param.philox_offset);
    }();

    dim3 kGridSize = HstuKernel::GridSize(
        param.num_batch,
        param.num_head,
        param.max_seqlen_q,
        param.hdim_v,
        almost_invariant_seqlen);
    dim3 kBlockSize = HstuKernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = HstuKernel::kBlockPerCu;

    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<kBlockPerCu>(
            HstuKernel{}, kGridSize, kBlockSize, 0, kargs));
  };
};

template <
    typename InOutDataType,
    bool kUseCausal,
    bool kUseSoftmax,
    bool kStoreLSE,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_group_forward_dispatch(
    HstuAttentionGroupFwdParams& param,
    hipStream_t stream) {
  int mtile_size = get_hstu_attention_fwd_mtile(
      param.num_batch, param.num_head, param.max_seqlen_q, param.max_seqlen_kv);

  const bool hdim_padded = (param.hdim_qk < MaxK || param.hdim_v < MaxK);

  BOOL_SWITCH(hdim_padded, kHdimPadded, [&] {
    if (!param.is_cross_attention && mtile_size == 128)
      group_forward_dispatch<
          InOutDataType,
          kUseCausal,
          kUseSoftmax,
          kStoreLSE,
          kHasBias,
          kHasDropout,
          MaxK,
          kHdimPadded,
          128>::Run(param, stream);
    else {
      // for cross-attention, we should give more opportunity to use split-kv
      // since the seqlen_kv is usually much bigger than seqlen_q, so the
      // main-loop along the seqlen_kv have enough iterations to counter-act the
      // cost brought by splitting

      const bool disable_fwd_splitkv = []() {
        const char* env_p = std::getenv("HSTU_DISABLE_SPLITKV");
        if (env_p == nullptr)
          return false;
        return static_cast<bool>(atoi(env_p));
      }();

      if (!disable_fwd_splitkv &&
          shall_use_splitkv(
              param.num_batch,
              param.num_head,
              param.max_seqlen_q,
              param.max_seqlen_kv)) {
        if (mtile_size == 128)
          group_forward_splitkv_dispatch<
              InOutDataType,
              kUseCausal,
              kUseSoftmax,
              kStoreLSE,
              kHasBias,
              kHasDropout,
              MaxK,
              kHdimPadded,
              128>::Run(param, stream);
        else
          group_forward_splitkv_dispatch<
              InOutDataType,
              kUseCausal,
              kUseSoftmax,
              kStoreLSE,
              kHasBias,
              kHasDropout,
              MaxK,
              kHdimPadded,
              64>::Run(param, stream);
      } else {
        if (mtile_size == 128)
          group_forward_dispatch<
              InOutDataType,
              kUseCausal,
              kUseSoftmax,
              kStoreLSE,
              kHasBias,
              kHasDropout,
              MaxK,
              kHdimPadded,
              128>::Run(param, stream);
        else
          group_forward_dispatch<
              InOutDataType,
              kUseCausal,
              kUseSoftmax,
              kStoreLSE,
              kHasBias,
              kHasDropout,
              MaxK,
              kHdimPadded,
              64>::Run(param, stream);
      }
    };
  });
};
