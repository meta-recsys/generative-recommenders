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
#include "hstu_attention_fwd_strategy.hpp"
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

#include "hstu_attention_host_util.hpp"
#include "hstu_attention_jagged_forward_splitkv_dispatch.hpp"

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
struct jagged_forward_dispatch {
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
      false, // kUseGroup
      true, // kIsJagged
      kHasBias,
      kHasDropout,
      kUseCausal,
      kUseSoftmax,
      kStoreLSE,
      HstuAttentionTileSetting>;

  static void Run(HstuAttentionNoGroupFwdParams& param, hipStream_t stream) {
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
      HstuAttentionNoGroupFwdParams& param,
      hipStream_t stream) {
    const bool almost_invariant_seqlen = param.grid_order_override >= 0
        ? param.grid_order_override == 1
        : is_almost_invariant_seqlen(param);

    auto kargs = [&] {
      return HstuKernel::MakeKargs(
          param.q_ptr,
          param.k_ptr,
          param.v_ptr,
          param.bias_ptr,
          param.o_ptr,
          param.lse_ptr,
          param.seq_q_offsets_ptr,
          param.is_cross_attention ? param.seq_kv_offsets_ptr
                                   : param.seq_q_offsets_ptr,
          param.max_seqlen_q,
          param.hdim_qk,
          param.hdim_v,
          param.num_head,
          param.scale_s,
          param.attn_scale,
          param.attn_scale_ptr,
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
          param.contextual_seqlen,
          param.window_size,
          param.min_full_attn_seqlen,
          param.p_drop,
          param.philox_seed,
          param.philox_offset);
    }();
    kargs.kv_truncate_len = param.kv_truncate_len;
    kargs.kv_keep_last = param.kv_keep_last;

    bool has_minfull_attn_seqlen = (param.min_full_attn_seqlen > 0);
    dim3 kGridSize = HstuKernel::GridSize(
        param.num_batch,
        param.num_head,
        param.max_seqlen_q,
        param.hdim_v,
        almost_invariant_seqlen,
        has_minfull_attn_seqlen);
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
void run_jagged_forward_dispatch(
    HstuAttentionNoGroupFwdParams& param,
    hipStream_t stream) {
  const bool hdim_padded = (param.hdim_qk < MaxK || param.hdim_v < MaxK);

  BOOL_SWITCH(hdim_padded, kHdimPadded, [&] {
    auto run = [&](int mtile_size, bool use_splitkv) {
      if (use_splitkv) {
        if (mtile_size == 128)
          jagged_forward_splitkv_dispatch<
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
          jagged_forward_splitkv_dispatch<
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
          jagged_forward_dispatch<
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
          jagged_forward_dispatch<
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

    if (param.is_jagged) {
      const HstuFwdStrategy& strategy =
          hstu_fwd_strategy_for_arch(get_hstu_fwd_arch());
      const HstuFwdCallFeatures features{
          param.is_cross_attention,
          kUseCausal,
          param.window_size > 0,
          kUseSoftmax,
          static_cast<int>(sizeof(InOutDataType)),
          param.hdim_qk,
          param.hdim_v,
          param.num_head,
          param.num_batch,
          param.total_seqlen_q,
          param.total_seqlen_kv};
      HstuFwdConfig cfg;
      param.config_rule =
          select_hstu_fwd_config_on_device(strategy, features, cfg);
      if (param.config_rule >= 0) {
        param.config_source = strategy.name.c_str();
        param.num_splits_override = cfg.num_splits;
        param.grid_order_override = cfg.grid_order;
        run(cfg.mtile, cfg.num_splits > 0);
        return;
      }
    }

    int mtile_size = get_hstu_attention_fwd_mtile(
        param.num_batch,
        param.num_head,
        param.max_seqlen_q,
        param.max_seqlen_kv,
        param.window_size);

    if (!param.is_cross_attention && mtile_size == 128) {
      run(128, false);
      return;
    }

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

    run(mtile_size,
        !disable_fwd_splitkv &&
            shall_use_splitkv(
                param.num_batch,
                param.num_head,
                param.max_seqlen_q,
                param.max_seqlen_kv));
  });
};
