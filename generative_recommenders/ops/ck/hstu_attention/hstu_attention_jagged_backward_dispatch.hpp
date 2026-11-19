// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/host/hip_check_error.hpp>
#include <ck_tile/host/kernel_launch.hpp>
#include <ck_tile/host/stream_config.hpp>
#include <ck_tile/ops/epilogue.hpp>

#include "hstu_attention_bool_switch.hpp"
#include "hstu_attention_bwd_helper.hpp"
#include "hstu_attention_bwd_pipeline_problem.hpp"
#include "hstu_attention_bwd_pipeline_traits.hpp"
#include "hstu_attention_bwd_setting.hpp"
#include "hstu_attention_bwd_type_config.hpp"
#include "hstu_attention_config.hpp"
#include "hstu_attention_hdim_switch.hpp"
#include "hstu_attention_params.hpp"
// default pipelines
#include "hstu_attention_no_softmax_bwd_pipeline_dk_dv.hpp"
#include "hstu_attention_no_softmax_bwd_pipeline_dq.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline_dk_dv.hpp"
#include "hstu_attention_with_softmax_bwd_pipeline_dq_delta.hpp"
// trload pipelines
#include "hstu_attention_no_softmax_bwd_trload_pipeline_dk_dv.hpp"
#include "hstu_attention_no_softmax_bwd_trload_pipeline_dq.hpp"
#include "hstu_attention_with_softmax_bwd_trload_pipeline_dk_dv.hpp"
#include "hstu_attention_with_softmax_bwd_trload_pipeline_dq_delta.hpp"
// tdm_trload pipelines
#include "hstu_attention_no_softmax_bwd_tdm_trload_pipeline_dk_dv.hpp"
#include "hstu_attention_no_softmax_bwd_tdm_trload_pipeline_dq.hpp"
#include "hstu_attention_with_softmax_bwd_tdm_trload_pipeline_dk_dv.hpp"
#include "hstu_attention_with_softmax_bwd_tdm_trload_pipeline_dq_delta.hpp"
// direct_lds_trload pipelines
#include "hstu_attention_no_softmax_bwd_direct_lds_trload_pipeline_dk_dv.hpp"
#include "hstu_attention_no_softmax_bwd_direct_lds_trload_pipeline_dq.hpp"
#include "hstu_attention_with_softmax_bwd_direct_lds_trload_pipeline_dk_dv.hpp"
#include "hstu_attention_with_softmax_bwd_direct_lds_trload_pipeline_dq_delta.hpp"

#include "hstu_attention_bwd_kernel_1.hpp"
#include "hstu_attention_bwd_kernel_2.hpp"
#include "hstu_attention_epilogue.hpp"
#include "hstu_attention_host_util.hpp"

// Backward dispatch for jagged (variable-length, non-group) mode.
// Backward never uses split-kv: training always provides sufficient CU
// coverage.

template <
    typename InOutDataType,
    bool kUseCausal,
    bool kUseSoftmax,
    bool kHasBias,
    bool kHasDropout,
    bool kHdimPadded,
    ck_tile::index_t MaxK>
struct jagged_backward_dispatch {
  static constexpr HstuBwdPipelineKind kPipelineKind =
      get_hstu_bwd_pipeline_kind<MaxK, kHdimPadded>();

  using HstuAttentionTileSettingForKernel1 =
      decltype(GetHstuAttentionBwdKernel1TileSetting<
               MaxK,
               kUseSoftmax,
               kPipelineKind>());
  using HstuAttentionTileSettingForKernel2 =
      decltype(GetHstuAttentionBwdKernel2TileSetting<
               MaxK,
               kUseSoftmax,
               kPipelineKind>());

  template <bool kIsCrossAttention>
  using HstuBwdBaseProblemTemp = ck_tile::HstuAttentionBwdPipelineBaseProblem<
      InOutDataType,
      typename HstuAttentionBwdTypeConfig<InOutDataType>::GemmAccDataType,
      typename HstuAttentionBwdTypeConfig<InOutDataType>::CompDataType,
      kIsCrossAttention,
      false, // kUseGroup
      true, // kIsJagged
      kHasBias,
      kUseCausal,
      kUseSoftmax,
      kHasDropout>;

  template <bool kIsCrossAttention>
  using HstuBwdPipelineProblemForKernel1Temp =
      ck_tile::HstuAttentionBwdPipelineProblemForKernel1<
          HstuBwdBaseProblemTemp<kIsCrossAttention>,
          HstuAttentionTileSettingForKernel1>;

  template <bool kIsCrossAttention>
  using HstuBwdPipelineProblemForKernel2Temp =
      ck_tile::HstuAttentionBwdPipelineProblemForKernel2<
          HstuBwdBaseProblemTemp<kIsCrossAttention>,
          HstuAttentionTileSettingForKernel2>;

  static void Run(HstuAttentionNoGroupBwdParams& param, hipStream_t stream) {
    BwdWorkspace ws;

    constexpr ck_tile::index_t occupancy_k1 = -1;
    constexpr ck_tile::index_t occupancy_k2 = -1;

    const bool pad_headdim_qk =
        !(param.hdim_qk % HstuAttentionTileSettingForKernel1::kQKHeaddim == 0);
    const bool pad_headdim_v =
        !(param.hdim_v % HstuAttentionTileSettingForKernel1::kVHeaddim == 0);

    // seqlen_q is not along the fastest memory dim; OOB is handled by the
    // buffer instructions. In jagged mode seqlen_k always requires padding
    // since it varies per sample.
    constexpr bool kPadSeqLenQ = false;
    constexpr bool kPadSeqLenK = true;

    BOOL_SWITCH(param.is_cross_attention, kIsCrossAttention, [&] {
      using HstuPipelineProblemForKernel1 =
          HstuBwdPipelineProblemForKernel1Temp<kIsCrossAttention>;
      using HstuPipelineProblemForKernel2 =
          HstuBwdPipelineProblemForKernel2Temp<kIsCrossAttention>;

      if constexpr (kPipelineKind == HstuBwdPipelineKind::DirectLdsTrLoad) {
        using HstuTraits = ck_tile::HstuAttentionBwdTraits<
            kPadSeqLenQ,
            kPadSeqLenK,
            false /*kPadHeadDimQK*/,
            false /*kPadHeadDimV*/,
            occupancy_k1,
            occupancy_k2>;

        using HstuEpilogueQGrad =
            ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename HstuAttentionBwdTypeConfig<
                    InOutDataType>::GemmAccDataType,
                InOutDataType,
                kPadSeqLenQ,
                false /*kPadHeadDimQK*/>>;

        using HstuEpilogueKGrad =
            ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename HstuAttentionBwdTypeConfig<
                    InOutDataType>::GemmAccDataType,
                InOutDataType,
                kPadSeqLenK,
                false /*kPadHeadDimQK*/>>;

        using HstuEpilogueVGrad =
            ck_tile::NRepetitions2DEpilogue<ck_tile::Default2DEpilogueProblem<
                typename HstuAttentionBwdTypeConfig<
                    InOutDataType>::GemmAccDataType,
                InOutDataType,
                kPadSeqLenK,
                false /*kPadHeadDimV*/>>;

        // we only use async_load_tile() for K/V/Q/OGrad when its maximum
        // vector_size is
        //  dwordx4 (ToDO: use dwordx3 when hdim is 96)
        constexpr ck_tile::index_t k_vector_bytes =
            HstuPipelineProblemForKernel1::GetKDramTileAccessMaxVectorSize() *
            sizeof(InOutDataType);
        constexpr ck_tile::index_t v_vector_bytes =
            HstuPipelineProblemForKernel1::GetVDramTileAccessMaxVectorSize() *
            sizeof(InOutDataType);
        constexpr ck_tile::index_t q_vector_bytes =
            HstuPipelineProblemForKernel2::GetQDramTileAccessMaxVectorSize() *
            sizeof(InOutDataType);
        constexpr ck_tile::index_t do_vector_bytes =
            HstuPipelineProblemForKernel2::
                GetOGradDramTileAccessMaxVectorSize() *
            sizeof(InOutDataType);

        static_assert(k_vector_bytes == 4 * sizeof(int), "Check failed!");
        static_assert(v_vector_bytes == 4 * sizeof(int), "Check failed!");
        static_assert(q_vector_bytes == 4 * sizeof(int), "Check failed!");
        static_assert(do_vector_bytes == 4 * sizeof(int), "Check failed!");
        static_assert(MaxK != 96, "Check failed!");

        using HstuPipelineKernel1 = std::conditional_t<
            kUseSoftmax,
            ck_tile::
                HstuAttentionWithSoftmaxBwdDirectLdsTrLoadPipelineQRKSVS_dQ_D<
                    HstuPipelineProblemForKernel1,
                    HstuTraits>,
            ck_tile::HstuAttentionNoSoftmaxBwdDirectLdsTrLoadPipelineQRKSVS_dQ<
                HstuPipelineProblemForKernel1,
                HstuTraits>>;

        using HstuPipelineKernel2 = std::conditional_t<
            kUseSoftmax,
            ck_tile::
                HstuAttentionWithSoftmaxBwdDirectLdsTrLoadPipelineKRVRQS_dK_dV<
                    HstuPipelineProblemForKernel2,
                    HstuTraits>,
            ck_tile::
                HstuAttentionNoSoftmaxBwdDirectLdsTrLoadPipelineKRVRQS_dK_dV<
                    HstuPipelineProblemForKernel2,
                    HstuTraits>>;

        using HstuKernel1 = ck_tile::
            HstuAttentionBwdKernel1<HstuPipelineKernel1, HstuEpilogueQGrad>;
        using HstuKernel2 = ck_tile::HstuAttentionBwdKernel2<
            HstuPipelineKernel2,
            HstuEpilogueKGrad,
            HstuEpilogueVGrad>;

        RunWithKernels<HstuKernel1, HstuKernel2>(param, ws, stream);
      } else {
        BOOL_SWITCH_2(
            pad_headdim_qk, kPadHeadDimQK, pad_headdim_v, kPadHeadDimV, [&] {
              using HstuTraits = ck_tile::HstuAttentionBwdTraits<
                  kPadSeqLenQ,
                  kPadSeqLenK,
                  kPadHeadDimQK,
                  kPadHeadDimV,
                  occupancy_k1,
                  occupancy_k2>;

              using HstuEpilogueQGrad = ck_tile::NRepetitions2DEpilogue<
                  ck_tile::Default2DEpilogueProblem<
                      typename HstuAttentionBwdTypeConfig<
                          InOutDataType>::GemmAccDataType,
                      InOutDataType,
                      kPadSeqLenQ,
                      kPadHeadDimQK>>;

              using HstuEpilogueKGrad = ck_tile::NRepetitions2DEpilogue<
                  ck_tile::Default2DEpilogueProblem<
                      typename HstuAttentionBwdTypeConfig<
                          InOutDataType>::GemmAccDataType,
                      InOutDataType,
                      kPadSeqLenK,
                      kPadHeadDimQK>>;

              using HstuEpilogueVGrad = ck_tile::NRepetitions2DEpilogue<
                  ck_tile::Default2DEpilogueProblem<
                      typename HstuAttentionBwdTypeConfig<
                          InOutDataType>::GemmAccDataType,
                      InOutDataType,
                      kPadSeqLenK,
                      kPadHeadDimV>>;

              if constexpr (kPipelineKind == HstuBwdPipelineKind::Default) {
                using HstuPipelineKernel1 = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::HstuAttentionWithSoftmaxBwdPipelineQRKSVS_dQ_D<
                        HstuPipelineProblemForKernel1,
                        HstuTraits>,
                    ck_tile::HstuAttentionNoSoftmaxBwdPipelineQRKSVS_dQ<
                        HstuPipelineProblemForKernel1,
                        HstuTraits>>;

                using HstuPipelineKernel2 = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::HstuAttentionWithSoftmaxBwdPipelineKRVRQS_dK_dV<
                        HstuPipelineProblemForKernel2,
                        HstuTraits>,
                    ck_tile::HstuAttentionNoSoftmaxBwdPipelineKRVRQS_dK_dV<
                        HstuPipelineProblemForKernel2,
                        HstuTraits>>;

                using HstuKernel1 = ck_tile::HstuAttentionBwdKernel1<
                    HstuPipelineKernel1,
                    HstuEpilogueQGrad>;
                using HstuKernel2 = ck_tile::HstuAttentionBwdKernel2<
                    HstuPipelineKernel2,
                    HstuEpilogueKGrad,
                    HstuEpilogueVGrad>;

                RunWithKernels<HstuKernel1, HstuKernel2>(param, ws, stream);
              } else if constexpr (
                  kPipelineKind == HstuBwdPipelineKind::TdmTrLoad) {
                using HstuPipelineKernel1 = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::
                        HstuAttentionWithSoftmaxBwdTdmTrLoadPipelineQRKSVS_dQ_D<
                            HstuPipelineProblemForKernel1,
                            HstuTraits>,
                    ck_tile::
                        HstuAttentionNoSoftmaxBwdTdmTrLoadPipelineQRKSVS_dQ<
                            HstuPipelineProblemForKernel1,
                            HstuTraits>>;

                using HstuPipelineKernel2 = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::
                        HstuAttentionWithSoftmaxBwdTdmTrLoadPipelineKRVRQS_dK_dV<
                            HstuPipelineProblemForKernel2,
                            HstuTraits>,
                    ck_tile::
                        HstuAttentionNoSoftmaxBwdTdmTrLoadPipelineKRVRQS_dK_dV<
                            HstuPipelineProblemForKernel2,
                            HstuTraits>>;

                using HstuKernel1 = ck_tile::HstuAttentionBwdKernel1<
                    HstuPipelineKernel1,
                    HstuEpilogueQGrad>;
                using HstuKernel2 = ck_tile::HstuAttentionBwdKernel2<
                    HstuPipelineKernel2,
                    HstuEpilogueKGrad,
                    HstuEpilogueVGrad>;

                RunWithKernels<HstuKernel1, HstuKernel2>(param, ws, stream);
              } else {
                using HstuPipelineKernel1 = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::
                        HstuAttentionWithSoftmaxBwdTrLoadPipelineQRKSVS_dQ_D<
                            HstuPipelineProblemForKernel1,
                            HstuTraits>,
                    ck_tile::HstuAttentionNoSoftmaxBwdTrLoadPipelineQRKSVS_dQ<
                        HstuPipelineProblemForKernel1,
                        HstuTraits>>;

                using HstuPipelineKernel2 = std::conditional_t<
                    kUseSoftmax,
                    ck_tile::
                        HstuAttentionWithSoftmaxBwdTrLoadPipelineKRVRQS_dK_dV<
                            HstuPipelineProblemForKernel2,
                            HstuTraits>,
                    ck_tile::
                        HstuAttentionNoSoftmaxBwdTrLoadPipelineKRVRQS_dK_dV<
                            HstuPipelineProblemForKernel2,
                            HstuTraits>>;

                using HstuKernel1 = ck_tile::HstuAttentionBwdKernel1<
                    HstuPipelineKernel1,
                    HstuEpilogueQGrad>;
                using HstuKernel2 = ck_tile::HstuAttentionBwdKernel2<
                    HstuPipelineKernel2,
                    HstuEpilogueKGrad,
                    HstuEpilogueVGrad>;

                RunWithKernels<HstuKernel1, HstuKernel2>(param, ws, stream);
              }
            });
      }
    });
  }

  template <typename HstuKernel1, typename HstuKernel2>
  static void RunWithKernels(
      HstuAttentionNoGroupBwdParams& param,
      BwdWorkspace& ws,
      hipStream_t stream) {
    // Delta layout in jagged mode: flat over the sequence dimension, strided by
    // head. seq_stride=1, nhead_stride=num_batch*max_seqlen_q -- no batch
    // stride needed.
    const ck_tile::index_t seq_stride_delta = 1;
    const ck_tile::index_t nhead_stride_delta =
        param.num_batch * param.max_seqlen_q;

    if constexpr (kUseSoftmax) {
      const size_t delta_bytes = static_cast<size_t>(param.num_batch) *
          static_cast<size_t>(param.num_head) *
          static_cast<size_t>(param.max_seqlen_q) * sizeof(float);
      HIP_CHECK_ERROR(hipMallocAsync(&ws.delta_ptr, delta_bytes, stream));
    }

    bool almost_invariant_seqlen = is_almost_invariant_seqlen(param);

    // ---- Kernel 1: compute dQ (and delta = D[sq] for softmax path) ----
    {
      const auto kargs = [&] {
        return HstuKernel1::MakeKargs(
            param.q_ptr,
            param.k_ptr,
            param.v_ptr,
            param.bias_ptr,
            param.o_ptr,
            param.do_ptr,
            param.dq_ptr,
            param.lse_ptr,
            ws.delta_ptr,
            param.seq_q_offsets_ptr,
            param.is_cross_attention ? param.seq_kv_offsets_ptr
                                     : param.seq_q_offsets_ptr,
            param.max_seqlen_q,
            param.hdim_qk,
            param.hdim_v,
            param.num_head,
            param.scale_s,
            param.attn_scale,
            almost_invariant_seqlen,
            param.seq_stride_q,
            param.seq_stride_k,
            param.seq_stride_v,
            param.seq_stride_o,
            param.seq_stride_do,
            param.seq_stride_dq,
            param.seq_stride_lse,
            seq_stride_delta,
            param.seq_stride_bias,
            param.nhead_stride_q,
            param.nhead_stride_k,
            param.nhead_stride_v,
            param.nhead_stride_o,
            param.nhead_stride_do,
            param.nhead_stride_dq,
            param.nhead_stride_lse,
            nhead_stride_delta,
            param.nhead_stride_bias,
            param.num_targets_ptr,
            param.contextual_seqlen,
            param.window_size,
            param.min_full_attn_seqlen,
            param.p_drop,
            param.philox_seed,
            param.philox_offset);
      }();

      bool has_minfull_attn_seqlen = (param.min_full_attn_seqlen > 0);
      dim3 kGridSize = HstuKernel1::GridSize(
          param.num_batch,
          param.num_head,
          param.max_seqlen_q,
          almost_invariant_seqlen,
          has_minfull_attn_seqlen);
      dim3 kBlockSize = HstuKernel1::BlockSize();
      constexpr ck_tile::index_t kBlockPerCu = HstuKernel1::kBlockPerCu;

      (void)ck_tile::launch_kernel(
          ck_tile::stream_config{stream, false},
          ck_tile::make_kernel<kBlockPerCu>(
              HstuKernel1{}, kGridSize, kBlockSize, 0, kargs));
    }

    // ---- Kernel 2: compute dK and dV ----
    {
      const auto kargs = [&] {
        return HstuKernel2::MakeKargs(
            param.q_ptr,
            param.k_ptr,
            param.v_ptr,
            param.bias_ptr,
            param.do_ptr,
            param.dk_ptr,
            param.dv_ptr,
            param.lse_ptr,
            ws.delta_ptr,
            param.seq_q_offsets_ptr,
            param.is_cross_attention ? param.seq_kv_offsets_ptr
                                     : param.seq_q_offsets_ptr,
            param.max_seqlen_q,
            param.hdim_qk,
            param.hdim_v,
            param.num_head,
            param.scale_s,
            param.attn_scale,
            almost_invariant_seqlen,
            param.seq_stride_q,
            param.seq_stride_k,
            param.seq_stride_v,
            param.seq_stride_do,
            param.seq_stride_dk,
            param.seq_stride_dv,
            param.seq_stride_lse,
            seq_stride_delta,
            param.seq_stride_bias,
            param.nhead_stride_q,
            param.nhead_stride_k,
            param.nhead_stride_v,
            param.nhead_stride_do,
            param.nhead_stride_dk,
            param.nhead_stride_dv,
            param.nhead_stride_lse,
            nhead_stride_delta,
            param.nhead_stride_bias,
            param.num_targets_ptr,
            param.contextual_seqlen,
            param.window_size,
            param.min_full_attn_seqlen,
            param.p_drop,
            param.philox_seed,
            param.philox_offset);
      }();

      dim3 kGridSize = HstuKernel2::GridSize(
          param.num_batch,
          param.num_head,
          param.max_seqlen_kv,
          almost_invariant_seqlen);
      dim3 kBlockSize = HstuKernel2::BlockSize();
      constexpr ck_tile::index_t kBlockPerCu = HstuKernel2::kBlockPerCu;

      (void)ck_tile::launch_kernel(
          ck_tile::stream_config{stream, false},
          ck_tile::make_kernel<kBlockPerCu>(
              HstuKernel2{}, kGridSize, kBlockSize, 0, kargs));
    }

    if constexpr (kUseSoftmax) {
      HIP_CHECK_ERROR(hipFreeAsync(ws.delta_ptr, stream));
    }
  }
};

template <
    typename InOutDataType,
    bool kUseCausal,
    bool kUseSoftmax,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_jagged_backward_dispatch(
    HstuAttentionNoGroupBwdParams& param,
    hipStream_t stream) {
  const bool hdim_padded = (param.hdim_qk < MaxK || param.hdim_v < MaxK);

  BOOL_SWITCH(hdim_padded, kHdimPadded, [&] {
    jagged_backward_dispatch<
        InOutDataType,
        kUseCausal,
        kUseSoftmax,
        kHasBias,
        kHasDropout,
        kHdimPadded,
        MaxK>::Run(param, stream);
  });
}
