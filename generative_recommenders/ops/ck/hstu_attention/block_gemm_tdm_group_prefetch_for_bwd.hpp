// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "block_gemm_areg_bsmem_creg_full_gemmk_slice_gemmn.hpp"
#include "block_gemm_areg_bsmem_trload_creg_slice_gemmk_full_gemmn.hpp"
#include "block_gemm_asmem_breg_creg_full_gemmk_slice_gemmm.hpp"
#include "hstu_attention_pipeline_policy_helper.hpp"

namespace ck_tile {

// Tdm-only P@V block gemm (gfx1250). Same A/B/C distributions as
// BlockGemmARegBSmemTrLoadCRegSliceGemmKFullGemmN (it derives from it), but the
// V ds_load_tr reads run ahead of the WMMAs in groups: the (kIter, nIter) steps
// are flattened to steps 0 .. kSteps - 1 and split into groups of
// kStepsPerGroup steps. The B tiles of steps [0, kPrefetchDepth) are loaded up
// front; group g then first issues, back to back, the loads of steps [g *
// kStepsPerGroup + kPrefetchDepth, (g + 1) * kStepsPerGroup + kPrefetchDepth)
// (clipped to kSteps) and then runs the WMMAs of its own steps [g *
// kStepsPerGroup, (g + 1) * kStepsPerGroup). So the B tile of step s is loaded
// by the group that runs step s - kPrefetchDepth (or up front), kPrefetchDepth
// steps before its own WMMAs. In ATT traces taken on gfx1250 of the Tdm kernel
// with the base gemms, which switch from WMMA to DS once per step, a DS load
// issued right after a WMMA waited about 30 cycles (a measured value, not a
// documented latency); this gemm switches once per group instead. The
// sched_group_barriers pin that order and the sched_barrier(0) closing each
// group keeps later loads from being hoisted into it. That bounds the DS data
// in flight to kPrefetchDepth + kStepsPerGroup steps only while the WMMAs stay
// in their group: the barriers act within a basic block, and a cross-attention
// M128 build (ROCm 7.2) was seen to sink the Q@K WMMAs past the mask branch,
// behind all of the loads.
template <
    typename Problem_,
    typename Policy_,
    index_t kPrefetchDepth_,
    index_t kStepsPerGroup_>
struct
    BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetchSliceGemmKFullGemmNWithPrologueHook
    : public BlockGemmARegBSmemTrLoadCRegSliceGemmKFullGemmN<
          Problem_,
          Policy_> {
  using Base =
      BlockGemmARegBSmemTrLoadCRegSliceGemmKFullGemmN<Problem_, Policy_>;
  using Problem = remove_cvref_t<Problem_>;
  using Policy = remove_cvref_t<Policy_>;
  using ADataType = remove_cvref_t<typename Problem::ADataType>;
  using BDataType = remove_cvref_t<typename Problem::BDataType>;
  using CDataType = remove_cvref_t<typename Problem::CDataType>;
  using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

  using Base::MakeABlockTileDistribution;
  using Base::MakeCBlockTile;

  static constexpr index_t kBlockSize = Problem::kBlockSize;
  static constexpr index_t kPrefetchDepth = kPrefetchDepth_;
  static constexpr index_t kStepsPerGroup = kStepsPerGroup_;

  // The run_ahead() of the base reads ahead with its own depth and would
  // silently ignore kPrefetchDepth_ and kStepsPerGroup_, so it is hidden here.
  template <index_t kPrefetchRequested, typename... Args>
  CK_TILE_DEVICE void run_ahead(Args&&...) const = delete;

  // C += A * B
  template <
      typename CBlockTensor,
      typename ABlockTensorTmp,
      typename BBlockWindowTmp,
      typename PrologueHook>
  CK_TILE_DEVICE void operator()(
      CBlockTensor& c_block_tensor,
      const ABlockTensorTmp& a_block_tensor_tmp,
      const BBlockWindowTmp& b_block_window_tmp,
      const PrologueHook& prologue_hook) const {
    static_assert(
        std::is_same_v<
            ADataType,
            remove_cv_t<typename ABlockTensorTmp::DataType>> &&
            std::is_same_v<
                BDataType,
                remove_cv_t<typename BBlockWindowTmp::DataType>> &&
            std::is_same_v<
                CDataType,
                remove_cv_t<typename CBlockTensor::DataType>>,
        "wrong!");

    constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
    constexpr index_t NPerBlock =
        BBlockWindowTmp{}.get_window_lengths()[number<1>{}];
    constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

    static_assert(
        MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
            KPerBlock == BlockGemmShape::kK,
        "wrong!");

    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
    constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
    constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

    constexpr index_t kSteps = KIterPerWarp * NIterPerWarp;

    static_assert(
        1 <= kPrefetchDepth_ && kPrefetchDepth_ <= kSteps,
        "kPrefetchDepth_ must be in [1, kSteps]");
    static_assert(
        1 <= kStepsPerGroup_ && kStepsPerGroup_ <= kSteps,
        "kStepsPerGroup_ must be in [1, kSteps]");
    static_assert(
        kSteps % kStepsPerGroup_ == 0, "kStepsPerGroup_ must divide kSteps");

    const index_t iNWarp = get_warp_id<false>() % NWarp;

    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

    auto a_block_tensor =
        make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

    a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

    constexpr auto b_warp_dstr_encode = typename InputTileDistributionTraits<
        typename WG::BWarpDstrEncoding,
        BDataType>::TransposedDstrEncode{};

    auto b_warp_window_tmp = make_tile_window(
        b_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<WG::kK>{}, number<WG::kN>{}),
        b_block_window_tmp.get_window_origin() +
            multi_index<2>{0, iNWarp * WG::kN},
        make_static_tile_distribution(b_warp_dstr_encode));

    static_assert(
        std::is_same_v<
            remove_cvref_t<decltype(c_block_dstr_encode)>,
            remove_cvref_t<
                decltype(CBlockTensor::get_tile_distribution()
                             .get_static_tile_distribution_encoding())>>,
        "wrong!");

    using AWarpDstr = typename WG::AWarpDstr;
    using CWarpDstr = typename WG::CWarpDstr;

    using AWarpTensor = typename WG::AWarpTensor;
    using CWarpTensor = typename WG::CWarpTensor;

    constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    constexpr auto a_warp_y_index_zeros =
        uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros =
        uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    using b_warp_tensor_type = decltype(load_tile_transpose(b_warp_window_tmp));

    // number of ds_load_tr instructions per B warp tile, one per vector access
    // of the window
    constexpr index_t kDsPerStep = b_warp_window_tmp.get_num_of_access();

    statically_indexed_array<b_warp_tensor_type, kSteps> b_warp_tensors;

    // Why the step offset can be folded in as a separate compile-time constant
    // (with trload read)
    // --------------------------------------------------------------------------
    // Every step reads its warp tile through the step-0 warp window and adds
    //
    //     kStepOffsetB = offset(R, C),   R = kIter * KPerBlockPerIter,
    //                                    C = nIter * NPerBlockPerIter
    //
    // to the step-0 lane address (folded into the ds_load immediate). This is
    // exact iff, for every element (r, c) a lane reads from the step-0 tile
    // (e.g. r = tid / 16 + tid % 8, c = (tid / 8 % 2) * 8),
    //
    //     offset(R + r, C + c) == offset(r, c) + offset(R, C) (*)
    //
    // with row_stride = pad_interval + pad_length:
    //
    // 1) hdim128 / hdim256: pad_interval = hdim, one pad per row
    //
    //        offset(row, col) = row * row_stride + col
    //
    //    linear in (row, col), so
    //
    //        offset(R + r, C + c) = (R + r) * row_stride + C + c
    //                             = (r * row_stride + c) + (R * row_stride + C)
    //                             = offset(r, c) + offset(R, C)
    //
    //    and (*) holds for any (R, C).
    //
    // 2) hdim64: pad_interval = 128 fp16/bf16 elements, one pad per row pair
    //
    //        offset(row, col) = (row / 2) * row_stride + (row % 2) * 64 + col
    //
    //    R is a multiple of WG::kK, hence even: (R + r) / 2 = R / 2 + r / 2 and
    //    (R + r) % 2 = r % 2, with R % 2 = 0. So
    //
    //        offset(R + r, C + c) = (R / 2 + r / 2) * row_stride + (r % 2) * 64
    //        + C + c
    //                             = ((r / 2) * row_stride + (r % 2) * 64 + c)
    //                               + ((R / 2) * row_stride + (R % 2) * 64 + C)
    //                             = offset(r, c) + offset(R, C)
    //
    //    and (*) holds.
    //
    // 3) hdim96: pad_interval = 128 elements as well
    //
    //    Identical to hdim128: 96 is padded to 128 (kN1), and Tdm fills the
    //    out-of-bound values with zeros when transferring from Dram to Lds.

    auto load_step = [&](auto i_step) {
      constexpr index_t kIter = i_step / NIterPerWarp;
      constexpr index_t nIter = i_step % NIterPerWarp;

      constexpr auto b_lds_desc =
          typename BBlockWindowTmp::BottomTensorView::TensorDesc{};
      constexpr index_t kStepOffsetB = b_lds_desc.calculate_offset(make_tuple(
          number<kIter * KPerBlockPerIter>{},
          number<nIter * NPerBlockPerIter>{}));

      load_tile_transpose_with_offset(
          b_warp_tensors(i_step),
          b_warp_window_tmp,
          kStepOffsetB,
          number<-1>{},
          bool_constant<true>{},
          bool_constant<true>{});
    };

    // prologue: the first kPrefetchDepth steps, back to back
    static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

    __builtin_amdgcn_sched_barrier(0);
    prologue_hook();
    __builtin_amdgcn_sched_barrier(0);

    static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
      constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
      constexpr index_t kNumLoads = kFirstLoad >= kSteps
          ? 0
          : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad
                                                  : kStepsPerGroup);

      // the loads kPrefetchDepth steps ahead first, back to back ...
      static_for<0, kNumLoads, 1>{}(
          [&](auto i_load) { load_step(number<kFirstLoad + i_load>{}); });

      // ... then the WMMAs of this group
      static_for<0, kStepsPerGroup, 1>{}([&](auto i_step_in_group) {
        constexpr index_t i_step = i_group * kStepsPerGroup + i_step_in_group;
        constexpr auto kIter = number<i_step / NIterPerWarp>{};
        constexpr auto nIter = number<i_step % NIterPerWarp>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
          AWarpTensor a_warp_tensor;

          a_warp_tensor.get_thread_buffer() =
              a_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

          CWarpTensor c_warp_tensor;

          c_warp_tensor.get_thread_buffer() =
              c_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

          WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensors[number<i_step>{}]);

          c_block_tensor.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
              c_warp_tensor.get_thread_buffer());
        });
      });

      // keep the order: the DS loads of the group, then its WMMAs
      if constexpr (kNumLoads > 0)
        __builtin_amdgcn_sched_group_barrier(0x100, kNumLoads * kDsPerStep, 0);
      __builtin_amdgcn_sched_group_barrier(
          0x008, kStepsPerGroup * MIterPerWarp, 0);

      __builtin_amdgcn_sched_barrier(0);
    });
  }

  // C = A * B
  template <
      typename ABlockTensorTmp,
      typename BBlockWindowTmp,
      typename PrologueHook>
    requires std::is_invocable_v<const PrologueHook&>
  CK_TILE_DEVICE auto operator()(
      const ABlockTensorTmp& a_block_tensor_tmp,
      const BBlockWindowTmp& b_block_window_tmp,
      const PrologueHook& prologue_hook) const {
    auto c_block_tensor = MakeCBlockTile();
    operator()(
        c_block_tensor, a_block_tensor_tmp, b_block_window_tmp, prologue_hook);
    return c_block_tensor;
  }
};

// Tdm-only Q@K block gemm (gfx1250), the K-side counterpart of the P@V gemm
// above: same distributions as BlockGemmARegBSmemCRegFullGemmKSliceGemmN (it
// derives from it) and the same step loop: group g first loads the K tiles of
// steps [g * kStepsPerGroup + kPrefetchDepth, (g + 1) * kStepsPerGroup +
// kPrefetchDepth) (clipped to kSteps), i.e. of its own steps shifted by
// kPrefetchDepth, and then runs the WMMAs of its own steps (same limits as the
// P@V gemm above).
template <
    typename Problem_,
    typename Policy_,
    index_t kPrefetchDepth_,
    index_t kStepsPerGroup_>
struct BlockGemmARegBSmemCRegTdmGroupPrefetchFullGemmKSliceGemmNWithPrologueHook
    : public BlockGemmARegBSmemCRegFullGemmKSliceGemmN<Problem_, Policy_> {
  using Base = BlockGemmARegBSmemCRegFullGemmKSliceGemmN<Problem_, Policy_>;
  using Problem = remove_cvref_t<Problem_>;
  using Policy = remove_cvref_t<Policy_>;
  using ADataType = remove_cvref_t<typename Problem::ADataType>;
  using BDataType = remove_cvref_t<typename Problem::BDataType>;
  using CDataType = remove_cvref_t<typename Problem::CDataType>;
  using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

  using Base::MakeABlockTileDistribution;
  using Base::MakeCBlockTile;

  static constexpr index_t kBlockSize = Problem::kBlockSize;
  static constexpr index_t kPrefetchDepth = kPrefetchDepth_;
  static constexpr index_t kStepsPerGroup = kStepsPerGroup_;

  // The run_ahead() of the base reads ahead with its own depth and would
  // silently ignore kPrefetchDepth_ and kStepsPerGroup_, so it is hidden here.
  template <index_t kPrefetchRequested, typename... Args>
  CK_TILE_DEVICE void run_ahead(Args&&...) const = delete;

  // C = A * B, written into c_block_tensor (the first k step overwrites C)
  template <
      typename CBlockTensor,
      typename ABlockTensorTmp,
      typename BBlockWindowTmp,
      typename PrologueHook>
  CK_TILE_DEVICE void operator()(
      CBlockTensor& c_block_tensor,
      const ABlockTensorTmp& a_block_tensor_tmp,
      const BBlockWindowTmp& b_block_window_tmp,
      const PrologueHook& prologue_hook) const {
    static_assert(
        std::is_same_v<
            ADataType,
            remove_cv_t<typename ABlockTensorTmp::DataType>> &&
            std::is_same_v<
                BDataType,
                remove_cv_t<typename BBlockWindowTmp::DataType>> &&
            std::is_same_v<
                CDataType,
                remove_cv_t<typename CBlockTensor::DataType>>,
        "wrong!");

    constexpr index_t MPerBlock = ABlockTensorTmp{}.get_lengths()[number<0>{}];
    constexpr index_t NPerBlock =
        BBlockWindowTmp{}.get_window_lengths()[number<0>{}];
    constexpr index_t KPerBlock = ABlockTensorTmp{}.get_lengths()[number<1>{}];

    static_assert(
        MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
            KPerBlock == BlockGemmShape::kK,
        "wrong!");

    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
    constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
    constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

    constexpr index_t kSteps = KIterPerWarp * NIterPerWarp;

    static_assert(
        1 <= kPrefetchDepth_ && kPrefetchDepth_ <= kSteps,
        "kPrefetchDepth_ must be in [1, kSteps]");
    static_assert(
        1 <= kStepsPerGroup_ && kStepsPerGroup_ <= kSteps,
        "kStepsPerGroup_ must be in [1, kSteps]");
    static_assert(
        kSteps % kStepsPerGroup_ == 0, "kStepsPerGroup_ must divide kSteps");

    const index_t iNWarp = get_warp_id<false>() % NWarp;

    static_assert(NWarp == 1, "Check failed!");

    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<NWarp>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp>>,
        tuple<sequence<1, 0>>,
        tuple<sequence<1, 0>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

    auto a_block_tensor =
        make_static_distributed_tensor<typename ABlockTensorTmp::DataType>(
            MakeABlockTileDistribution());

    a_block_tensor.get_thread_buffer() = a_block_tensor_tmp.get_thread_buffer();

    auto b_warp_window_tmp = make_tile_window(
        b_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<WG::kN>{}, number<WG::kK>{}),
        b_block_window_tmp.get_window_origin() +
            multi_index<2>{iNWarp * WG::kN, 0},
        make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

    static_assert(
        std::is_same_v<
            remove_cvref_t<decltype(c_block_dstr_encode)>,
            remove_cvref_t<
                decltype(CBlockTensor::get_tile_distribution()
                             .get_static_tile_distribution_encoding())>>,
        "wrong!");

    using AWarpDstr = typename WG::AWarpDstr;
    using CWarpDstr = typename WG::CWarpDstr;

    using AWarpTensor = typename WG::AWarpTensor;
    using CWarpTensor = typename WG::CWarpTensor;

    constexpr auto a_warp_y_lengths =
        to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    constexpr auto a_warp_y_index_zeros =
        uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros =
        uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    using b_warp_tensor_type = decltype(load_tile(b_warp_window_tmp));

    // number of ds_load instructions per B warp tile, one per vector access of
    // the window
    constexpr index_t kDsPerStep = b_warp_window_tmp.get_num_of_access();

    statically_indexed_array<b_warp_tensor_type, kSteps> b_warp_tensors;

    // Why the step offset can be folded in as a separate compile-time constant
    // (with normal read)
    // --------------------------------------------------------------------------
    // Every step reads its warp tile through the step-0 warp window and adds
    //
    //     kStepOffsetB = offset(R, C),   R = nIter * NPerBlockPerIter,
    //                                    C = kIter * KPerBlockPerIter
    //
    // to the step-0 lane address (folded into the ds_load immediate). This is
    // exact iff, for every element (r, c) a lane reads from the step-0 tile (r
    // < WG::kN, c < WG::kK, e.g. r = lane % 16, c = lane / 16 * 8 + ...),
    //
    //     offset(R + r, C + c) == offset(r, c) + offset(R, C) (*)
    //
    // The Tdm Lds layout of K is row-major (Cols = hdim) with pad_length
    // elements inserted after every pad_interval = P elements (see
    // MakeRowMajorLdsPaddedBlockDescriptor()):
    //
    //     offset(row, col) = f(row * hdim + col),   f(L) = (L / P) * stride + L
    //     % P, stride = P + pad_length
    //
    // 1) hdim128 / hdim256: P = hdim, one pad per row
    //
    //        offset(row, col) = row * stride + col
    //
    //    linear in (row, col), so (*) holds for any (R, C).
    //
    // 2) hdim64: P = 128 (the 256 B bank span), one pad per row pair
    //
    //        offset(row, col) = (row / 2) * stride + (row % 2) * 64 + col
    //
    //    R is a multiple of WG::kN, hence even: (R + r) / 2 = R / 2 + r / 2 and
    //    (R + r) % 2 = r % 2. The col term is linear, so (*) holds.
    //
    // 3) hdim96: P = 32 (the lowest set bit of the 192 B row), three pads per
    // row
    //
    //        offset(row, col) = (3 * row + col / 32) * stride + col % 32
    //
    //    C is a multiple of WG::kK = 32: (C + c) / 32 = C / 32 + c / 32 and
    //    (C + c) % 32 = c % 32. The row term is linear, so (*) holds.

    auto load_step = [&](auto i_step) {
      constexpr index_t kIter = i_step / NIterPerWarp;
      constexpr index_t nIter = i_step % NIterPerWarp;

      constexpr auto b_lds_desc =
          typename BBlockWindowTmp::BottomTensorView::TensorDesc{};
      constexpr index_t kStepOffsetB = b_lds_desc.calculate_offset(make_tuple(
          number<nIter * NPerBlockPerIter>{},
          number<kIter * KPerBlockPerIter>{}));

      b_warp_tensors(i_step) =
          load_tile_with_offset(b_warp_window_tmp, kStepOffsetB);
    };

    static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

    __builtin_amdgcn_sched_barrier(0);
    prologue_hook();
    __builtin_amdgcn_sched_barrier(0);

    static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
      constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
      constexpr index_t kNumLoads = kFirstLoad >= kSteps
          ? 0
          : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad
                                                  : kStepsPerGroup);

      static_for<0, kNumLoads, 1>{}(
          [&](auto i_load) { load_step(number<kFirstLoad + i_load>{}); });

      static_for<0, kStepsPerGroup, 1>{}([&](auto i_step_in_group) {
        constexpr index_t i_step = i_group * kStepsPerGroup + i_step_in_group;
        constexpr auto kIter = number<i_step / NIterPerWarp>{};
        constexpr auto nIter = number<i_step % NIterPerWarp>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
          AWarpTensor a_warp_tensor;

          a_warp_tensor.get_thread_buffer() =
              a_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

          CWarpTensor c_warp_tensor;

          if constexpr (kIter == 0) {
            c_warp_tensor =
                WG{}(a_warp_tensor, b_warp_tensors[number<i_step>{}]);
          } else {
            c_warp_tensor.get_thread_buffer() =
                c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            WG{}(
                c_warp_tensor, a_warp_tensor, b_warp_tensors[number<i_step>{}]);
          }

          c_block_tensor.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
              c_warp_tensor.get_thread_buffer());
        });
      });

      if constexpr (kNumLoads > 0)
        __builtin_amdgcn_sched_group_barrier(0x100, kNumLoads * kDsPerStep, 0);
      __builtin_amdgcn_sched_group_barrier(
          0x008, kStepsPerGroup * MIterPerWarp, 0);

      if constexpr (i_group < kSteps / kStepsPerGroup - 1)
        __builtin_amdgcn_sched_barrier(0);
    });
  }

  // C = A * B
  template <
      typename ABlockTensorTmp,
      typename BBlockWindowTmp,
      typename PrologueHook>
    requires std::is_invocable_v<const PrologueHook&>
  CK_TILE_DEVICE auto operator()(
      const ABlockTensorTmp& a_block_tensor_tmp,
      const BBlockWindowTmp& b_block_window_tmp,
      const PrologueHook& prologue_hook) const {
    auto c_block_tensor = MakeCBlockTile();
    operator()(
        c_block_tensor, a_block_tensor_tmp, b_block_window_tmp, prologue_hook);
    return c_block_tensor;
  }
};

// Tdm-only block gemm with A in Lds and B in registers (gfx1250), used by the
// bwd kernel-2 gemms (e.g. S = Q @ K, dP = dO @ V). Same distributions as
// BlockGemmASmemBRegCRegFullGemmKSliceGemmM (it derives from it). The base
// loops over mIter, then kIter, loads one A warp tile from Lds per (mIter,
// kIter) and runs the NIterPerWarp WMMAs of that A tile against the B tile in
// registers. Here those (mIter, kIter) pairs are flattened to steps 0 .. kSteps
// - 1 (i_step = mIter * KIterPerWarp + kIter, the base's order) and run in
// groups as in BlockGemmARegBSmemTrLoadCRegTdmGroupPrefetchSliceGemmKFullGemmN:
// the A tiles of steps [0, kPrefetchDepth) are loaded up front, and group g
// first issues the loads of steps [g * kStepsPerGroup + kPrefetchDepth, (g + 1)
// * kStepsPerGroup + kPrefetchDepth) (clipped to kSteps) back to back, then
// runs the kStepsPerGroup * NIterPerWarp WMMAs of its own steps (same limits as
// the P@V gemm above).
template <
    typename Problem_,
    typename Policy_,
    index_t kPrefetchDepth_,
    index_t kStepsPerGroup_>
struct BlockGemmASmemBRegCRegTdmGroupPrefetchFullGemmKSliceGemmMWithPrologueHook
    : public BlockGemmASmemBRegCRegFullGemmKSliceGemmM<Problem_, Policy_> {
  using Base = BlockGemmASmemBRegCRegFullGemmKSliceGemmM<Problem_, Policy_>;
  using Problem = remove_cvref_t<Problem_>;
  using Policy = remove_cvref_t<Policy_>;
  using ADataType = remove_cvref_t<typename Problem::ADataType>;
  using BDataType = remove_cvref_t<typename Problem::BDataType>;
  using CDataType = remove_cvref_t<typename Problem::CDataType>;
  using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

  using Base::MakeBBlockTileDistribution;
  using Base::MakeCBlockTile;

  static constexpr index_t kBlockSize = Problem::kBlockSize;
  static constexpr index_t kPrefetchDepth = kPrefetchDepth_;
  static constexpr index_t kStepsPerGroup = kStepsPerGroup_;

  // C = A * B, written into c_block_tensor (the first k step overwrites C, as
  // in the base)
  template <
      typename CBlockTensor,
      typename ABlockWindowTmp,
      typename BBlockTensorTmp,
      typename PrologueHook>
  CK_TILE_DEVICE void operator()(
      CBlockTensor& c_block_tensor,
      const ABlockWindowTmp& a_block_window_tmp,
      const BBlockTensorTmp& b_block_tensor_tmp,
      const PrologueHook& prologue_hook) const {
    static_assert(
        std::is_same_v<
            ADataType,
            remove_cv_t<typename ABlockWindowTmp::DataType>> &&
            std::is_same_v<
                BDataType,
                remove_cv_t<typename BBlockTensorTmp::DataType>> &&
            std::is_same_v<
                CDataType,
                remove_cv_t<typename CBlockTensor::DataType>>,
        "wrong!");

    constexpr index_t MPerBlock =
        ABlockWindowTmp{}.get_window_lengths()[number<0>{}];
    constexpr index_t NPerBlock = BBlockTensorTmp{}.get_lengths()[number<0>{}];
    constexpr index_t KPerBlock =
        ABlockWindowTmp{}.get_window_lengths()[number<1>{}];

    static_assert(
        MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
            KPerBlock == BlockGemmShape::kK,
        "wrong!");

    constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    constexpr index_t MWarp = config.template at<1>();
    constexpr index_t NWarp = config.template at<2>();

    constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
    constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
    constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

    constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
    constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

    constexpr index_t kSteps = MIterPerWarp * KIterPerWarp;

    static_assert(
        1 <= kPrefetchDepth_ && kPrefetchDepth_ <= kSteps,
        "kPrefetchDepth_ must be in [1, kSteps]");
    static_assert(
        1 <= kStepsPerGroup_ && kStepsPerGroup_ <= kSteps,
        "kStepsPerGroup_ must be in [1, kSteps]");
    static_assert(
        kSteps % kStepsPerGroup_ == 0, "kStepsPerGroup_ must divide kSteps");

    static_assert(MWarp == 1, "Check failed!");

    constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2>,
        sequence<0, 0>>{};

    constexpr auto c_block_dstr_encode =
        detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

    auto b_block_tensor =
        make_static_distributed_tensor<typename BBlockTensorTmp::DataType>(
            MakeBBlockTileDistribution());

    b_block_tensor.get_thread_buffer() = b_block_tensor_tmp.get_thread_buffer();

    // MWarp == 1, so every warp reads A from the window origin
    auto a_warp_window_tmp = make_tile_window(
        a_block_window_tmp.get_bottom_tensor_view(),
        make_tuple(number<WG::kM>{}, number<WG::kK>{}),
        a_block_window_tmp.get_window_origin(),
        make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

    static_assert(
        std::is_same_v<
            remove_cvref_t<decltype(c_block_dstr_encode)>,
            remove_cvref_t<
                decltype(CBlockTensor::get_tile_distribution()
                             .get_static_tile_distribution_encoding())>>,
        "wrong!");

    using BWarpDstr = typename WG::BWarpDstr;
    using CWarpDstr = typename WG::CWarpDstr;

    using BWarpTensor = typename WG::BWarpTensor;
    using CWarpTensor = typename WG::CWarpTensor;

    constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

    constexpr auto b_warp_y_index_zeros =
        uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros =
        uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    using a_warp_tensor_type = decltype(load_tile(a_warp_window_tmp));

    // number of ds_load instructions per A warp tile, one per vector access of
    // the window
    constexpr index_t kDsPerStep = a_warp_window_tmp.get_num_of_access();

    statically_indexed_array<a_warp_tensor_type, kSteps> a_warp_tensors;

    // The step offset is folded into kStepOffsetA as kStepOffsetB is in
    // BlockGemmARegBSmemCRegTdmGroupPrefetchFullGemmKSliceGemmN (A and B are
    // both read with normal ds_load from row-major padded Lds layouts); see the
    // comments there for why this is exact.
    auto load_step = [&](auto i_step) {
      constexpr index_t mIter = i_step / KIterPerWarp;
      constexpr index_t kIter = i_step % KIterPerWarp;

      constexpr auto a_lds_desc =
          typename ABlockWindowTmp::BottomTensorView::TensorDesc{};
      constexpr index_t kStepOffsetA = a_lds_desc.calculate_offset(make_tuple(
          number<mIter * MPerBlockPerIter>{},
          number<kIter * KPerBlockPerIter>{}));

      a_warp_tensors(i_step) =
          load_tile_with_offset(a_warp_window_tmp, kStepOffsetA);
    };

    // prologue: the first kPrefetchDepth steps, back to back
    static_for<0, kPrefetchDepth, 1>{}([&](auto i_step) { load_step(i_step); });

    __builtin_amdgcn_sched_barrier(0);
    prologue_hook();
    __builtin_amdgcn_sched_barrier(0);

    static_for<0, kSteps / kStepsPerGroup, 1>{}([&](auto i_group) {
      constexpr index_t kFirstLoad = i_group * kStepsPerGroup + kPrefetchDepth;
      constexpr index_t kNumLoads = kFirstLoad >= kSteps
          ? 0
          : (kSteps - kFirstLoad < kStepsPerGroup ? kSteps - kFirstLoad
                                                  : kStepsPerGroup);

      // the loads kPrefetchDepth steps ahead first, back to back ...
      static_for<0, kNumLoads, 1>{}(
          [&](auto i_load) { load_step(number<kFirstLoad + i_load>{}); });

      // ... then the WMMAs of this group
      static_for<0, kStepsPerGroup, 1>{}([&](auto i_step_in_group) {
        constexpr index_t i_step = i_group * kStepsPerGroup + i_step_in_group;
        constexpr auto mIter = number<i_step / KIterPerWarp>{};
        constexpr auto kIter = number<i_step % KIterPerWarp>{};

        const auto& a_warp_tensor = a_warp_tensors[number<i_step>{}];

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
          BWarpTensor b_warp_tensor;

          b_warp_tensor.get_thread_buffer() =
              b_block_tensor.get_y_sliced_thread_data(
                  merge_sequences(
                      sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                  merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

          CWarpTensor c_warp_tensor;

          if constexpr (kIter == 0) {
            c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensor);
          } else {
            c_warp_tensor.get_thread_buffer() =
                c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
          }

          c_block_tensor.set_y_sliced_thread_data(
              merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
              merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
              c_warp_tensor.get_thread_buffer());
        });
      });

      // keep the order: the DS loads of the group, then its WMMAs
      if constexpr (kNumLoads > 0)
        __builtin_amdgcn_sched_group_barrier(0x100, kNumLoads * kDsPerStep, 0);
      __builtin_amdgcn_sched_group_barrier(
          0x008, kStepsPerGroup * NIterPerWarp, 0);

      if constexpr (i_group < kSteps / kStepsPerGroup - 1)
        __builtin_amdgcn_sched_barrier(0);
    });
  }

  // C = A * B
  template <
      typename ABlockWindowTmp,
      typename BBlockTensorTmp,
      typename PrologueHook>
    requires std::is_invocable_v<const PrologueHook&>
  CK_TILE_DEVICE auto operator()(
      const ABlockWindowTmp& a_block_window_tmp,
      const BBlockTensorTmp& b_block_tensor_tmp,
      const PrologueHook& prologue_hook) const {
    auto c_block_tensor = MakeCBlockTile();
    operator()(
        c_block_tensor, a_block_window_tmp, b_block_tensor_tmp, prologue_hook);
    return c_block_tensor;
  }
};

} // namespace ck_tile
