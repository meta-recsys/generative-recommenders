// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>

#include "hstu_attention_config.hpp"

namespace ck_tile {

namespace detail {

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t f_integer_log2() {
  static_assert(
      X > 0 && (X & (X - 1)) == 0,
      "integer_log2 requires a positive power of two");

  index_t value = X;
  index_t result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
};

#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
template <index_t PadIntervalBytes, index_t PadLengthBytes>
CK_TILE_HOST_DEVICE constexpr bool IsTdmPaddingValid() {
  // What the descriptor field can encode.
  constexpr bool is_interval_valid = PadIntervalBytes >= 8 &&
      PadIntervalBytes <= 1024 &&
      (PadIntervalBytes & (PadIntervalBytes - 1)) == 0;
  constexpr bool is_length_valid =
      PadLengthBytes >= 4 && PadLengthBytes <= 512 && PadLengthBytes % 4 == 0;

  // Two further conditions, both conservative and both inert for every value
  // the callers in this header can produce. They are kept as a tripwire, not as
  // a documented hardware requirement -- neither is derivable from anything in
  // this tree, unlike the two above, which come straight from the descriptor's
  // pad_interval:3 / pad_amount:7 bitfields.
  //
  // is_interval_copyable keeps intervals to whole 64-byte chunks. It used to
  // ask for 128 bytes, an assumption about how the engine moves data for which
  // no source was found in the descriptor header, the toolchain, or upstream
  // ck_tile; the hdim96 K interval of 64 bytes (see GetTdmPadIntervalBytes())
  // was validated on gfx1250. What reaches it from the callers below:
  // power-of-two rows give a power of two >= 256, and 192-byte rows (hdim96)
  // give 64 for K and 256 for V. Rows of 80/144/160/224 bytes would give 16 or
  // 32 for K and still trip it.
  //
  // is_length_aligned restates a property the derivation already guarantees
  // rather than establishing one: PadLengthBytes is ScalarPerVector *
  // sizeof(T), i.e. exactly one Lds read vector, and the interval is a multiple
  // of that, so a row start after a pad is aligned for the read that follows it
  // whatever the vector width happens to be. The literal 16 holds for fp16/bf16
  // (the only types hstu instantiates); fp32 and non-mixed-precision fp8/bf8
  // would give 8 and be rejected. Widen or drop this if hstu ever gains those.
  constexpr bool is_interval_copyable =
      PadIntervalBytes >= 64 && PadIntervalBytes % 64 == 0;
  constexpr bool is_length_aligned = PadLengthBytes % 16 == 0;

  return is_interval_valid && is_length_valid && is_interval_copyable &&
      is_length_aligned;
};

template <index_t PadIntervalBytes, index_t PadLengthBytes>
CK_TILE_HOST_DEVICE constexpr auto GetTdmRawPaddingConfig() {
  static_assert(
      IsTdmPaddingValid<PadIntervalBytes, PadLengthBytes>(),
      "Invalid tdm padding config!");

  constexpr index_t raw_pad_interval =
      f_integer_log2<PadIntervalBytes / 4>() - 1;
  constexpr index_t raw_pad_length = PadLengthBytes / 4 - 1;

  return make_tuple(number<raw_pad_interval>{}, number<raw_pad_length>{});
};

// Lds pad interval for one Tdm-staged row of kRowBytes. Power-of-two rows (hdim
// 64/128/256) keep max(row, bank span). Other rows (hdim96: 192 B) are padded
// every lowest-set-bit bytes when read as a normal gemm operand (K): measured
// on gfx1250, a 256 B K interval (and the Lds layout it implies) was up to 19%
// slower than 64 B. The ds_read_tr side (V) keeps the bank-span rule; 64 B
// there moved times by <= 1.1%.
template <index_t kRowBytes, bool kIsNormalRead>
CK_TILE_HOST_DEVICE constexpr index_t GetTdmPadIntervalBytes() {
  constexpr index_t BankSpanBytes = 64 * 4; // gfx1250 has 64 banks
  if constexpr (is_power_of_two_integer(kRowBytes) || !kIsNormalRead)
    return max(kRowBytes, BankSpanBytes);
  else
    return kRowBytes & (-kRowBytes);
}

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetTdmLdsPaddingConfigForNormalRead() {
  if constexpr (IsGemmInputA) { // access gemm inputA from LDS
    using AEncoding = typename WarpGemm::AWarpDstrEncoding;
    using ADataType = typename WarpGemm::ADataType;

    constexpr index_t ScalarPerVector =
        AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];

    constexpr index_t kKBytesPerBlock = kKPerBlock * sizeof(ADataType);
    constexpr index_t PadIntervalBytes =
        GetTdmPadIntervalBytes<kKBytesPerBlock, true>();
    constexpr index_t PadLengthBytes = ScalarPerVector * sizeof(ADataType);

    return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
  } else { // acccess gemm inputB from LDS
    using BEncoding = typename WarpGemm::BWarpDstrEncoding;
    using BDataType = typename WarpGemm::BDataType;

    constexpr index_t ScalarPerVector =
        BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];

    constexpr index_t kKBytesPerBlock = kKPerBlock * sizeof(BDataType);
    constexpr index_t PadIntervalBytes =
        GetTdmPadIntervalBytes<kKBytesPerBlock, true>();
    constexpr index_t PadLengthBytes = ScalarPerVector * sizeof(BDataType);

    return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
  }
}

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetTdmLdsPaddingConfigForTrLoadRead() {
  if constexpr (IsGemmInputA) { // access gemm inputA from LDS
    using ADataType = typename WarpGemm::ADataType;
    using AEncoding = typename WarpGemm::AWarpDstrEncoding;
    using AEncodingForTrLoad =
        typename InputTileDistributionTraits<AEncoding, ADataType>::
            TransposedDstrEncode;

    constexpr index_t ScalarPerVector =
        AEncodingForTrLoad::detail::ys_lengths_[AEncoding::NDimY - 1];

    constexpr index_t kAKLane = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

    constexpr index_t kKBytesPerBlock = kKPerBlock * sizeof(ADataType);
    constexpr index_t PadIntervalBytes =
        GetTdmPadIntervalBytes<kKBytesPerBlock, false>();
    constexpr index_t PadLengthBytes =
        kAKLane * ScalarPerVector * sizeof(ADataType);

    return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
  } else { // acccess gemm inputB from LDS
    using BDataType = typename WarpGemm::BDataType;
    using BEncoding = typename WarpGemm::BWarpDstrEncoding;
    using BEncodingForTrLoad =
        typename InputTileDistributionTraits<BEncoding, BDataType>::
            TransposedDstrEncode;

    constexpr index_t ScalarPerVector =
        BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];

    constexpr index_t kBKLane = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

    constexpr index_t kKBytesPerBlock = kKPerBlock * sizeof(BDataType);
    constexpr index_t PadIntervalBytes =
        GetTdmPadIntervalBytes<kKBytesPerBlock, false>();
    constexpr index_t PadLengthBytes =
        kBKLane * ScalarPerVector * sizeof(BDataType);

    return make_tuple(number<PadIntervalBytes>{}, number<PadLengthBytes>{});
  }
}
#endif

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetLdsPaddingConfigForNormalRead() {
#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
  constexpr index_t BankSpanBytes = 64 * 4;
#else
  constexpr index_t BankSpanBytes = 32 * 4;
#endif

  if constexpr (IsGemmInputA) { // access gemm inputA from LDS
    using AEncoding = typename WarpGemm::AWarpDstrEncoding;
    using ADataType = typename WarpGemm::ADataType;

    constexpr index_t ScalarPerVector =
        AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];

    constexpr index_t BankSpanElements = BankSpanBytes / sizeof(ADataType);
    constexpr index_t PadInterval = max(kKPerBlock, BankSpanElements);
    constexpr index_t PadLength = ScalarPerVector;

    return make_tuple(number<PadInterval>{}, number<PadLength>{});
  } else { // acccess gemm inputB from LDS
    using BEncoding = typename WarpGemm::BWarpDstrEncoding;
    using BDataType = typename WarpGemm::BDataType;

    constexpr index_t ScalarPerVector =
        BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];

    constexpr index_t BankSpanElements = BankSpanBytes / sizeof(BDataType);
    constexpr index_t PadInterval = max(kKPerBlock, BankSpanElements);
    constexpr index_t PadLength = ScalarPerVector;

    return make_tuple(number<PadInterval>{}, number<PadLength>{});
  }
}

template <typename WarpGemm, bool IsGemmInputA, index_t kKPerBlock>
CK_TILE_HOST_DEVICE constexpr auto GetLdsPaddingConfigForTrLoadRead() {
#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
  constexpr index_t BankSpanBytes = 64 * 4;
#else
  constexpr index_t BankSpanBytes = 32 * 4;
#endif

  if constexpr (IsGemmInputA) { // access gemm inputA from LDS
    using ADataType = typename WarpGemm::ADataType;
    using AEncoding = typename WarpGemm::AWarpDstrEncoding;
    using AEncodingForTrLoad =
        typename InputTileDistributionTraits<AEncoding, ADataType>::
            TransposedDstrEncode;

    constexpr index_t ScalarPerVector =
        AEncodingForTrLoad::detail::ys_lengths_[AEncoding::NDimY - 1];

    constexpr index_t kAKLane = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

    constexpr index_t BankSpanElements = BankSpanBytes / sizeof(ADataType);
    constexpr index_t PadInterval = max(kKPerBlock, BankSpanElements);
    constexpr index_t PadLength = kAKLane * ScalarPerVector;

    return make_tuple(number<PadInterval>{}, number<PadLength>{});
  } else { // acccess gemm inputB from LDS
    using BDataType = typename WarpGemm::BDataType;
    using BEncoding = typename WarpGemm::BWarpDstrEncoding;
    using BEncodingForTrLoad =
        typename InputTileDistributionTraits<BEncoding, BDataType>::
            TransposedDstrEncode;

    constexpr index_t ScalarPerVector =
        BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];

    constexpr index_t kBKLane = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

    constexpr index_t BankSpanElements = BankSpanBytes / sizeof(BDataType);
    constexpr index_t PadInterval = max(kKPerBlock, BankSpanElements);
    constexpr index_t PadLength = kBKLane * ScalarPerVector;

    return make_tuple(number<PadInterval>{}, number<PadLength>{});
  }
}

template <
    index_t NumBuffers,
    index_t Rows,
    index_t Cols,
    index_t PadInterval,
    index_t PadLength>
CK_TILE_HOST_DEVICE constexpr auto MakeRowMajorLdsPaddedBlockDescriptor() {
  constexpr index_t LogicBufferSize = Rows * Cols;

  static_assert(LogicBufferSize >= PadInterval, "Check failed!");
  // The unmerge below re-reads the padded buffer as Rows x Cols, so a partial
  // trailing interval would silently drop its remainder and leave the last rows
  // short.
  static_assert(
      LogicBufferSize % PadInterval == 0,
      "Rows * Cols must be a whole number of pad intervals");

  constexpr index_t NumIntervals = LogicBufferSize / PadInterval;
  constexpr index_t SingleBufferSize = NumIntervals * (PadInterval + PadLength);

  constexpr auto interval_desc = make_naive_tensor_descriptor(
      make_tuple(
          number<NumBuffers>{}, number<NumIntervals>{}, number<PadInterval>{}),
      make_tuple(
          number<SingleBufferSize>{},
          number<PadInterval + PadLength>{},
          number<1>{}),
      number<8>{},
      number<1>{});

  constexpr auto flat_desc = transform_tensor_descriptor(
      interval_desc,
      make_tuple(
          make_pass_through_transform(number<NumBuffers>{}),
          make_merge_transform_v3_division_mod(
              make_tuple(number<NumIntervals>{}, number<PadInterval>{}))),
      make_tuple(sequence<0>{}, sequence<1, 2>{}),
      make_tuple(sequence<0>{}, sequence<1>{}));

  constexpr auto normal_layout_desc = transform_tensor_descriptor(
      flat_desc,
      make_tuple(
          make_pass_through_transform(number<NumBuffers>{}),
          make_unmerge_transform(make_tuple(number<Rows>{}, number<Cols>{}))),
      make_tuple(sequence<0>{}, sequence<1>{}),
      make_tuple(sequence<0>{}, sequence<1, 2>{}));

  return transform_tensor_descriptor(
      normal_layout_desc,
      make_tuple(
          make_merge_transform_v3_division_mod(
              make_tuple(number<NumBuffers>{}, number<Rows>{})),
          make_pass_through_transform(number<Cols>{})),
      make_tuple(sequence<0, 1>{}, sequence<2>{}),
      make_tuple(sequence<0>{}, sequence<1>{}));
};

template <
    typename DataType,
    index_t NumBuffers,
    index_t Rows,
    index_t Cols,
    index_t PadIntervalBytes,
    index_t PadLengthBytes>
CK_TILE_HOST_DEVICE constexpr auto MakeRowMajorLdsPaddedBlockDescriptor() {
  constexpr index_t PadInterval = PadIntervalBytes / sizeof(DataType);
  constexpr index_t PadLength = PadLengthBytes / sizeof(DataType);

  return MakeRowMajorLdsPaddedBlockDescriptor<
      NumBuffers,
      Rows,
      Cols,
      PadInterval,
      PadLength>();
};

template <index_t NumBuffers, index_t Rows, index_t Cols>
CK_TILE_HOST_DEVICE constexpr auto MakeRowMajorLdsPlainBlockDescriptor() {
  constexpr index_t SingleBufferSize = Rows * Cols;

  constexpr auto naive_desc = make_naive_tensor_descriptor(
      make_tuple(number<NumBuffers>{}, number<Rows>{}, number<Cols>{}),
      make_tuple(number<SingleBufferSize>{}, number<Cols>{}, number<1>{}),
      number<8>{},
      number<1>{});

  return transform_tensor_descriptor(
      naive_desc,
      make_tuple(
          make_merge_transform(
              make_tuple(number<NumBuffers>{}, number<Rows>{})),
          make_pass_through_transform(number<Cols>{})),
      make_tuple(sequence<0, 1>{}, sequence<2>{}),
      make_tuple(sequence<0>{}, sequence<1>{}));
}

template <typename WarpGemm, bool IsGemmInputA>
CK_TILE_HOST_DEVICE constexpr index_t GetSwizzleUnitForNormalRead() {
  if constexpr (IsGemmInputA) { // access gemm inputA from LDS
    using AEncoding = typename WarpGemm::AWarpDstrEncoding;

    constexpr index_t ScalarPerVector =
        AEncoding::detail::ys_lengths_[AEncoding::NDimY - 1];

    return ScalarPerVector;
  } else { // acccess gemm inputB from LDS
    using BEncoding = typename WarpGemm::BWarpDstrEncoding;

    constexpr index_t ScalarPerVector =
        BEncoding::detail::ys_lengths_[BEncoding::NDimY - 1];

    return ScalarPerVector;
  }
}

template <typename WarpGemm, bool IsGemmInputA>
CK_TILE_HOST_DEVICE constexpr index_t GetSwizzleUnitForTrLoadRead() {
  if constexpr (IsGemmInputA) { // access gemm inputA from LDS
    using ADataType = typename WarpGemm::ADataType;
    using AEncoding = typename WarpGemm::AWarpDstrEncoding;
    using AEncodingForTrLoad =
        typename InputTileDistributionTraits<AEncoding, ADataType>::
            TransposedDstrEncode;

    constexpr index_t ScalarPerVector =
        AEncodingForTrLoad::detail::ys_lengths_[AEncoding::NDimY - 1];

    constexpr index_t kAKLane = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

    return kAKLane * ScalarPerVector;
  } else { // acccess gemm inputB from LDS
    using BDataType = typename WarpGemm::BDataType;
    using BEncoding = typename WarpGemm::BWarpDstrEncoding;
    using BEncodingForTrLoad =
        typename InputTileDistributionTraits<BEncoding, BDataType>::
            TransposedDstrEncode;

    constexpr index_t ScalarPerVector =
        BEncodingForTrLoad::detail::ys_lengths_[BEncoding::NDimY - 1];

    constexpr index_t kBKLane = WarpGemm::WarpGemmAttribute::Impl::kABKLane;

    return kBKLane * ScalarPerVector;
  }
}

template <
    typename Problem,
    index_t NumBuffers,
    index_t kN,
    index_t kK,
    index_t kKPack>
CK_TILE_HOST_DEVICE static constexpr auto MakeSwizzledNativeDesc() {
  constexpr index_t ElementBytes = sizeof(typename Problem::QKVDataType);

#if defined(__hstu_gfx95__) || defined(__hstu_gfx125__)
  constexpr index_t BankSpanBytes = 64 * 4;
#else
  constexpr index_t BankSpanBytes = 32 * 4;
#endif

  constexpr index_t SingleBufferSize = kN * kK;

  if constexpr (kK * ElementBytes < BankSpanBytes) {
    constexpr index_t NLdsLayer = BankSpanBytes / (kK * ElementBytes);

    // 4D packed physical layout [NumBuffers, kN/NLdsLayer,
    // (kK/kKPack)*NLdsLayer, kKPack].
    constexpr auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(
            number<NumBuffers>{},
            number<kN / NLdsLayer>{},
            number<kK / kKPack * NLdsLayer>{},
            number<kKPack>{}),
        make_tuple(
            number<SingleBufferSize>{},
            number<kK * NLdsLayer>{},
            number<kKPack>{},
            number<1>{}),
        number<kKPack>{},
        number<1>{});

    // XOR-swizzle the (kN/NLdsLayer, kK-group*NLdsLayer) dims -> scatter banks.
    constexpr auto desc_permuted = transform_tensor_descriptor(
        desc_0,
        make_tuple(
            make_pass_through_transform(number<NumBuffers>{}),
            make_xor_transform(make_tuple(
                number<kN / NLdsLayer>{}, number<kK / kKPack * NLdsLayer>{})),
            make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

    // Split the kK-group dim back into [kK/kKPack, NLdsLayer].
    constexpr auto desc_split = transform_tensor_descriptor(
        desc_permuted,
        make_tuple(
            make_pass_through_transform(number<NumBuffers>{}),
            make_pass_through_transform(number<kN / NLdsLayer>{}),
            make_unmerge_transform(
                make_tuple(number<kK / kKPack>{}, number<NLdsLayer>{})),
            make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
        make_tuple(
            sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}));

    // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
    //   kN = (kN/NLdsLayer) * NLdsLayer
    //   kK = (kK/kKPack) * kKPack
    return transform_tensor_descriptor(
        desc_split,
        make_tuple(
            make_pass_through_transform(number<NumBuffers>{}),
            make_merge_transform_v3_division_mod(
                make_tuple(number<NLdsLayer>{}, number<kN / NLdsLayer>{})),
            make_merge_transform_v3_division_mod(
                make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<0>{}, sequence<3, 1>{}, sequence<2, 4>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
  } else {
    // 4D packed physical layout [NumBuffers, kN, kK/kKPack, kKPack].
    constexpr auto desc_0 = make_naive_tensor_descriptor(
        make_tuple(
            number<NumBuffers>{},
            number<kN>{},
            number<kK / kKPack>{},
            number<kKPack>{}),
        make_tuple(
            number<SingleBufferSize>{},
            number<kK>{},
            number<kKPack>{},
            number<1>{}),
        number<kKPack>{},
        number<1>{});

    // XOR-swizzle the (kN, kK-group) dims -> scatter banks.
    constexpr auto desc_permuted = transform_tensor_descriptor(
        desc_0,
        make_tuple(
            make_pass_through_transform(number<NumBuffers>{}),
            make_xor_transform(make_tuple(number<kN>{}, number<kK / kKPack>{})),
            make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

    // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
    //   kK = (kK/kKPack) * kKPack
    return transform_tensor_descriptor(
        desc_permuted,
        make_tuple(
            make_pass_through_transform(number<NumBuffers>{}),
            make_pass_through_transform(number<kN>{}),
            make_merge_transform_v3_division_mod(
                make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
  }
}

// Simple swizzling through xor without folding the NLdsLayer into xor space of
// bank-groups , used by normal reading of k-lds and trload reading of v-lds,
// when hdim is 64, 96 this could leave bank-conflicts not solved completely
template <
    typename Problem,
    index_t NumBuffers,
    index_t kN,
    index_t kK,
    index_t kKPack>
CK_TILE_HOST_DEVICE static constexpr auto MakeSimplySwizzledNativeDesc() {
  constexpr index_t SingleBufferSize = kN * kK;

  // 4D packed physical layout [NumBuffers, kN, kK/kKPack, kKPack].
  constexpr auto desc_0 = make_naive_tensor_descriptor(
      make_tuple(
          number<NumBuffers>{},
          number<kN>{},
          number<kK / kKPack>{},
          number<kKPack>{}),
      make_tuple(
          number<SingleBufferSize>{},
          number<kK>{},
          number<kKPack>{},
          number<1>{}),
      number<kKPack>{},
      number<1>{});

  // XOR-swizzle the (kN, kK-group) dims -> scatter banks.
  constexpr auto desc_permuted = transform_tensor_descriptor(
      desc_0,
      make_tuple(
          make_pass_through_transform(number<NumBuffers>{}),
          make_xor_transform(make_tuple(number<kN>{}, number<kK / kKPack>{})),
          make_pass_through_transform(number<kKPack>{})),
      make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}),
      make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

  // Re-merge to the logical 3D physical view [NumBuffers, kN, kK]:
  //   kK = (kK/kKPack) * kKPack
  return transform_tensor_descriptor(
      desc_permuted,
      make_tuple(
          make_pass_through_transform(number<NumBuffers>{}),
          make_pass_through_transform(number<kN>{}),
          make_merge_transform_v3_division_mod(
              make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
      make_tuple(sequence<0>{}, sequence<1>{}, sequence<2, 3>{}),
      make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));
}

}; // namespace detail
}; // namespace ck_tile
