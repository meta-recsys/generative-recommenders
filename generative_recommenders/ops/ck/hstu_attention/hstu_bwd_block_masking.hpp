// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/config.hpp>
#include <ck_tile/core/numeric/integer.hpp>
#include <ck_tile/core/numeric/integral_constant.hpp>
#include <ck_tile/core/numeric/math.hpp>
#include <type_traits>

namespace ck_tile {

template <bool kUseCausal>
struct HstuBwdCrossAttentionBlockMaskWithLocal {
  static constexpr bool kUseLocal = true;
  static constexpr bool IsMasking = true;
  static constexpr bool kIsCrossAttention = true;

  int seqlen_q;
  int seqlen_k;
  int contextual_seqlen;

  int max_attn_len;
  int min_full_attn_seqlen;

  int max_q_uih_len;
  int max_k_uih_len;
  int max_row_id;
  int max_col_id;
  int diff_q_kv_len;

  CK_TILE_HOST_DEVICE HstuBwdCrossAttentionBlockMaskWithLocal(
      int seqlen_q_,
      int seqlen_k_,
      int contextual_seqlen_,
      int max_attn_len_,
      int min_full_attn_seqlen_,
      int num_target_)
      : seqlen_q(seqlen_q_),
        seqlen_k(seqlen_k_),
        contextual_seqlen(contextual_seqlen_),
        max_attn_len(max_attn_len_),
        min_full_attn_seqlen(min_full_attn_seqlen_) {
    max_q_uih_len = seqlen_q - num_target_;
    max_k_uih_len = seqlen_k; // assuming target_in_kv == false

    // in case user provided max_attn_len_ could be bigger than max_uih_len
    max_attn_len = min(max_k_uih_len, min(max_q_uih_len, max_attn_len));

    // assuming min_full_attn_seqlen has higher priority, ensure contextual
    // scope not collide with min_full_attn_seqlen scope
    contextual_seqlen =
        min(contextual_seqlen, max_q_uih_len - min_full_attn_seqlen);

    if (contextual_seqlen > 0) {
      max_row_id = max_q_uih_len - (contextual_seqlen - 1);
      max_col_id = max_k_uih_len - (contextual_seqlen - 1);
    } else {
      max_row_id = max_q_uih_len;
      max_col_id = max_k_uih_len;
    }

    diff_q_kv_len = max_k_uih_len - max_q_uih_len;
    max_row_id += diff_q_kv_len;
  };

  // to get the loop length along Y axis, return index:[start, end),
  // end-start=length use this if need loop over Y axis tile by tile (eg.
  // seqlen_q loop-over) i_x is the start offset of the current tile along the
  // seqlen_k dimension
  template <index_t XTile, index_t YTile>
  CK_TILE_DEVICE constexpr auto
  GetTileRangeAlongY(index_t i_x, number<XTile>, number<YTile>) const {
    if (contextual_seqlen > 0) {
      return ck_tile::make_tuple(0, seqlen_q);
    } else {
      if constexpr (kUseCausal) {
        index_t y_start = min(
            max(i_x - diff_q_kv_len, 0), max_q_uih_len - min_full_attn_seqlen);
        index_t y_start_aligned = y_start - y_start % YTile;

        return ck_tile::make_tuple(y_start_aligned, seqlen_q);
      } else {
        index_t y_start =
            min(max(i_x - diff_q_kv_len - max_attn_len, 0),
                max_q_uih_len - min_full_attn_seqlen);
        index_t y_start_aligned = y_start - y_start % YTile;

        return ck_tile::make_tuple(y_start_aligned, seqlen_q);
      }
    }
  }

  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(int row, int col) const {
    int row_id;
    int col_id;

    row += diff_q_kv_len;

    if (contextual_seqlen > 0) {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = max(row - contextual_seqlen + 1, diff_q_kv_len);
      col_id = max(col - contextual_seqlen + 1, 0);

      row_id = min(row_id, max_row_id);
      col_id = min(col_id, max_col_id);

      if (row_id == diff_q_kv_len && col_id < max_col_id)
        return true;
    } else {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = min(row, max_row_id);
      col_id = min(col, max_col_id);
    };

    // use row_id/col_id to check the dist between two q/k token pair, token
    // pairs on the diagonal line are always considerred
    if constexpr (kUseCausal) {
      bool in_min_full_scope = (min_full_attn_seqlen > 0)
          ? (row_id >= max_row_id - min_full_attn_seqlen)
          : false;

      return (
          ((row_id > col_id) || (row == col)) &&
          ((row_id - col_id <= max_attn_len) || in_min_full_scope));
    } else {
      // Non-causal: only apply sliding window constraint, no diagonal inclusion
      // logic This matches PyTorch reference which just returns boundary mask
      // for non-causal
      bool in_min_full_scope = (min_full_attn_seqlen > 0)
          ? (row_id >= max_row_id - min_full_attn_seqlen)
          : false;

      return ((abs(row_id - col_id) <= max_attn_len) || in_min_full_scope);
    }
  };

  // if the whole tile inside the masking area, no need for pixel-by-pixel
  // checking
  template <index_t TileWidth, index_t TileHeight>
  CK_TILE_DEVICE constexpr bool IsFullTileInsideMask(
      index_t i_tile_top,
      index_t i_tile_left,
      number<TileWidth>,
      number<TileHeight>) const {
    if constexpr (kUseCausal) {
      index_t i_tile_right = i_tile_left + TileWidth;

      bool is_tile_in_bottom_scope =
          (i_tile_top >= (max_q_uih_len - min_full_attn_seqlen));

      if (is_tile_in_bottom_scope &&
          i_tile_right <= min(i_tile_top + diff_q_kv_len + 1, max_k_uih_len))
        return true;
    } else {
      index_t i_tile_right = i_tile_left + TileWidth;
      index_t i_tile_bottom = i_tile_top + TileHeight;

      bool is_tile_in_bottom_scope =
          (i_tile_top >= (max_q_uih_len - min_full_attn_seqlen));

      // 1) tile is completely in [max_q_uih_len-min_full_attn_seqlen,
      // max_q_uih_len] 2) some row of tile is in [max_q_uih_len, seqlen_q],
      // requires i_tile_right <= max_k_uih_len to return true
      if (is_tile_in_bottom_scope &&
          ((i_tile_bottom <= max_q_uih_len && i_tile_right <= seqlen_k) ||
           i_tile_right <= max_k_uih_len))
        return true;
    };

    return false;
  }
};

template <bool kUseCausal>
struct HstuBwdSelfAttentionBlockMaskWithLocal {
  static constexpr bool kUseLocal = true;
  static constexpr bool IsMasking = true;
  static constexpr bool kIsCrossAttention = false;

  int seqlen;
  int contextual_seqlen;

  int max_attn_len;
  int min_full_attn_seqlen;

  int max_uih_len;
  int max_id;

  CK_TILE_HOST_DEVICE HstuBwdSelfAttentionBlockMaskWithLocal(
      int seqlen_,
      int contextual_seqlen_,
      int max_attn_len_,
      int min_full_attn_seqlen_,
      int num_target_)
      : seqlen(seqlen_),
        contextual_seqlen(contextual_seqlen_),
        max_attn_len(max_attn_len_),
        min_full_attn_seqlen(min_full_attn_seqlen_) {
    max_uih_len = seqlen - num_target_;

    // in case user provided max_attn_len_ could be bigger than max_uih_len
    max_attn_len = min(max_uih_len, max_attn_len);

    // assuming min_full_attn_seqlen has higher priority, ensure contextual
    // scope not collide with min_full_attn_seqlen scope
    contextual_seqlen =
        min(contextual_seqlen, max_uih_len - min_full_attn_seqlen);

    if (contextual_seqlen > 0)
      max_id = max_uih_len - (contextual_seqlen - 1);
    else
      max_id = max_uih_len;
  };

  // to get the loop length along Y axis, return index:[start, end),
  // end-start=length use this if need loop over Y axis tile by tile (eg.
  // seqlen_q loop-over) i_x is the start offset of the current tile along the
  // seqlen_k dimension
  template <index_t XTile, index_t YTile>
  CK_TILE_DEVICE constexpr auto
  GetTileRangeAlongY(index_t i_x, number<XTile>, number<YTile>) const {
    if (contextual_seqlen > 0) {
      return ck_tile::make_tuple(0, seqlen);
    } else {
      if constexpr (kUseCausal) {
        index_t y_start = min(i_x, max_uih_len - min_full_attn_seqlen);
        index_t y_start_aligned = y_start - y_start % YTile;

        return ck_tile::make_tuple(y_start_aligned, seqlen);
      } else {
        index_t y_start =
            min(max(i_x - max_attn_len, 0), max_uih_len - min_full_attn_seqlen);
        index_t y_start_aligned = y_start - y_start % YTile;

        return ck_tile::make_tuple(y_start_aligned, seqlen);
      }
    }
  }

  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(int row, int col) const {
    int row_id;
    int col_id;

    if (contextual_seqlen > 0) {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = max(row - contextual_seqlen + 1, 0);
      col_id = max(col - contextual_seqlen + 1, 0);

      row_id = min(row_id, max_id);
      col_id = min(col_id, max_id);

      if (row_id == 0 && col_id < max_id)
        return true;
    } else {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = min(row, max_id);
      col_id = min(col, max_id);
    };

    // use row_id/col_id to check the dist between two q/k token pair, token
    // pairs on the diagonal line are always considerred
    if constexpr (kUseCausal) {
      bool in_min_full_scope = (min_full_attn_seqlen > 0)
          ? (row_id >= max_id - min_full_attn_seqlen)
          : false;

      return (
          ((row_id > col_id) || (row == col)) &&
          ((row_id - col_id <= max_attn_len) || in_min_full_scope));
    } else {
      bool in_min_full_scope = (min_full_attn_seqlen > 0)
          ? (row_id >= max_id - min_full_attn_seqlen)
          : false;

      return (
          ((row_id != col_id) || (row == col)) &&
          ((abs(row_id - col_id) <= max_attn_len) || in_min_full_scope));
    }
  };

  // if the whole tile inside the masking area, no need for pixel-by-pixel
  // checking
  template <index_t TileWidth, index_t TileHeight>
  CK_TILE_DEVICE constexpr bool IsFullTileInsideMask(
      index_t i_tile_top,
      index_t i_tile_left,
      number<TileWidth>,
      number<TileHeight>) const {
    if constexpr (kUseCausal) {
      index_t i_tile_right = i_tile_left + TileWidth;

      bool is_tile_in_bottom_scope =
          (i_tile_top >= (max_uih_len - min_full_attn_seqlen));

      if (is_tile_in_bottom_scope &&
          i_tile_right <= min(i_tile_top + 1, max_uih_len))
        return true;
    } else {
      index_t i_tile_right = i_tile_left + TileWidth;
      index_t i_tile_bottom = i_tile_top + TileHeight;

      bool is_tile_in_bottom_scope =
          (i_tile_top >= (max_uih_len - min_full_attn_seqlen));

      // 1) tile is completely in [max_uih_len-min_full_attn_seqlen,
      // max_uih_len] 2) some row of tile is in [max_uih_len, seqlen], requires
      // i_tile_right <= max_uih_len to return true
      if (is_tile_in_bottom_scope &&
          ((i_tile_bottom <= max_uih_len && i_tile_right <= seqlen) ||
           i_tile_right <= max_uih_len))
        return true;
    };

    return false;
  }
};

template <bool kUseCausal>
struct HstuBwdCrossAttentionBlockMaskNoLocal {
  static constexpr bool kUseLocal = false;
  static constexpr bool IsMasking = kUseCausal;
  static constexpr bool kIsCrossAttention = true;

  int seqlen_q;
  int seqlen_k;
  int contextual_seqlen;

  int max_q_uih_len;
  int max_k_uih_len;
  int max_row_id;
  int max_col_id;
  int diff_q_kv_len;

  CK_TILE_HOST_DEVICE
  HstuBwdCrossAttentionBlockMaskNoLocal(
      int seqlen_q_,
      int seqlen_k_,
      int contextual_seqlen_,
      int num_target_)
      : seqlen_q(seqlen_q_),
        seqlen_k(seqlen_k_),
        contextual_seqlen(contextual_seqlen_) {
    max_q_uih_len = seqlen_q - num_target_;
    max_k_uih_len = seqlen_k; // assuming target_in_kv == false

    if (contextual_seqlen > 0) {
      max_row_id = max_q_uih_len - (contextual_seqlen - 1);
      max_col_id = max_k_uih_len - (contextual_seqlen - 1);
    } else {
      max_row_id = max_q_uih_len;
      max_col_id = max_k_uih_len;
    }

    diff_q_kv_len = max_k_uih_len - max_q_uih_len;
    max_row_id += diff_q_kv_len;
  };

  // to get the loop length along Y axis, return index:[start, end),
  // end-start=length use this if need loop over Y axis tile by tile (eg.
  // seqlen_q loop-over) i_x is the start offset of the current tile along the
  // seqlen_k dimension
  template <index_t XTile, index_t YTile>
  CK_TILE_DEVICE constexpr auto
  GetTileRangeAlongY(index_t i_x, number<XTile>, number<YTile>) const {
    if (contextual_seqlen > 0) {
      return ck_tile::make_tuple(0, seqlen_q);
    } else {
      if constexpr (kUseCausal) {
        index_t y_start = min(max(i_x - diff_q_kv_len, 0), max_q_uih_len);
        index_t y_start_aligned = y_start - y_start % YTile;

        return ck_tile::make_tuple(y_start_aligned, seqlen_q);
      } else {
        return ck_tile::make_tuple(0, seqlen_q);
      }
    }
  }

  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(int row, int col) const {
    int row_id;
    int col_id;

    row += diff_q_kv_len;

    if (contextual_seqlen > 0) {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = max(row - contextual_seqlen + 1, diff_q_kv_len);
      col_id = max(col - contextual_seqlen + 1, 0);

      row_id = min(row_id, max_row_id);
      col_id = min(col_id, max_col_id);

      if (row_id == diff_q_kv_len && col_id < max_col_id)
        return true;
    } else {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = min(row, max_row_id);
      col_id = min(col, max_col_id);
    };

    // use row_id/col_id to check the dist between two q/k token pair, token
    // pairs on the diagonal line are always considerred
    if constexpr (IsMasking) {
      return (row_id > col_id) || (row == col);
    } else {
      // Non-causal: no masking needed, everything in bounds is allowed
      // This matches PyTorch reference which just returns boundary mask for
      // non-causal
      return true;
    };
  };

  // if the whole tile inside the masking area, no need for pixel-by-pixel
  // checking
  template <index_t TileWidth, index_t TileHeight>
  CK_TILE_DEVICE constexpr bool IsFullTileInsideMask(
      index_t i_tile_top,
      index_t i_tile_left,
      number<TileWidth>,
      number<TileHeight>) const {
    if constexpr (kUseCausal) {
      index_t i_tile_right = i_tile_left + (TileWidth - 1);
      index_t i_tile_bottom = i_tile_top + (TileHeight - 1);

      // assume num_target > 0 with high probability, don't check whether
      // num_target is 0; so if num_target is 0, IsTokenPairInsideMask() will be
      // called for the bottom tile
      if (i_tile_bottom >= max_q_uih_len ||
          i_tile_right > i_tile_top + diff_q_kv_len || i_tile_right >= seqlen_k)
        return false;

      return true;
    } else {
      index_t i_tile_right = i_tile_left + (TileWidth - 1);
      index_t i_tile_bottom = i_tile_top + (TileHeight - 1);

      // assume num_target > 0 with high probability, don't check whether
      // num_target is 0; so if num_target is 0, IsTokenPairInsideMask() will be
      // called for the bottom tile
      if (i_tile_bottom >= max_q_uih_len || i_tile_right >= max_k_uih_len)
        return false;

      return true;
    }
  };
};

template <bool kUseCausal>
struct HstuBwdSelfAttentionBlockMaskNoLocal {
  static constexpr bool kUseLocal = false;
  static constexpr bool IsMasking = kUseCausal;
  static constexpr bool kIsCrossAttention = false;

  int seqlen;
  int contextual_seqlen;

  int max_uih_len;
  int max_id;

  CK_TILE_HOST_DEVICE
  HstuBwdSelfAttentionBlockMaskNoLocal(
      int seqlen_,
      int contextual_seqlen_,
      int num_target_)
      : seqlen(seqlen_), contextual_seqlen(contextual_seqlen_) {
    max_uih_len = seqlen - num_target_;

    if (contextual_seqlen > 0)
      max_id = max_uih_len - (contextual_seqlen - 1);
    else
      max_id = max_uih_len;
  };

  // to get the loop length along Y axis, return index:[start, end),
  // end-start=length use this if need loop over Y axis tile by tile (eg.
  // seqlen_q loop-over) i_x is the start offset of the current tile along the
  // seqlen_k dimension
  template <index_t XTile, index_t YTile>
  CK_TILE_DEVICE constexpr auto
  GetTileRangeAlongY(index_t i_x, number<XTile>, number<YTile>) const {
    if (contextual_seqlen > 0) {
      return ck_tile::make_tuple(0, seqlen);
    } else {
      if constexpr (kUseCausal) {
        index_t y_start = min(i_x, max_uih_len);
        index_t y_start_aligned = y_start - y_start % YTile;

        return ck_tile::make_tuple(y_start_aligned, seqlen);
      } else {
        return ck_tile::make_tuple(0, seqlen);
      }
    }
  }

  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(int row, int col) const {
    int row_id;
    int col_id;

    if (contextual_seqlen > 0) {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = max(row - contextual_seqlen + 1, 0);
      col_id = max(col - contextual_seqlen + 1, 0);

      row_id = min(row_id, max_id);
      col_id = min(col_id, max_id);

      if (row_id == 0 && col_id < max_id)
        return true;
    } else {
      // row_id/col_id is clamped from physical row/col according to
      // contextual_seqlen and max_uih_len
      row_id = min(row, max_id);
      col_id = min(col, max_id);
    };

    // use row_id/col_id to check the dist between two q/k token pair, token
    // pairs on the diagonal line are always considerred
    if constexpr (IsMasking) {
      return (row_id > col_id) || (row == col);
    } else {
      return (row_id != col_id) || (row == col);
    };
  };

  // if the whole tile inside the masking area, no need for pixel-by-pixel
  // checking
  template <index_t TileWidth, index_t TileHeight>
  CK_TILE_DEVICE constexpr bool IsFullTileInsideMask(
      index_t i_tile_top,
      index_t i_tile_left,
      number<TileWidth>,
      number<TileHeight>) const {
    if constexpr (kUseCausal) {
      index_t i_tile_right = i_tile_left + (TileWidth - 1);
      index_t i_tile_bottom = i_tile_top + (TileHeight - 1);

      // assume num_target > 0 with high probability, don't check whether
      // num_target is 0; so if num_target is 0, IsTokenPairInsideMask() will be
      // called for the bottom tile
      if (i_tile_bottom >= max_uih_len || i_tile_right > i_tile_top ||
          i_tile_right >= seqlen)
        return false;

      return true;
    } else {
      index_t i_tile_right = i_tile_left + (TileWidth - 1);
      index_t i_tile_bottom = i_tile_top + (TileHeight - 1);

      // assume num_target > 0 with high probability, don't check whether
      // num_target is 0; so if num_target is 0, IsTokenPairInsideMask() will be
      // called for the bottom tile
      if (i_tile_bottom >= max_uih_len || i_tile_right >= max_uih_len)
        return false;

      return true;
    }
  };
};

template <bool kIsCrossAttention, bool kUseCausal, bool kUseLocal>
struct HstuBwdBlockMasking {
  using Type = std::conditional_t<
      kUseLocal,
      std::conditional_t<
          kIsCrossAttention,
          HstuBwdCrossAttentionBlockMaskWithLocal<kUseCausal>,
          HstuBwdSelfAttentionBlockMaskWithLocal<kUseCausal>>,
      std::conditional_t<
          kIsCrossAttention,
          HstuBwdCrossAttentionBlockMaskNoLocal<kUseCausal>,
          HstuBwdSelfAttentionBlockMaskNoLocal<kUseCausal>>>;
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_bwd_cross_attention_block_mask_with_local(
    int seqlen_q_,
    int seqlen_k_,
    int contextual_seqlen_,
    int num_target,
    int max_attn_len_,
    int min_full_attn_seqlen_) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == true);

  return HstuBlockMaskType{
      seqlen_q_,
      seqlen_k_,
      contextual_seqlen_,
      max_attn_len_,
      min_full_attn_seqlen_,
      num_target};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_bwd_self_attention_block_mask_with_local(
    int seqlen_,
    int contextual_seqlen_,
    int num_target,
    int max_attn_len_,
    int min_full_attn_seqlen_) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == false);

  return HstuBlockMaskType{
      seqlen_,
      contextual_seqlen_,
      max_attn_len_,
      min_full_attn_seqlen_,
      num_target};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_bwd_cross_attention_block_mask_without_local(
    int seqlen_q_,
    int seqlen_k_,
    int contextual_seqlen_,
    int num_target) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == true);

  return HstuBlockMaskType{
      seqlen_q_, seqlen_k_, contextual_seqlen_, num_target};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_bwd_self_attention_block_mask_without_local(
    int seqlen_,
    int contextual_seqlen_,
    int num_target) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == false);

  return HstuBlockMaskType{seqlen_, contextual_seqlen_, num_target};
};

} // namespace ck_tile
