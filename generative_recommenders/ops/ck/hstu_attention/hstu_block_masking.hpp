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
struct HstuCrossAttentionBlockMaskWithLocal {
  static constexpr bool kUseLocal = true;
  static constexpr bool IsMasking = true;
  static constexpr bool kIsCrossAttention = true;

  // is_tile_in_upper_scope is false only when min_full_attn_seqlen > 0 and the
  // current tile is inside scope [max_uih_len - min_full_attn_seqlen,
  // seqlen_q); for other cases and tiles, is_tile_in_upper_scope is true
  bool is_tile_in_upper_scope;
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

  CK_TILE_HOST_DEVICE HstuCrossAttentionBlockMaskWithLocal(
      bool is_tile_in_upper_scope_,
      int seqlen_q_,
      int seqlen_k_,
      int contextual_seqlen_,
      int max_attn_len_,
      int min_full_attn_seqlen_,
      int num_target_)
      : is_tile_in_upper_scope(is_tile_in_upper_scope_),
        seqlen_q(seqlen_q_),
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

  // to get the loop length along X axis, return index:[start, end),
  // end-start=length use this if need loop over X axis tile by tile (eg.
  // seqlen_k loop-over) i_y is the start offset of the current tile along the
  // seqlen_q dimension
  template <bool kHasDropout, index_t YTile, index_t XTile>
  CK_TILE_DEVICE constexpr auto GetTileRangeAlongX(
      bool_constant<kHasDropout>,
      index_t i_y,
      number<YTile>,
      number<XTile>) const {
    // handle two special cases first
    if (!is_tile_in_upper_scope) {
      if constexpr (kUseCausal) {
        index_t x_end = min(i_y + YTile + diff_q_kv_len, seqlen_k);
        return ck_tile::make_tuple(0, x_end);
      } else {
        // tile is partitially or completely in
        // [max_uih_len-min_full_attn_seqlen, max_q_uih_len)
        if (i_y < max_q_uih_len) {
          return ck_tile::make_tuple(0, seqlen_k);
        } else // tile is completely inside [max_q_uih_len, seqlen_q)
        {
          index_t x_end = min(i_y + YTile + diff_q_kv_len, seqlen_k);
          return ck_tile::make_tuple(0, x_end);
        };
      };
    };

    if constexpr (kHasDropout) {
      index_t boundary = max_q_uih_len - min_full_attn_seqlen;
      // the last tile of first split could be a cross-boundary tile
      if (i_y < boundary && i_y + YTile > boundary)
        return ck_tile::make_tuple(0, seqlen_k);
    };

    // is_tile_in_upper_scope is true, either min_full_attn_seqlen is 0 or tile
    // is in [0, max_uih_len-min_full_attn_seqlen)
    if constexpr (!kUseCausal) {
      if (i_y >= min(contextual_seqlen, 1) + max_attn_len) {
        // some row of the tile in [contextual_seqlen+max_attn_len,
        // max_q_uih_len)
        if (i_y < max_q_uih_len) {
          index_t x_start = i_y + diff_q_kv_len - max_attn_len;
          index_t x_start_aligned = x_start - x_start % XTile;

          // some rows of the tile in [max_q_uih_len - max_attn_len,
          // max_q_uih_len)
          if (i_y + YTile > max_q_uih_len - max_attn_len) {
            return ck_tile::make_tuple(x_start_aligned, seqlen_k);
          } else // whole tile in [contextual_seqlen+max_attn_len, max_q_uih_len
                 // -max_attn_len)
          {
            index_t x_end = i_y + YTile + diff_q_kv_len + max_attn_len;
            return ck_tile::make_tuple(x_start_aligned, x_end);
          };
        } else // whole tile in [max_uih_len, seqlen)
        {
          index_t x_start = max_k_uih_len - max_attn_len;
          index_t x_end = min(i_y + YTile + diff_q_kv_len, seqlen_k);

          return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
        }
      } else // for i_y < contextual_seqlen + max_attn_len
      {
        if (i_y <
            contextual_seqlen) // some row of the tile in [0, contextual_seqlen)
        {
          index_t x_end = min(
              max(i_y + YTile + diff_q_kv_len + max_attn_len, max_k_uih_len),
              seqlen_k);
          return ck_tile::make_tuple(0, x_end);
        } else // whole tile in [contextual_seqlen, seqlen)
        {
          index_t x_end =
              min(i_y + YTile + diff_q_kv_len + max_attn_len, seqlen_k);
          return ck_tile::make_tuple(0, x_end);
        }
      }
    } else // kUseCausal && kUseLocal
    {
      if (i_y >= min(contextual_seqlen, 1) + max_attn_len) {
        index_t x_end = min(i_y + YTile + diff_q_kv_len, seqlen_k);

        // some row of the tile in [contextual_seqlen+max_attn_len,
        // max_q_uih_len)
        if (i_y < max_q_uih_len) {
          index_t x_start = i_y + diff_q_kv_len - max_attn_len;
          return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
        } else // whole tile in [max_q_uih_len, seqlen_q)
        {
          index_t x_start = max_k_uih_len - max_attn_len;
          return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
        }
      } else // for i_y < contextual_seqlen + max_attn_len
      {
        if (i_y <
            contextual_seqlen) // some row of the tile in [0, contextual_seqlen)
        {
          index_t x_end =
              min(max(i_y + YTile + diff_q_kv_len, max_k_uih_len), seqlen_k);
          return ck_tile::make_tuple(0, x_end);
        } else // whole tile in [contextual_seqlen, seqlen)
        {
          index_t x_end = min(i_y + YTile + diff_q_kv_len, seqlen_k);
          return ck_tile::make_tuple(0, x_end);
        }
      }
    };
  }

  template <bool kHasContextual = true>
  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(
      int row,
      int col,
      bool_constant<kHasContextual> = {}) const {
    int row_id;
    int col_id;

    row += diff_q_kv_len;

    if (kHasContextual && contextual_seqlen > 0) {
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

      if (!is_tile_in_upper_scope &&
          i_tile_right <= min(i_tile_top + diff_q_kv_len + 1, max_k_uih_len))
        return true;
    } else {
      index_t i_tile_right = i_tile_left + TileWidth;
      index_t i_tile_bottom = i_tile_top + TileHeight;

      // 1) tile is completely in [max_q_uih_len-min_full_attn_seqlen,
      // max_q_uih_len] 2) some row of tile is in [max_q_uih_len, seqlen_q],
      // requires i_tile_right <= max_k_uih_len to return true
      if (!is_tile_in_upper_scope &&
          ((i_tile_bottom <= max_q_uih_len && i_tile_right <= seqlen_k) ||
           i_tile_right <= max_k_uih_len))
        return true;
    };

    return false;
  }
};

template <bool kUseCausal>
struct HstuSelfAttentionBlockMaskWithLocal {
  static constexpr bool kUseLocal = true;
  static constexpr bool IsMasking = true;
  static constexpr bool kIsCrossAttention = false;

  // is_tile_in_upper_scope is false only when min_full_attn_seqlen > 0 and the
  // current tile is inside scope [max_uih_len - min_full_attn_seqlen,
  // seqlen_q); for other cases and tiles, is_tile_in_upper_scope is true
  bool is_tile_in_upper_scope;
  int seqlen;
  int contextual_seqlen;

  int max_attn_len;
  int min_full_attn_seqlen;

  int max_uih_len;
  int max_id;

  CK_TILE_HOST_DEVICE HstuSelfAttentionBlockMaskWithLocal(
      bool is_tile_in_upper_scope_,
      int seqlen_,
      int contextual_seqlen_,
      int max_attn_len_,
      int min_full_attn_seqlen_,
      int num_target_)
      : is_tile_in_upper_scope(is_tile_in_upper_scope_),
        seqlen(seqlen_),
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

  // to get the loop length along X axis, return index:[start, end),
  // end-start=length use this if need loop over X axis tile by tile (eg.
  // seqlen_k loop-over) i_y is the start offset of the current tile along the
  // seqlen_q dimension
  template <bool kHasDropout, index_t YTile, index_t XTile>
  CK_TILE_DEVICE constexpr auto GetTileRangeAlongX(
      bool_constant<kHasDropout>,
      index_t i_y,
      number<YTile>,
      number<XTile>) const {
    // handle two special cases first
    if (!is_tile_in_upper_scope) {
      if constexpr (kUseCausal) {
        index_t x_end = min(i_y + YTile, seqlen);
        return ck_tile::make_tuple(0, x_end);
      } else {
        // tile is partitially or completely in
        // [max_uih_len-min_full_attn_seqlen, max_uih_len)
        if (i_y < max_uih_len) {
          return ck_tile::make_tuple(0, seqlen);
        } else // tile is completely inside [max_uih_len, seqlen)
        {
          index_t x_end = min(i_y + YTile, seqlen);
          return ck_tile::make_tuple(0, x_end);
        };
      };
    };

    if constexpr (kHasDropout) {
      index_t boundary = max_uih_len - min_full_attn_seqlen;
      // the last tile of first split could be a cross-boundary tile
      if (i_y < boundary && i_y + YTile > boundary)
        return ck_tile::make_tuple(0, seqlen);
    };

    // is_tile_in_upper_scope is true, either min_full_attn_seqlen is 0 or tile
    // is in [0, max_uih_len-min_full_attn_seqlen)
    if constexpr (!kUseCausal) {
      if (i_y >= min(contextual_seqlen, 1) + max_attn_len) {
        // some row of the tile in [contextual_seqlen+max_attn_len, max_uih_len)
        if (i_y < max_uih_len) {
          index_t x_start = i_y - max_attn_len;
          index_t x_start_aligned = x_start - x_start % XTile;

          // some rows of the tile in [max_uih_len -max_attn_len, max_uih_len)
          if (i_y + YTile > max_uih_len - max_attn_len) {
            return ck_tile::make_tuple(x_start_aligned, seqlen);
          } else // whole tile in [contextual_seqlen+max_attn_len, max_uih_len
                 // -max_attn_len)
          {
            index_t x_end = i_y + YTile + max_attn_len;
            return ck_tile::make_tuple(x_start_aligned, x_end);
          };
        } else // whole tile in [max_uih_len, seqlen)
        {
          index_t x_start = max_uih_len - max_attn_len;
          index_t x_end = min(i_y + YTile, seqlen);

          return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
        }
      } else // for i_y < contextual_seqlen + max_attn_len
      {
        if (i_y <
            contextual_seqlen) // some row of the tile in [0, contextual_seqlen)
        {
          index_t x_end =
              min(max(i_y + YTile + max_attn_len, max_uih_len), seqlen);
          return ck_tile::make_tuple(0, x_end);
        } else // whole tile in [contextual_seqlen, seqlen)
        {
          index_t x_end = min(i_y + YTile + max_attn_len, seqlen);
          return ck_tile::make_tuple(0, x_end);
        }
      }
    } else // kUseCausal && kUseLocal
    {
      if (i_y >= min(contextual_seqlen, 1) + max_attn_len) {
        index_t x_end = min(i_y + YTile, seqlen);

        // some row of the tile in [contextual_seqlen+max_attn_len, max_uih_len)
        if (i_y < max_uih_len) {
          index_t x_start = i_y - max_attn_len;
          return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
        } else // whole tile in [max_uih_len, seqlen)
        {
          index_t x_start = max_uih_len - max_attn_len;
          return ck_tile::make_tuple(x_start - x_start % XTile, x_end);
        }
      } else // for i_y < contextual_seqlen + max_attn_len
      {
        if (i_y <
            contextual_seqlen) // some row of the tile in [0, contextual_seqlen)
        {
          index_t x_end = min(max(i_y + YTile, max_uih_len), seqlen);
          return ck_tile::make_tuple(0, x_end);
        } else // whole tile in [contextual_seqlen, seqlen)
        {
          index_t x_end = min(i_y + YTile, seqlen);
          return ck_tile::make_tuple(0, x_end);
        }
      }
    };
  }

  template <bool kHasContextual = true>
  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(
      int row,
      int col,
      bool_constant<kHasContextual> = {}) const {
    int row_id;
    int col_id;

    if (kHasContextual && contextual_seqlen > 0) {
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

      if (!is_tile_in_upper_scope &&
          i_tile_right <= min(i_tile_top + 1, max_uih_len))
        return true;
    } else {
      index_t i_tile_right = i_tile_left + TileWidth;
      index_t i_tile_bottom = i_tile_top + TileHeight;

      // 1) tile is completely in [max_uih_len-min_full_attn_seqlen,
      // max_uih_len] 2) some row of tile is in [max_uih_len, seqlen], requires
      // i_tile_right <= max_uih_len to return true
      if (!is_tile_in_upper_scope &&
          ((i_tile_bottom <= max_uih_len && i_tile_right <= seqlen) ||
           i_tile_right <= max_uih_len))
        return true;
    };

    return false;
  }
};

template <bool kUseCausal>
struct HstuCrossAttentionBlockMaskNoLocal {
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
  HstuCrossAttentionBlockMaskNoLocal(
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

  // to get the loop length along X axis, return index:[start, end),
  // end-start=length use this if need loop over X axis tile by tile (eg.
  // seqlen_k loop-over) i_y is the start offset of the current tile along the
  // seqlen_q dimension
  template <index_t YTile, index_t XTile>
  CK_TILE_DEVICE constexpr auto
  GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const {
    if constexpr (!IsMasking) {
      return ck_tile::make_tuple(0, seqlen_k);
    } else {
      index_t x_end = min(i_y + YTile + diff_q_kv_len, seqlen_k);

      if (i_y < contextual_seqlen) {
        if (i_y + YTile + diff_q_kv_len > max_k_uih_len) {
          return ck_tile::make_tuple(0, x_end);
        } else {
          return ck_tile::make_tuple(0, max_k_uih_len);
        };
      } else {
        return ck_tile::make_tuple(0, x_end);
      };
    };
  }

  template <bool kHasContextual = true>
  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(
      int row,
      int col,
      bool_constant<kHasContextual> = {}) const {
    int row_id;
    int col_id;

    row += diff_q_kv_len;

    if (kHasContextual && contextual_seqlen > 0) {
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
struct HstuSelfAttentionBlockMaskNoLocal {
  static constexpr bool kUseLocal = false;
  static constexpr bool IsMasking = kUseCausal;
  static constexpr bool kIsCrossAttention = false;

  int seqlen;
  int contextual_seqlen;

  int max_uih_len;
  int max_id;

  CK_TILE_HOST_DEVICE
  HstuSelfAttentionBlockMaskNoLocal(
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

  // to get the loop length along X axis, return index:[start, end),
  // end-start=length use this if need loop over X axis tile by tile (eg.
  // seqlen_k loop-over) i_y is the start offset of the current tile along the
  // seqlen_q dimension
  template <index_t YTile, index_t XTile>
  CK_TILE_DEVICE constexpr auto
  GetTileRangeAlongX(index_t i_y, number<YTile>, number<XTile>) const {
    if constexpr (!IsMasking) {
      return ck_tile::make_tuple(0, seqlen);
    } else {
      index_t x_end = min(i_y + YTile, seqlen);

      if (i_y < contextual_seqlen) {
        if (i_y + YTile > max_uih_len) {
          return ck_tile::make_tuple(0, x_end);
        } else {
          return ck_tile::make_tuple(0, max_uih_len);
        };
      } else {
        return ck_tile::make_tuple(0, x_end);
      };
    };
  }

  template <bool kHasContextual = true>
  CK_TILE_HOST_DEVICE bool IsTokenPairInsideMask(
      int row,
      int col,
      bool_constant<kHasContextual> = {}) const {
    int row_id;
    int col_id;

    if (kHasContextual && contextual_seqlen > 0) {
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
struct HstuBlockMasking {
  using Type = std::conditional_t<
      kUseLocal,
      std::conditional_t<
          kIsCrossAttention,
          HstuCrossAttentionBlockMaskWithLocal<kUseCausal>,
          HstuSelfAttentionBlockMaskWithLocal<kUseCausal>>,
      std::conditional_t<
          kIsCrossAttention,
          HstuCrossAttentionBlockMaskNoLocal<kUseCausal>,
          HstuSelfAttentionBlockMaskNoLocal<kUseCausal>>>;
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_cross_attention_block_mask_with_local(
    bool is_tile_in_upper_scope_,
    int seqlen_q_,
    int seqlen_k_,
    int contextual_seqlen_,
    int num_target,
    int max_attn_len_,
    int min_full_attn_seqlen_) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == true);

  return HstuBlockMaskType{
      is_tile_in_upper_scope_,
      seqlen_q_,
      seqlen_k_,
      contextual_seqlen_,
      max_attn_len_,
      min_full_attn_seqlen_,
      num_target};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_self_attention_block_mask_with_local(
    bool is_tile_in_upper_scope_,
    int seqlen_,
    int contextual_seqlen_,
    int num_target,
    int max_attn_len_,
    int min_full_attn_seqlen_) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == false);

  return HstuBlockMaskType{
      is_tile_in_upper_scope_,
      seqlen_,
      contextual_seqlen_,
      max_attn_len_,
      min_full_attn_seqlen_,
      num_target};
};

template <typename HstuBlockMaskType>
CK_TILE_HOST_DEVICE constexpr auto
make_hstu_cross_attention_block_mask_without_local(
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
make_hstu_self_attention_block_mask_without_local(
    int seqlen_,
    int contextual_seqlen_,
    int num_target) {
  static_assert(HstuBlockMaskType::kIsCrossAttention == false);

  return HstuBlockMaskType{seqlen_, contextual_seqlen_, num_target};
};

} // namespace ck_tile
