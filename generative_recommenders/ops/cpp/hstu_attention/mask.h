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

#include <cute/tensor.hpp>

#include "utils.h"

namespace hstu {

using namespace cute;

template <int BlockM, int BlockN, typename TiledMma, bool SwapAB = false>
struct Mask {
  int const thread_idx;
  int const max_q_len;
  int const max_kv_len;
  int const max_attn_len;
  int const min_full_attn_seq_len;
  int const contextual_seq_len;
  int const max_uih_len;

  CUTLASS_DEVICE
  Mask(
      const int thread_idx,
      const int max_q_len,
      const int max_kv_len,
      const int max_attn_len,
      const int min_full_attn_seq_len,
      const int contextual_seq_len,
      const int max_uih_len)
      : thread_idx(thread_idx),
        max_q_len(max_q_len),
        max_kv_len(max_kv_len),
        max_attn_len(max_attn_len),
        min_full_attn_seq_len(min_full_attn_seq_len),
        contextual_seq_len(contextual_seq_len),
        max_uih_len(max_uih_len){};

  template <
      bool Seqlenq_mask = false,
      bool Seqlenk_mask = false,
      bool Causal_mask = false,
      bool Local_mask = false,
      bool Contexual_mask = false,
      bool Target_mask = false, // If Target_mask, Seqlenk_mask will be disabled
      bool Cross = false,
      bool Softmax = false,
      typename Engine,
      typename Layout>
  CUTLASS_DEVICE void apply(
      Tensor<Engine, Layout>& tSrS,
      const int m_block,
      const int n_block) const {
    static_assert(
        !(Causal_mask && Local_mask), "Cannot be both causal and local");
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    if constexpr (Cross) {
      static_assert(
          (!Local_mask) && (!Contexual_mask) && (!Target_mask),
          "Local, contexual, and target masks not supported under cross attention");
    }
    if (!Seqlenq_mask && !Seqlenk_mask && !Causal_mask && !Local_mask &&
        !Target_mask) {
      return;
    }

    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

    static constexpr int Qdim = !SwapAB ? 0 : 1, Kdim = !SwapAB ? 1 : 0;

    Tensor cS = cute::make_identity_tensor(Shape<
                                           Int<!SwapAB ? BlockM : BlockN>,
                                           Int<!SwapAB ? BlockN : BlockM>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(
        tSrS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(
        tScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
    Tensor t0ScS = thread0_mma.partition_C(cS);
    Tensor t0ScS_rowcol = make_tensor(
        t0ScS.data(),
        hstu::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
    // We want to use the col indices of thread0 to compare, since that is known
    // at compile time. So we subtract the limit by the first col index of this
    // thread
    int const thread_kdim_offset = get<Kdim>(tScS_rowcol(_0{}, _0{}));
    int const thread_qdim_offset = get<Qdim>(tScS_rowcol(_0{}, _0{}));
    int const seqlen_k_limit =
        max_kv_len - n_block * BlockN - thread_kdim_offset;
    int const uihlen_k_limit =
        max_uih_len - n_block * BlockN - thread_kdim_offset;
    int const seqlen_q_limit =
        max_q_len - m_block * BlockM - thread_qdim_offset;
    if constexpr (Seqlenq_mask) {
#pragma unroll
      for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        if (int(get<Qdim>(t0ScS_rowcol(m, _0{}))) >= seqlen_q_limit) {
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            if constexpr (Softmax) {
              tSrS_rowcol(m, n) = -INFINITY;
            } else {
              tSrS_rowcol(m, n) = 0.0f;
            }
          }
        }
      }
    }
    if constexpr (Cross) {
      if constexpr (!Causal_mask) {
        if constexpr (Seqlenk_mask) {
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
            if (t0_col_idx >= seqlen_k_limit) {
#pragma unroll
              for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                if constexpr (Softmax) {
                  tSrS_rowcol(m, n) = -INFINITY;
                } else {
                  tSrS_rowcol(m, n) = 0.0f;
                }
              }
            }
          }
        }
      } else {
        int const causal_row_offset = max_kv_len - max_uih_len + 1 -
            n_block * BlockN - thread_kdim_offset;
#pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
          if constexpr (Seqlenq_mask) {
            if (int(get<Qdim>(t0ScS_rowcol(m, _0{}))) >= seqlen_q_limit) {
              continue;
            }
          }
          int const row_idx = get<Qdim>(t0ScS_rowcol(m, _0{})) +
              m_block * BlockM + thread_qdim_offset;
          int const col_limit_right = !Seqlenk_mask
              ? row_idx + causal_row_offset
              : __viaddmin_s32(row_idx, causal_row_offset, seqlen_k_limit);
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
            if (t0_col_idx >= col_limit_right) {
              if constexpr (Softmax) {
                tSrS_rowcol(m, n) = -INFINITY;
              } else {
                tSrS_rowcol(m, n) = 0.0f;
              }
            }
          }
        }
      }
    } else {
      if constexpr (!Causal_mask && !Local_mask) {
        if constexpr (Seqlenk_mask || Target_mask) {
#pragma unroll
          for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
            if constexpr (Target_mask) {
              if (t0_col_idx >= uihlen_k_limit) {
                bool const oob_predicate = (t0_col_idx >= seqlen_k_limit);
                int const col_offset =
                    t0_col_idx - seqlen_k_limit + seqlen_q_limit;
#pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                  int const t0_row_idx = int(get<Qdim>(t0ScS_rowcol(m, _0{})));
                  if ((t0_row_idx != col_offset) || oob_predicate) {
                    if constexpr (Softmax) {
                      tSrS_rowcol(m, n) = -INFINITY;
                    } else {
                      tSrS_rowcol(m, n) = 0.0f;
                    }
                  }
                }
              }
            } else if constexpr (Seqlenk_mask) {
              if (t0_col_idx >= seqlen_k_limit) {
#pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                  if constexpr (Softmax) {
                    tSrS_rowcol(m, n) = -INFINITY;
                  } else {
                    tSrS_rowcol(m, n) = 0.0f;
                  }
                }
              }
            }
          }
        }
      } else { // Causal_mask or Local_mask
        int const causal_row_offset = 1 - n_block * BlockN - thread_kdim_offset;
        if constexpr (Causal_mask) {
#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            if constexpr (Seqlenq_mask) {
              if (int(get<Qdim>(t0ScS_rowcol(m, _0{}))) >= seqlen_q_limit) {
                continue;
              }
            }
            if constexpr (Contexual_mask) {
              if (int(get<Qdim>(t0ScS_rowcol(m, _0{}))) <
                  contextual_seq_len - m_block * BlockM - thread_qdim_offset) {
#pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                  int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
                  if (t0_col_idx >= uihlen_k_limit) {
                    if constexpr (Softmax) {
                      tSrS_rowcol(m, n) = -INFINITY;
                    } else {
                      tSrS_rowcol(m, n) = 0.0f;
                    }
                  }
                }
                continue;
              }
            }
            int const row_idx = get<Qdim>(t0ScS_rowcol(m, _0{})) +
                m_block * BlockM + thread_qdim_offset;
            if constexpr (!Target_mask) {
              int const col_limit_right = !Seqlenk_mask
                  ? row_idx + causal_row_offset
                  : __viaddmin_s32(row_idx, causal_row_offset, seqlen_k_limit);
#pragma unroll
              for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
                if (t0_col_idx >= col_limit_right) {
                  if constexpr (Softmax) {
                    tSrS_rowcol(m, n) = -INFINITY;
                  } else {
                    tSrS_rowcol(m, n) = 0.0f;
                  }
                }
              }
            } else {
#pragma unroll
              for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
                int const col_idx =
                    t0_col_idx + n_block * BlockN + thread_kdim_offset;
                bool const uih_cond =
                    (t0_col_idx >= row_idx + causal_row_offset) &&
                    (row_idx < max_uih_len);
                bool const target_cond = (row_idx != col_idx) &&
                    (row_idx >= max_uih_len) && (col_idx >= max_uih_len);
                bool const seqlen_k_cond = (t0_col_idx >= seqlen_k_limit);
                if (uih_cond || target_cond || seqlen_k_cond) {
                  if constexpr (Softmax) {
                    tSrS_rowcol(m, n) = -INFINITY;
                  } else {
                    tSrS_rowcol(m, n) = 0.0f;
                  }
                }
              }
            }
          }
        } else { // Local_mask
          int const local_row_offset_left =
              causal_row_offset - 1 - max_attn_len;
          int const col_limit_sink = 0 - n_block * BlockN - thread_kdim_offset;
#pragma unroll
          for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            if constexpr (Seqlenq_mask) {
              if (int(get<Qdim>(t0ScS_rowcol(m, _0{}))) >= seqlen_q_limit) {
                continue;
              }
            }
            if constexpr (Contexual_mask) {
              if (int(get<Qdim>(t0ScS_rowcol(m, _0{}))) <
                  contextual_seq_len - m_block * BlockM - thread_qdim_offset) {
#pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                  int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
                  if (t0_col_idx >= uihlen_k_limit) {
                    if constexpr (Softmax) {
                      tSrS_rowcol(m, n) = -INFINITY;
                    } else {
                      tSrS_rowcol(m, n) = 0.0f;
                    }
                  }
                }
                continue;
              }
            }
            int const row_idx = get<Qdim>(t0ScS_rowcol(m, _0{})) +
                m_block * BlockM + thread_qdim_offset;
            int col_limit_left = row_idx + local_row_offset_left;
            if constexpr (Contexual_mask) {
              // row contexual without sink
              if (col_limit_left + n_block * BlockN + thread_kdim_offset <
                  contextual_seq_len) {
                col_limit_left = 0;
              }
            }
            if constexpr (!Target_mask) {
              int const col_limit_right = !Seqlenk_mask
                  ? row_idx + causal_row_offset
                  : __viaddmin_s32(row_idx, causal_row_offset, seqlen_k_limit);
#pragma unroll
              for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(m, n)));
                if (row_idx < max_uih_len - min_full_attn_seq_len) {
                  bool const local_left_cond = Contexual_mask
                      ? (t0_col_idx < col_limit_left &&
                         t0_col_idx >= col_limit_sink)
                      : (t0_col_idx < col_limit_left);
                  if (local_left_cond) {
                    if constexpr (Softmax) {
                      tSrS_rowcol(m, n) = -INFINITY;
                    } else {
                      tSrS_rowcol(m, n) = 0.0f;
                    }
                  }
                }
                if (t0_col_idx >= col_limit_right) {
                  if constexpr (Softmax) {
                    tSrS_rowcol(m, n) = -INFINITY;
                  } else {
                    tSrS_rowcol(m, n) = 0.0f;
                  }
                }
              }
            } else {
              int const col_limit_right = row_idx + causal_row_offset;
#pragma unroll
              for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                int const t0_col_idx = int(get<Kdim>(t0ScS_rowcol(_0{}, n)));
                if (row_idx < max_uih_len) {
                  if (row_idx < max_uih_len - min_full_attn_seq_len) {
                    bool const local_left_cond = Contexual_mask
                        ? (t0_col_idx < col_limit_left &&
                           t0_col_idx >= col_limit_sink)
                        : (t0_col_idx < col_limit_left);
                    if (local_left_cond) {
                      if constexpr (Softmax) {
                        tSrS_rowcol(m, n) = -INFINITY;
                      } else {
                        tSrS_rowcol(m, n) = 0.0f;
                      }
                    }
                  }
                  if (t0_col_idx >= col_limit_right) {
                    if constexpr (Softmax) {
                      tSrS_rowcol(m, n) = -INFINITY;
                    } else {
                      tSrS_rowcol(m, n) = 0.0f;
                    }
                  }
                } else {
                  int const col_idx =
                      t0_col_idx + n_block * BlockN + thread_kdim_offset;
                  bool const target_cond = (row_idx != col_idx) &&
                      (row_idx >= max_uih_len) && (col_idx >= max_uih_len);
                  bool const seqlen_k_cond = (t0_col_idx >= seqlen_k_limit);
                  if (target_cond || seqlen_k_cond) {
                    if constexpr (Softmax) {
                      tSrS_rowcol(m, n) = -INFINITY;
                    } else {
                      tSrS_rowcol(m, n) = 0.0f;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  };
};

} // namespace hstu
