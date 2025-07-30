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

namespace hstu {

// We consolidate all the info related to sequence length here. This is so that
// we can do all the gmem reads once at the beginning of each tile, rather than
// having to repeat these reads to compute various things like n_block_min,
// n_block_max, etc.

template <bool Jagged, int kBlock>
struct SeqlenInfo {
  int const offset, offset_padded;
  int const seqlen;

  CUTLASS_DEVICE
  SeqlenInfo(
      int const bidb,
      int const seqlen_static,
      int const* const seq_offsets)
      : offset(!Jagged ? 0 : seq_offsets[bidb]),
        offset_padded(
            !Jagged ? 0
                    : (seq_offsets[bidb] + bidb * kBlock) / kBlock * kBlock),
        seqlen(
            !Jagged ? seqlen_static
                    : (seq_offsets[bidb + 1] - seq_offsets[bidb])) {}
};

template <bool Jagged, bool Cross, bool Has_targets, int kBlockM>
struct SeqlenInfoQKBwd {
  int const offset_q, offset_k, offset_q_padded;
  int const seqlen_q, seqlen_kv, uihlen_q;

  CUTLASS_DEVICE
  SeqlenInfoQKBwd(
      int const bidb,
      int const max_q_len,
      int const max_kv_len,
      int const* const seq_offsets,
      int const* const seq_offsets_q,
      int const* const num_targets)
      : offset_q(
            !Jagged ? 0 : (Cross ? seq_offsets_q[bidb] : seq_offsets[bidb])),
        offset_k(!Jagged ? 0 : seq_offsets[bidb])
        // If jagged, the layout for dQaccum is that we pad
        // each sequence in the batch by an extra kBlockM, so that the write for
        // each sequence doesn't touch the next sequence. Sequence i starts at
        // seq_offsets[i] + i * kBlockM and ends at seq_offsets[i + 1] + i *
        // kBlockM However, the start must align to multiples of kBlockM.
        ,
        offset_q_padded(
            !Jagged ? 0
                : Cross
                ? ((seq_offsets_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM)
                : ((seq_offsets[bidb] + bidb * kBlockM) / kBlockM * kBlockM)),
        seqlen_q(
            !Jagged ? max_q_len
                    : (Cross ? (seq_offsets_q[bidb + 1] - seq_offsets_q[bidb])
                             : (seq_offsets[bidb + 1] - seq_offsets[bidb]))),
        seqlen_kv(
            !Jagged ? max_kv_len : (seq_offsets[bidb + 1] - seq_offsets[bidb])),
        uihlen_q(
            !Jagged
                ? (Has_targets ? max_q_len - num_targets[bidb] : max_q_len)
                : (Has_targets
                       ? (Cross ? (seq_offsets_q[bidb + 1] -
                                   seq_offsets_q[bidb] - num_targets[bidb])
                                : (seq_offsets[bidb + 1] - seq_offsets[bidb] -
                                   num_targets[bidb]))
                       : (Cross
                              ? (seq_offsets_q[bidb + 1] - seq_offsets_q[bidb])
                              : (seq_offsets[bidb + 1] - seq_offsets[bidb])))) {
  }
};

template <bool Jagged, bool Cross, bool Has_targets>
struct SeqlenInfoQKFwd {
  int const offset_q, offset_k;
  int const seqlen_q, seqlen_kv, uihlen_q;

  CUTLASS_DEVICE
  SeqlenInfoQKFwd(
      int const bidb,
      int const max_q_len,
      int const max_kv_len,
      int const* const seq_offsets,
      int const* const seq_offsets_q,
      int const* const num_targets)
      : offset_q(
            !Jagged ? 0 : (Cross ? seq_offsets_q[bidb] : seq_offsets[bidb])),
        offset_k(!Jagged ? 0 : seq_offsets[bidb]),
        seqlen_q(
            !Jagged ? max_q_len
                    : (Cross ? (seq_offsets_q[bidb + 1] - seq_offsets_q[bidb])
                             : (seq_offsets[bidb + 1] - seq_offsets[bidb]))),
        seqlen_kv(
            !Jagged ? max_kv_len : (seq_offsets[bidb + 1] - seq_offsets[bidb])),
        uihlen_q(
            !Jagged
                ? (Has_targets ? max_q_len - num_targets[bidb] : max_q_len)
                : (Has_targets
                       ? (Cross ? (seq_offsets_q[bidb + 1] -
                                   seq_offsets_q[bidb] - num_targets[bidb])
                                : (seq_offsets[bidb + 1] - seq_offsets[bidb] -
                                   num_targets[bidb]))
                       : (Cross
                              ? (seq_offsets_q[bidb + 1] - seq_offsets_q[bidb])
                              : (seq_offsets[bidb + 1] - seq_offsets[bidb])))) {
  }
};

} // namespace hstu
