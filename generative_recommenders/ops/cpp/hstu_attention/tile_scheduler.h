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

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"

#include "named_barrier.h"

namespace hstu {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
  int const num_blocks, num_head, num_batch;
  int const max_seq_len, headdim,
      element_size; // Used to calculate L2 swizzling
  int* const tile_count_semaphore = nullptr;
  int* const seq_offsets = nullptr;
  int* const sort_by_length_indices = nullptr;
};

///////////////////////////////////////////////////////////////////////////////

template <
    bool Jagged = false,
    int kBlock = 128,
    bool Sort_by_length_indices = false>
class SingleTileScheduler {
 public:
  using SharedStorage = int;

  // Device side kernel params
  struct Params {
    int const num_blocks, num_head, num_batch;
    int const max_seq_len;
    int* const seq_offsets;
    int* const sort_by_length_indices;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    return {
        args.num_blocks,
        args.num_head,
        args.num_batch,
        args.max_seq_len,
        !Jagged ? nullptr : args.seq_offsets,
        !Sort_by_length_indices ? nullptr : args.sort_by_length_indices};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    std::printf(
        "SingleTileScheduler::get_grid_shape: %d, %d, %d\n",
        params.num_blocks,
        params.num_head,
        params.num_batch);
#endif
    return {
        uint32_t(params.num_blocks),
        uint32_t(params.num_head),
        uint32_t(params.num_batch)};
  }

  struct WorkTileInfo {
    int block_idx = 0;
    int bidh = 0;
    int bidb = 0;
    bool is_valid_tile = false;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return is_valid_tile;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      return {block_idx, bidh, bidb, 0 /*split_idx*/};
    }
  };

  CUTLASS_DEVICE
  SingleTileScheduler(SharedStorage* const smem_scheduler) {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    int bidb = int(blockIdx.z);
    if constexpr (Sort_by_length_indices) {
      bidb = params.sort_by_length_indices[bidb];
    }
    WorkTileInfo work_info{int(blockIdx.x), int(blockIdx.y), bidb, true};
    if constexpr (Jagged) {
      int seqlen =
          (params.seq_offsets ? params.seq_offsets[work_info.bidb + 1] -
                   params.seq_offsets[work_info.bidb]
                              : params.max_seq_len);
      work_info.is_valid_tile = work_info.block_idx * kBlock < seqlen;
    }
    return work_info;
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work)
      const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {-1, -1, -1, false};
  }
};

///////////////////////////////////////////////////////////////////////////////

class StaticPersistentTileScheduler {
 public:
  using SharedStorage = int;

  // Device side kernel params
  struct Params {
    int total_blocks;
    cutlass::FastDivmod m_block_divmod, head_divmod;
    cutlass::FastDivmod nsplits_divmod;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    return {
        args.num_blocks * args.num_head * args.num_batch,
        cutlass::FastDivmod(args.num_blocks),
        cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(1)};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    std::printf("StaticPersistentTileScheduler::get_grid_shape %d\n", num_sm);
#endif
    return {uint32_t(num_sm)};
  }

  struct WorkTileInfo {
    int tile_idx;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      int block, bidh, bidb;
      bidb = params.head_divmod.divmod(
          bidh, params.m_block_divmod.divmod(block, tile_idx));
      int split_idx = 0;
      return {block, bidh, bidb, split_idx};
    }
  };

  CUTLASS_DEVICE
  StaticPersistentTileScheduler(SharedStorage* const smem_scheduler){};

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    return {int(blockIdx.x)};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work)
      const {}

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};
  }
};

template <
    int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup,
    int NumProducerThreads = cutlass::NumThreadsPerWarp,
    bool WarpSpecialized = true>
class DynamicPersistentTileScheduler {
  // This scheduler targets the causal (or local) case where each tile takes
  // different amount of time. We use longest-processing-time-first scheduling:
  // the longest remaining tile is assigned to the first SM that's free.
  // SM indicates they are free by incrementing a semaphore.
  // However, we have to make sure K & V still fit into L2 cache, so we perform
  // scheduling on "sections" of the head & batch dimension, each section
  // consisting of e.g. 8 heads. This is the L2 swizzling part. The size of each
  // section is precomputed based on the size of K & V and the L2 cache size.

  static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
  static constexpr int NumThreads =
      WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

 public:
  using SharedStorage = int;

 protected:
  SharedStorage* const tile_count_smem;

 public:
  // Device side kernel params
  struct Params {
    int const total_blocks;
    cutlass::FastDivmod const m_block_divmod, head_divmod;
    cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
    cutlass::FastDivmod const l2_minor_residual_divmod;
    int const num_hb_quotient;
    int* const tile_count_semaphore;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    int const size_one_kv_head =
        args.max_seq_len * args.headdim * args.element_size * 2;
    int const size_l2 = 32 * 1024 * 1024; // 32 MB for K & V
    // Swizzle is the size of each "section". Round swizzle to a power of 2
    // If not PackGQA already, the size of each section can increase by
    // qhead_per_khead
    int const swizzle = (1 << cutlass::find_log2(size_l2 / size_one_kv_head));
    // If we're in the last section (called residual), we don't want to divide
    // by swizzle. Instead we want to divide by the remainder.
    int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
    int const num_split_blocks = args.num_blocks;
    // printf("num_split_blocks = %d, num_head = %d, num_batch = %d, swizzle =
    // %d, PackGQA = %d, qhead_per_khead = %d, num_hb_remainder = %d\n",
    // num_split_blocks, args.num_head, args.num_batch, swizzle, int(PackGQA),
    // args.qhead_per_khead, num_hb_remainder);
    assert(args.tile_count_semaphore != nullptr);
    return {
        num_split_blocks * args.num_head * args.num_batch,
        cutlass::FastDivmod(args.num_blocks),
        cutlass::FastDivmod(args.num_head),
        cutlass::FastDivmod(swizzle),
        cutlass::FastDivmod(swizzle * num_split_blocks),
        // don't divide by 0
        cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
        (args.num_head * args.num_batch) / swizzle,
        args.tile_count_semaphore};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    std::printf("DynamicPersistentTileScheduler::get_grid_shape %d\n", num_sm);
#endif
    return {uint32_t(num_sm)};
  }

  struct WorkTileInfo {
    int tile_idx;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      int block, bidh, bidb;
      int l2_mod, bidhb, bidhb_residual;
      bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
      // If we're in the last section (called residual), we don't want to divide
      // by swizzle. Instead we want to divide by the remainder.
      if (bidhb < params.num_hb_quotient) {
        block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
      } else {
        block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
      }
      bidb = params.head_divmod.divmod(
          bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
      int split_idx = 0;
      // Longest-processing-time-first
      block = params.m_block_divmod.divisor - 1 - block;
      return {block, bidh, bidb, split_idx};
    }
  };

  CUTLASS_DEVICE
  DynamicPersistentTileScheduler(SharedStorage* const smem_scheduler)
      : tile_count_smem(smem_scheduler){};

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    return {int(blockIdx.x)};
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    if (WarpSpecialized || cutlass::canonical_warp_idx_sync() > 0) {
      hstu::named_barrier_arrive(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
    }
  }

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work)
      const {
    if (threadIdx.x % NumProducerThreads == 0) {
      current_work.tile_idx =
          atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
    }
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      // thread 0 already has the right tile_idx, just need to broadcast to the
      // rest of warp 0
      int new_tile_idx =
          __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
      hstu::named_barrier_sync(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
      if (threadIdx.x % NumProducerThreads == 0) {
        *tile_count_smem = current_work.tile_idx;
      }
      hstu::named_barrier_arrive(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
      return {new_tile_idx};
    } else {
      hstu::named_barrier_sync(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
      int tile_idx = *tile_count_smem;
      hstu::named_barrier_arrive(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
      return {tile_idx};
    }
  }
};

template <
    int kBlock,
    int NumMmaThreads = 2 * cutlass::NumThreadsPerWarpGroup,
    int NumProducerThreads = cutlass::NumThreadsPerWarp,
    bool WarpSpecialized = true>
class VarlenDynamicPersistentTileScheduler {
  static_assert(WarpSpecialized || NumProducerThreads == NumMmaThreads);
  static constexpr int NumThreads =
      WarpSpecialized ? NumMmaThreads + NumProducerThreads : NumMmaThreads;

 public:
  using SharedStorage = int4;

 protected:
  SharedStorage* const work_info_smem;

 public:
  // Device side kernel params
  struct Params {
    int num_head, num_batch;
    int const max_seq_len;
    cutlass::FastDivmod nsplits_divmod;
    int* const tile_count_semaphore;
    int* const seq_offsets;
  };

  static Params to_underlying_arguments(TileSchedulerArguments const& args) {
    // If Split, for the purpose of scheduling, we pretend that instead there
    // are (args.num_splits * args.num_head) number of heads.
    assert(args.tile_count_semaphore != nullptr);
    return {
        args.num_head,
        args.num_batch,
        args.max_seq_len,
        cutlass::FastDivmod(1),
        args.tile_count_semaphore,
        args.seq_offsets};
  }

  static dim3 get_grid_shape(Params const& params, int num_sm) {
#ifdef HSTU_FLASH_ATTN_DEBUG_INFO
    std::printf(
        "VarlenDynamicPersistentTileScheduler::get_grid_shape %d\n", num_sm);
#endif
    return {uint32_t(num_sm)};
  }

  struct WorkTileInfo {
    int tile_idx, block, bidh, bidb;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const {
      // if (blockIdx.x >= 0 && (threadIdx.x == 128 || threadIdx.x == 0)) {
      // printf("blockIdx.x = %d, threadIdx.x = %d, checking valid, bidb = %d,
      // params.num_batch = %d\n", blockIdx.x, threadIdx.x, bidb,
      // params.num_batch); }
      return bidb < params.num_batch;
    }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      return {block, bidh, bidb, 0 /*split_idx*/};
    }
  };

  CUTLASS_DEVICE
  VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler)
      : work_info_smem(smem_scheduler){};

  CUTLASS_DEVICE
  WorkTileInfo tile_idx_to_work_tile(
      Params const& params,
      int next_tile_idx,
      WorkTileInfo const& current_work) const {
    auto prefix_sum = [](int val) {
      auto lane = threadIdx.x % cutlass::NumThreadsPerWarp;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 1; i < cutlass::NumThreadsPerWarp; i <<= 1) {
        int32_t partial_sum = __shfl_up_sync(0xffffffff, val, i);
        if (lane >= i) {
          val += partial_sum;
        }
      }
      return val;
    };

    auto get_num_m_blocks = [&](int bidb_start) {
      auto lane = threadIdx.x % cutlass::NumThreadsPerWarp;
      int seqlen;
      if (params.seq_offsets) {
        int cur_cu_seqlen = lane + bidb_start <= params.num_batch
            ? params.seq_offsets[lane + bidb_start]
            : 0;
        int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
        seqlen = next_cu_seqlen - cur_cu_seqlen;
      } else {
        seqlen = params.max_seq_len;
      }
      return lane + bidb_start < params.num_batch &&
              lane < cutlass::NumThreadsPerWarp - 1
          ? cute::ceil_div(seqlen, kBlock)
          : 0;
    };

    int num_m_blocks =
        get_num_m_blocks(current_work.bidb); // Different for each lane
    // Cumulative number of blocks for the next 31 batches
    int num_m_blocks_cumulative = prefix_sum(num_m_blocks);
    // Total number of blocks for the next 31 batches
    int m_blocks_in_group = __shfl_sync(
        0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
    int group_end_tile = current_work.tile_idx - current_work.block -
        current_work.bidh * __shfl_sync(0xffffffff, num_m_blocks, 0 /*lane*/) +
        m_blocks_in_group * params.num_head; // Same for all lanes
    int bidb = current_work.bidb;
    // if (blockIdx.x <= 9 && threadIdx.x == 0) {
    //     printf("Before while, blockIdx.x = %d, threadIdx.x = %d, bidb = %d,
    //     num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d,
    //     m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb,
    //     num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
    // }
    while (group_end_tile <= next_tile_idx) {
      bidb += cutlass::NumThreadsPerWarp - 1;
      if (bidb >= params.num_batch) {
        // if (blockIdx.x <= 9 && threadIdx.x == 0) {
        //     printf("Returning early, blockIdx.x = %d, threadIdx.x = %d, bidb
        //     = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d,
        //     m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb,
        //     num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
        // }
        return {next_tile_idx, 0, 0, params.num_batch};
      }
      num_m_blocks = get_num_m_blocks(bidb);
      num_m_blocks_cumulative = prefix_sum(num_m_blocks);
      m_blocks_in_group = __shfl_sync(
          0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
      group_end_tile += m_blocks_in_group * params.num_head;
      // if (blockIdx.x <= 9 && threadIdx.x == 0) {
      //     printf("Bottom of while, blockIdx.x = %d, threadIdx.x = %d, bidb =
      //     %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile = %d,
      //     m_blocks_in_group = %d\n", blockIdx.x, threadIdx.x, bidb,
      //     num_m_blocks, next_tile_idx, group_end_tile, m_blocks_in_group);
      // }
    }
    int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
    // The next problem to process is the first one that does not have ending
    // tile position that is greater than or equal to tile index.
    int batch_idx_in_group = __popc(__ballot_sync(
        0xffffffff,
        group_start_tile + num_m_blocks_cumulative * params.num_head <=
            next_tile_idx));
    bidb += batch_idx_in_group;
    num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
    int mh_block = next_tile_idx - group_start_tile -
        (batch_idx_in_group == 0 ? 0
                                 : __shfl_sync(
                                       0xffffffff,
                                       num_m_blocks_cumulative,
                                       batch_idx_in_group - 1)) *
            params.num_head;
    int bidh = mh_block / num_m_blocks;
    int block = mh_block - bidh * num_m_blocks;
    // if (blockIdx.x <= 9 && threadIdx.x == 0) {
    //     printf("blockIdx.x = %d, threadIdx.x = %d, batch_idx_in_group = %d,
    //     bidb = %d, num_m_blocks = %d, next_tile_idx = %d, group_end_tile =
    //     %d, m_blocks_in_group = %d, mh_block = %d, bidh = %d, block = %d\n",
    //     blockIdx.x, threadIdx.x, batch_idx_in_group, bidb, num_m_blocks,
    //     next_tile_idx, group_end_tile, m_blocks_in_group, mh_block, bidh,
    //     block);
    // }
    return {next_tile_idx, block, bidh, bidb};
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo get_initial_work(Params const& params) const {
    if constexpr (IsProducerWarp) {
      WorkTileInfo work_info =
          tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
      if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
        *work_info_smem = make_int4(
            work_info.tile_idx,
            work_info.block,
            work_info.bidh,
            work_info.bidb);
      }
      hstu::named_barrier_arrive(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
      return work_info;
    } else {
      return get_next_work<false>(params, {0, 0, 0, 0});
    }
  }

  CUTLASS_DEVICE
  void init_consumer() const {
    // Don't arrive at the TileCountSmemEmpty barrier here, because
    // get_initial_work will do that
  }

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work)
      const {
    if (threadIdx.x % NumProducerThreads == 0) {
      current_work.tile_idx =
          atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
    }
  }

  template <bool IsProducerWarp = false>
  CUTLASS_DEVICE WorkTileInfo
  get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    if constexpr (IsProducerWarp) {
      // thread 0 has the next tile_idx, just need to broadcast to the rest of
      // warp 0
      int new_tile_idx =
          __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
      WorkTileInfo work_info = {
          __shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/),
          current_work.block,
          current_work.bidh,
          current_work.bidb};
      work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
      hstu::named_barrier_sync(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
      if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
        *work_info_smem = make_int4(
            work_info.tile_idx,
            work_info.block,
            work_info.bidh,
            work_info.bidb);
      }
      hstu::named_barrier_arrive(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
      return work_info;
    } else {
      hstu::named_barrier_sync(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemFull) /*id*/);
      int4 work_info = *work_info_smem;
      hstu::named_barrier_arrive(
          NumThreads,
          static_cast<uint32_t>(FwdNamedBarriers::TileCountSmemEmpty) /*id*/);
      return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
    }
  }
};

} // namespace hstu
