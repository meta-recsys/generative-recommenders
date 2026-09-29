"""CuTeDSL jagged grouped GEMM with broadcast bias, for Blackwell (sm_100a).

    out[s:e] = jagged[s:e] @ dense[g] + bias[g]      s, e = seq_offsets[g], [g + 1]

The CuTeDSL backend for `generative_recommenders.ops.jagged_tensors
.jagged_dense_bmm_broadcast_add`, reached through `HammerKernel.CUTEDSL`. Forward only.

Blackwell-native: tcgen05 MMA with `cta_group=2` over a 2x1 cluster, TMA bulk-tensor
loads into swizzled shared memory, an fp32 accumulator in tensor memory, the bias folded
into the epilogue, and warp specialisation (4 epilogue + 1 MMA + 1 TMA warp, 192
threads). Persistent grid-stride loop with N-fast rasterisation.

Correctness gates live in generative_recommenders/ops/tests/jagged_tensors_test.py
(test_jagged_dense_bmm_broadcast_add_cutedsl and ..._cutedsl_shapes).

"""

import functools
import os
from dataclasses import dataclass

# pyrefly: ignore [missing-import]  # cuda-bindings ships no type stubs
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from generative_recommenders.ops.utils import is_sm100_plus


def _env_int(name: str, default: int) -> int:
    """Tuning knob overridable from the environment, for sweeps without editing code."""
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Cfg:
    # MMA tiler M, split across the CTA pair when two_cta: each CTA owns cta_tile_m rows
    # of C, so the epilogue always works in 128x256 regardless.
    mma_tile_m: int = _env_int("KA_MMA_TILE_M", 256)
    # tcgen05 cta_group=2: one MMA instruction drives a 2-CTA cluster.
    two_cta: bool = bool(_env_int("KA_TWO_CTA", 1))
    # 256. A is re-read once per N-tile.
    tile_n: int = _env_int("KA_TILE_N", 256)
    tile_k: int = _env_int("KA_TILE_K", 64)
    # Epilogue subtile width: the accumulator is drained epi_n columns at a time so the
    # live register fragment stays small. Draining the full tile_n=256 at once needs 256
    # fp32 + 256 bf16 per thread, which blows the 255-register budget and spills to local
    # memory. At epi_n=64 the kernel sits at 126 registers with zero spill.
    epi_n: int = _env_int("KA_EPI_N", 64)
    # 3 stages. This keeps dynamic smem at 98.4 KB
    stages: int = _env_int("KA_STAGES", 3)

    @property
    def cta_tile_m(self) -> int:
        """Rows of C this CTA owns. The epilogue works in CTA tiles, the MMA in mma_tile_m."""
        return self.mma_tile_m // (2 if self.two_cta else 1)

    @property
    def cluster_shape_mn(self):
        """cta_group=2 requires a cluster M that is a multiple of 2."""
        return (2, 1) if self.two_cta else (1, 1)


@functools.lru_cache(maxsize=1)
def get_cfg() -> Cfg:
    """Build the tuning config on first use, not at import.

    The Cfg field defaults call _env_int, so constructing it at module scope would read
    os.environ as an import side effect, which the Python style guide forbids.
    """
    return Cfg()


# Warp specialisation, following the upstream CUTLASS Blackwell dense_gemm_persistent example.
# Three roles:
#   0-3  epilogue -- tmem -> rmem, bias, store. 128 threads, which is what makes the
#                    t2r partitioning line up one row of the tile per thread.
#   4    mma      -- issues tcgen05 MMA only.
#   5    tma      -- issues TMA loads only.
# Splitting TMA issue from MMA issue is what removes the need for explicit prefetching:
# the load warp runs ahead on its own, which that example calls out as the point of
# the design.
EPILOGUE_WARP_ID = (0, 1, 2, 3)
MMA_WARP_ID = 4
TMA_WARP_ID = 5
THREADS = 32 * (len(EPILOGUE_WARP_ID) + 2)  # 192
EPILOGUE_THREADS = 32 * len(EPILOGUE_WARP_ID)  # 128

# NamedBarrier ids. 0 collides with __syncthreads(); 1 is the reference's epilogue sync.
TMEM_ALLOC_BAR_ID = 2
TMEM_DEALLOC_BAR_ID = 3
# Epilogue r2s/TMA handshake, over the epilogue warps only.
EPI_SYNC_BAR_ID = 4
# The retrieve barrier must span exactly the warps that read the tmem pointer -- the MMA
# warp and the epilogue warps. The TMA warp never touches tmem and must NOT arrive here;
# a wrong count deadlocks rather than errors.
TMEM_ALLOC_BAR_THREADS = 32 * (len(EPILOGUE_WARP_ID) + 1)  # 160

# Two accumulator buffers, so the epilogue of tile n overlaps the MMA of tile n+1.
# One measures slower: the MMA then waits on the epilogue for every tile.
NUM_ACC_STAGE = 2


def _ab_stages(cfg: Cfg, ab_dtype, occupancy: int = 1) -> int:
    """A/B pipeline depth that fits the smem budget, as the upstream CUTLASS Blackwell dense_gemm_persistent
    example sizes it.
    """
    if cfg.stages:
        return cfg.stages
    per_stage = cfg.tile_k * (cfg.cta_tile_m + cfg.tile_n) * (ab_dtype.width // 8)
    mbar_helpers_bytes = 1024
    cap = cutlass.utils.get_smem_capacity_in_bytes()
    return max(2, (cap // occupancy - mbar_helpers_bytes) // per_stage)


class JaggedDenseBmmBroadcastAdd:
    """The compiled kernel object, named to match the public op deliberately.

    CuTeDSL bakes the class name into the compiled kernel symbol, so this is the name a
    profile shows: `kernel_cutlass_kernel_<module>JaggedDenseBmmBroadcastAdd_object_at_...`.
    `cutlass` in that prefix already identifies the backend, so the name does not repeat
    it.
    """

    def __init__(self, ab_dtype, acc_dtype, cfg: Cfg):
        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.cfg = cfg
        self.cta_group = tcgen05.CtaGroup.TWO if cfg.two_cta else tcgen05.CtaGroup.ONE
        # Major mode is about which mode is CONTIGUOUS IN MEMORY, not which one we
        # contract over.
        #   A: jagged (G*M, K) contiguous -> (M, K, G) has K at stride 1  -> K-major.
        #   B: dense  (G, K, N) contiguous -> (N, K, G) has N at stride 1 -> MN-major.
        # Declaring B K-major builds a TMA descriptor whose box walks the wrong axis;
        # most of it lands out of bounds, TMA zero-fills, and the result comes back
        # about 25x too small rather than obviously garbage.
        self.a_major = tcgen05.OperandMajorMode.K
        self.b_major = tcgen05.OperandMajorMode.MN

    @cute.jit
    def __call__(
        self,
        mA,
        mB,
        mBias,
        mC,
        mSeq,
        max_seq_len: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major,
            self.b_major,
            self.acc_dtype,
            self.cta_group,
            (cfg.mma_tile_m, cfg.tile_n),
        )
        mma_tiler = (cfg.mma_tile_m, cfg.tile_n, cfg.tile_k)
        # With two_cta the cluster is 2x1 and thr_id.shape is 2, so the v mode absorbs
        # the pair: vmnk = (2,1,1,1) and there is no A/B multicast beyond the pair itself.
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*cfg.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        # Two accumulators so the epilogue can drain tile n while the MMA fills n+1.
        # 128 lanes x 256 cols x 2 stages = 512 tmem columns -- the ENTIRE tmem.
        acc_shape = cute.append(
            tiled_mma.partition_shape_C((cfg.mma_tile_m, cfg.tile_n)), NUM_ACC_STAGE
        )
        tmem_cols = cutlass.utils.get_num_tmem_alloc_cols(
            tiled_mma.make_fragment_C(acc_shape)
        )

        stages = _ab_stages(cfg, self.ab_dtype)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, self.ab_dtype, stages
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, self.ab_dtype, stages
        )

        # Pick the TMA op via the helper: it returns CopyBulkTensorTileG2SOp(CtaGroup),
        # and the CtaGroup argument is what makes the atom tcgen05-compatible. A bare
        # CopyBulkTensorTileG2SOp() builds a different variant that tma_partition rejects.
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cfg.cluster_shape_mn, tiled_mma.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            cfg.cluster_shape_mn, tiled_mma.thr_id
        )
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            cute.slice_(a_smem_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            cute.slice_(b_smem_layout, (None, None, None, 0)),
            mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        n = cute.size(mC, mode=[1])
        g = cute.size(mB, mode=[2])
        # m_tiles is sized by max_seq_len (the PADDED bound), matching Triton
        # (triton_jagged.py:347 `if start_m >= seq_len: return`). Tiles
        # past a group's real length are skipped at runtime.
        #
        n_tiles = cute.ceil_div(n, cfg.tile_n)
        # pyrefly: ignore [unsupported-operation]  # max_seq_len is a
        # cutlass.Constexpr, which erases to Unknown for the type checker
        m_tiles = (max_seq_len + cfg.mma_tile_m - 1) // cfg.mma_tile_m
        total_clusters = m_tiles * n_tiles * g
        # pyrefly: ignore [bad-specialization]  # Constexpr has no ordering bound
        n_persist = min(max_active_clusters, total_clusters)
        grid = (cfg.cluster_shape_mn[0] * n_persist, 1, 1)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            mBias,
            mC,
            mSeq,
            a_smem_layout,
            b_smem_layout,
            acc_shape,
            tmem_cols,
            cluster_layout_vmnk,
            stages,
            n_tiles,
            m_tiles,
            total_clusters,
        ).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            cluster=(*cfg.cluster_shape_mn, 1),
            stream=stream,
        )

    # noqa C901: the complexity is three warp-role blocks (TMA / MMA / epilogue) in
    # one @cute.kernel. They cannot be split into helpers -- CuTeDSL rejects closures
    # that capture variables inside dynamic control flow, which is why the per-tile
    # decode is inlined three times rather than factored out.
    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tiled_mma,
        tma_atom_a,
        mA,
        tma_atom_b,
        mB,
        mBias,
        mC,
        mSeq,
        a_smem_layout,
        b_smem_layout,
        acc_shape: cutlass.Constexpr,
        tmem_cols: cutlass.Constexpr,
        cluster_layout_vmnk,
        stages: cutlass.Constexpr,
        n_tiles,
        m_tiles,
        total_clusters,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, by, bz = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        # Pass layout and swizzle SEPARATELY (.outer / .inner), not the composed layout.
        # Both spell the same tensor, but they build different memref types:
        #   split:    !cute.memref<bf16, smem, align<1024>, S<3,4,3>, "((128,16),1,4,2):...">
        #   composed: !cute.memref<bf16, smem, align<1024>, "S<3,4,3> o 0 o ((128,16),...)">
        # tma_partition only matches the first; the second fails its operand check with
        # "unable to partition input tensors for TMA". Alignment is not the issue -- the
        # allocator rounds both up to 1024 anyway.
        sA = smem.allocate_tensor(
            self.ab_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner
        )
        sB = smem.allocate_tensor(
            self.ab_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        # Two pipelines, both from cutlass.pipeline rather than raw mbarriers. The
        # hand-rolled version this replaces had a single "buffer full" barrier with the
        # phase bit hardcoded to 0, which is only correct on each stage's first use.
        #   ab  : TMA (producer) -> tcgen05 MMA (consumer), 2*stages mbarriers
        #         (full + empty -- the empty half is what stops the TMA for k+stages
        #         from overwriting smem the MMA for k is still reading)
        #   acc : tcgen05 MMA (producer) -> epilogue (consumer). The MMA writes tmem
        #         asynchronously, so the epilogue needs a real handoff, not barrier().
        # pyrefly: ignore [unsupported-operation]  # stages is a Constexpr
        ab_mbar = smem.allocate_array(cutlass.Int64, 2 * stages)
        acc_mbar = smem.allocate_array(cutlass.Int64, 2 * NUM_ACC_STAGE)
        tmem_holding_buf = smem.allocate_array(cutlass.Int32, 1)
        # Freeing tmem in a 2-CTA pair needs a cross-CTA mbarrier, not just a NamedBarrier.
        tmem_dealloc_mbar = smem.allocate_array(cutlass.Int64, 1)

        # One thread issues the TMA; one arrival consumes each full barrier.
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        tx_count = atom_thr_size * (
            cute.size_in_bytes(
                self.ab_dtype, cute.slice_(a_smem_layout, (None, None, None, 0))
            )
            + cute.size_in_bytes(
                self.ab_dtype, cute.slice_(b_smem_layout, (None, None, None, 0))
            )
        )
        ab_producer, ab_consumer = cutlass.pipeline.PipelineTmaUmma.create(
            barrier_storage=ab_mbar,
            num_stages=stages,
            producer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            ),
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread, 1
            ),
            tx_count=tx_count,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=cfg.two_cta,
        ).make_participants()

        # Consumer is the epilogue warps only -- 128 threads, not all 192. The TMA and
        # MMA warps never read the accumulator and must not arrive on this barrier.
        acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_mbar,
            num_stages=NUM_ACC_STAGE,
            producer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            ),
            # NOT EPILOGUE_THREADS. This counts ARRIVALS, and each epilogue warp makes
            # one (elect_one), across both CTAs of the pair: 4 warps x 2 CTAs = 8
            # (same as the upstream CUTLASS Blackwell dense_gemm_persistent example).
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                len(EPILOGUE_WARP_ID) * (2 if cfg.two_cta else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=cfg.two_cta,
        )

        # Accumulator lives in tensor memory. With specialised warps the allocating warp
        # has to be named, as in the upstream CUTLASS Blackwell dense_gemm_persistent example: epilogue warp 0 allocates, and
        # the retrieve barrier publishes the pointer to the MMA warp as well.
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tmem = cutlass.utils.TmemAllocator(
            tmem_holding_buf,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=TMEM_ALLOC_BAR_ID, num_threads=TMEM_ALLOC_BAR_THREADS
            ),
            allocator_warp_id=EPILOGUE_WARP_ID[0],
            is_two_cta=cfg.two_cta,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar,
        )
        tmem_dealloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR_ID, num_threads=EPILOGUE_THREADS
        )

        # Cluster-wide handshake around barrier init, as the upstream CUTLASS Blackwell dense_gemm example does. A plain
        # sync_threads() only covers this CTA; the partner could reach a barrier before
        # we have initialised it.
        if cutlass.const_expr(cfg.two_cta):
            cutlass.pipeline.pipeline_init_arrive(
                cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
            )

        # B is loop-invariant: it is indexed by group through a real stride, so it needs
        # no per-tile offset. A and C DO, so their tiling moves inside the loop below.
        mma_tiler = (cfg.mma_tile_m, cfg.tile_n, cfg.tile_k)
        gB_nkl = cute.local_tile(
            mB, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
        )

        # Which half of the MMA pair this CTA is. With cta_group=1 thr_id.shape is 1, so
        # v is always 0 and every CTA is a leader -- the 1-CTA path is unchanged.
        mma_tile_coord_v = bx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cluster_id = bx // cute.size(tiled_mma.thr_id.shape)
        n_clusters = cute.arch.grid_dim()[0] // cute.size(tiled_mma.thr_id.shape)

        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        blk_in_cluster_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgB = thr_mma.partition_B(gB_nkl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            blk_in_cluster_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # A 2-CTA MMA reads operands staged by BOTH CTAs, so each TMA must arrive on the
        # partner's barrier too. Required whenever cta_group=2, even with no A/B mcast.
        a_mcast_mask = None
        b_mcast_mask = None
        if cutlass.const_expr(cfg.two_cta):
            a_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, blk_in_cluster_vmnk, mcast_mode=2
            )
            b_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, blk_in_cluster_vmnk, mcast_mode=1
            )

        # SMEM-side MMA fragments. These are loop-invariant: the jagged offset only
        # affects GMEM addressing, and by the time data reaches SMEM it is group-local.
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        k_tiles = cute.size(gB_nkl, mode=[3])  # same K as A
        num_kblks = cute.size(tCrA, mode=[2])

        # Must precede every warp block: the TMA and MMA warps touch cluster barriers
        # immediately, so waiting only before the epilogue lets them run ahead of the
        # partner CTA's barrier init.
        if cutlass.const_expr(cfg.two_cta):
            cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # TMA warp -- issues loads and nothing else.
        #
        # No prefetch prologue, a dedicated load warp runs
        # ahead structurally, bounded only by `stages` empty buffers.
        #
        if warp_idx == TMA_WARP_ID:
            for t in cutlass.range(cluster_id, total_clusters, n_clusters, unroll=1):
                # Decode one linear tile index. This block is IDENTICAL in all three
                # warp roles -- if one warp skips a tile another does not, the A/B
                # pipeline desyncs and hangs.
                n_tile = t % n_tiles
                rest = t // n_tiles
                mtile = rest % m_tiles
                bz = rest // m_tiles
                seq_start = mSeq[bz]
                seq_len = mSeq[bz + 1] - seq_start
                # Tiles past this group's length -- and every tile of an EMPTY group.
                active = mtile * cfg.mma_tile_m < seq_len
                if active:
                    # A's TMA view shifted to this group's first row. domain_offset must
                    # be applied to the rank-3 (M,K,L) view BEFORE local_tile.
                    # Alignment holds: K is 16-divisible, so seq_start*K*2B is a multiple
                    # of 32 B however ragged the lengths are.
                    mA_g = cute.domain_offset((seq_start, 0, 0), mA)
                    gA_mkl = cute.local_tile(
                        mA_g,
                        cute.slice_(mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_a,
                        blk_in_cluster_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3),
                    )
                    # group mode is a singleton on the flat A, so index 0
                    gA = tAgA[(None, mtile, None, 0)]
                    gB = tBgB[(None, n_tile, None, bz)]

                    ab_producer.reset()
                    for k in cutlass.range(k_tiles, unroll=1):
                        producer_handle = ab_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_a,
                            gA[(None, k)],
                            tAsA[(None, producer_handle.index)],
                            tma_bar_ptr=producer_handle.barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            gB[(None, k)],
                            tBsB[(None, producer_handle.index)],
                            tma_bar_ptr=producer_handle.barrier,
                            mcast_mask=b_mcast_mask,
                        )
            ab_producer.tail()

        #
        # MMA warp -- issues tcgen05 MMA and nothing else.
        #
        if warp_idx == MMA_WARP_ID:
            # NOT guarded by is_leader_cta: this barrier spans mma + epilogue on EVERY
            # CTA (160 threads).
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Only the leader ISSUES the MMA: one tcgen05 instruction drives the pair
            # and writes both CTAs' tmem.
            if is_leader_cta:
                acc_producer_state = cutlass.pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Producer, NUM_ACC_STAGE
                )
                for t in cutlass.range(
                    cluster_id, total_clusters, n_clusters, unroll=1
                ):
                    # Decode one linear tile index. This block is IDENTICAL in all three
                    # warp roles -- if one warp skips a tile another does not, the A/B
                    # pipeline desyncs and hangs.
                    n_tile = t % n_tiles
                    rest = t // n_tiles
                    mtile = rest % m_tiles
                    bz = rest // m_tiles
                    seq_start = mSeq[bz]
                    seq_len = mSeq[bz + 1] - seq_start
                    # Tiles past this group's length -- and every tile of an EMPTY group.
                    active = mtile * cfg.mma_tile_m < seq_len
                    # (the MMA warp reads A/B from SMEM, so it needs no jagged offset)
                    if active:
                        tCtAcc = tCtAcc_base[
                            (None, None, None, acc_producer_state.index)
                        ]
                        ab_consumer.reset()
                        acc_pipeline.producer_acquire(acc_producer_state)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for _k in cutlass.range(k_tiles):
                            consumer_handle = ab_consumer.wait_and_advance()
                            for kb in cutlass.range_constexpr(num_kblks):
                                crd = (None, None, kb, consumer_handle.index)
                                cute.gemm(
                                    tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            consumer_handle.release()

                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()

                acc_pipeline.producer_tail(acc_producer_state)

        #
        # Epilogue warps -- tmem -> registers, broadcast bias, vectorised store with a
        # per-element predicated fallback for partial-N tiles.
        #
        if warp_idx < MMA_WARP_ID:
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, NUM_ACC_STAGE
            )
            # All 128 epilogue threads must finish reading tmem before ONE THREAD PER WARP
            # releases the accumulator: the consumer group is 4 warps x 2 CTAs = 8
            # arrivals, so an unguarded release from 128 threads over-completes the
            # barrier and faults the launch.
            epi_sync = cutlass.pipeline.NamedBarrier(
                barrier_id=EPI_SYNC_BAR_ID, num_threads=EPILOGUE_THREADS
            )

            for t in cutlass.range(cluster_id, total_clusters, n_clusters, unroll=1):
                n_tile = t % n_tiles
                rest = t // n_tiles
                mtile = rest % m_tiles
                bz = rest // m_tiles
                seq_start = mSeq[bz]
                seq_len = mSeq[bz + 1] - seq_start
                active = mtile * cfg.mma_tile_m < seq_len
                if active:
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    epi_tile = (cfg.cta_tile_m, cfg.epi_n)
                    tmem_load = sm100_utils.get_tmem_load_op(
                        (cfg.cta_tile_m, cfg.tile_n, cfg.tile_k),
                        cutlass.utils.LayoutEnum.ROW_MAJOR,
                        self.ab_dtype,
                        self.acc_dtype,
                        epi_tile,
                        cfg.two_cta,
                    )
                    tAcc_epi = cute.flat_divide(
                        tCtAcc_base[((None, None), 0, 0, acc_consumer_state.index)],
                        epi_tile,
                    )
                    tiled_t2r = tcgen05.make_tmem_copy(
                        tmem_load, tAcc_epi[(None, None, 0, 0)]
                    )
                    thr_t2r = tiled_t2r.get_slice(tidx)
                    tAcc = cute.group_modes(thr_t2r.partition_S(tAcc_epi), 3, 5)

                    # C shifted to this group's first row, same as A.
                    mC_g = cute.domain_offset((seq_start, 0, 0), mC)
                    gC_mnl = cute.local_tile(
                        mC_g,
                        cute.slice_(mma_tiler, (None, None, 0)),
                        (None, None, None),
                    )
                    tCgC = thr_mma.partition_C(gC_mnl)
                    gC = tCgC[((None, None), 0, 0, mtile, n_tile, 0)]

                    gBias = cute.local_tile(mBias, (cfg.tile_n,), (n_tile, bz))
                    biasTile = cute.make_tensor(
                        gBias.iterator,
                        cute.make_layout((cfg.cta_tile_m, cfg.tile_n), stride=(0, 1)),
                    )
                    cC = cute.make_identity_tensor((cfg.cta_tile_m, cfg.tile_n))

                    tC = cute.group_modes(
                        thr_t2r.partition_D(cute.flat_divide(gC, epi_tile)), 3, 5
                    )
                    tCoord = cute.group_modes(
                        thr_t2r.partition_D(cute.flat_divide(cC, epi_tile)), 3, 5
                    )
                    tBias = cute.group_modes(
                        thr_t2r.partition_D(cute.flat_divide(biasTile, epi_tile)), 3, 5
                    )

                    frag_shape = tC[(None, None, None, 0)].shape
                    rAcc = cute.make_fragment(frag_shape, self.acc_dtype)
                    rBias = cute.make_fragment(frag_shape, self.ab_dtype)
                    rC = cute.make_rmem_tensor(frag_shape, self.ab_dtype)
                    simt_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), self.ab_dtype
                    )

                    _probe = tCoord[(None, None, None, 0)]
                    _modes = [
                        getattr(_sb, "mode", None)
                        for _sb in cute.flatten(_probe.layout.stride)
                    ]
                    assert all(m == [1] for m in _modes if m is not None), (
                        f"epilogue fragment is not row-contiguous (stride modes {_modes}); "
                        "the hoisted M predicate below would store wrong rows"
                    )

                    m_base = mtile * cfg.mma_tile_m + mma_tile_coord_v * cfg.cta_tile_m
                    m_max = seq_len - m_base
                    n_max = cute.size(mC, mode=[1]) - n_tile * cfg.tile_n
                    last = cute.size(rAcc) - 1
                    for sub in cutlass.range_constexpr(cute.size(tAcc, mode=[3])):
                        cute.copy(tiled_t2r, tAcc[(None, None, None, sub)], rAcc)
                        cute.copy(simt_atom, tBias[(None, None, None, sub)], rBias)
                        rAcc.store(rAcc.load() + rBias.load().to(self.acc_dtype))
                        rC.store(rAcc.load().to(self.ab_dtype))
                        tC_s = tC[(None, None, None, sub)]
                        tCo_s = tCoord[(None, None, None, sub)]
                        # Two whole-fragment tests instead of 2 per element: the row (M is
                        # constant, so element 0 speaks for all of them) and the far end of
                        # the column run (they are contiguous, so the last one in bounds
                        # means all of them are).
                        mi0, _ = tCo_s[0]
                        _, ni_last = tCo_s[last]
                        if mi0 < m_max and ni_last < n_max:
                            cute.autovec_copy(rC, tC_s)
                        else:
                            # Cold: the ragged M tail, or an N tile that C does not fill
                            # (N not a multiple of epi_n). Per-element, as before.
                            for i in cutlass.range_constexpr(cute.size(rAcc)):
                                mi, ni = tCo_s[i]
                                if mi < m_max and ni < n_max:
                                    tC_s[i] = rC[i]

                    epi_sync.arrive_and_wait()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


_COMPILED = {}
_MAX_CLUSTERS: dict = {}


def _max_active_clusters() -> int:
    """CTAs the machine holds at once -- this sizes the persistent grid.

    Cached because the query is a driver call on the hot path, but keyed PER DEVICE: it
    is a device property, and a single global would size the grid for whichever GPU
    happened to be current first on a multi-GPU host.
    """
    d = torch.cuda.current_device()
    if d not in _MAX_CLUSTERS:
        _MAX_CLUSTERS[d] = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    return _MAX_CLUSTERS[d]


def _stream():
    """The CALLER's current stream, resolved per call -- never cached.

    Caching the first stream seen per device meant that a caller running under a
    non-default stream (torch.cuda.stream(...), a CUDA graph capture, or any library that
    switches streams) had the kernel launched on a stale stream instead, silently breaking
    stream ordering against their other work.
    """
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _cutedsl_jagged_dense_bmm_broadcast_add_fwd(
    seq_offsets, jagged, dense, bias, max_seq_len: int
):
    """Jagged: group g owns rows [seq_offsets[g], seq_offsets[g+1]) of `jagged`.

    A and C are FLAT (sum_L, K) / (sum_L, N) with a singleton group mode. The group's
    row offset is applied on device with cute.domain_offset, which shifts the TMA
    coordinate origin -- see the comment at the offset site in `kernel`.
    """
    assert is_sm100_plus(), (
        "cutedsl_jagged_dense_bmm_broadcast_add requires a Blackwell datacenter GPU "
        "(sm_100-sm_103); it uses tcgen05 MMA with cta_group=2"
    )
    assert jagged.is_cuda and jagged.dim() == 2
    g, k, n = dense.shape[0], dense.shape[1], dense.shape[2]
    total = jagged.shape[0]
    assert seq_offsets.numel() == g + 1, "seq_offsets must be (G+1,)"
    # Assert the documented contract at the call site. Without these, fp16 fails deep
    # inside cute.compile (which pins cutlass.BFloat16) and a non-16-divisible N/K
    # produces misaligned TMA loads.
    assert jagged.dtype == torch.bfloat16, (
        f"CUTEDSL jagged_dense_bmm_broadcast_add is bf16-only, got {jagged.dtype}"
    )
    assert dense.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16, (
        f"dense and bias must be bf16 too, got dense={dense.dtype} bias={bias.dtype}"
    )
    assert tuple(bias.shape) == (g, n), (
        f"bias must be (G, N) = ({g}, {n}), got {tuple(bias.shape)}"
    )
    # Contiguity is a HARD requirement. from_dlpack bakes each tensor's
    # strides into the compiled kernel, and only mA/mC mark mode 0 dynamic -- mB and mBias
    # are fully static. The compile cache key carries dtypes and sizes but NOT strides.
    assert jagged.is_contiguous(), "jagged must be contiguous"
    assert dense.is_contiguous(), (
        "dense must be contiguous: its strides are baked into the compiled kernel and "
        "the compile cache is not keyed on layout"
    )
    assert bias.is_contiguous(), "bias must be contiguous (same reason as dense)"
    assert k % 16 == 0 and n % 16 == 0, (
        f"K and N must be 16-divisible (the shipped contract), got K={k} N={n}"
    )

    out = torch.empty((total, n), dtype=jagged.dtype, device=jagged.device)

    jagged = jagged.detach()
    dense = dense.detach()
    bias = bias.detach()

    a3 = jagged.unsqueeze(-1)  # (sum_L, K, 1)
    c3 = out.unsqueeze(-1)  # (sum_L, N, 1)
    b3 = dense.permute(2, 1, 0)  # (N, K, G)
    bias2 = bias.permute(1, 0)  # (N, G)      unchanged

    mA = from_dlpack(a3, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1, 2)
    )
    mB = from_dlpack(b3, assumed_align=16)
    mC = from_dlpack(c3, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1, 2)
    )
    mBias = from_dlpack(bias2, assumed_align=16)
    mSeq = from_dlpack(seq_offsets, assumed_align=8)

    key = (
        torch.cuda.current_device(),
        n,
        k,
        g,
        int(max_seq_len),
        jagged.dtype,
        seq_offsets.dtype,
    )
    if key not in _COMPILED:
        op = JaggedDenseBmmBroadcastAdd(cutlass.BFloat16, cutlass.Float32, get_cfg())
        _COMPILED[key] = cute.compile(
            op,
            mA,
            mB,
            mBias,
            mC,
            mSeq,
            int(max_seq_len),
            _max_active_clusters(),
            _stream(),
        )
    _COMPILED[key](mA, mB, mBias, mC, mSeq, _stream())
    return out


class _CuTeDSLJaggedDenseBmmBroadcastAddFunction(torch.autograd.Function):
    """Forward-only autograd wrapper.

    The kernel detaches its inputs so cute's from_dlpack will accept them.
    Routing through an autograd.Function keeps the output connected
    and turns the missing backward into a loud failure at .backward() instead.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
        bias: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        return _cutedsl_jagged_dense_bmm_broadcast_add_fwd(
            seq_offsets, jagged, dense, bias, max_seq_len
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, d_out: torch.Tensor):
        raise NotImplementedError(
            "The CuTeDSL jagged_dense_bmm_broadcast_add backend is forward-only. "
            "Backward needs two further kernels (dJagged, and dDense/dBias reduced over "
            "the jagged dimension). Use HammerKernel.TRITON or HammerKernel.PYTORCH for "
            "training."
        )


def cutedsl_jagged_dense_bmm_broadcast_add(
    seq_offsets, jagged, dense, bias, max_seq_len: int
):
    """Blackwell CuTeDSL jagged_dense_bmm_broadcast_add. Forward only.

    Public entry point for the HammerKernel.CUTEDSL dispatch. Forward works in a
    grad-enabled context; calling .backward() on the result raises NotImplementedError.
    """
    return _CuTeDSLJaggedDenseBmmBroadcastAddFunction.apply(
        seq_offsets, jagged, dense, bias, max_seq_len
    )
