"""CuTeDSL jagged grouped GEMM with broadcast bias, for Blackwell (sm_100a).

    out[s:e] = jagged[s:e] @ dense[g] + bias[g]      s, e = seq_offsets[g], [g + 1]

The CuTeDSL backend for `generative_recommenders.ops.jagged_tensors
.jagged_dense_bmm_broadcast_add`, reached through `HammerKernel.CUTEDSL`. Forward only.

Blackwell-native: tcgen05 MMA with `cta_group=2` over a 2x1 cluster, TMA bulk-tensor
loads into swizzled shared memory, an fp32 accumulator in tensor memory, the bias folded
into the epilogue, and warp specialisation (4 epilogue + 1 MMA + 1 TMA warp, 192
threads). Persistent grid-stride loop with N-fast rasterisation.

    2026-09-23, B200, production shape (G=40, sum_M=841,815, K=2176, N=768,
    max_seq_len=32,743): 4.53-4.94 ms / 569-621 TFLOP/s against a 5.40 ms production
    Triton bar, i.e. 0.84-0.92x. Output is bit-identical to Triton. Compute-bound at
    84.7% SOL, 126 registers/thread, no spill.

    That range is the honest one: 8 invocations on an idle GPU spanned 4.53-4.94 ms while
    Triton measured 5.40 EVERY time. The asymmetry is structural, not measurement error --
    this kernel is persistent and runs a single wave sized to max_active_clusters, so it
    takes whatever SMs happen to be free, where Triton's 6,144-CTA / ~21-wave grid
    self-averages. Quote the range, not the best run, and time both arms inside a SINGLE
    invocation; cross-invocation comparisons here are noise.

    Correctness gates live in generative_recommenders/ops/tests/jagged_tensors_test.py
    (test_jagged_dense_bmm_broadcast_add_cutedsl and ..._cutedsl_shapes).

Design notes that are load-bearing, in the order they cost the most to learn:

  * `tile_n = 256`, not 128. A is re-read once per N-tile, so with N=768 this is 3 passes
    over the 3.4 GB A matrix rather than 6. Largest single win in the whole kernel.
  * Warp specialisation is what feeds the async units; the win is not occupancy.
    Configurations that fit 2 CTAs/SM measure SLOWER, because A/B pipeline depth matters
    more than resident warps. Do not trade `stages` away for occupancy without measuring.
  * N-fast rasterisation, hand-rolled rather than the stock scheduler, which rasterises
    M-major. Pure scheduling: consecutive tiles share the same A rows, so A is fetched
    about once instead of once per N-tile. Measured L2 hit rate 66.8%. The ~15% this was
    worth was measured on the pre-jagged version of the kernel.
  * The epilogue is sub-tiled `epi_n` columns at a time. Draining all of `tile_n` at once
    needs 256 fp32 + 256 bf16 per thread, which blows the 255-register budget and spills.
  * The epilogue store is VECTORISED, not per element. Each thread owns one contiguous
    row of the tile, so the M predicate is loop-invariant and hoists out and the columns
    go out as `STG.E.128`. Asserted from the static layout stride at trace time; see the
    comment at the store. The per-element path it replaced cost 2.86x.
  * `pipeline_init_wait` must precede every warp block. Without it a CTA can use the A/B
    mbarriers before its cluster partner has initialised them, which deadlocks the launch
    nondeterministically.

Comments below cite the upstream CUTLASS Blackwell `dense_gemm` and
`dense_gemm_persistent` examples as prior art. Those ship with the CUTLASS source tree,
NOT with the nvidia-cutlass-dsl wheel, so they are not resolvable from fbsource.

Stream-K was considered and rejected. It fixes wave quantisation, and this kernel is
persistent: NCU reports launch__waves_per_multiprocessor = 1, so there is no partial
final wave to recover.
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
    # tcgen05 cta_group=2: one MMA instruction drives a 2-CTA cluster, and it is what
    # makes the 256-wide N tile reachable at all. Worth 6-7% on top of the tile change.
    two_cta: bool = bool(_env_int("KA_TWO_CTA", 1))
    # 256, not 128. A is re-read once per N-tile: at the production N=768 that is 3 passes
    # over the 3.4 GB A matrix instead of 6, and it is the single largest win in the
    # kernel. 256 is the tcgen05 maximum; 192 lands in between, as the traffic model
    # predicts.
    tile_n: int = _env_int("KA_TILE_N", 256)
    tile_k: int = _env_int("KA_TILE_K", 64)
    # Epilogue subtile width: the accumulator is drained epi_n columns at a time so the
    # live register fragment stays small. Draining the full tile_n=256 at once needs 256
    # fp32 + 256 bf16 per thread, which blows the 255-register budget and spills to local
    # memory. At epi_n=64 the kernel sits at 126 registers with zero spill (verified by
    # NCU: launch__registers_per_thread=126, derived__local_spilling_requests=0).
    # 32/64/128 all measured the same; only the un-subtiled 256 regresses.
    epi_n: int = _env_int("KA_EPI_N", 64)
    # 3, not auto. This keeps dynamic smem at 98.4 KB (NCU:
    # launch__shared_mem_per_block_dynamic), where the auto-sized 4 stages needs ~132 KB.
    # 0 = size from the smem budget instead.
    #
    # NOTE: the 2/3/4/5 sweep behind this default was run on the earlier uniform-M,
    # TMA-store, non-persistent version of this kernel. The reasoning still holds but the
    # optimum has not been re-swept since the jagged rewrite and the vectorised epilogue.
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

    A sweep once found 3/4/5 all within noise, but that was with one warp doing both
    TMA and MMA issue, where depth could not help. With a separate load warp, depth is
    exactly what lets it run ahead, so size it properly: at 32 KB per stage and ~228 KB
    of smem this gives ~7 rather than 3.
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

        # NO TMA STORE on the jagged path. C is flat, so rows past this group's seq_end
        # belong to the NEXT group and are written by a different CTA; an unpredicated
        # tile store would corrupt a neighbour, and TMA cannot express a per-group extent
        # without a tensormap update. The predicated scalar store is also free here: the
        # epilogue is fully hidden behind the mainloop (E/M = 0.134, acc EMPTY = 33
        # samples), so making it slower costs no wall time.
        # G comes from B, NOT from C: on the jagged path C is flat (sum_L, N, 1) and its
        # mode-2 is the SINGLETON group mode, so reading g from it yields 1 and
        # total_clusters covers only group 0 -- every other group is silently skipped.
        n = cute.size(mC, mode=[1])
        g = cute.size(mB, mode=[2])
        # PERSISTENT + N-FAST, now over a JAGGED row space.
        #
        # m_tiles is sized by max_seq_len (the PADDED bound), matching Triton
        # (triton_jagged.py:347 `if start_m >= seq_len: return`) and the existing sm80 CuTeDSL
        # kernel. Tiles
        # past a group's real length are skipped at runtime. At the production shape 35.5%
        # of tiles are empty -- that is a property of the workload, not waste we added,
        # and in a persistent grid-stride loop a skipped tile costs a decode and a branch.
        #
        # N-fast decode is preserved: it is worth 15% and the stock scheduler rasterises
        # M-major, which is why we hand-roll the loop.
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
        #
        # x atom_thr_size, as the upstream CUTLASS Blackwell dense_gemm example does: with cta_group=2 the TMA arrival is
        # cluster-wide, so the leader's full barrier receives the bytes BOTH CTAs of the
        # pair load. Expecting only this CTA's share lets the barrier complete at half
        # the data and desync the next phase -- which presents as a hang, not a wrong
        # answer.
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
            # (same as the upstream CUTLASS Blackwell dense_gemm_persistent example). Sizing it 128 makes producer_acquire wait
            # for 128 arrivals that never come -- a deadlock that only shows once an
            # accumulator is REUSED, i.e. from the 3rd tile onward with 2 stages. The
            # non-persistent kernel had the same wrong value and never noticed, because
            # it never called consumer_release.
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
        # partner CTA's barrier init. Dropping this is what caused
        # the stage-2 deadlock -- racecheck reported the ab mbarrier init racing the
        # TMA warp's first wait/arrive on it, and stage-1, which keeps the wait, is
        # racecheck-clean at the same shape.
        if cutlass.const_expr(cfg.two_cta):
            cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # TMA warp -- issues loads and nothing else.
        #
        # No prefetch prologue and no try_acquire/try_wait peeking: those existed to
        # pipeline loads against MMA inside a single warp. A dedicated load warp runs
        # ahead structurally, bounded only by `stages` empty buffers.
        #
        if warp_idx == TMA_WARP_ID:
            for t in cutlass.range(cluster_id, total_clusters, n_clusters, unroll=1):
                # Decode one linear tile index. This block is IDENTICAL in all three
                # warp roles -- if one warp skips a tile another does not, the A/B
                # pipeline desyncs and hangs. (Inlined rather than a helper: CuTeDSL
                # rejects closures that capture variables inside dynamic control flow.)
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
                    # be applied to the rank-3 (M,K,L) view BEFORE local_tile -- applied
                    # after tma_partition the coordinate profile no longer matches
                    # ("unable to compute crd2idx ... '!cute.coord<(?)>'"). Same placement
                    # the existing sm80 CuTeDSL kernel uses on its cp.async path
                    # (prime_perf_optimizer/.../cutedsl/jagged_dense_bmm_cute.py:397).
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
            # CTA (160 threads). Skipping it on the follower hangs the pair -- that is
            # exactly what the first 2-CTA attempt did.
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
                    # pipeline desyncs and hangs. (Inlined rather than a helper: CuTeDSL
                    # rejects closures that capture variables inside dynamic control flow.)
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
            # Descriptors below stay INSIDE the loop deliberately. Hoisting them above it
            # does not compile: a cute.tiled_copy cannot be a loop-carried block argument
            # ("failed to legalize unresolved materialization ... remained live after
            # conversion"). Rebuilding them per tile is cheap because the live fragment is
            # only 64 elements/thread -- the kernel sits at 126 registers with zero spill.
            for t in cutlass.range(cluster_id, total_clusters, n_clusters, unroll=1):
                # Decode one linear tile index. This block is IDENTICAL in all three
                # warp roles -- if one warp skips a tile another does not, the A/B
                # pipeline desyncs and hangs. (Inlined rather than a helper: CuTeDSL
                # rejects closures that capture variables inside dynamic control flow.)
                n_tile = t % n_tiles
                rest = t // n_tiles
                mtile = rest % m_tiles
                bz = rest // m_tiles
                seq_start = mSeq[bz]
                seq_len = mSeq[bz + 1] - seq_start
                # Tiles past this group's length -- and every tile of an EMPTY group.
                active = mtile * cfg.mma_tile_m < seq_len
                if active:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                    # Sub-tiled drain: holding the whole 128x256 tile needed 256 fp32 +
                    # 256 bf16 per thread, blew the 255-register budget and spilled
                    # 1.6 GB. epi_n at a time keeps the live fragment small.
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
                    # bf16 staging fragment: converting the whole fragment in one
                    # rC.store(rAcc.load().to(...)) is a vector op the compiler folds into
                    # F2FP.BF16.F32.PACK_AB pairs, where the old per-element
                    # `tC_s[i] = rAcc[i].to(...)` emitted one convert AND one 2-byte store
                    # each.
                    rC = cute.make_rmem_tensor(frag_shape, self.ab_dtype)
                    simt_atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), self.ab_dtype
                    )

                    # Rows past this group's seq_len belong to the NEXT group in the flat C
                    # and are written by a different CTA, so an unpredicated tile store
                    # would corrupt a neighbour. tCoord's M is the tile-local row; m_base
                    # converts it to a group-local row.
                    #
                    # But the predicate does NOT have to be per element. Trace-time proof:
                    # the coordinate fragment's only non-zero stride is 1@1, a pure step
                    # along mode 1 (N), so every element a thread owns lies in the SAME row
                    # and differs only in column -- M is loop-invariant and hoists out, and
                    # the columns are contiguous so the store vectorises. Asserted rather
                    # than assumed: if a future tiler changes the t2r ownership pattern this
                    # fires instead of silently writing the wrong rows.
                    # Duck-typed on .mode rather than isinstance(ScaledBasis): plain ints
                    # (the zero strides) satisfy that isinstance inside the DSL trace.
                    _probe = tCoord[(None, None, None, 0)]
                    # Written as one pure-Python expression, not a loop with an `if`:
                    # CuTeDSL rewrites `if` statements inside a @cute.kernel into TRACED
                    # control flow (if_region/then_block), so a Python-level guard here is
                    # not evaluated at trace time and the check misfires. Comprehensions
                    # and all() are plain expressions, so they run during tracing.
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
    # tcgen05 MMA and the 2-CTA cluster are Blackwell-only. Without this the failure is
    # a CUDA error from deep inside the JIT compile, which is hard to attribute.
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
    # produces misaligned TMA loads -- both far from the caller and hard to attribute.
    assert jagged.dtype == torch.bfloat16, (
        f"CUTEDSL jagged_dense_bmm_broadcast_add is bf16-only, got {jagged.dtype}"
    )
    assert dense.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16, (
        f"dense and bias must be bf16 too, got dense={dense.dtype} bias={bias.dtype}"
    )
    assert tuple(bias.shape) == (g, n), (
        f"bias must be (G, N) = ({g}, {n}), got {tuple(bias.shape)}"
    )
    # Contiguity is a HARD requirement, not a preference. from_dlpack bakes each tensor's
    # strides into the compiled kernel, and only mA/mC mark mode 0 dynamic -- mB and mBias
    # are fully static. The compile cache key carries dtypes and sizes but NOT strides, so
    # a differently-laid-out dense (e.g. the contiguous=False path in the shared test
    # harness, which builds it via transpose().contiguous().transpose()) would reuse a
    # kernel whose TMA descriptor baked the previous layout and read B with the wrong
    # strides -- silently wrong output, no error. Fail here instead.
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

    # A/C keep a SINGLETON group mode: make_tiled_tma_atom_A and local_tile both want
    # rank 3, and a bare 2-D tensor is rejected ("failed to construct a valid coordinate
    # ... resulting in an incorrect profile"). Rows stay flat; the group is selected by
    # offsetting the row coordinate, not by a stride.
    # detach(): cute's from_dlpack refuses tensors with requires_grad=True ("Can't
    # export tensors that require gradient"), and production callers do pass those.
    # Detaching is safe ONLY because this function is wrapped in the autograd.Function
    # below, which is what re-establishes the graph edge -- call the wrapper, never this
    # directly, or gradients are silently dropped.
    jagged = jagged.detach()
    dense = dense.detach()
    bias = bias.detach()

    a3 = jagged.unsqueeze(-1)  # (sum_L, K, 1)
    c3 = out.unsqueeze(-1)  # (sum_L, N, 1)
    b3 = dense.permute(2, 1, 0)  # (N, K, G)
    bias2 = bias.permute(1, 0)  # (N, G)      unchanged

    # int64 straight through: that is the op contract (seq_offsets is (B+1,) int64) and
    # a .to(torch.int32) here silently WRAPS past 2^31 rather than erroring. It also cost
    # an extra direct_copy_kernel_cuda launch per call. Production Triton carries the full
    # width too -- LDG.E.64 x2 at +0x8, then IADD3/IMAD.X for the subtract and
    # ISETP/ISETP.EX for the compare. We now compile to the identical idiom.

    # A and C are (sum_L, ., 1) and sum_L VARIES PER BATCH, so mode 0 must be dynamic.
    # from_dlpack bakes every dim as a compile-time constant by default, and sum_L is not
    # (and must not be) part of the compile key below -- baking it meant a kernel compiled
    # for one batch silently wrote the wrong number of rows for the next one with the same
    # (N, K, G, max_seq_len). K/N stay static: the MMA tiler and k_tiles need them.
    # stride_order must be explicit: (sum_L, ., 1) has TWO stride-1 modes (the K/N mode
    # and the singleton group mode), so auto-deduction raises "The layout could not be
    # deduced". (0, 1, 2) is the row-major order torch reports for this view.
    mA = from_dlpack(a3, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1, 2)
    )
    mB = from_dlpack(b3, assumed_align=16)
    mC = from_dlpack(c3, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1, 2)
    )
    mBias = from_dlpack(bias2, assumed_align=16)
    mSeq = from_dlpack(seq_offsets, assumed_align=8)

    # max_seq_len sizes the grid, so it is part of the compile key. seq_offsets.dtype is
    # in it too now that we no longer normalise it host-side: mSeq's element type is baked
    # into the compiled kernel, so an int32 call after an int64 one would otherwise reuse
    # the int64 build and misread the offsets.
    # Device is part of the key: the compiled kernel bakes in a persistent grid sized
    # from that device's max_active_clusters, so reusing one device's build on another
    # would launch a wrongly-sized grid.
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

    The kernel detaches its inputs so cute's from_dlpack will accept them. Returning that
    detached output directly would leave it disconnected from the graph, so a caller in a
    grad-enabled context would get None/zero gradients with no error -- a silent wrong
    answer in training. Routing through an autograd.Function keeps the output connected
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
    grad-enabled context; calling .backward() on the result raises NotImplementedError
    rather than silently producing no gradients.
    """
    return _CuTeDSLJaggedDenseBmmBroadcastAddFunction.apply(
        seq_offsets, jagged, dense, bias, max_seq_len
    )
