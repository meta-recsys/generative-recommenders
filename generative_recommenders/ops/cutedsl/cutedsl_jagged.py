"""CuTeDSL jagged grouped GEMM with broadcast bias, for Blackwell (sm_100a).

    out[s:e] = jagged[s:e] @ dense[g] + bias[g]      s, e = seq_offsets[g], [g + 1]

The CuTeDSL backend for `generative_recommenders.ops.jagged_tensors
.jagged_dense_bmm_broadcast_add`, reached through `HammerKernel.CUTEDSL`. Forward
(`cutedsl_jagged_dense_bmm_broadcast_add`) and backward
(`_cutedsl_jagged_dense_bmm_broadcast_add_djagged`,
`_cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias`).

Blackwell-native: tcgen05 MMA with `cta_group=2` over a 2x1 cluster, TMA bulk-tensor
loads into swizzled shared memory, an fp32 accumulator in tensor memory, the bias folded
into the epilogue, and warp specialisation (4 epilogue + 1 MMA + 1 TMA warp, 192
threads). Persistent grid-stride loop with N-fast rasterisation.

Correctness gates live in
generative_recommenders/ops/tests/jagged_dense_bmm_broadcast_add_cutedsl_test.py.

"""

import functools
import os
import re
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


_MIN_CUTLASS_DSL_VERSION = (4, 5, 2)


def _check_cutlass_dsl_version(version: str) -> None:
    found = tuple(int(x) for x in re.findall(r"\d+", version)[:3])
    if found < _MIN_CUTLASS_DSL_VERSION:
        raise ImportError(
            f"cutedsl_jagged needs nvidia-cutlass-dsl >= 4.5.2, found {version}. In fbcode, "
            "add constraint_overrides = ["
            '"ovr_config//third-party/pypi/nvidia-cutlass-dsl/constraints:4.5.2", '
            '"ovr_config//third-party/cuda/constraints:13.3"] to the binary or test target.'
        )


_check_cutlass_dsl_version(getattr(cutlass, "__version__", "0"))


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


def _max_active_clusters(cluster_size: int = 1) -> int:
    """Clusters of `cluster_size` CTAs the machine holds at once -- this sizes the
    persistent grid.

    Cached because the query is a driver call on the hot path, but keyed PER DEVICE: it
    is a device property, and a single global would size the grid for whichever GPU
    happened to be current first on a multi-GPU host.
    """
    key = (torch.cuda.current_device(), cluster_size)
    if key not in _MAX_CLUSTERS:
        _MAX_CLUSTERS[key] = cutlass.utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
    return _MAX_CLUSTERS[key]


def _stream():
    """The CALLER's current stream, resolved per call -- never cached.

    Caching the first stream seen per device meant that a caller running under a
    non-default stream (torch.cuda.stream(...), a CUDA graph capture, or any library that
    switches streams) had the kernel launched on a stale stream instead, silently breaking
    stream ordering against their other work.
    """
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def cutedsl_jagged_dense_bmm_broadcast_add_supported(
    jagged: torch.Tensor, dense: torch.Tensor, bias: torch.Tensor
) -> bool:
    """Whether the CuTeDSL kernels (fwd, d_jagged, d_dense + d_bias) handle these inputs.

    They are compiled for bf16 only, and TMA needs 16 B-aligned row strides, so the feature
    dims (jagged.shape[1] and dense.shape[2]) must be multiples of 8. Other inputs give
    silently wrong results, so ops/jagged_tensors.py falls back to Triton for them.
    Layout (contiguity, base alignment) is not a condition: the dispatchers fix it with a copy.
    """
    return (
        jagged.dtype == dense.dtype == bias.dtype == torch.bfloat16
        and jagged.shape[-1] % 8 == 0
        and dense.shape[-1] % 8 == 0
    )


def _require_supported(fn: str, tensors, dims) -> None:
    """Raise instead of computing a wrong result (see cutedsl_jagged_dense_bmm_broadcast_add_supported)."""
    dtypes = sorted({str(t.dtype) for t in tensors if t.dtype != torch.bfloat16})
    bad_dims = [d for d in dims if d % 8]
    if dtypes or bad_dims:
        raise ValueError(
            f"{fn}: the CuTeDSL kernels need bf16 inputs and feature dims divisible by 8 "
            f"(non-bf16 dtypes: {dtypes}, dims not divisible by 8: {bad_dims}). "
            "Use the Triton kernel for these inputs."
        )


def _tma_ready(t: torch.Tensor) -> torch.Tensor:
    """Contiguous with a 16 B-aligned base, as from_dlpack(assumed_align=16) and TMA need.
    A no-op for ordinary tensors; a sliced view starting off a 16 B boundary is copied."""
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


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
    assert tuple(bias.shape) == (g, n), (
        f"bias must be (G, N) = ({g}, {n}), got {tuple(bias.shape)}"
    )
    _require_supported(
        "cutedsl_jagged_dense_bmm_broadcast_add", (jagged, dense, bias), (k, n)
    )
    # Contiguity is a HARD requirement. from_dlpack bakes each tensor's strides into the
    # compiled kernel, and only mA/mC mark mode 0 dynamic -- mB and mBias are fully
    # static. The compile cache key carries dtypes and sizes but NOT strides, so
    # non-contiguous (or off-16 B) inputs are copied rather than rejected.
    jagged, dense, bias = _tma_ready(jagged), _tma_ready(dense), _tma_ready(bias)

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
    """Autograd wrapper: forward kernel, and the d_jagged and fused d_dense + d_bias
    kernels for backward.

    The kernels detach their inputs so cute's from_dlpack will accept them; routing
    through an autograd.Function keeps the output connected to the graph.
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
        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.max_seq_len = max_seq_len
        return _cutedsl_jagged_dense_bmm_broadcast_add_fwd(
            seq_offsets, jagged, dense, bias, max_seq_len
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, d_out: torch.Tensor):
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_out = d_out.contiguous()
        d_jagged = _cutedsl_jagged_dense_bmm_broadcast_add_djagged(
            seq_offsets, d_out, dense, ctx.max_seq_len
        )
        d_dense, d_bias = _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
            seq_offsets, jagged, d_out, ctx.max_seq_len
        )
        return None, d_jagged, d_dense, d_bias, None


def cutedsl_jagged_dense_bmm_broadcast_add(
    seq_offsets, jagged, dense, bias, max_seq_len: int
):
    """Blackwell CuTeDSL jagged_dense_bmm_broadcast_add, forward and backward.

    Public entry point for the HammerKernel.CUTEDSL dispatch. Inputs must pass
    cutedsl_jagged_dense_bmm_broadcast_add_supported (ValueError otherwise).
    """
    return _CuTeDSLJaggedDenseBmmBroadcastAddFunction.apply(
        seq_offsets, jagged, dense, bias, max_seq_len
    )


# ─────────────────────────────────────────────────────────────────────────────
# Backward kernel constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_ACC_STAGE_JAGGED = 2

_COMPILED_BWD_DJAGGED: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# d_jagged backward: d_out (M×K) @ dense[g].T (K×D) → d_jagged (M×D)
# Each tile runs the full ceil(K / tile_k) contraction in one accumulator pass.
# ─────────────────────────────────────────────────────────────────────────────


class JaggedDenseBmmBroadcastAddDJagged:
    """d_jagged backward: d_out @ dense[g].T  (K-major B, no copy)."""

    def __init__(self, ab_dtype, acc_dtype, cfg: Cfg):
        from dataclasses import replace as _replace

        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.cfg = _replace(cfg, stages=2)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.cfg.two_cta else tcgen05.CtaGroup.ONE
        )
        self.a_major = tcgen05.OperandMajorMode.K  # d_out is K-major
        self.b_major = tcgen05.OperandMajorMode.K  # dense.permute(1,2,0) K stride-1

    @cute.jit
    def __call__(
        self,
        mA,
        mB,
        mC,
        mSeq,
        max_seq_len: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        D_static: cutlass.Constexpr,
        mTmapAddrs,
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
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*cfg.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        acc_shape = cute.append(
            tiled_mma.partition_shape_C((cfg.mma_tile_m, cfg.tile_n)),
            NUM_ACC_STAGE_JAGGED,
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
        # GEMM K is mode 1 of both A = d_out (M, K, 1) and B = dense (D, K, G). Static, so
        # the per-tile k-loop has a compile-time trip count (12 at K=768).
        k_tiles = cute.ceil_div(cute.size(mA, mode=[1]), cfg.tile_k)
        n_tiles = cute.ceil_div(n, cfg.tile_n)
        # pyrefly: ignore [unsupported-operation]  # max_seq_len is a Constexpr
        m_tiles = (max_seq_len + cfg.mma_tile_m - 1) // cfg.mma_tile_m
        total_clusters = m_tiles * n_tiles * g
        # pyrefly: ignore [bad-specialization]  # Constexpr has no ordering bound
        n_persist = min(max_active_clusters, total_clusters)
        grid = (cfg.cluster_shape_mn[0] * n_persist, 1, 1)

        EPI_STAGES = 4
        epi_tile_shape = (
            cfg.cta_tile_m,
            cfg.tile_n // EPI_STAGES,
        )
        epi_smem_staged = sm100_utils.make_smem_layout_epi(
            self.ab_dtype,
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            epi_tile_shape,
            EPI_STAGES,
        )
        # Base atom (for coordinate tensor mC_tma) uses full mC shape
        _tma_c_base = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            epi_smem_staged,
            epi_tile_shape,
        )
        tma_atom_c = _tma_c_base.atom
        tma_c = _tma_c_base.tma_tensor

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
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
            tma_atom_c,
            tma_c,
            EPI_STAGES,
            epi_smem_staged,
            mTmapAddrs,
            D_static,
            k_tiles,
        ).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            cluster=(*cfg.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tiled_mma,
        tma_atom_a,
        mA,
        tma_atom_b,
        mB,
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
        tma_atom_c,
        mC_tma,
        epi_stages: cutlass.Constexpr,
        epi_smem_staged,
        mTmapAddrs,
        D_static: cutlass.Constexpr,
        k_tiles: cutlass.Constexpr,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, by, bz = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            self.ab_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner
        )
        sB = smem.allocate_tensor(
            self.ab_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        sD = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=epi_smem_staged.outer,
            byte_alignment=128,
            swizzle=epi_smem_staged.inner,
        )
        # pyrefly: ignore [unsupported-operation]  # stages is a Constexpr
        ab_mbar = smem.allocate_array(cutlass.Int64, 2 * stages)
        acc_mbar = smem.allocate_array(cutlass.Int64, 2 * NUM_ACC_STAGE_JAGGED)
        tmem_holding_buf = smem.allocate_array(cutlass.Int32, 1)
        tmem_dealloc_mbar = smem.allocate_array(cutlass.Int64, 1)

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

        acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_mbar,
            num_stages=NUM_ACC_STAGE_JAGGED,
            producer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            ),
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                len(EPILOGUE_WARP_ID) * (2 if cfg.two_cta else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=cfg.two_cta,
        )

        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tmem = cutlass.utils.TmemAllocator(
            tmem_holding_buf,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=TMEM_ALLOC_BAR_ID, num_threads=TMEM_ALLOC_BAR_THREADS
            ),
            allocator_warp_id=EPILOGUE_WARP_ID[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar,
        )
        tmem_dealloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR_ID, num_threads=EPILOGUE_THREADS
        )

        if cutlass.const_expr(cfg.two_cta):
            cutlass.pipeline.pipeline_init_arrive(
                cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True
            )

        mma_tiler = (cfg.mma_tile_m, cfg.tile_n, cfg.tile_k)
        epi_tile = (
            cfg.cta_tile_m,
            # pyrefly: ignore [unsupported-operation]  # epi_stages is a Constexpr
            cfg.tile_n // epi_stages,
        )

        # K-major B: mB mode[0]=K (stride-1). Precompute B tile view outside tile loop.
        gB_nkl = cute.local_tile(
            mB, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
        )

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

        a_mcast_mask = None
        b_mcast_mask = None
        if cutlass.const_expr(cfg.two_cta):
            a_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, blk_in_cluster_vmnk, mcast_mode=2
            )
            b_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, blk_in_cluster_vmnk, mcast_mode=1
            )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        num_kblks = cute.size(tCrA, mode=[2])

        if cutlass.const_expr(cfg.two_cta):
            cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # TMA warp: M-fast rasterisation, k_tiles stages per tile
        if warp_idx == TMA_WARP_ID:
            for t in cutlass.range(cluster_id, total_clusters, n_clusters, unroll=1):
                mtile = t % m_tiles
                rest = t // m_tiles
                n_tile = rest % n_tiles
                bz_t = rest // n_tiles
                seq_start = mSeq[bz_t]
                seq_len = mSeq[bz_t + 1] - seq_start
                active = mtile * cfg.mma_tile_m < seq_len
                if active:
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
                    gA = tAgA[(None, mtile, None, 0)]
                    gB = tBgB[(None, n_tile, None, bz_t)]
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

        # MMA warp: k_tiles stages per tile, ACCUMULATE=False on the first
        if warp_idx == MMA_WARP_ID:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            if is_leader_cta:
                acc_producer_state = cutlass.pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Producer, NUM_ACC_STAGE_JAGGED
                )
                for t in cutlass.range(
                    cluster_id, total_clusters, n_clusters, unroll=1
                ):
                    mtile = t % m_tiles
                    rest = t // m_tiles
                    n_tile = rest % n_tiles
                    bz_t = rest // n_tiles
                    seq_start = mSeq[bz_t]
                    seq_len = mSeq[bz_t + 1] - seq_start
                    active = mtile * cfg.mma_tile_m < seq_len
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

        # Epilogue: single drain per tile — TMEM → stmatrix → sD → async TMA (per-group OOB descriptor)
        if warp_idx < MMA_WARP_ID:
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, NUM_ACC_STAGE_JAGGED
            )
            epi_sync = cutlass.pipeline.NamedBarrier(
                barrier_id=EPI_SYNC_BAR_ID, num_threads=EPILOGUE_THREADS
            )
            epilogue_pipeline = cutlass.pipeline.PipelineTmaStore.create(
                # pyrefly: ignore [bad-argument-type]  # epi_stages is a Constexpr
                num_stages=epi_stages,
                producer_group=cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread
                ),
            )
            epilogue_pipeline.producer_acquire()
            _tmap_mgr_k = cutlass.utils.TensorMapManager(
                cutlass.utils.TensorMapUpdateMode.GMEM, 128
            )

            # Prolog: initialize this CTA's private descriptor slot with the base atom
            # (full mC shape). Per-tile update_tensormap patches the M-extent for each
            # group before the TMA copy. One slot per CTA, not per cluster: with a shared
            # slot, one CTA can rewrite the extent for its next group while its partner
            # is still storing the current tile, so rows past a short group's end land
            # in the next group (seen with short tiles, e.g. K <= 128).
            _cid_addr = mTmapAddrs[bx]
            _slot_ptr_gmem = _tmap_mgr_k.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64, _cid_addr, cute.AddressSpace.gmem, assumed_align=128
                ),
                cute.AddressSpace.gmem,
            )
            _slot_ptr_gen = _tmap_mgr_k.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64, _cid_addr, cute.AddressSpace.gmem, assumed_align=128
                ),
                cute.AddressSpace.generic,
            )
            _tmap_mgr_k.init_tensormap_from_atom(tma_atom_c, _slot_ptr_gmem, warp_id=0)
            _tmap_mgr_k.fence_tensormap_initialization()

            for t in cutlass.range(cluster_id, total_clusters, n_clusters, unroll=1):
                mtile = t % m_tiles
                rest = t // m_tiles
                n_tile = rest % n_tiles
                bz_t = rest // n_tiles
                seq_start = mSeq[bz_t]
                seq_len = mSeq[bz_t + 1] - seq_start
                active = mtile * cfg.mma_tile_m < seq_len
                if active:
                    tmem_load = sm100_utils.get_tmem_load_op(
                        (cfg.cta_tile_m, cfg.tile_n, cfg.tile_k),
                        cutlass.utils.LayoutEnum.ROW_MAJOR,
                        self.ab_dtype,
                        self.acc_dtype,
                        epi_tile,
                        cfg.two_cta,
                    )

                    # Single consumer_wait per tile — drain TMEM directly → stmatrix → sD
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tAcc_epi0 = cute.flat_divide(
                        tCtAcc_base[((None, None), 0, 0, acc_consumer_state.index)],
                        epi_tile,
                    )
                    tiled_t2r0 = tcgen05.make_tmem_copy(
                        tmem_load, tAcc_epi0[(None, None, 0, 0)]
                    )
                    thr_t2r0 = tiled_t2r0.get_slice(tidx)
                    tAcc0 = cute.group_modes(thr_t2r0.partition_S(tAcc_epi0), 3, 5)

                    copy_atom_r2s = sm100_utils.get_smem_store_op(
                        cutlass.utils.LayoutEnum.ROW_MAJOR,
                        self.ab_dtype,
                        self.acc_dtype,
                        tiled_t2r0,
                    )
                    tiled_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_t2r0)
                    thr_r2s = tiled_r2s.get_slice(tidx)
                    tSD = thr_r2s.partition_D(sD)

                    # frag_shape must match the T2R copy's TMEM-physical register layout,
                    # NOT the stmatrix R2S destination shape (which is 32x smaller).
                    # Use a fake fp32 SMEM tensor (address 0) with the accumulator tile's
                    # shape to extract the T2R partition_D shape at trace time — no
                    # actual writes.
                    _fake_sc_ptr = cute.make_ptr(
                        self.acc_dtype, 0, cute.AddressSpace.smem, assumed_align=16
                    )
                    _fake_sc = cute.make_tensor(
                        _fake_sc_ptr,
                        cute.make_layout((cfg.cta_tile_m, cfg.tile_n)),
                    )
                    _fake_tSC = cute.group_modes(
                        thr_t2r0.partition_D(cute.flat_divide(_fake_sc, epi_tile)), 3, 5
                    )
                    frag_shape = _fake_tSC[(None, None, None, 0)].shape
                    rAcc = cute.make_fragment(frag_shape, self.acc_dtype)
                    rC_out = cute.make_rmem_tensor(frag_shape, self.ab_dtype)

                    # TMA S2G epilogue: TMEM → sD → GMEM with per-group descriptor.
                    # TMA hardware suppresses OOB stores (rows ≥ seq_len[g]) automatically.
                    mC_tma_g = cute.domain_offset((seq_start, 0, 0), mC_tma)
                    gC_mnl_tma = cute.local_tile(
                        mC_tma_g,
                        cute.slice_(mma_tiler, (None, None, 0)),
                        (None, None, None),
                    )
                    tCgC_tma_p = thr_mma.partition_C(gC_mnl_tma)
                    gC_full_tma = tCgC_tma_p[((None, None), 0, 0, mtile, n_tile, 0)]
                    gC_epi_raw = cute.flat_divide(gC_full_tma, epi_tile)
                    gC_epi = gC_epi_raw[(None, None, 0, None)]
                    tCsD, tCgC_tma_part = cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sD, 0, 2),
                        cute.group_modes(gC_epi, 0, 2),
                    )
                    tCgC_grouped = cute.group_modes(
                        tCgC_tma_part, 1, cute.rank(tCgC_tma_part)
                    )

                    # Update descriptor: patch M-extent to seq_offsets[g+1] so TMA suppresses
                    # stores beyond the group boundary (OOB → NOP on the last M-tile).
                    _seq_end = mSeq[bz_t + 1]
                    _mC_g_upd = cute.make_tensor(
                        mC.iterator,
                        cute.make_layout(
                            (_seq_end, D_static, 1), stride=(D_static, 1, 0)
                        ),
                    )
                    _tmap_mgr_k.update_tensormap(
                        (_mC_g_upd,),
                        (tma_atom_c,),
                        (_slot_ptr_gmem,),
                        warp_id=0,
                        tensormap_smem_ptr=(),
                    )
                    epi_sync.arrive_and_wait()

                    # All 4 warps: stmatrix all sub-tiles into sD (128 threads, max throughput)
                    for sub in cutlass.range_constexpr(cute.size(tAcc0, mode=[3])):
                        c_buf = sub % epi_stages
                        cute.copy(tiled_t2r0, tAcc0[(None, None, None, sub)], rAcc)
                        rC_out.store(rAcc.load().to(self.ab_dtype))
                        rC_retiled = tiled_r2s.retile(rC_out)
                        cute.copy(tiled_r2s, rC_retiled, tSD[(None, None, None, c_buf)])

                    cute.arch.fence_view_async_shared()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                    # Every warp's stmatrix into sD lands before warp 0 issues the stores.
                    epi_sync.arrive_and_wait()

                    # Warp 0: fence descriptor + all EPI_STAGES TMA stores (non-blocking with EPI_STAGES=4)
                    if warp_idx == EPILOGUE_WARP_ID[0]:
                        _tmap_mgr_k.fence_tensormap_update(_slot_ptr_gmem)
                        for sub in cutlass.range_constexpr(cute.size(tAcc0, mode=[3])):
                            c_buf = sub % epi_stages
                            cute.copy(
                                tma_atom_c,
                                tCsD[(None, c_buf)],
                                tCgC_grouped[(None, sub)],
                                tma_desc_ptr=_slot_ptr_gen,
                            )
                            epilogue_pipeline.producer_commit()
                            epilogue_pipeline.producer_acquire()

            epilogue_pipeline.producer_tail()
            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ─────────────────────────────────────────────────────────────────────────────
# d_dense + d_bias backward: TMA-OOB + 2-CTA cluster UMMA, d_bias fused
# ─────────────────────────────────────────────────────────────────────────────

_COMPILED_BWD_DDENSE_DBIAS: dict = {}


class JaggedDenseBmmBroadcastAddDDenseDBias:
    """d_dense + d_bias backward: 2-CTA cluster UMMA.

    mma_tile_m=256 across 2 CTAs (cta_tile_m=128 each), tile_n=_TILE_N.
    TMA-OOB via TensorMapManager handles non-aligned jagged sequences.
    Groups are scheduled longest first (mOrder, an on-device argsort of the lengths).
    CTA1 MMA warp idles (the leader's cta_group=2 UMMA covers both CTAs).
    One CTA per SM with a deep A/B pipeline (see _MAX_AB_STAGES).

    d_bias is fused on the tensor cores: d_bias[g, k] = sum_m d_out[m, k] is the row sum
    of A over the reduction dim, so the MMA warp also issues A (the same SMEM stage) x an
    all-ones B of width _BIAS_N into spare TMEM columns. Each N tile does this only for its
    own k-blocks (k_local % n_tiles == n_tile), which spreads the extra UMMAs evenly; the
    epilogue of each CTA writes its partial row sums to (G, n_tiles, K) and the dispatcher
    sums them in fixed order. It rides the existing A/B and accumulator pipelines, so no
    extra synchronisation.
    """

    _BIAS_N = 16
    _MAX_AB_STAGES = 8
    _TILE_N = 256
    _EPI_STAGES_C = _TILE_N // 64  # = tile_n // epi_n

    def __init__(self, ab_dtype, acc_dtype, cfg: Cfg):
        from dataclasses import replace as _replace

        self.ab_dtype = ab_dtype
        self.acc_dtype = acc_dtype
        self.c_dtype = ab_dtype
        self.cfg = _replace(
            cfg,
            mma_tile_m=256,
            two_cta=True,
            stages=self._MAX_AB_STAGES,
            tile_n=self._TILE_N,
        )
        assert self.cfg.tile_n // self.cfg.epi_n == self._EPI_STAGES_C
        self.cta_group = tcgen05.CtaGroup.TWO
        self.a_major = tcgen05.OperandMajorMode.MN
        self.b_major = tcgen05.OperandMajorMode.MN

    def _fast_epilogue(self, D_static, K_static) -> bool:
        """Output tiles exactly (no partial tiles): direct st.global epilogue, no SMEM
        staging. After the operand swap D_static is the MMA M extent, K_static the N."""
        return D_static % self.cfg.mma_tile_m == 0 and K_static % self.cfg.tile_n == 0

    def _num_ab_stages(self, D_static, K_static) -> int:
        """A/B stages that fit one CTA's SMEM next to the epilogue staging, capped.

        Not _ab_stages(): that counts the full tile_n of B per CTA, but with cta_group=2
        each CTA holds only tile_n/2 columns of B, and it ignores the staging buffer.
        32 KB per stage at tile_n=256: fast path 7, slow path (64 KB staging) 5.
        """
        cfg = self.cfg
        ab_bytes = self.ab_dtype.width // 8
        per_stage = cfg.tile_k * (cfg.cta_tile_m + cfg.tile_n // 2) * ab_bytes
        staging = 0
        if not self._fast_epilogue(D_static, K_static):
            staging = (
                cfg.cta_tile_m
                * cfg.epi_n
                * (self.c_dtype.width // 8)
                * self._EPI_STAGES_C
            )
        ones = (self._BIAS_N // 2) * cfg.tile_k * ab_bytes
        mbar_bytes = 1024
        budget = (
            cutlass.utils.get_smem_capacity_in_bytes() - staging - ones - mbar_bytes
        )
        return min(self._MAX_AB_STAGES, budget // per_stage)

    @cute.jit
    def __call__(
        self,
        mA,
        mB,
        mC,
        mDBias,  # (G, n_tiles, K) fp32 per-N-tile partial row sums
        mSeq,
        mOrder,  # (G,) int32 group ids, longest group first
        mTmapAddrsA,
        mTmapAddrsB,
        mTmapAddrsC,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        D_static: cutlass.Constexpr,
        K_static: cutlass.Constexpr,
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
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*cfg.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        _NUM_ACC = 1
        acc_shape = cute.append(
            tiled_mma.partition_shape_C((cfg.mma_tile_m, cfg.tile_n)), _NUM_ACC
        )
        # d_bias UMMA: same A operand (MN-major), all-ones B (K-major: tile_k is the
        # contiguous dim, so the 128B swizzle atom fits however few rows B has).
        tiled_mma_bias = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            (cfg.mma_tile_m, self._BIAS_N),
        )
        bias_acc_shape = tiled_mma_bias.partition_shape_C(
            (cfg.mma_tile_m, self._BIAS_N)
        )
        ones_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_bias, (cfg.mma_tile_m, self._BIAS_N, cfg.tile_k), self.ab_dtype, 1
        )
        # Main accumulator, then the d_bias accumulator in the next TMEM columns.
        tmem_cols = cutlass.utils.get_num_tmem_alloc_cols(
            [
                tiled_mma.make_fragment_C(acc_shape),
                tiled_mma_bias.make_fragment_C(bias_acc_shape),
            ]
        )
        stages = self._num_ab_stages(D_static, K_static)
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler, self.ab_dtype, stages
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler, self.ab_dtype, stages
        )
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
        # ── TMA store for C (SMEM → GMEM epilogue) ────────────────────────────
        _EPI_STAGES_C = self._EPI_STAGES_C
        # Store sub-tile = the epilogue's epi_n drain width.
        _epi_tile_c = (cfg.cta_tile_m, cfg.epi_n)
        # COLUMN_MAJOR: first dim (cta_tile_m=128K) is fast — matches K-fast mC=[K,D,G].
        _epi_smem_c = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            cutlass.utils.LayoutEnum.COL_MAJOR,
            _epi_tile_c,
            _EPI_STAGES_C,
        )
        _tma_c_base = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            _epi_smem_c,
            _epi_tile_c,
        )
        tma_atom_c = _tma_c_base.atom
        tma_c = _tma_c_base.tma_tensor
        d_size = cute.size(mA, mode=[0])
        k_size = cute.size(mB, mode=[0])
        g = cute.size(mC, mode=[2])
        m_tiles = (d_size + cfg.mma_tile_m - 1) // cfg.mma_tile_m
        n_tiles = cute.ceil_div(k_size, cfg.tile_n)
        total_tiles = m_tiles * n_tiles * g
        # pyrefly: ignore [bad-specialization]  # Constexpr has no ordering bound
        n_persist = min(max_active_clusters, total_tiles)
        grid = (n_persist * cute.size(tiled_mma.thr_id.shape), 1, 1)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            mC,
            mSeq,
            mOrder,
            mTmapAddrsA,
            mTmapAddrsB,
            mA,
            mB,
            a_smem_layout,
            b_smem_layout,
            acc_shape,
            tmem_cols,
            cluster_layout_vmnk,
            stages,
            n_tiles,
            m_tiles,
            total_tiles,
            D_static,
            K_static,
            tma_atom_c,
            tma_c,
            _epi_smem_c,
            _EPI_STAGES_C,
            mTmapAddrsC,
            mDBias,
            tiled_mma_bias,
            bias_acc_shape,
            ones_smem_layout,
        ).launch(
            grid=grid,
            block=[THREADS, 1, 1],
            cluster=(*cfg.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(  # noqa: C901
        self,
        tiled_mma,
        tma_atom_a,
        mA,
        tma_atom_b,
        mB,
        mC,
        mSeq,
        mOrder,  # (G,) int32 group ids, longest group first
        mTmapAddrsA,
        mTmapAddrsB,
        mA_gmem,
        mB_gmem,
        a_smem_layout,
        b_smem_layout,
        acc_shape: cutlass.Constexpr,
        tmem_cols: cutlass.Constexpr,
        cluster_layout_vmnk,
        stages: cutlass.Constexpr,
        n_tiles,
        m_tiles,
        total_tiles,
        D_static: cutlass.Constexpr,
        K_static: cutlass.Constexpr,
        tma_atom_c,
        mC_tma,
        epi_smem_c_layout,
        epi_stages_c: cutlass.Constexpr,
        mTmapAddrsC,
        mDBias,
        tiled_mma_bias,
        bias_acc_shape: cutlass.Constexpr,
        ones_smem_layout,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bx, by, bz_block = cute.arch.block_idx()
        _NUM_ACC = 1

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            self.ab_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner
        )
        sB = smem.allocate_tensor(
            self.ab_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner
        )
        sOnes = smem.allocate_tensor(
            self.ab_dtype, ones_smem_layout.outer, 1024, swizzle=ones_smem_layout.inner
        )
        # sC_staging only allocated for non-aligned shapes (slow TMA staging path).
        # For aligned shapes the const_expr branch takes st.global and this is dead.
        if cutlass.const_expr(not self._fast_epilogue(D_static, K_static)):
            sC_staging = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=epi_smem_c_layout.outer
                if hasattr(epi_smem_c_layout, "outer")
                else epi_smem_c_layout,
                byte_alignment=128,
                swizzle=epi_smem_c_layout.inner
                if hasattr(epi_smem_c_layout, "inner")
                else None,
            )
        # pyrefly: ignore [unsupported-operation]  # stages is a Constexpr
        ab_mbar = smem.allocate_array(cutlass.Int64, 2 * stages)
        acc_mbar = smem.allocate_array(cutlass.Int64, 2 * _NUM_ACC)
        tmem_hold = smem.allocate_array(cutlass.Int32, 1)
        tmem_dealloc = smem.allocate_array(cutlass.Int64, 1)

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

        acc_pipeline = cutlass.pipeline.PipelineUmmaAsync.create(
            barrier_storage=acc_mbar,
            num_stages=_NUM_ACC,
            producer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread
            ),
            consumer_group=cutlass.pipeline.CooperativeGroup(
                cutlass.pipeline.Agent.Thread,
                len(EPILOGUE_WARP_ID) * (2 if cfg.two_cta else 1),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=cfg.two_cta,
        )

        for i in cutlass.range(tidx, cute.size(sOnes), THREADS, unroll=1):
            sOnes[i] = self.ab_dtype(1.0)
        cute.arch.fence_view_async_shared()

        cutlass.pipeline.pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk, is_relaxed=False
        )

        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        tCtAccBias_fake = tiled_mma_bias.make_fragment_C(bias_acc_shape)
        bias_tmem_col = tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake)
        tmem = cutlass.utils.TmemAllocator(
            tmem_hold,
            barrier_for_retrieve=cutlass.pipeline.NamedBarrier(
                barrier_id=TMEM_ALLOC_BAR_ID, num_threads=TMEM_ALLOC_BAR_THREADS
            ),
            allocator_warp_id=EPILOGUE_WARP_ID[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc,
        )
        tmem_dealloc_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=TMEM_DEALLOC_BAR_ID, num_threads=EPILOGUE_THREADS
        )

        mma_tiler = (cfg.mma_tile_m, cfg.tile_n, cfg.tile_k)
        epi_tile_shape = (cfg.cta_tile_m, cfg.epi_n)

        mma_tile_coord_v = bx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cluster_id = bx // cute.size(tiled_mma.thr_id.shape)
        n_clusters = cute.arch.grid_dim()[0] // cute.size(tiled_mma.thr_id.shape)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        blk_in_cluster_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )

        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, blk_in_cluster_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, blk_in_cluster_vmnk, mcast_mode=1
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        num_kblks = cute.size(tCrA, mode=[2])

        gA_nkl = cute.local_tile(
            mA, cute.slice_(mma_tiler, (None, 0, None)), (None, None, None)
        )
        tAsA, _ = cpasync.tma_partition(
            tma_atom_a,
            blk_in_cluster_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA_nkl), 0, 3),
        )
        gB_nkl = cute.local_tile(
            mB, cute.slice_(mma_tiler, (0, None, None)), (None, None, None)
        )
        tBsB, _ = cpasync.tma_partition(
            tma_atom_b,
            blk_in_cluster_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
        )

        cutlass.pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ── TMA warp — TMA-OOB: update descriptor once per outer tile ─────────
        if warp_idx == TMA_WARP_ID:
            _cid_addr_a = mTmapAddrsA[bx]
            _cid_addr_b = mTmapAddrsB[bx]
            _tmap_mgr = cutlass.utils.TensorMapManager(
                cutlass.utils.TensorMapUpdateMode.GMEM, 128
            )
            _slot_a_gmem = _tmap_mgr.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64,
                    _cid_addr_a,
                    cute.AddressSpace.gmem,
                    assumed_align=128,
                ),
                cute.AddressSpace.gmem,
            )
            _slot_a_gen = _tmap_mgr.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64,
                    _cid_addr_a,
                    cute.AddressSpace.gmem,
                    assumed_align=128,
                ),
                cute.AddressSpace.generic,
            )
            _slot_b_gmem = _tmap_mgr.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64,
                    _cid_addr_b,
                    cute.AddressSpace.gmem,
                    assumed_align=128,
                ),
                cute.AddressSpace.gmem,
            )
            _slot_b_gen = _tmap_mgr.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64,
                    _cid_addr_b,
                    cute.AddressSpace.gmem,
                    assumed_align=128,
                ),
                cute.AddressSpace.generic,
            )
            cpasync.copy_tensormap(tma_atom_a, _slot_a_gmem)
            cpasync.copy_tensormap(tma_atom_b, _slot_b_gmem)
            cute.arch.fence_acq_rel_cta()

            _mA_base_addr = mA_gmem.iterator.toint()
            _mB_base_addr = mB_gmem.iterator.toint()

            for t in cutlass.range(cluster_id, total_tiles, n_clusters, unroll=1):
                n_tile = t % n_tiles
                mtile = (t // n_tiles) % m_tiles
                bz = mOrder[(t // n_tiles) // m_tiles]  # longest group first
                seq_start = mSeq[bz]
                seq_len = mSeq[bz + 1] - seq_start
                if seq_len > 0:
                    k_tiles_g = cute.ceil_div(seq_len, cfg.tile_k)
                    _gA_g = cute.local_tile(
                        cute.domain_offset((0, seq_start, 0), mA),
                        cute.slice_(mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    _, _tAgA_g = cpasync.tma_partition(
                        tma_atom_a,
                        blk_in_cluster_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(thr_mma.partition_A(_gA_g), 0, 3),
                    )
                    _gB_g = cute.local_tile(
                        cute.domain_offset((0, seq_start, 0), mB),
                        cute.slice_(mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    _, _tBgB_g = cpasync.tma_partition(
                        tma_atom_b,
                        blk_in_cluster_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(thr_mma.partition_B(_gB_g), 0, 3),
                    )
                    gA_m = _tAgA_g[(None, mtile, None, 0)]
                    gB_n = _tBgB_g[(None, n_tile, None, 0)]

                    _mA_upd = cute.make_tensor(
                        cute.make_ptr(
                            self.ab_dtype,
                            _mA_base_addr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(
                            (D_static, seq_start + seq_len, 1), stride=(1, D_static, 0)
                        ),
                    )
                    _mB_upd = cute.make_tensor(
                        cute.make_ptr(
                            self.ab_dtype,
                            _mB_base_addr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(
                            (K_static, seq_start + seq_len, 1), stride=(1, K_static, 0)
                        ),
                    )
                    _tmap_mgr.update_tensormap(
                        (_mA_upd, _mB_upd),
                        (tma_atom_a, tma_atom_b),
                        (_slot_a_gmem, _slot_b_gmem),
                        TMA_WARP_ID,
                        (None, None),
                    )
                    _tmap_mgr.fence_tensormap_update(_slot_a_gen)
                    _tmap_mgr.fence_tensormap_update(_slot_b_gen)

                    ab_producer.reset()
                    for k_local in cutlass.range(k_tiles_g, unroll=1):
                        handle = ab_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_a,
                            gA_m[(None, k_local)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=a_mcast_mask,
                            tma_desc_ptr=_slot_a_gen,
                        )
                        cute.copy(
                            tma_atom_b,
                            gB_n[(None, k_local)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=b_mcast_mask,
                            tma_desc_ptr=_slot_b_gen,
                        )
            ab_producer.tail()

        # ── MMA warp ──────────────────────────────────────────────────────────
        if warp_idx == MMA_WARP_ID:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAccBias = cute.make_tensor(
                tmem_ptr + bias_tmem_col, tCtAccBias_fake.layout
            )
            tCrA_bias = tiled_mma_bias.make_fragment_A(sA)
            tCrOnes = tiled_mma_bias.make_fragment_B(sOnes)
            if is_leader_cta:
                acc_producer_state = cutlass.pipeline.make_pipeline_state(
                    cutlass.pipeline.PipelineUserType.Producer, _NUM_ACC
                )
                for t in cutlass.range(cluster_id, total_tiles, n_clusters, unroll=1):
                    n_tile = t % n_tiles
                    mtile = (t // n_tiles) % m_tiles
                    bz = mOrder[(t // n_tiles) // m_tiles]  # longest group first
                    seq_start = mSeq[bz]
                    seq_len = mSeq[bz + 1] - seq_start
                    if seq_len > 0:
                        k_tiles_g = cute.ceil_div(seq_len, cfg.tile_k)
                        tCtAcc = tCtAcc_base[
                            (None, None, None, acc_producer_state.index)
                        ]
                        ab_consumer.reset()
                        acc_pipeline.producer_acquire(acc_producer_state)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for k_local in cutlass.range(k_tiles_g, unroll=1):
                            consumer_handle = ab_consumer.wait_and_advance()
                            cute.arch.fence_view_async_shared()
                            is_bias_kblk = k_local % n_tiles == n_tile
                            for kb in cutlass.range_constexpr(num_kblks):
                                crd = (None, None, kb, consumer_handle.index)
                                cute.gemm(
                                    tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                tiled_mma_bias.set(
                                    tcgen05.Field.ACCUMULATE, k_local > n_tile or kb > 0
                                )
                                if is_bias_kblk:
                                    cute.gemm(
                                        tiled_mma_bias,
                                        tCtAccBias,
                                        tCrA_bias[crd],
                                        tCrOnes[(None, None, kb, 0)],
                                        tCtAccBias,
                                    )
                            consumer_handle.release()
                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()
                acc_pipeline.producer_tail(acc_producer_state)
            else:
                # CTA1 idles: the leader's cta_group=2 UMMAs cover both CTAs.
                pass

        # ── Epilogue warps — both CTAs write their cta_tile_m D-row half ──────
        if warp_idx < MMA_WARP_ID:
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_consumer_state = cutlass.pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, _NUM_ACC
            )
            epi_sync = cutlass.pipeline.NamedBarrier(
                barrier_id=EPI_SYNC_BAR_ID, num_threads=EPILOGUE_THREADS
            )
            # TMA store pipeline: SMEM → GMEM (epi_stages_c slots, one per epi_n sub-tile)
            epi_tma_pipeline = cutlass.pipeline.PipelineTmaStore.create(
                # pyrefly: ignore [bad-argument-type]  # epi_stages_c is a Constexpr
                num_stages=epi_stages_c,
                producer_group=cutlass.pipeline.CooperativeGroup(
                    cutlass.pipeline.Agent.Thread
                ),
            )
            epi_tma_pipeline.producer_acquire()
            # Init per-cluster TMA descriptor slot for C (dense output, no per-tile update)
            _tmap_mgr_c = cutlass.utils.TensorMapManager(
                cutlass.utils.TensorMapUpdateMode.GMEM, 128
            )
            _cid_addr_c = mTmapAddrsC[cluster_id]
            _slot_c_gmem = _tmap_mgr_c.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64,
                    _cid_addr_c,
                    cute.AddressSpace.gmem,
                    assumed_align=128,
                ),
                cute.AddressSpace.gmem,
            )
            _slot_c_gen = _tmap_mgr_c.get_tensormap_ptr(
                cute.make_ptr(
                    cutlass.Int64,
                    _cid_addr_c,
                    cute.AddressSpace.gmem,
                    assumed_align=128,
                ),
                cute.AddressSpace.generic,
            )
            _tmap_mgr_c.init_tensormap_from_atom(tma_atom_c, _slot_c_gmem, warp_id=0)
            _tmap_mgr_c.fence_tensormap_initialization()
            tmem_load = sm100_utils.get_tmem_load_op(
                (cfg.cta_tile_m, cfg.tile_n, cfg.tile_k),
                cutlass.utils.LayoutEnum.ROW_MAJOR,
                self.ab_dtype,
                self.acc_dtype,
                epi_tile_shape,
                cfg.two_cta,
            )
            # d_bias readback: this CTA's (cta_tile_m, _BIAS_N) slice of the bias
            # accumulator, one TMEM lane (= one K row) per epilogue thread.
            tCtAccBias = cute.make_tensor(
                tmem_ptr + bias_tmem_col, tCtAccBias_fake.layout
            )
            tAccBias = tCtAccBias[((None, None), 0, 0)]
            tiled_t2r_bias = tcgen05.make_tmem_copy(
                cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), self.acc_dtype
                ),
                tAccBias,
            )
            thr_t2r_bias = tiled_t2r_bias.get_slice(tidx)
            tTR_tBias = thr_t2r_bias.partition_S(tAccBias)
            tTR_cBias = thr_t2r_bias.partition_D(
                cute.make_identity_tensor((cfg.cta_tile_m, self._BIAS_N))
            )
            rBias = cute.make_rmem_tensor(tTR_cBias.shape, self.acc_dtype)
            bias_row = tTR_cBias[0][0]

            for t in cutlass.range(cluster_id, total_tiles, n_clusters, unroll=1):
                n_tile = t % n_tiles
                mtile = (t // n_tiles) % m_tiles
                bz = mOrder[(t // n_tiles) // m_tiles]  # longest group first
                seq_start = mSeq[bz]
                seq_len = mSeq[bz + 1] - seq_start
                if seq_len > 0:
                    # 2-CTA: each CTA writes its own cta_tile_m rows.
                    m_base = mtile * cfg.mma_tile_m + mma_tile_coord_v * cfg.cta_tile_m

                    acc_pipeline.consumer_wait(acc_consumer_state)
                    if n_tile < cute.ceil_div(seq_len, cfg.tile_k):
                        cute.copy(tiled_t2r_bias, tTR_tBias, rBias)
                        bias_k = m_base + bias_row
                        if bias_k < D_static:
                            mDBias[bz, n_tile, bias_k] = rBias[0]
                    tAcc_epi0 = cute.flat_divide(
                        tCtAcc_base[((None, None), 0, 0, acc_consumer_state.index)],
                        epi_tile_shape,
                    )
                    tiled_t2r0 = tcgen05.make_tmem_copy(
                        tmem_load, tAcc_epi0[(None, None, 0, 0)]
                    )
                    thr_t2r0 = tiled_t2r0.get_slice(tidx)
                    tAcc0 = cute.group_modes(thr_t2r0.partition_S(tAcc_epi0), 3, 5)

                    if cutlass.const_expr(self._fast_epilogue(D_static, K_static)):
                        # ── FAST PATH: st.global (no boundary check, no SMEM staging) ──
                        # All tiles are full — output dims exactly divide tile sizes.
                        gC_mnl = cute.local_tile(
                            mC,
                            cute.slice_(mma_tiler, (None, None, 0)),
                            (None, None, None),
                        )
                        tCgC = thr_mma.partition_C(gC_mnl)
                        gC = tCgC[((None, None), 0, 0, mtile, n_tile, bz)]
                        tC = cute.group_modes(
                            thr_t2r0.partition_D(cute.flat_divide(gC, epi_tile_shape)),
                            3,
                            5,
                        )
                        frag_shape = tC[(None, None, None, 0)].shape
                        rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
                        rC_out = cute.make_rmem_tensor(frag_shape, self.c_dtype)

                        for sub in cutlass.range_constexpr(cute.size(tAcc0, mode=[3])):
                            cute.copy(tiled_t2r0, tAcc0[(None, None, None, sub)], rAcc)
                            rC_out.store(rAcc.load().to(self.c_dtype))
                            cute.autovec_copy(rC_out, tC[(None, None, None, sub)])

                        epi_sync.arrive_and_wait()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                        epi_sync.arrive_and_wait()

                    else:
                        # ── SLOW PATH: TMA staging (non-aligned shapes, boundary suppression) ──
                        copy_atom_r2s = sm100_utils.get_smem_store_op(
                            cutlass.utils.LayoutEnum.COL_MAJOR,
                            self.c_dtype,
                            self.acc_dtype,
                            tiled_t2r0,
                        )
                        tiled_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_t2r0)
                        thr_r2s = tiled_r2s.get_slice(tidx)
                        # pyrefly: ignore [unbound-name]  # allocated under the same const_expr(not _fast_epilogue) condition
                        tSD = thr_r2s.partition_D(sC_staging)

                        gC_mnl_tma = cute.local_tile(
                            mC_tma,
                            cute.slice_(mma_tiler, (None, None, 0)),
                            (None, None, None),
                        )
                        tCgC_tma = thr_mma.partition_C(gC_mnl_tma)
                        gC_full_tma = tCgC_tma[((None, None), 0, 0, mtile, n_tile, bz)]
                        gC_epi_tma = cute.flat_divide(gC_full_tma, epi_tile_shape)
                        bSG_sC, bSG_gC = cpasync.tma_partition(
                            tma_atom_c,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sC_staging, 0, 2),
                            cute.group_modes(gC_epi_tma[(None, None, 0, None)], 0, 2),
                        )
                        bSG_gC_grouped = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                        _fake_sc_ptr = cute.make_ptr(
                            self.acc_dtype, 0, cute.AddressSpace.smem, assumed_align=16
                        )
                        _fake_sc = cute.make_tensor(
                            _fake_sc_ptr,
                            cute.make_layout((cfg.cta_tile_m, cfg.tile_n)),
                        )
                        _fake_tSC = cute.group_modes(
                            thr_t2r0.partition_D(
                                cute.flat_divide(_fake_sc, epi_tile_shape)
                            ),
                            3,
                            5,
                        )
                        frag_shape = _fake_tSC[(None, None, None, 0)].shape
                        rAcc = cute.make_fragment(frag_shape, self.acc_dtype)
                        rC_staging = cute.make_rmem_tensor(frag_shape, self.c_dtype)

                        for sub in cutlass.range_constexpr(cute.size(tAcc0, mode=[3])):
                            c_buf = sub % epi_stages_c
                            cute.copy(tiled_t2r0, tAcc0[(None, None, None, sub)], rAcc)
                            rC_staging.store(rAcc.load().to(self.c_dtype))
                            rC_retiled = tiled_r2s.retile(rC_staging)
                            cute.copy(
                                tiled_r2s, rC_retiled, tSD[(None, None, None, c_buf)]
                            )

                        cute.arch.fence_view_async_shared()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()
                        epi_sync.arrive_and_wait()

                        if warp_idx == EPILOGUE_WARP_ID[0]:
                            for sub in cutlass.range_constexpr(
                                cute.size(tAcc0, mode=[3])
                            ):
                                c_buf = sub % epi_stages_c
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sC[(None, c_buf)],
                                    bSG_gC_grouped[(None, sub)],
                                    tma_desc_ptr=_slot_c_gen,
                                )
                                epi_tma_pipeline.producer_commit()
                                epi_tma_pipeline.producer_acquire()
                            epi_tma_pipeline.producer_tail()
                        epi_sync.arrive_and_wait()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch functions
# ─────────────────────────────────────────────────────────────────────────────


def _cutedsl_jagged_dense_bmm_broadcast_add_djagged(
    seq_offsets: torch.Tensor,
    d_out: torch.Tensor,
    dense: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """d_jagged = d_out @ dense[g].T  (K-major B, no copy)."""
    assert is_sm100_plus()
    G, D, N = dense.shape
    total = d_out.shape[0]
    _require_supported(
        "_cutedsl_jagged_dense_bmm_broadcast_add_djagged", (d_out, dense), (D, N)
    )
    d_out, dense = _tma_ready(d_out), _tma_ready(dense)
    d_jagged = torch.empty((total, D), dtype=d_out.dtype, device=d_out.device)

    dense_d = dense.detach()
    b3_bwd = dense_d.permute(1, 2, 0)  # (D, K, G), K stride-1

    a3 = d_out.detach().unsqueeze(-1)
    c3 = d_jagged.unsqueeze(-1)

    mA = from_dlpack(a3, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1, 2)
    )
    mB = from_dlpack(b3_bwd, assumed_align=16)
    mC = from_dlpack(c3, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1, 2)
    )
    mSeq = from_dlpack(seq_offsets, assumed_align=8)

    # The kernel reads the offsets on device (mSeq), so they are not in the key: one
    # compile serves every batch of these dims, with no host sync.
    key = (
        torch.cuda.current_device(),
        D,
        N,
        G,
        int(max_seq_len),
        d_out.dtype,
        seq_offsets.dtype,
    )
    if key not in _COMPILED_BWD_DJAGGED:
        _n_cta_slots = 2 * _max_active_clusters()
        _tmap_buf = torch.empty(
            (_n_cta_slots, 128), dtype=torch.uint8, device=d_out.device
        )
        _tmap_base = int(_tmap_buf.data_ptr())
        _tmap_addrs_t = torch.tensor(
            [_tmap_base + slot * 128 for slot in range(_n_cta_slots)],
            dtype=torch.int64,
            device=d_out.device,
        )
        _mTmapAddrs = from_dlpack(_tmap_addrs_t).mark_layout_dynamic()
        op = JaggedDenseBmmBroadcastAddDJagged(
            cutlass.BFloat16, cutlass.Float32, get_cfg()
        )
        _COMPILED_BWD_DJAGGED[key] = (
            cute.compile(
                op,
                mA,
                mB,
                mC,
                mSeq,
                int(max_seq_len),
                _max_active_clusters(),
                _stream(),
                D,
                _mTmapAddrs,
            ),
            _tmap_buf,  # keep alive
            _tmap_addrs_t,  # keep alive
            _mTmapAddrs,
        )
    _compiled_op, _tmap_buf, _tmap_addrs_t, _mTmapAddrs = _COMPILED_BWD_DJAGGED[key]
    _compiled_op(mA, mB, mC, mSeq, _stream(), _mTmapAddrs)
    return d_jagged


def _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    d_out: torch.Tensor,
    max_seq_len: int,
) -> tuple:
    """d_dense[g] = jagged[g].T @ d_out[g] and d_bias[g] = d_out[g].sum(0), fused in one
    2-CTA kernel (TMA-OOB). Returns (d_dense, d_bias)."""
    assert is_sm100_plus()
    D = jagged.shape[1]
    G = int(seq_offsets.numel()) - 1
    K = d_out.shape[1]
    _require_supported(
        "_cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias", (jagged, d_out), (D, K)
    )
    d_out_c = _tma_ready(d_out)
    jagged_c = _tma_ready(jagged)
    # Operand swap: A=d_out.T[K,M] (stride_M=1536B), B=jagged.T[D,M] (stride_M=4352B).
    # Kernel output: C[k,d]=d_dense[g,d,k] written to mC=[K,D,G] (K-fast, COLUMN_MAJOR SMEM).
    # The kernel writes d_dense[g,d,k] in the input dtype directly — return as-is.
    # Zero-filled only because tiles of empty groups (seq_len == 0) are skipped.
    d_dense = torch.zeros(G, D, K, dtype=jagged.dtype, device=jagged.device)
    n_tiles_d = -(-D // JaggedDenseBmmBroadcastAddDDenseDBias._TILE_N)
    d_bias_partial = torch.zeros(
        G, n_tiles_d, K, dtype=torch.float32, device=jagged.device
    )
    c3 = d_dense.permute(2, 1, 0)  # (K, D, G) — K-fast, correct for new A=d_out.T

    seq_offsets_i32 = seq_offsets.to(torch.int32)
    _jag_3d = jagged_c.detach().permute(1, 0).unsqueeze(-1)  # (D, total_M, 1)
    _dout_3d = d_out_c.detach().permute(1, 0).unsqueeze(-1)  # (K, total_M, 1)
    mA = from_dlpack(_dout_3d, assumed_align=16).mark_compact_shape_dynamic(
        mode=1, stride_order=(2, 1, 0)
    )  # A = d_out.T [K, M]
    mB = from_dlpack(_jag_3d, assumed_align=16).mark_compact_shape_dynamic(
        mode=1, stride_order=(2, 1, 0)
    )  # B = jagged.T [D, M]
    mC = from_dlpack(c3, assumed_align=16)
    mDBias = from_dlpack(d_bias_partial, assumed_align=16)
    mSeq = from_dlpack(seq_offsets_i32, assumed_align=8)
    _order = torch.argsort(
        seq_offsets_i32[1:] - seq_offsets_i32[:-1], descending=True, stable=True
    ).to(torch.int32)
    mOrder = from_dlpack(_order, assumed_align=4)
    key = (
        torch.cuda.current_device(),
        D,
        K,
        G,
        jagged.dtype,
        seq_offsets_i32.dtype,
    )
    n_persist = _max_active_clusters(cluster_size=2)

    if key not in _COMPILED_BWD_DDENSE_DBIAS:
        n_cta_slots = 2 * n_persist
        _tmap_buf_a = torch.empty(
            (n_cta_slots, 128), dtype=torch.uint8, device=jagged.device
        )
        _tmap_buf_b = torch.empty(
            (n_cta_slots, 128), dtype=torch.uint8, device=jagged.device
        )
        _addrs_a = torch.tensor(
            [int(_tmap_buf_a.data_ptr()) + i * 128 for i in range(n_cta_slots)],
            dtype=torch.int64,
            device=jagged.device,
        )
        _addrs_b = torch.tensor(
            [int(_tmap_buf_b.data_ptr()) + i * 128 for i in range(n_cta_slots)],
            dtype=torch.int64,
            device=jagged.device,
        )
        mTmapAddrsA = from_dlpack(_addrs_a).mark_layout_dynamic()
        mTmapAddrsB = from_dlpack(_addrs_b).mark_layout_dynamic()
        _tmap_buf_c = torch.empty(
            (n_persist, 128), dtype=torch.uint8, device=jagged.device
        )
        _addrs_c = torch.tensor(
            [int(_tmap_buf_c.data_ptr()) + i * 128 for i in range(n_persist)],
            dtype=torch.int64,
            device=jagged.device,
        )
        mTmapAddrsC = from_dlpack(_addrs_c).mark_layout_dynamic()
        op = JaggedDenseBmmBroadcastAddDDenseDBias(
            cutlass.BFloat16, cutlass.Float32, get_cfg()
        )
        _COMPILED_BWD_DDENSE_DBIAS[key] = (
            cute.compile(
                op,
                mA,
                mB,
                mC,
                mDBias,
                mSeq,
                mOrder,
                mTmapAddrsA,
                mTmapAddrsB,
                mTmapAddrsC,
                n_persist,
                _stream(),
                K,
                D,
            ),
            _tmap_buf_a,
            _tmap_buf_b,
            _tmap_buf_c,
            _addrs_a,
            _addrs_b,
            _addrs_c,
            mTmapAddrsA,
            mTmapAddrsB,
            mTmapAddrsC,
        )

    (
        _compiled_op,
        _tba,
        _tbb,
        _tbc,
        _aa,
        _ab,
        _ac,
        mTmapAddrsA,
        mTmapAddrsB,
        mTmapAddrsC,
    ) = _COMPILED_BWD_DDENSE_DBIAS[key]
    _compiled_op(
        mA,
        mB,
        mC,
        mDBias,
        mSeq,
        mOrder,
        mTmapAddrsA,
        mTmapAddrsB,
        mTmapAddrsC,
        _stream(),
    )
    return d_dense, d_bias_partial.sum(dim=1).to(d_out.dtype)
