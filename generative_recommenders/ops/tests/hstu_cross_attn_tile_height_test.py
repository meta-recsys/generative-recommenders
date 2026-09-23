# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# pyre-strict

"""Tile-height sweep for the CUTLASS inference cross-attention fwd kernel
(T288157648): kBlockM 64 (current) vs 128, across q length.

kBlockM 64 sets AtomLayoutM to 1, i.e. one MMA warpgroup and no softmax/GEMM
ping-pong; 128 gives two of each but needs enough q blocks to still fill the
GPU. This sweep locates the crossover so the dispatch guard can key on wave
count. Protocol:

  * All per-iteration setup (int32 offset casts, the attn_scale host->device
    tensor, the fp32 cast) is hoisted out of the timed region; only the kernel
    launch and its output allocation remain.
  * Arms are interleaved round-robin (ABAB...) for REPS passes, so clock
    drift and thermal droop hit every arm equally instead of accumulating
    against whichever arm ran last.
  * Reported as median over REPS with min/max, not a single sample.
  * The realized KV lengths and the seed are logged so a run is reproducible
    and the actual spread of the jagged batch is visible.

Runs on an H100 remote-execution GPU worker, so it works from a GPU-less
devserver:
  buck2 test @//mode/opt -c fbcode.enable_gpu_sections=true \\
    //generative_recommenders/ops/tests:hstu_cross_attn_tile_height_test \\
    -- --print-passing-details
"""

import logging
import math
import os
import statistics
import unittest
from typing import Dict, List, Optional, Tuple

import torch
from generative_recommenders.common import generate_sparse_seq_len
from generative_recommenders.ops.cpp.cuda_hstu_attention import (
    cuda_hstu_mha,
    LARGE_BLOCKM_AUTO,
    LARGE_BLOCKM_OFF,
    LARGE_BLOCKM_ON,
)

logger: logging.Logger = logging.getLogger(__name__)

# The production RankFM inference shape from T288157648.
BATCH_SIZE = 8
NUM_HEADS = 6
HEAD_DIM = 128
MAX_KV_LEN = 32768
KV_SPARSITY = 0.5
Q_LEN = 1300
SEED = 1001

REPS = 7
WARMUP_ITERS = 20
TIMED_ITERS = 50

# large_blockm_fwd, pinned rather than left on AUTO so the sweep measures both
# tile heights at every q instead of whatever the wave-count guard would pick.
ARMS: List[int] = [LARGE_BLOCKM_OFF, LARGE_BLOCKM_ON]
# q lengths to sweep; 1300 is production.
Q_SWEEP: List[int] = [256, 512, 768, 1024, 1300, 2048]


def _arm_label(large_blockm: int) -> str:
    return "kBlockM=128" if large_blockm == LARGE_BLOCKM_ON else "kBlockM=64"


def _num_ctas(q_len: int, block_m: int) -> int:
    """CTAs one cross-attention call launches: ceil(q/kBlockM) * heads * batch.

    This is the quantity the dispatch guard compares against num_sm, so the
    sweep reports it directly rather than recomputing it inline.
    """
    return -(-q_len // block_m) * NUM_HEADS * BATCH_SIZE


def _hopper_unavailable() -> bool:
    if not torch.cuda.is_available():
        return True
    major, _ = torch.cuda.get_device_capability()
    return major != 9


class _Inputs:
    """Everything the kernel needs, in final dtype, allocated once."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_kv_len: int,
        q_len: int,
        sparsity: float,
        seed: int = SEED,
    ) -> None:
        torch.manual_seed(seed)
        device = torch.device("cuda")
        # generate_sparse_seq_len at sparsity=0.5 takes the >= 0.5 branch with
        # min_seq_len = int((2*0.5-1.0)*max) = 0, i.e. uniform over [0, max).
        # Clamping to 1 only lifts an exact 0 (p = 1/max per sequence): the
        # CUTLASS provider stalls forever on an empty sequence in the batch.
        lengths_kv = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_kv_len,
            sparsity=sparsity,
            device=device,
        ).clamp(min=1, max=max_kv_len)
        self.lengths_kv: List[int] = lengths_kv.tolist()
        self.seed = seed
        self.q_len = q_len
        self.max_kv_len = max_kv_len
        self.batch_size = batch_size
        self.num_heads = num_heads

        seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        seq_offsets[1:] = torch.cumsum(lengths_kv, dim=0)
        seq_offsets_q = torch.arange(
            0, (batch_size + 1) * q_len, q_len, dtype=torch.int64, device=device
        )
        total_q = int(seq_offsets_q[-1].item())
        total_kv = int(seq_offsets[-1].item())

        self.q: torch.Tensor = torch.empty(
            (total_q, num_heads, head_dim), dtype=torch.bfloat16, device=device
        ).uniform_(-0.1, 0.1)
        # k and v are not shared for this model.
        self.k: torch.Tensor = torch.empty(
            (total_kv, num_heads, head_dim), dtype=torch.bfloat16, device=device
        ).uniform_(-0.1, 0.1)
        self.v: torch.Tensor = torch.empty(
            (total_kv, num_heads, head_dim), dtype=torch.bfloat16, device=device
        ).uniform_(-0.1, 0.1)

        # Hoisted out of the timed region: final dtypes, allocated once.
        self.max_seq_len: int = max_kv_len + q_len
        self.alpha: float = 1.0 / (head_dim**0.5)
        self.seq_offsets_i32: torch.Tensor = seq_offsets.to(torch.int32)
        self.seq_offsets_q_i32: torch.Tensor = seq_offsets_q.to(torch.int32)
        self.num_targets: torch.Tensor = torch.full(
            (batch_size,), q_len, dtype=torch.int32, device=device
        )
        self.attn_scale: torch.Tensor = torch.tensor(
            1.0 / self.max_seq_len, dtype=torch.float32, device=device
        )

    def run(self, large_blockm: int) -> torch.Tensor:
        return cuda_hstu_mha(
            max_seq_len=self.max_seq_len,
            alpha=self.alpha,
            q=self.q,
            k=self.k,
            v=self.v,
            seq_offsets=self.seq_offsets_i32,
            causal=False,
            num_targets=self.num_targets,
            attn_scale=self.attn_scale,
            max_q_len=self.q_len,
            seq_offsets_q=self.seq_offsets_q_i32,
            # softmax activation on every head
            num_softmax_heads=self.num_heads,
            training=False,
            large_blockm_fwd=large_blockm,
        )

    def describe(self) -> str:
        lens = sorted(self.lengths_kv)
        return (
            f"seed={self.seed} b={self.batch_size} h={self.num_heads} "
            f"kv_cap={self.max_kv_len} q={self.q_len}\n"
            f"  realized KV lengths (sorted): {lens}\n"
            f"  min={lens[0]} median={statistics.median(lens):.0f} max={lens[-1]} "
            f"mean={statistics.mean(lens):.0f} total={sum(lens)}"
        )


def _sm_clock_mhz() -> Optional[int]:
    try:
        import pynvml  # pyre-ignore[21]

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:
        return None


def _time_once(inp: _Inputs, large_blockm: int) -> float:
    """One sample: mean ms over TIMED_ITERS launches."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(TIMED_ITERS):
        inp.run(large_blockm)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / TIMED_ITERS


@unittest.skipIf(_hopper_unavailable(), "requires an sm90 (H100) GPU")
class HstuCrossAttnTileHeightTest(unittest.TestCase):
    def test_tile_height_parity(self) -> None:
        """kBlockM only groups q rows; per-row accumulation order is unchanged.

        Reported as a real tolerance comparison rather than an exact check, so
        that any accumulation-grouping drift shows up as a number.
        """
        for q_len in (Q_LEN, 256, 64, 1):
            with self.subTest(q_len=q_len):
                inp = _Inputs(
                    BATCH_SIZE, NUM_HEADS, HEAD_DIM, MAX_KV_LEN, q_len, KV_SPARSITY
                )
                base = inp.run(large_blockm=LARGE_BLOCKM_OFF).float()
                large = inp.run(large_blockm=LARGE_BLOCKM_ON).float()
                abs_err = (large - base).abs()
                denom = base.abs().clamp(min=1e-6)
                print(
                    f"\nkBlockM 64 vs 128 at q={q_len}: "
                    f"max_abs={abs_err.max().item():.3e} "
                    f"mean_abs={abs_err.mean().item():.3e} "
                    f"max_rel={(abs_err / denom).max().item():.3e} "
                    f"base_absmax={base.abs().max().item():.3e} "
                    f"bitwise_identical={bool(torch.equal(large, base))}",
                    flush=True,
                )
                torch.testing.assert_close(large, base, rtol=2e-2, atol=2e-2)

    def test_auto_matches_pinned_arms(self) -> None:
        """Exercise the AUTO wave-count guard, not just the two pinned arms.

        AUTO resolves to one tile height or the other inside the kernel, and
        which one it picks is not observable from Python. What is checkable is
        that the guard runs on both sides of the crossover and that whichever
        tile it lands on still produces the same numbers. q=1300 and 2048 are
        past one full wave (so the guard should take the 128-row tile) and
        q=64/256 are below it.
        """
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        for q_len in (64, 256, Q_LEN, 2048):
            with self.subTest(q_len=q_len):
                inp = _Inputs(
                    BATCH_SIZE, NUM_HEADS, HEAD_DIM, MAX_KV_LEN, q_len, KV_SPARSITY
                )
                auto = inp.run(large_blockm=LARGE_BLOCKM_AUTO)
                waves = _num_ctas(q_len, 128) / num_sm
                predicted = 128 if _num_ctas(q_len, 128) >= num_sm else 64
                print(
                    f"\nAUTO at q={q_len}: waves@128={waves:.2f}, "
                    f"guard predicts the {predicted}-row tile",
                    flush=True,
                )
                for arm in ARMS:
                    torch.testing.assert_close(
                        auto,
                        inp.run(large_blockm=arm),
                        rtol=2e-2,
                        atol=2e-2,
                        msg=f"AUTO disagrees with {_arm_label(arm)} at q={q_len}",
                    )

    def test_tile_height_q_sweep(self) -> None:
        """Locate the q length where kBlockM=128 stops paying off."""
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        lines: List[str] = [
            f"device={torch.cuda.get_device_name(0)} num_sm={num_sm}",
            f"protocol: interleaved round-robin, reps={REPS}, "
            f"warmup={WARMUP_ITERS}, timed_iters={TIMED_ITERS} per sample",
            "waves = ceil(q/kBlockM) * heads * batch / num_sm",
        ]
        header = (
            "q",
            "waves@64",
            "waves@128",
            "bm64 med",
            "bm128 med",
            "bm64 min",
            "bm128 min",
            "speedup",
            "clk lo-hi",
        )
        rows: List[Tuple[str, ...]] = []
        first = True
        for q_len in Q_SWEEP:
            inp = _Inputs(
                BATCH_SIZE, NUM_HEADS, HEAD_DIM, MAX_KV_LEN, q_len, KV_SPARSITY
            )
            if first:
                lines.append("")
                lines.append(inp.describe())
                first = False

            for large_blockm in ARMS:
                for _ in range(WARMUP_ITERS):
                    inp.run(large_blockm)
            torch.cuda.synchronize()

            samples: Dict[int, List[float]] = {a: [] for a in ARMS}
            clocks: List[int] = []
            for _rep in range(REPS):
                for arm in ARMS:
                    samples[arm].append(_time_once(inp, arm))
                mhz = _sm_clock_mhz()
                if mhz is not None:
                    clocks.append(mhz)

            # A timing sweep with no assertions passes even when the kernel is
            # broken, so assert on the sweep itself: every arm must actually
            # have run to completion, and the two tile heights must still
            # agree numerically at this q (the parity test only covers a few
            # fixed shapes).
            for arm in ARMS:
                self.assertEqual(
                    len(samples[arm]),
                    REPS,
                    f"{_arm_label(arm)} at q={q_len} did not produce REPS samples",
                )
                for sample in samples[arm]:
                    self.assertTrue(
                        math.isfinite(sample) and sample > 0.0,
                        f"{_arm_label(arm)} at q={q_len} timed {sample} ms",
                    )
            torch.testing.assert_close(
                inp.run(LARGE_BLOCKM_ON),
                inp.run(LARGE_BLOCKM_OFF),
                rtol=2e-2,
                atol=2e-2,
                msg=f"tile heights disagree numerically at q={q_len}",
            )

            med64 = statistics.median(samples[LARGE_BLOCKM_OFF])
            med128 = statistics.median(samples[LARGE_BLOCKM_ON])
            rows.append(
                (
                    str(q_len),
                    f"{_num_ctas(q_len, 64) / num_sm:.2f}",
                    f"{_num_ctas(q_len, 128) / num_sm:.2f}",
                    f"{med64:.3f}",
                    f"{med128:.3f}",
                    f"{min(samples[LARGE_BLOCKM_OFF]):.3f}",
                    f"{min(samples[LARGE_BLOCKM_ON]):.3f}",
                    f"{med64 / med128:.3f}",
                    f"{min(clocks)}-{max(clocks)}" if clocks else "n/a",
                )
            )
            del inp
            torch.cuda.empty_cache()

        widths = [max(len(r[i]) for r in (rows + [header])) for i in range(len(header))]
        lines.append("")
        lines.append("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        lines.append("  ".join("-" * w for w in widths))
        for r in rows:
            lines.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        report = "\n".join(lines)
        print("\n" + report, flush=True)
        logger.info("\n%s", report)
        out_dir = os.environ.get("TEST_RESULT_DIR")
        if out_dir:
            with open(os.path.join(out_dir, "hstu_q_sweep.txt"), "w") as f:
                f.write(report + "\n")


if __name__ == "__main__":
    unittest.main()
