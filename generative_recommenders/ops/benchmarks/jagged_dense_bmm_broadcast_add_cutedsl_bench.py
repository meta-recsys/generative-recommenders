"""Benchmarks for the CuTeDSL (Blackwell tcgen05) backend of jagged_dense_bmm_broadcast_add,
against Triton. Needs a B200 (sm_100); the Triton/PyTorch-only benchmark is
jagged_dense_bmm_broadcast_add_bench.

  sweep      fwd / bwd over seq_len (triton.testing.perf_report), CuTeDSL vs Triton.
  components backward per kernel (d_jagged, fused d_dense + d_bias) and the full autograd
             backward, CuTeDSL vs Triton, at the production shape. Variants are interleaved
             within each round and timed at several do_bench windows: devserver B200s are
             power-capped at 750 W, so a short window measures boost clocks and a long one the
             sustained clock.
  tail-tile  fused d_dense + d_bias at D = 2048 / 2176 / 2304: what the half-empty last N
             tile (D % 256 = 128 at the production D=2176) costs.

buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_dense_bmm_broadcast_add_cutedsl_bench -- sweep
buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_dense_bmm_broadcast_add_cutedsl_bench -- components --shape uniform --shape jagged
(The .par launcher takes `-d` for itself: use `--jagged-dim` / `--out-dim`, or `-d2176`.)
"""

import math
from typing import Callable, Dict, List, Tuple

import click
import pandas as pd
import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.cutedsl.cutedsl_jagged import (
    _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias,
    _cutedsl_jagged_dense_bmm_broadcast_add_djagged,
)
from generative_recommenders.ops.jagged_tensors import jagged_dense_bmm_broadcast_add
from generative_recommenders.ops.triton.triton_jagged import (
    triton_jagged_dense_bmm_broadcast_add_bwd_ddense_dbias,
    triton_jagged_dense_bmm_broadcast_add_bwd_djagged,
)
from generative_recommenders.ops.utils import is_sm100_plus

KERNELS: Dict[str, HammerKernel] = {
    "triton": HammerKernel.TRITON,
    "cutedsl": HammerKernel.CUTEDSL,
}


def _seq_offsets(lengths: torch.Tensor) -> torch.Tensor:
    seq_offsets = torch.zeros(
        (lengths.numel() + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths.to(seq_offsets.device), dim=0)
    return seq_offsets


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16, device=torch.device("cuda"))


def _with_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Speedup = Triton time / CuTeDSL time (> 1x: CuTeDSL is faster)."""
    df["Speedup"] = (df["Triton"] / df["CuTeDSL"]).map("{:.2f}x".format)
    return df


def _full_backward(
    kernel: HammerKernel,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    d_out: torch.Tensor,
) -> Callable[[], None]:
    """The autograd backward of one forward, re-run on each call (what training pays)."""
    j, d, b = (t.detach().requires_grad_(True) for t in (jagged, dense, bias))
    out = jagged_dense_bmm_broadcast_add(
        max_seq_len, seq_offsets, j, d, b, kernel=kernel
    )

    def fn() -> None:
        out.backward(d_out, retain_graph=True)
        j.grad = d.grad = b.grad = None

    return fn


@click.group()
def cli() -> None:
    if not is_sm100_plus():
        raise click.UsageError(
            "the CuTeDSL kernels need a Blackwell datacenter GPU (sm_100-sm_103); use "
            "jagged_dense_bmm_broadcast_add_bench for Triton/PyTorch"
        )


@cli.command()
@click.option("--batch-size", type=int, default=384, show_default=True)
@click.option("--max-seq-len", type=int, default=4096, show_default=True)
@click.option("-d", "--jagged-dim", "d", type=int, default=512, show_default=True)
@click.option("-k", "--out-dim", "k", type=int, default=512, show_default=True)
@click.option("--fwd-only", is_flag=True)
def sweep(batch_size: int, max_seq_len: int, d: int, k: int, fwd_only: bool) -> None:
    """fwd / bwd time over seq_len, CuTeDSL vs Triton (bf16)."""
    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(8, int(round(math.log2(max_seq_len))) + 1)],
            line_arg="provider",
            line_vals=["triton", "cutedsl"],
            line_names=["Triton", "CuTeDSL"],
            styles=[("red", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"jagged_dense_bmm_broadcast_add-{mode}-b{batch_size}-D{d}-K{k}-bf16",
            args={"batch_size": batch_size, "D": d, "K": k, "mode": mode},
        )
        for mode in (["fwd"] if fwd_only else ["fwd", "bwd"])
    ]

    @triton.testing.perf_report(configs)
    def bench(
        batch_size: int, seq_len: int, D: int, K: int, mode: str, provider: str
    ) -> float:
        # Seeded per point so both providers are timed on the same lengths: at small
        # batch sizes, total rows vary widely between draws.
        lengths = torch.randint(
            seq_len + 1,
            size=(batch_size,),
            generator=torch.Generator().manual_seed(seq_len),
        )
        seq_offsets = _seq_offsets(lengths)
        jagged = _randn(int(seq_offsets[-1].item()), D).requires_grad_()
        dense = _randn(batch_size, D, K).requires_grad_()
        bias = _randn(batch_size, K).requires_grad_()
        fn = lambda: jagged_dense_bmm_broadcast_add(  # noqa E731
            max_seq_len=seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
            kernel=KERNELS[provider],
        )
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
        return triton.testing.do_bench(fn, warmup=25, rep=100)

    for config, df in zip(configs, bench.run(return_df=True)):
        # Newer Triton labels the columns "<line name> (<ylabel>)".
        df.columns = [c.removesuffix(f" ({config.ylabel})") for c in df.columns]
        print(f"{config.plot_name}:")
        print(_with_speedup(df).to_string())


@cli.command()
@click.option(
    "--shape",
    type=click.Choice(["uniform", "jagged"]),
    multiple=True,
    default=["uniform"],
    show_default=True,
    help="uniform: every group 21045 rows; jagged: lengths uniform in [0, max-seq-len].",
)
@click.option("--batch-size", type=int, default=40, show_default=True)
@click.option("--max-seq-len", type=int, default=32743, show_default=True)
@click.option("-d", "--jagged-dim", "d", type=int, default=2176, show_default=True)
@click.option("-k", "--out-dim", "k", type=int, default=768, show_default=True)
@click.option(
    "--rep-ms",
    type=str,
    default="100,1000",
    show_default=True,
    help="do_bench windows (comma-separated ms).",
)
@click.option("--rounds", type=int, default=2, show_default=True)
def components(
    shape: Tuple[str, ...],
    batch_size: int,
    max_seq_len: int,
    d: int,
    k: int,
    rep_ms: str,
    rounds: int,
) -> None:
    """Backward per kernel and in full, CuTeDSL vs Triton (best of `rounds`)."""
    for s in shape:
        _components_one(
            s,
            batch_size,
            max_seq_len,
            d,
            k,
            [int(w) for w in rep_ms.split(",")],
            rounds,
        )


def _components_one(
    shape: str,
    batch_size: int,
    max_seq_len: int,
    d: int,
    k: int,
    windows: List[int],
    rounds: int,
) -> None:
    torch.manual_seed(0)
    lengths = (
        torch.full((batch_size,), 21045)
        if shape == "uniform"
        else torch.randint(0, max_seq_len + 1, (batch_size,))
    )
    seq_offsets = _seq_offsets(lengths)
    total = int(seq_offsets[-1].item())
    jagged, dense, bias = (
        _randn(total, d),
        _randn(batch_size, d, k),
        _randn(batch_size, k),
    )
    d_out = _randn(total, k)
    msl = max_seq_len
    variants: Dict[str, Callable[[], object]] = {
        "d_jagged triton": lambda: triton_jagged_dense_bmm_broadcast_add_bwd_djagged(
            msl, seq_offsets, d_out, dense, jagged
        ),
        "d_jagged cutedsl": lambda: _cutedsl_jagged_dense_bmm_broadcast_add_djagged(
            seq_offsets, d_out, dense, msl
        ),
        "d_dense+d_bias triton": lambda: (
            triton_jagged_dense_bmm_broadcast_add_bwd_ddense_dbias(
                msl, seq_offsets, jagged, d_out, dense
            )
        ),
        "d_dense+d_bias cutedsl": lambda: (
            _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                seq_offsets, jagged, d_out, msl
            )
        ),
        "full bwd triton": _full_backward(
            HammerKernel.TRITON, msl, seq_offsets, jagged, dense, bias, d_out
        ),
        "full bwd cutedsl": _full_backward(
            HammerKernel.CUTEDSL, msl, seq_offsets, jagged, dense, bias, d_out
        ),
    }
    for fn in variants.values():  # compile, autotune, warm
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    times: Dict[Tuple[str, int], List[float]] = {
        (n, w): [] for n in variants for w in windows
    }
    for _ in range(rounds):
        for w in windows:
            for n, fn in variants.items():
                times[(n, w)].append(triton.testing.do_bench(fn, warmup=25, rep=w))
    best = {key: min(v) for key, v in times.items()}

    parts = ["d_jagged", "d_dense+d_bias", "full bwd"]
    for w in windows:
        df = pd.DataFrame(
            {
                "kernel": parts,
                "Triton": [best[(f"{p} triton", w)] for p in parts],
                "CuTeDSL": [best[(f"{p} cutedsl", w)] for p in parts],
            }
        )
        print(
            f"jagged_dense_bmm_broadcast_add-bwd-{shape}-b{batch_size}-D{d}-K{k}-bf16 "
            f"(total_M={total:,}, {w} ms window):"
        )
        print(_with_speedup(df).to_string())


@cli.command("tail-tile")
@click.option("--rounds", type=int, default=3, show_default=True)
def tail_tile(rounds: int) -> None:
    """Fused d_dense + d_bias at D = 2048 / 2176 / 2304 (G=40, 21045 rows/group, K=768).

    2048 is 8 full 256-wide N tiles, 2304 is 9, and the production 2176 is 8 plus one with
    128 of its 256 columns valid. If that tile costs only its useful half,
    t(2176) ~= t(2048) * 8.5 / 8; if it costs a full tile, t(2176) ~= t(2304)."""
    num_groups, rows, k = 40, 21045, 768
    seq_offsets = _seq_offsets(torch.full((num_groups,), rows))
    d_out = _randn(num_groups * rows, k) * 0.01
    inputs = {d: _randn(num_groups * rows, d) for d in (2048, 2176, 2304)}
    for jagged in inputs.values():  # compile + warm
        _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
            seq_offsets, jagged, d_out, rows
        )
    torch.cuda.synchronize()
    times: Dict[int, List[float]] = {d: [] for d in inputs}
    for _ in range(rounds):  # interleaved: the power cap drifts between rounds
        for d, jagged in inputs.items():
            times[d].append(
                triton.testing.do_bench(
                    lambda jagged=jagged: _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias(
                        seq_offsets, jagged, d_out, rows
                    ),
                    warmup=200,
                    rep=1000,
                )
            )
    for d, t in times.items():
        print(f"[tail-tile] D={d}: " + " ".join(f"{x:.3f}" for x in t) + " ms")
    for r in range(rounds):
        t2048, t2176, t2304 = times[2048][r], times[2176][r], times[2304][r]
        wasted = (t2176 - t2048 * 8.5 / 8) / t2176
        print(
            f"[tail-tile] round {r}: t2176/t2048={t2176 / t2048:.3f} (useful-work ratio "
            f"1.0625, full-tile ratio 1.125)  t2176/t2304={t2176 / t2304:.3f}  -> time on "
            f"the empty half: {wasted:.1%}"
        )


if __name__ == "__main__":
    cli()
