# pyre-strict
import math
import pickle
from typing import Callable, Dict, List, Tuple

import click
import pandas as pd
import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.jagged_tensors import jagged_dense_bmm_broadcast_add
from generative_recommenders.ops.triton.triton_jagged import (
    jagged_dense_bmm_broadcast_add_kernel,
    triton_jagged_dense_bmm,
    triton_jagged_dense_bmm_broadcast_add_bwd_ddense_dbias,
    triton_jagged_dense_bmm_broadcast_add_bwd_djagged,
    triton_jagged_dense_broadcast_add,
)
from generative_recommenders.ops.utils import is_sm100_plus

# buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_dense_bmm_broadcast_add_bench -- --fwd-only

# Backward per kernel (d_jagged, d_dense + d_bias, full backward), Triton vs CuTeDSL:
# buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_dense_bmm_broadcast_add_bench -- --bwd-breakdown --batch-size 40 --max-seq-len 32743 -d 2176 -k 768 [--fixed-len 21045]

# To dump the jagged_dense_bmm_broadcast_add_kernel cache
# buck2 run @mode/opt //generative_recommenders/ops/benchmarks:jagged_dense_bmm_broadcast_add_bench -- --fwd-only --dump-cache-dir=/home/${USER}/fbsource/fbcode/generative_recommenders/ops/triton/jagged_dense_bmm_broadcast_add_kernel_cache.pkl


def get_kernel(provider: str) -> HammerKernel:
    if provider == "triton":
        return HammerKernel.TRITON
    elif provider == "pytorch":
        return HammerKernel.PYTORCH
    elif provider == "cutedsl":
        return HammerKernel.CUTEDSL
    else:
        raise ValueError(f"Unknown provider {provider}")


def jagged_dense_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Wrapper function for jagged_dense_broadcast_add with kernel selection.
    Computing out = jagged + dense (broadcasted)
    jagged has shape (sum_B(M_i), N), dense has shape (B, N)
    out has shape (sum_B(M_i), N)
    """
    if kernel == HammerKernel.TRITON:
        return triton_jagged_dense_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
        )
    elif kernel == HammerKernel.PYTORCH:
        # PyTorch implementation - manual implementation using standard operations
        B, N = dense.shape
        outputs = []
        for i in range(B):
            start_idx = seq_offsets[i]
            end_idx = seq_offsets[i + 1]
            if start_idx < end_idx:
                jagged_seq = jagged[start_idx:end_idx]  # (seq_len, N)
                dense_seq = dense[i]  # (N,)
                output_seq = jagged_seq + dense_seq  # (seq_len, N)
                outputs.append(output_seq)
        return (
            torch.cat(outputs, dim=0)
            if outputs
            else torch.empty(0, N, device=jagged.device, dtype=jagged.dtype)
        )
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")


def jagged_dense_bmm(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    """
    Wrapper function for jagged_dense_bmm with kernel selection.
    Computing out = jagged x dense
    jagged has shape (sum_B(M_i), K), dense has shape (B, K, N)
    out has shape (sum_B(M_i), N)
    """
    if kernel == HammerKernel.TRITON:
        return triton_jagged_dense_bmm(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
        )
    elif kernel == HammerKernel.PYTORCH:
        # PyTorch implementation - manual implementation using standard operations
        B, K, N = dense.shape
        outputs = []
        for i in range(B):
            start_idx = seq_offsets[i]
            end_idx = seq_offsets[i + 1]
            if start_idx < end_idx:
                jagged_seq = jagged[start_idx:end_idx]  # (seq_len, K)
                dense_seq = dense[i]  # (K, N)
                output_seq = torch.mm(jagged_seq, dense_seq)  # (seq_len, N)
                outputs.append(output_seq)
        return (
            torch.cat(outputs, dim=0)
            if outputs
            else torch.empty(0, N, device=jagged.device, dtype=jagged.dtype)
        )
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")


def _supported(
    provider: str, dtype: torch.dtype, batch_size: int, seq_len: int, D: int, K: int
) -> bool:
    if provider == "cutedsl":
        # sm_100+ and bf16 only; other dtypes would time the Triton fallback.
        return dtype == torch.bfloat16 and is_sm100_plus()
    if provider == "pytorch":
        # The PyTorch path pads to (B, seq_len, D) and overflows int32 indexing.
        return batch_size * seq_len * max(D, K) < 2**31
    return True


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
    j, w, b = (t.detach().requires_grad_(True) for t in (jagged, dense, bias))
    out = jagged_dense_bmm_broadcast_add(
        max_seq_len, seq_offsets, j, w, b, kernel=kernel
    )

    def fn() -> None:
        out.backward(d_out, retain_graph=True)
        j.grad = w.grad = b.grad = None

    return fn


def _bwd_breakdown(
    batch_size: int, max_seq_len: int, d: int, k: int, fixed_len: int, rounds: int = 2
) -> None:
    """d_jagged, fused d_dense + d_bias, and the full backward, Triton vs CuTeDSL (bf16).
    Variants are interleaved within each round and the best of `rounds` is kept: B200s
    on devservers are power-capped, so the clock drifts between rounds."""
    if not is_sm100_plus():
        raise click.UsageError(
            "--bwd-breakdown needs a Blackwell GPU (sm_100) for CuTeDSL"
        )
    from generative_recommenders.ops.cutedsl.cutedsl_jagged import (
        _cutedsl_jagged_dense_bmm_broadcast_add_ddense_dbias,
        _cutedsl_jagged_dense_bmm_broadcast_add_djagged,
    )

    torch.manual_seed(0)
    lengths = (
        torch.full((batch_size,), fixed_len)
        if fixed_len
        else torch.randint(0, max_seq_len + 1, (batch_size,))
    )
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths.to(seq_offsets.device), dim=0)
    total = int(seq_offsets[-1].item())

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.bfloat16, device=torch.device("cuda"))

    jagged, dense, bias = randn(total, d), randn(batch_size, d, k), randn(batch_size, k)
    d_out = randn(total, k)
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
    times: Dict[str, List[float]] = {n: [] for n in variants}
    for _ in range(rounds):
        for n, fn in variants.items():
            times[n].append(triton.testing.do_bench(fn, warmup=200, rep=1000))
    best: Dict[str, float] = {n: min(t) for n, t in times.items()}

    parts: Tuple[str, ...] = ("d_jagged", "d_dense+d_bias", "full bwd")
    df = pd.DataFrame(
        {
            "kernel": list(parts),
            "Triton": [best[f"{p} triton"] for p in parts],
            "CuTeDSL": [best[f"{p} cutedsl"] for p in parts],
        }
    )
    lens = f"len={fixed_len}" if fixed_len else f"len~U[0,{max_seq_len}]"
    print(
        f"jagged_dense_bmm_broadcast_add-bwd-breakdown-b{batch_size}-D{d}-K{k}-bf16 "
        f"({lens}, total_M={total:,}):"
    )
    print(_with_speedup(df).to_string())


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=384,
)
@click.option(
    "--max-seq-len",
    type=int,
    default=4096,
    show_default=True,
)
@click.option(
    "-d",
    type=int,
    default=512,
    show_default=True,
)
@click.option(
    "-k",
    type=int,
    default=512,
    show_default=True,
)
@click.option("--dtype", type=str, default="bf16")
@click.option("--fwd-only", is_flag=True)
@click.option("--dump-cache-dir", type=str, default="")
@click.option(
    "--bwd-breakdown",
    is_flag=True,
    help="Instead of the seq_len sweep: time each backward kernel and the full backward, "
    "Triton vs CuTeDSL, at --batch-size/--max-seq-len/-d/-k (bf16, sm_100 only).",
)
@click.option(
    "--fixed-len",
    type=int,
    default=0,
    help="With --bwd-breakdown: every group this long, instead of lengths drawn "
    "uniformly from [0, max_seq_len].",
)
def main(
    batch_size: int,
    max_seq_len: int,
    d: int,
    k: int,
    dtype: str,
    fwd_only: bool,
    dump_cache_dir: str,
    bwd_breakdown: bool,
    fixed_len: int,
) -> None:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    max_seq_len_log2 = int(round(math.log2(max_seq_len)))
    if dtype == "fp32":
        pt_dtype = torch.float32
    elif dtype == "fp16":
        pt_dtype = torch.float16
    elif dtype == "bf16":
        pt_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {dtype}.")
    if bwd_breakdown:
        _bwd_breakdown(batch_size, max_seq_len, d, k, fixed_len)
        return

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(8, max_seq_len_log2 + 1)],
            line_arg="provider",
            line_vals=["triton", "pytorch", "triton_nonfused", "cutedsl"],
            line_names=["Triton", "Pytorch", "Triton_Nonfused", "CuTeDSL"],
            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"jagged_dense_bmm_broadcast_add-{mode}-b{batch_size}-D{d}-K{k}-{dtype}",
            args={
                "batch_size": batch_size,
                "D": d,
                "K": k,
                "dtype": pt_dtype,
                "mode": mode,
            },
        )
        for mode in (["fwd"] if fwd_only else ["fwd", "bwd"])
    ]

    @triton.testing.perf_report(configs)
    def bench_jagged_dense_bmm_broadcast_add(
        batch_size: int,
        seq_len: int,
        D: int,
        K: int,
        mode: str,
        provider: str,
        dtype: torch.dtype,
    ) -> float:
        assert mode in ["fwd", "bwd"]
        warmup = 25
        rep = 100

        max_seq_len = seq_len
        # Seeded per point so every provider is timed on the same lengths: at small
        # batch sizes the total row count varies widely between draws.
        lengths = torch.randint(
            max_seq_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
            generator=torch.Generator(device=torch.device("cuda")).manual_seed(seq_len),
        )
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        jagged_size = int(seq_offsets[-1].item())
        jagged = (
            torch.empty((jagged_size, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        dense = (
            torch.empty((batch_size, D, K), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias = (
            torch.empty((batch_size, K), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        # -1 is this file's convention for an unsupported provider/mode combination.
        if not _supported(provider, dtype, batch_size, seq_len, D, K):
            return -1

        if provider in ["triton", "pytorch", "cutedsl"]:
            fn = lambda: jagged_dense_bmm_broadcast_add(  # noqa E731
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                jagged=jagged,
                dense=dense,
                bias=bias,
                kernel=get_kernel(provider),
            )
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        elif provider == "triton_nonfused":
            fn = lambda: jagged_dense_broadcast_add(  # noqa E731
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                jagged=jagged_dense_bmm(
                    max_seq_len=max_seq_len,
                    seq_offsets=seq_offsets,
                    jagged=jagged,
                    dense=dense,
                    kernel=HammerKernel.TRITON,
                ),
                dense=bias,
                kernel=HammerKernel.TRITON,
            )
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        else:
            raise ValueError(f"unsupported provider: {provider}")

    bench_jagged_dense_bmm_broadcast_add.run(print_data=True)
    if dump_cache_dir:
        with open(dump_cache_dir, "wb") as data:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(jagged_dense_bmm_broadcast_add_kernel.cache, data)


if __name__ == "__main__":
    main()
