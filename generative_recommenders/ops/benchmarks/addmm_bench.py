# pyre-unsafe
import time
from typing import List, Optional, Tuple

import click
import pandas as pd

import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.triton.triton_addmm import (
    triton_addmm_fwd,
    triton_addmm_fwd_tma_persistent,
    triton_addmm_fwd_tma_ws_tlx,
)

try:
    # @manual=//triton:triton
    import triton.language.extra.tlx as tlx  # type: ignore

    HAS_TLX = True
except ImportError:
    tlx = None
    HAS_TLX = False


def get_kernel(provider: str) -> HammerKernel:
    if provider == "triton":
        return HammerKernel.TRITON
    elif provider == "pytorch":
        return HammerKernel.PYTORCH
    else:
        raise ValueError(f"Unknown provider {provider}")


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float16":
        return torch.float16
    else:
        raise ValueError(f"Not supported dtype {dtype}")


@click.command()
@click.option("--m", type=int, default=0)
@click.option("--k", type=int, default=4096)
@click.option("--n", type=int, default=4096)
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--return-result", type=bool, default=False)
@click.option("--broadcast-y", type=bool, is_flag=True, default=False)
def main(
    m: int,
    k: int,
    n: int,
    dtype: str,
    return_result: bool,
    broadcast_y: bool,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
    if m == 0:
        batch_sizes = [64, 128, 256, 512]
    else:
        batch_sizes = [m]
    line_vals = [
        "pytorch",
        "triton",
        "triton_tma_persistent",
        "triton_tma_persistent_ws",
    ]
    line_names = [
        "PyTorch",
        "Triton",
        "Triton TMA Persistent",
        "Triton TMA Persistent WS",
    ]
    styles = [
        ("red", "-"),
        ("green", "-"),
        ("orange", "-"),
        ("purple", "-"),
    ]
    if HAS_TLX == True:
        line_vals.append("triton_tma_ws_tlx")
        line_names.append("Triton TMA WS TLX")
        styles.append(("magenta", "-"))
    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=batch_sizes,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel="ms",
            plot_name=f"addmm-K-{k}-N-{n}-mode-{mode}-dtype-{dtype}-broadcast_y-{broadcast_y}",
            args={
                "K": k,
                "N": n,
                "dtype": dtype,
                "broadcast_y": broadcast_y,
            },
        )
        for mode in ["fwd"]
    ]

    @triton.testing.perf_report(configs)
    def bench_addmm(
        batch_size: int,
        K: int,
        N: int,
        dtype: str,
        provider: str,
        broadcast_y: bool,
    ) -> float:
        warmup = 20
        rep = 2000
        x = torch.randn(
            (batch_size, K), dtype=get_dtype(dtype), device=torch.device("cuda")
        ).requires_grad_(True)
        weight = torch.randn(
            (N, K), dtype=get_dtype(dtype), device=torch.device("cuda")
        ).requires_grad_(True)
        if broadcast_y:
            y = torch.randn(
                (N), dtype=get_dtype(dtype), device=torch.device("cuda")
            ).requires_grad_(True)
        else:
            y = torch.randn(
                (batch_size, N), dtype=get_dtype(dtype), device=torch.device("cuda")
            ).requires_grad_(True)

        # Make sure tensors are contiguous for TMA kernels
        weight_t_contiguous = weight.T.contiguous()

        if provider == "pytorch":
            fn = lambda: torch.addmm(y, x, weight.T)  # noqa E731
        elif provider == "triton_tma_persistent":
            fn = lambda: triton_addmm_fwd_tma_persistent(
                x, weight_t_contiguous, y, warp_specialize=False
            )  # noqa E731
        elif provider == "triton_tma_persistent_ws":
            fn = lambda: triton_addmm_fwd_tma_persistent(
                x, weight_t_contiguous, y, warp_specialize=True
            )  # noqa E731
        elif provider == "triton_tma_ws_tlx":
            fn = lambda: triton_addmm_fwd_tma_ws_tlx(x, weight_t_contiguous, y)  # noqa E731
        elif provider == "triton":
            fn = lambda: triton_addmm_fwd(x, weight_t_contiguous, y)  # noqa E731
        else:
            raise ValueError(f"Unknown provider: {provider}")
        time.sleep(2)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    df = bench_addmm.run(print_data=True, return_df=return_result)

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()
