# pyre-unsafe
from typing import List, Optional, Tuple

import click
import pandas as pd

import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.mm import addmm


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
    else:
        raise ValueError(f"Not supported dtype {dtype}")


@click.command()
@click.option("--m", type=int, default=4096)
@click.option("--k", type=int, default=4096)
@click.option("--dtype", type=str, default="bfloat16")
@click.option("--return-result", type=bool, default=False)
def main(
    m: int,
    k: int,
    dtype: str,
    return_result: bool,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
    batch_sizes = [64, 128, 256, 512]
    line_vals = ["triton", "pytorch"] if dtype == "bfloat16" else ["triton", "pytorch"]
    line_names = ["Triton", "PyTorch"] if dtype == "bfloat16" else ["Triton", "PyTorch"]
    styles = (
        [
            ("red", "-"),
            ("blue", "-"),
        ]
        if dtype == "bfloat16"
        else [
            ("red", "-"),
            ("blue", "-"),
        ]
    )
    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=batch_sizes,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_names,
            styles=styles,
            ylabel="ms",
            plot_name=f"addmm-M-{m}-K-{k}-mode-{mode}-dtype-{dtype}",
            args={
                "M": m,
                "K": k,
                "dtype": dtype,
            },
        )
        for mode in ["fwd"]
    ]

    @triton.testing.perf_report(configs)
    def bench_addmm(
        batch_size: int,
        M: int,
        K: int,
        dtype: str,
        provider: str,
    ) -> float:
        warmup = 20
        rep = 20
        x = torch.randn(
            (batch_size, K), dtype=get_dtype(dtype), device=torch.device("cuda")
        ).requires_grad_(True)
        weight = torch.randn(
            (M, K), dtype=get_dtype(dtype), device=torch.device("cuda")
        ).requires_grad_(True)
        y = torch.randn(
            (batch_size, M), dtype=get_dtype(dtype), device=torch.device("cuda")
        ).requires_grad_(True)

        fn = lambda: addmm(  # noqa E73
            y,
            x,
            weight,
            kernel=get_kernel(provider),
        )
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    df = bench_addmm.run(print_data=True, return_df=return_result)

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()
