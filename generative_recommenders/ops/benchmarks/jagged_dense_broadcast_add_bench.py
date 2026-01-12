# pyre-strict
import math
import pickle
from typing import List, Optional, Tuple

import click
import pandas as pd
import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.triton.triton_jagged import (
    jagged_dense_broadcast_add_kernel,
    triton_jagged_dense_broadcast_add,
)

# buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_dense_broadcast_add_bench


# To dump the jagged_dense_broadcast_add_kernel cache, run:
# buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_dense_broadcast_add_bench -- --dump-ragged-tuner-cache-dir=/home/${USER}/fbsource/fbcode/generative_recommenders/ops/triton/jagged_dense_broadcast_add_kernel_cache.pkl


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


@click.command()
@click.option(
    "--batch-size",
    type=int,
    default=512,
)
@click.option(
    "--max-seq-len",
    type=int,
    default=8192,
    show_default=True,
)
@click.option(
    "-d",
    type=int,
    default=64,
    show_default=True,
)
@click.option("--dtype", type=str, default="fp32")
@click.option("--fwd-only", is_flag=True)
@click.option("--dump-ragged-tuner-cache-dir", type=str, default="")
@click.option("--return-result", type=bool, default=False)
def main(
    batch_size: int,
    max_seq_len: int,
    d: int,
    dtype: str,
    fwd_only: bool,
    dump_ragged_tuner_cache_dir: str,
    return_result: bool,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
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

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[2**i for i in range(5, max_seq_len_log2 + 1)],
            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "Pytorch"],
            styles=[("red", "-"), ("blue", "-")],
            ylabel="ms",
            plot_name=f"jagged_dense_broadcast_add-b{batch_size}-D{d}-{dtype}-{mode}",
            args={
                "batch_size": batch_size,
                "D": d,
                "dtype": pt_dtype,
                "mode": mode,
            },
        )
        for mode in (["fwd"] if fwd_only else ["fwd", "fwd+bwd"])
    ]

    @triton.testing.perf_report(configs)
    def bench_jagged_dense_broadcast_add(
        batch_size: int,
        seq_len: int,
        D: int,
        mode: str,
        provider: str,
        dtype: torch.dtype,
    ) -> float:
        assert mode in ["fwd", "bwd", "fwd+bwd"]
        warmup = 25
        rep = 100

        max_seq_len = seq_len
        lengths = torch.randint(
            max_seq_len + 1, size=(batch_size,), device=torch.device("cuda")
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
            torch.empty((batch_size, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if provider == "triton":
            fn = lambda: jagged_dense_broadcast_add(  # noqa E731
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                jagged=jagged,
                dense=dense,
                kernel=HammerKernel.TRITON,
            )
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        elif provider == "pytorch":
            fn = lambda: jagged_dense_broadcast_add(  # noqa E731
                max_seq_len=max_seq_len,
                seq_offsets=seq_offsets,
                jagged=jagged,
                dense=dense,
                kernel=HammerKernel.PYTORCH,
            )
            if mode == "bwd":
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)  # noqa E731
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        else:
            raise ValueError(f"unsupported provider: {provider}")

    df = bench_jagged_dense_broadcast_add.run(print_data=True, return_df=return_result)

    if dump_ragged_tuner_cache_dir:
        with open(dump_ragged_tuner_cache_dir, "wb") as data:
            # @lint-ignore PYTHONPICKLEISBAD
            pickle.dump(jagged_dense_broadcast_add_kernel.cache, data)

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()
