# pyre-strict
from typing import List

import click
import torch

# @manual=//triton:triton
import triton
from hammer.ops.jagged import jagged_transpose_1D

# buck2 run @//mode/opt -c fbcode.nvcc_arch=h100 //generative_recommenders/ops/cpp/benchmarks:jagged_transpose_1d_bench

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


@click.command()
@click.option("--data-type", type=str, default="float32")
@click.option("--size1", type=int, default=32)
@click.option("--size2", type=int, default=16)
@click.option("--max-len-log2", type=int, default=19)
@click.option("--seq-sparsity", type=float, default=0.8)
def main(
    data_type: str,
    size1: int,
    size2: int,
    max_len_log2: int,
    seq_sparsity: float,
) -> None:
    if data_type == "float32":
        dtype = torch.float32
    elif data_type == "float16":
        dtype = torch.float16
    elif data_type == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {data_type}.")

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["max_len"],
            x_vals=[2**i for i in range(4, max_len_log2)],
            line_arg="method",
            line_vals=[
                "custom_cuda",
                "hammer_pytorch",
            ],
            line_names=["Custom CUDA", "Hammer PyTorch"],
            styles=[("green", "-"), ("orange", "--")],
            ylabel="ms",
            plot_name=f"jagged_transpose_1d_size1_{size1}_size2_{size2}_sparsity{seq_sparsity}_{data_type}",
            args={
                "dtype": dtype,
                "size1": size1,
                "size2": size2,
                "seq_sparsity": seq_sparsity,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_jagged_transpose_1d(
        max_len: int,
        size1: int,
        size2: int,
        method: str,
        dtype: torch.dtype,
        seq_sparsity: float,
    ) -> float:
        warmup = 50
        rep = 500
        torch.manual_seed(1001)

        lengths = torch.randint(
            1, int(max_len * seq_sparsity) + 1, (size1 * size2,), dtype=torch.int32
        )
        offsets = torch.zeros(
            (size1 * size2 + 1,), dtype=lengths.dtype, device=lengths.device
        )
        offsets[1:] = torch.cumsum(lengths.view(-1), dim=0)

        values = torch.randn(int(offsets[-1].item()), dtype=dtype)

        lengths = lengths.cuda()
        offsets = offsets.cuda()
        values = values.cuda()

        if method == "custom_cuda":
            fn = lambda: torch.ops.hstu.jagged_transpose_1d(  # noqa E731
                values=values,
                offsets=offsets,
                lengths=lengths,
                max_len=max_len,
                size1=size1,
                size2=size2,
            )
        elif method == "hammer_pytorch":
            fn = lambda: jagged_transpose_1D(  # noqa E731
                values=values,
                offsets=offsets,
                lengths=lengths,
                max_len=max_len,
                size1=size1,
                size2=size2,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_jagged_transpose_1d.run(print_data=True)


if __name__ == "__main__":
    main()
