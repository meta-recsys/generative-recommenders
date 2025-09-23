# pyre-strict
from typing import List

import click

import torch

# @manual=//triton:triton
import triton
from hammer.ops.jagged import split_1D_jagged_jagged

# buck2 run @//mode/opt -c fbcode.nvcc_arch=h100 //generative_recommenders/ops/cpp/benchmarks:split_1d_jagged_jagged_bench

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


@click.command()
@click.option("--data-type", type=str, default="float32")
@click.option("--batch-size", type=int, default=512)
@click.option("--max-seq-len-log2", type=int, default=20)
@click.option("--seq-sparsity", type=float, default=0.8)
def main(
    data_type: str,
    batch_size: int,
    max_seq_len_log2: int,
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
            x_names=["max_seq_len"],
            x_vals=[2**i for i in range(6, max_seq_len_log2)],
            line_arg="method",
            line_vals=[
                "custom_cuda",
                "hammer_pytorch",
            ],
            line_names=["Custom CUDA", "Hammer PyTorch"],
            styles=[("green", "-"), ("orange", "--")],
            ylabel="ms",
            plot_name=f"split_1d_jagged_jagged_batch{batch_size}_sparsity{seq_sparsity}_{data_type}",
            args={
                "dtype": dtype,
                "batch_size": batch_size,
                "seq_sparsity": seq_sparsity,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_split_1d_jagged_jagged(
        max_seq_len: int,
        batch_size: int,
        method: str,
        dtype: torch.dtype,
        seq_sparsity: float,
    ) -> float:
        warmup = 50
        rep = 500
        torch.manual_seed(1001)

        lengths_left = torch.randint(
            1, int(max_seq_len * seq_sparsity) + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            1, int(max_seq_len * seq_sparsity) + 1, (batch_size,), dtype=torch.int32
        )

        offsets_left = torch.zeros(
            (batch_size + 1,), dtype=lengths_left.dtype, device=lengths_left.device
        )
        offsets_left[1:] = torch.cumsum(lengths_left.view(-1), dim=0)
        offsets_right = torch.zeros(
            (batch_size + 1,), dtype=lengths_right.dtype, device=lengths_right.device
        )
        offsets_right[1:] = torch.cumsum(lengths_right.view(-1), dim=0)

        combined_offsets = offsets_left + offsets_right
        combined_values = torch.randn(int(combined_offsets[-1].item()), dtype=dtype)

        max_seq_len_combined = int((lengths_left + lengths_right).max().item())

        lengths_left = lengths_left.cuda()
        lengths_right = lengths_right.cuda()
        combined_values = combined_values.cuda()
        offsets_left = offsets_left.cuda()
        offsets_right = offsets_right.cuda()

        if method == "custom_cuda":
            fn = lambda: torch.ops.hstu.split_1d_jagged_jagged(  # noqa E731
                lengths_left, lengths_right, combined_values
            )
        elif method == "hammer_pytorch":
            fn = lambda: split_1D_jagged_jagged(  # noqa E731
                max_seq_len_combined, combined_values, offsets_left, offsets_right
            )
        else:
            raise ValueError(f"unknown method: {method}")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_split_1d_jagged_jagged.run(print_data=True)


if __name__ == "__main__":
    main()
