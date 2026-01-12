# pyre-strict
from typing import List

import click
import torch

# @manual=//triton:triton
import triton
from hammer.ops.jagged import replace_last_n_with_jagged
from hammer.utils import HammerKernel

# buck2 run @//mode/opt -c fbcode.nvcc_arch=h100 //generative_recommenders/ops/cpp/benchmarks:replace_last_n_with_jagged_bench

torch.ops.load_library("//generative_recommenders/ops/cpp:cpp_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


@click.command()
@click.option("--data-type", type=str, default="float32")
@click.option("--batch-size", type=int, default=512)
@click.option("--embedding-dim", type=int, default=64)
@click.option("--max-seq-len-log2", type=int, default=16)
@click.option("--seq-sparsity", type=float, default=0.8)
def main(
    data_type: str,
    batch_size: int,
    embedding_dim: int,
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
                "hammer_triton",
            ],
            line_names=[
                "Custom CUDA",
                "Hammer PyTorch",
                "Hammer Triton",
            ],
            styles=[
                ("green", "-"),
                ("orange", "--"),
                ("purple", "-."),
            ],
            ylabel="ms",
            plot_name=f"replace_last_n_with_jagged_batch{batch_size}_dim{embedding_dim}_sparsity{seq_sparsity}_{data_type}",
            args={
                "dtype": dtype,
                "batch_size": batch_size,
                "embedding_dim": embedding_dim,
                "seq_sparsity": seq_sparsity,
            },
        )
    ]

    @triton.testing.perf_report(configs)
    def bench_replace_last_n_with_jagged(
        max_seq_len: int,
        batch_size: int,
        method: str,
        dtype: torch.dtype,
        embedding_dim: int,
        seq_sparsity: float,
    ) -> float:
        warmup = 50
        rep = 500
        torch.manual_seed(1001)

        min_left_len = max(1, int(max_seq_len * seq_sparsity * 0.3))
        max_left_len = int(max_seq_len * seq_sparsity)

        lengths_left = torch.randint(
            min_left_len, max_left_len + 1, (batch_size,), dtype=torch.int32
        )
        lengths_right = torch.randint(
            1, min_left_len + 1, (batch_size,), dtype=torch.int32
        )

        lengths_right = torch.min(lengths_right, lengths_left)

        total_left = int(lengths_left.sum().item())
        total_right = int(lengths_right.sum().item())

        values_left = torch.randn(total_left, embedding_dim, dtype=dtype)
        values_right = torch.randn(total_right, embedding_dim, dtype=dtype)

        offsets_left = torch.zeros(
            (batch_size + 1,), dtype=lengths_left.dtype, device=lengths_left.device
        )
        offsets_left[1:] = torch.cumsum(lengths_left.view(-1), dim=0)
        offsets_right = torch.zeros(
            (batch_size + 1,), dtype=lengths_right.dtype, device=lengths_right.device
        )
        offsets_right[1:] = torch.cumsum(lengths_right.view(-1), dim=0)

        lengths_left = lengths_left.cuda()
        lengths_right = lengths_right.cuda()
        values_left = values_left.cuda()
        values_right = values_right.cuda()
        offsets_left = offsets_left.cuda()
        offsets_right = offsets_right.cuda()

        if method == "custom_cuda":
            fn = lambda: torch.ops.hstu.replace_last_n_with_jagged(  # noqa E731
                lengths_left, values_left, lengths_right, values_right
            )
        elif method == "hammer_pytorch":
            fn = lambda: replace_last_n_with_jagged(  # noqa E731
                max_seq_len_left=max_seq_len,
                offsets_left=offsets_left,
                values_left=values_left,
                offsets_right=offsets_right,
                values_right=values_right,
            )
        elif method == "hammer_triton":
            fn = lambda: replace_last_n_with_jagged(  # noqa E731
                max_seq_len_left=max_seq_len,
                offsets_left=offsets_left,
                values_left=values_left,
                offsets_right=offsets_right,
                values_right=values_right,
                kernel=HammerKernel.TRITON,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_replace_last_n_with_jagged.run(print_data=True)


if __name__ == "__main__":
    main()
