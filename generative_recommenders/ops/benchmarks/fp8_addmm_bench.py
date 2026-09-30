# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark portable row-wise FP8 addmm against BF16."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import click
import torch
from generative_recommenders.ops.fp8 import fp8_rowwise_addmm, quantize_fp8_per_row


@dataclass(frozen=True)
class BenchmarkResult:
    provider: str
    m: int
    n: int
    k: int
    latency_ms: float
    tflops: float


def _measure_cuda_ms(
    benchmark: Callable[[], None], warmup: int, repetitions: int
) -> float:
    for _ in range(warmup):
        benchmark()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        benchmark()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def benchmark_fp8_addmm(
    provider: str,
    m: int,
    n: int,
    k: int,
    warmup: int,
    repetitions: int,
) -> BenchmarkResult:
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 0.1
    w = torch.randn(k, n, device=device, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(n, device=device, dtype=torch.bfloat16) * 0.1
    x_fp8, x_scale = quantize_fp8_per_row(x)
    w_fp8, w_scale = quantize_fp8_per_row(w.t().contiguous())

    if provider == "bf16":

        def run() -> None:
            torch.addmm(bias, x, w)

    elif provider == "fp8_prequantized":

        def run() -> None:
            fp8_rowwise_addmm(
                input=bias,
                mat1_fp8=x_fp8,
                mat1_scale=x_scale,
                mat2_fp8=w_fp8,
                mat2_scale=w_scale,
                is_inference=True,
            )

    elif provider == "fp8_end_to_end":

        def run() -> None:
            fp8_rowwise_addmm(
                input=bias,
                mat1=x,
                mat2=w,
                is_inference=True,
            )

    else:
        raise ValueError(f"unsupported provider: {provider}")

    latency_ms = _measure_cuda_ms(run, warmup, repetitions)
    return BenchmarkResult(
        provider=provider,
        m=m,
        n=n,
        k=k,
        latency_ms=latency_ms,
        tflops=2.0 * m * n * k / (latency_ms * 1.0e9),
    )


@click.command()
@click.option(
    "--provider",
    type=click.Choice(("bf16", "fp8_prequantized", "fp8_end_to_end", "all")),
    default="all",
    show_default=True,
)
@click.option("--m", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--n", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--k", type=click.IntRange(min=1), default=512, show_default=True)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--repetitions", type=click.IntRange(min=1), default=100, show_default=True
)
def main(
    provider: str,
    m: int,
    n: int,
    k: int,
    warmup: int,
    repetitions: int,
) -> None:
    """Benchmark BF16 and portable row-wise FP8 addmm on CUDA."""
    if not torch.cuda.is_available():
        raise click.ClickException("this benchmark requires a CUDA device")
    providers = (
        ("bf16", "fp8_prequantized", "fp8_end_to_end")
        if provider == "all"
        else (provider,)
    )
    for selected_provider in providers:
        result = benchmark_fp8_addmm(
            provider=selected_provider,
            m=m,
            n=n,
            k=k,
            warmup=warmup,
            repetitions=repetitions,
        )
        click.echo(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
