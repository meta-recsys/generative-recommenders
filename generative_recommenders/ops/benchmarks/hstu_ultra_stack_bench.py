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

"""Benchmark complete HSTU Ultra layer stacks with BF16 or FP8 projections."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import median

import click
import torch
from generative_recommenders.common import HammerKernel
from generative_recommenders.modules.hstu_ultra import (
    hstu_ultra_stack_configs,
    HSTUUltraStack,
)
from generative_recommenders.ops.hstu_ultra import HSTU_ULTRA_CONFIG_NAMES


@dataclass(frozen=True)
class BenchmarkResult:
    config: str
    provider: str
    batch_size: int
    sequence_length: int
    num_layers: int
    trials: int
    latency_ms: float
    latency_min_ms: float
    latency_max_ms: float
    tokens_per_second: float


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


def benchmark_hstu_ultra_stack(
    config_name: str,
    provider: str,
    batch_size: int,
    sequence_length: int,
    warmup: int,
    repetitions: int,
    trials: int,
) -> BenchmarkResult:
    fp8_addmm_fwd = provider == "fp8"
    config = hstu_ultra_stack_configs(fp8_addmm_fwd=fp8_addmm_fwd)[config_name]
    attention = config.attention
    if sequence_length <= attention.max_targets:
        raise ValueError("sequence_length must exceed the configured target count")
    if sequence_length > attention.max_sequence_length:
        raise ValueError(
            f"sequence_length must not exceed {attention.max_sequence_length}"
        )

    device = torch.device("cuda")
    stack = (
        HSTUUltraStack(
            config=config,
            is_inference=True,
            kernel=HammerKernel.TRITON,
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    if fp8_addmm_fwd:
        stack.prepare_fp8_weights()
    tokens = batch_size * sequence_length
    x = torch.randn(tokens, config.embedding_dim, device=device, dtype=torch.bfloat16)
    lengths = torch.full(
        (batch_size,), sequence_length, dtype=torch.int32, device=device
    )
    offsets = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    offsets = offsets * sequence_length
    num_targets = torch.full(
        (batch_size,), attention.max_targets, dtype=torch.int32, device=device
    )

    def run() -> None:
        with torch.inference_mode():
            stack(
                x=x,
                x_lengths=lengths,
                x_offsets=offsets,
                max_seq_len=sequence_length,
                num_targets=num_targets,
            )

    measurements = [_measure_cuda_ms(run, warmup, repetitions) for _ in range(trials)]
    latency_ms = median(measurements)
    return BenchmarkResult(
        config=config.name,
        provider=provider,
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_layers=config.num_layers,
        trials=trials,
        latency_ms=latency_ms,
        latency_min_ms=min(measurements),
        latency_max_ms=max(measurements),
        tokens_per_second=tokens * 1000.0 / latency_ms,
    )


def _parse_sequence_lengths(raw_sequence_lengths: str) -> tuple[int, ...]:
    try:
        sequence_lengths = tuple(
            int(value.strip()) for value in raw_sequence_lengths.split(",")
        )
    except ValueError as error:
        raise click.BadParameter(
            "sequence lengths must be comma-separated integers"
        ) from error
    if not sequence_lengths or any(length <= 0 for length in sequence_lengths):
        raise click.BadParameter("sequence lengths must be positive")
    return sequence_lengths


@click.command()
@click.option(
    "--config-name",
    "config_name",
    type=click.Choice(HSTU_ULTRA_CONFIG_NAMES),
    default="hstu_ultra_semi_local",
    show_default=True,
)
@click.option(
    "--provider",
    type=click.Choice(("bf16", "fp8", "all")),
    default="all",
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--sequence-lengths",
    default="512",
    show_default=True,
    help="Comma-separated sequence lengths.",
)
@click.option("--warmup", type=click.IntRange(min=1), default=10, show_default=True)
@click.option(
    "--repetitions", type=click.IntRange(min=1), default=100, show_default=True
)
@click.option("--trials", type=click.IntRange(min=1), default=3, show_default=True)
def main(
    config_name: str,
    provider: str,
    batch_size: int,
    sequence_lengths: str,
    warmup: int,
    repetitions: int,
    trials: int,
) -> None:
    """Benchmark a complete HSTU Ultra stage stack on CUDA."""
    if not torch.cuda.is_available():
        raise click.ClickException("this benchmark requires a CUDA device")
    providers = ("bf16", "fp8") if provider == "all" else (provider,)
    for sequence_length in _parse_sequence_lengths(sequence_lengths):
        for selected_provider in providers:
            torch.manual_seed(1001)
            result = benchmark_hstu_ultra_stack(
                config_name=config_name,
                provider=selected_provider,
                batch_size=batch_size,
                sequence_length=sequence_length,
                warmup=warmup,
                repetitions=repetitions,
                trials=trials,
            )
            click.echo(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
