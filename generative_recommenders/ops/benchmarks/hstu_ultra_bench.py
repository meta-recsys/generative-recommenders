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

"""Benchmark the dedicated PyTorch HSTU Ultra attention module."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

import click
import torch
from generative_recommenders.ops.hstu_ultra import (
    hstu_ultra_attention_configs,
    HSTU_ULTRA_CONFIG_NAMES,
    HSTUUltraAttentionConfig,
    pytorch_hstu_ultra_mha,
)


@dataclass(frozen=True)
class HSTUUltraInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    seq_offsets: torch.Tensor
    num_targets: torch.Tensor | None
    max_seq_len: int
    tokens: int


@dataclass(frozen=True)
class BenchmarkResult:
    config: str
    mode: str
    batch_size: int
    sequence_length: int
    latency_ms: float
    tokens: int
    tokens_per_second: float


def build_hstu_ultra_inputs(
    config: HSTUUltraAttentionConfig,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> HSTUUltraInputs:
    """Build fixed-length jagged inputs for an HSTU Ultra configuration."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sequence_length <= config.max_targets:
        raise ValueError("sequence_length must exceed the configured target count")
    if sequence_length > config.max_sequence_length:
        raise ValueError(
            f"sequence_length must not exceed {config.max_sequence_length}"
        )

    tokens = batch_size * sequence_length
    seq_offsets = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * sequence_length
    )
    num_targets = None
    if config.max_targets > 0:
        num_targets = torch.full(
            (batch_size,),
            config.max_targets,
            dtype=torch.int32,
            device=device,
        )
    q = torch.empty(
        tokens,
        config.heads,
        config.attention_dim,
        dtype=dtype,
        device=device,
    ).uniform_(-0.01, 0.01)
    k = torch.empty_like(q).uniform_(-0.01, 0.01)
    v = torch.empty(
        tokens,
        config.heads,
        config.value_dim,
        dtype=dtype,
        device=device,
    ).uniform_(-0.01, 0.01)
    return HSTUUltraInputs(
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_seq_len=sequence_length,
        tokens=tokens,
    )


def _forward(
    config: HSTUUltraAttentionConfig,
    inputs: HSTUUltraInputs,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return pytorch_hstu_ultra_mha(
        config=config,
        max_seq_len=inputs.max_seq_len,
        q=q,
        k=k,
        v=v,
        seq_offsets=inputs.seq_offsets,
        num_targets=inputs.num_targets,
    )


def _benchmark_callable(
    config: HSTUUltraAttentionConfig,
    inputs: HSTUUltraInputs,
    mode: str,
) -> Callable[[], None]:
    if mode == "fwd":

        def run_forward() -> None:
            _forward(config, inputs, inputs.q, inputs.k, inputs.v)

        return run_forward
    if mode != "fwd_bwd":
        raise ValueError(f"unsupported benchmark mode: {mode}")

    q = inputs.q.detach().requires_grad_()
    k = inputs.k.detach().requires_grad_()
    v = inputs.v.detach().requires_grad_()
    output_gradient = torch.randn_like(v)

    def run_forward_backward() -> None:
        q.grad = None
        k.grad = None
        v.grad = None
        _forward(config, inputs, q, k, v).backward(output_gradient)

    return run_forward_backward


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


def benchmark_hstu_ultra(
    config: HSTUUltraAttentionConfig,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype,
    mode: str,
    warmup: int,
    repetitions: int,
) -> BenchmarkResult:
    inputs = build_hstu_ultra_inputs(
        config=config,
        batch_size=batch_size,
        sequence_length=sequence_length,
        dtype=dtype,
        device=torch.device("cuda"),
    )
    latency_ms = _measure_cuda_ms(
        _benchmark_callable(config, inputs, mode), warmup, repetitions
    )
    return BenchmarkResult(
        config=config.name,
        mode=mode,
        batch_size=batch_size,
        sequence_length=sequence_length,
        latency_ms=latency_ms,
        tokens=inputs.tokens,
        tokens_per_second=inputs.tokens * 1000.0 / latency_ms,
    )


def _parse_sequence_lengths(
    raw_sequence_lengths: str | None, config: HSTUUltraAttentionConfig
) -> tuple[int, ...]:
    if raw_sequence_lengths is None:
        return config.default_sequence_lengths
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
    default="hstu_ultra_l1",
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--sequence-lengths",
    default=None,
    help="Comma-separated sequence lengths; defaults come from the configuration.",
)
@click.option(
    "--data-type",
    type=click.Choice(("bf16", "fp16", "fp32")),
    default="bf16",
    show_default=True,
)
@click.option(
    "--mode",
    type=click.Choice(("fwd", "fwd_bwd")),
    default="fwd",
    show_default=True,
)
@click.option("--warmup", type=click.IntRange(min=1), default=5, show_default=True)
@click.option(
    "--repetitions", type=click.IntRange(min=1), default=20, show_default=True
)
def main(
    config_name: str,
    batch_size: int,
    sequence_lengths: str | None,
    data_type: str,
    mode: str,
    warmup: int,
    repetitions: int,
) -> None:
    """Run the PyTorch baseline for a named HSTU Ultra configuration."""
    if not torch.cuda.is_available():
        raise click.ClickException("this benchmark requires a CUDA device")
    torch.manual_seed(1001)
    config = hstu_ultra_attention_configs()[config_name]
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    for sequence_length in _parse_sequence_lengths(sequence_lengths, config):
        result = benchmark_hstu_ultra(
            config=config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=dtypes[data_type],
            mode=mode,
            warmup=warmup,
            repetitions=repetitions,
        )
        click.echo(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
