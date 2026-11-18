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

# pyre-strict

"""
Benchmarks the Composable Kernel (CK) HSTU attention against the Triton (and
optionally PyTorch) HSTU attention on AMD GPUs.

    python -m generative_recommenders.ops.benchmarks.ck_hstu_attention_bench \
        --bench-backward True --report-flops True
"""

from typing import Callable, List, Optional, Tuple

import click
import pandas as pd
import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import (
    apply_sampling,
    generate_sparse_seq_len,
    HammerKernel,
)
from generative_recommenders.ops.ck.ck_hstu_attention import (
    ck_hstu_mha,
    is_ck_hstu_available,
)
from generative_recommenders.ops.hstu_attention import hstu_mha

PROVIDERS = ["triton", "pytorch", "ck_jagged", "ck_batched"]


def _flops(
    batch_size: int,
    attn_dim: int,
    hidden_dim: int,
    nheads: int,
    seq_offsets: torch.Tensor,
    mode: str = "fwd",
) -> float:
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    ratio = 2.0  # triangular masking
    f1 = 0.0
    f2 = 0.0
    for i in range(batch_size):
        seq_len = int((seq_offsets[i + 1] - seq_offsets[i]).item())
        # (QK^T), dQ = d(QK^T)K, dK^T = Q^Td(QK^T)
        f1 += 2 * nheads * attn_dim * seq_len**2 // ratio
        # (QK^T)V, d(QK^T) = dOV^T, dV = (QK^T)^TdO,
        f2 += 2 * nheads * hidden_dim * seq_len**2 // ratio
    if mode == "fwd":
        return f1 + f2  # computes (QK^T) and (QK^T)V
    elif mode == "bwd":
        return 3 * f1 + 2 * f2  # computes (QK^T), dQ, dK, dV, d(QK^T)
    else:
        return 4 * f1 + 3 * f2


def _bench_fn(
    attn: Callable[[], torch.Tensor],
    mode: str,
    warmup: int,
    rep: int,
) -> float:
    if mode == "fwd":
        return triton.testing.do_bench(attn, warmup=warmup, rep=rep)
    if mode == "bwd":
        out = attn()
        dout = torch.randn_like(out)
        return triton.testing.do_bench(
            lambda: out.backward(dout, retain_graph=True), warmup=warmup, rep=rep
        )

    def fwd_bwd() -> None:
        out = attn()
        out.backward(torch.ones_like(out))

    return triton.testing.do_bench(fwd_bwd, warmup=warmup, rep=rep)


@click.command()
@click.option("--heads", type=int, default=4)
@click.option("--attn-dim", type=int, default=128)
@click.option("--hidden-dim", type=int, default=128)
@click.option("--max-seq-len-log2", type=int, default=15)
@click.option("--batch-sizes", type=str, default="4,8,16,32")
@click.option("--data-type", type=str, default="bf16")
@click.option("--seq-sparsity", type=float, default=1.0)
@click.option("--target-size", type=int, default=20)
@click.option("--bench-forward", type=bool, default=True)
@click.option("--bench-backward", type=bool, default=False)
@click.option("--bench-pytorch", type=bool, default=False)
@click.option("--bench-ck-batched", type=bool, default=False)
@click.option("--report-flops", type=bool, default=False)
@click.option("--return-result", type=bool, default=False)
@click.option("--max-attn-len", type=int, default=256)
@click.option("--min-full-attn-seq-len", type=int, default=256)
@click.option("--contextual-seq-len", type=int, default=0)
@click.option("--sampling-alpha", type=float, default=2.0)
@click.option("--save-path", type=str, default="")
def main(  # noqa: C901
    heads: int,
    attn_dim: int,
    hidden_dim: int,
    max_seq_len_log2: int,
    batch_sizes: str,
    data_type: str,
    seq_sparsity: float,
    target_size: int,
    bench_forward: bool,
    bench_backward: bool,
    bench_pytorch: bool,
    bench_ck_batched: bool,
    report_flops: bool,
    return_result: bool,
    max_attn_len: int,
    min_full_attn_seq_len: int,
    contextual_seq_len: int,
    sampling_alpha: float,
    save_path: str,
) -> Optional[Tuple[List[triton.testing.Benchmark], List[pd.DataFrame]]]:
    if not is_ck_hstu_available():
        raise RuntimeError(
            "The CK HSTU attention benchmark needs an AMD GPU and the CK HSTU "
            "attention extension (see generative_recommenders/ops/ck/README.md)."
        )
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if data_type == "fp16":
        dtype = torch.float16
    elif data_type == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported data type: {data_type}.")

    line_vals = ["triton", "ck_jagged"]
    styles = [("blue", "-"), ("red", "-")]
    if bench_ck_batched:
        # Dense [B, N, H, D] inputs of the full sequence length.
        line_vals.append("ck_batched")
        styles.append(("red", "--"))
    if bench_pytorch:
        line_vals.append("pytorch")
        styles.append(("green", "-"))

    modes = []
    if bench_forward:
        modes.append("fwd")
    if bench_backward:
        modes.append("bwd")
    assert len(modes) > 0

    x_vals = sorted([2**i for i in range(10, max_seq_len_log2)] + [3072])

    configs: List[triton.testing.Benchmark] = [
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=x_vals,
            line_arg="provider",
            line_vals=line_vals,
            line_names=line_vals,
            styles=styles,
            ylabel="TFLOPS" if report_flops else "ms",
            plot_name=f"ck-hstu-attn-b{batch_size}-h{heads}-d{attn_dim}-v{hidden_dim}-sparsity{seq_sparsity}-{mode}-{dtype}-target{target_size}-mattn{max_attn_len}-full{min_full_attn_seq_len}-c{contextual_seq_len}-sl_alpha{sampling_alpha}",
            args={
                "batch_size": batch_size,
                "heads": heads,
                "attn_dim": attn_dim,
                "hidden_dim": hidden_dim,
                "dtype": dtype,
                "mode": mode,
                "seq_sparsity": seq_sparsity,
                "target_size": target_size,
                "report_flops": report_flops,
                "max_attn_len": max_attn_len,
                "min_full_attn_seq_len": min_full_attn_seq_len,
                "contextual_seq_len": contextual_seq_len,
                "sampling_alpha": sampling_alpha,
            },
        )
        for mode in modes
        for batch_size in [int(b) for b in batch_sizes.split(",")]
    ]

    @triton.testing.perf_report(configs)
    def _bench_hstu_attention(
        batch_size: int,
        heads: int,
        seq_len: int,
        attn_dim: int,
        hidden_dim: int,
        mode: str,
        provider: str,
        dtype: torch.dtype,
        seq_sparsity: float,
        target_size: int,
        report_flops: bool,
        max_attn_len: int,
        min_full_attn_seq_len: int,
        contextual_seq_len: int,
        sampling_alpha: float,
    ) -> float:
        assert provider in PROVIDERS
        warmup = 25
        rep = 1000
        torch.manual_seed(1001)  # for reproducibility
        device = torch.device("cuda")
        alpha = 1.0 / attn_dim
        if provider == "ck_batched":
            lengths = torch.full((batch_size,), seq_len, device=device)
        else:
            lengths = generate_sparse_seq_len(
                size=batch_size,
                max_seq_len=seq_len,
                sparsity=seq_sparsity,
                device=device,
            )
            lengths = apply_sampling(lengths, sampling_alpha, max_seq_len=seq_len)
        num_targets = None
        if target_size != 0:
            num_targets = torch.clamp(lengths, max=target_size).to(torch.int32)
        max_attn_len = min(max_attn_len, seq_len)
        seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device=device)
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        L = int(seq_offsets[-1].item())
        x = torch.empty(
            (L, heads, attn_dim * 2 + hidden_dim), dtype=dtype, device=device
        ).uniform_(-0.01, 0.01)
        q, k, v = torch.split(x, [attn_dim, attn_dim, hidden_dim], dim=-1)
        if mode != "fwd":
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)

        if provider in ("ck_jagged", "ck_batched"):
            ck_seq_offsets: Optional[torch.Tensor] = seq_offsets.to(torch.int32)
            ck_q, ck_k, ck_v = q, k, v
            if provider == "ck_batched":
                ck_seq_offsets = None
                ck_q, ck_k, ck_v = (
                    t.view(batch_size, seq_len, heads, t.shape[-1]) for t in (q, k, v)
                )
            min_seq_len = int(lengths.min().item())

            def attn() -> torch.Tensor:
                return ck_hstu_mha(
                    max_seq_len=seq_len,
                    alpha=alpha,
                    q=ck_q,
                    k=ck_k,
                    v=ck_v,
                    seq_offsets_q=ck_seq_offsets,
                    seq_offsets_kv=ck_seq_offsets,
                    causal=True,
                    num_targets=num_targets,
                    max_attn_len=max_attn_len,
                    min_full_attn_seq_len=min_full_attn_seq_len,
                    contextual_seq_len=contextual_seq_len,
                    max_seq_len_kv=seq_len,
                    min_seq_len_q=min_seq_len,
                    min_seq_len_kv=min_seq_len,
                )

        else:
            kernel = (
                HammerKernel.TRITON if provider == "triton" else HammerKernel.PYTORCH
            )

            def attn() -> torch.Tensor:
                return hstu_mha(
                    max_seq_len=seq_len,
                    alpha=alpha,
                    q=q,
                    k=k,
                    v=v,
                    seq_offsets=seq_offsets,
                    causal=True,
                    num_targets=num_targets,
                    max_attn_len=max_attn_len,
                    contextual_seq_len=contextual_seq_len,
                    min_full_attn_seq_len=min_full_attn_seq_len,
                    sort_by_length=True,
                    kernel=kernel,
                )

        ms = _bench_fn(attn, mode, warmup, rep)
        if report_flops:
            all_flops = _flops(
                batch_size, attn_dim, hidden_dim, heads, seq_offsets, mode
            )
            return all_flops / ms / 1e9
        return ms

    df = _bench_hstu_attention.run(
        print_data=True,
        show_plots=False,
        save_path=save_path,
        return_df=return_result,
    )

    if return_result:
        return configs, df


if __name__ == "__main__":
    main()
