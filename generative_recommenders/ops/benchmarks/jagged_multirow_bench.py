# pyre-strict
import argparse
from typing import Tuple

import torch

# @manual=//triton:triton
import triton
from generative_recommenders.ops.triton.triton_jagged import (
    set_split_concat_2d_jagged_multirow_kernel,
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)

# buck2 run @mode/{opt,inplace} //generative_recommenders/ops/benchmarks:jagged_multirow_bench


def make_lengths(
    B: int, mean_len: int, max_len: int, seed: int, uniform: bool = False
) -> torch.Tensor:
    if uniform:
        return torch.full((B,), max_len, dtype=torch.long)
    g = torch.Generator(device="cpu").manual_seed(seed)
    la = torch.empty(B).exponential_(1.0 / mean_len, generator=g)
    la = la.clamp(1, max_len).round().long()
    la[0] = max_len  # force the tail so grid dim0 == max_len
    return la


def build_concat_inputs(
    B: int,
    mean_a: int,
    max_a: int,
    len_b: int,
    D: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
    uniform: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    la = make_lengths(B, mean_a, max_a, seed, uniform)
    lb = torch.full((B,), len_b, dtype=torch.long)
    combined_max = int((la + lb).max().item())
    offsets_a = torch.zeros(B + 1, dtype=torch.long)
    offsets_b = torch.zeros(B + 1, dtype=torch.long)
    offsets_a[1:] = torch.cumsum(la, 0)
    offsets_b[1:] = torch.cumsum(lb, 0)
    total_a, total_b = int(la.sum()), int(lb.sum())
    va = torch.randn(total_a, D, dtype=dtype, device=device)
    vb = torch.randn(total_b, D, dtype=dtype, device=device)
    return (
        va,
        vb,
        offsets_a.to(device),
        offsets_b.to(device),
        combined_max,
        total_a,
        total_b,
    )


def bench(
    B: int,
    dtype: torch.dtype,
    multirow: bool,
    device: torch.device,
    D: int = 128,
    quiet: bool = False,
    uniform: bool = False,
) -> Tuple[float, float]:
    mean_a, max_a, len_b = 1086, 3052, 16
    va, vb, off_a, off_b, cmax, total_a, total_b = build_concat_inputs(
        B, mean_a, max_a, len_b, D, dtype, device, uniform=uniform
    )
    set_split_concat_2d_jagged_multirow_kernel(multirow)
    elem = torch.tensor([], dtype=dtype).element_size()

    with torch.no_grad():
        out = triton_concat_2D_jagged(cmax, va, vb, off_a, off_b)

        def do_concat() -> None:
            triton_concat_2D_jagged(cmax, va, vb, off_a, off_b)

        ms_c = triton.testing.do_bench(do_concat, warmup=15, rep=50)
        gbps_c = (total_a + total_b) * D * elem * 2 / (ms_c * 1e-3) / 1e9
        values = out.contiguous()

        def do_split() -> None:
            triton_split_2D_jagged(values, cmax, off_a, off_b)

        ms_s = triton.testing.do_bench(do_split, warmup=15, rep=50)
        gbps_s = (total_a + total_b) * D * elem * 2 / (ms_s * 1e-3) / 1e9

    if not quiet:
        tag = "MULTIROW" if multirow else "single-row"
        print(
            f"[{tag}] B={B} D={D} dtype={dtype} max_seq_len={cmax} "
            f"working/grid={(total_a + total_b) / (cmax * B):.3f}",
            flush=True,
        )
        print(f"    concat fwd: {ms_c:.3f} ms  ({gbps_c:6.1f} GB/s)", flush=True)
        print(f"    split  fwd: {ms_s:.3f} ms  ({gbps_s:6.1f} GB/s)", flush=True)
    return ms_c, ms_s


def sweep(
    B: int, dtype: torch.dtype, device: torch.device, uniform: bool = False
) -> None:
    print(f"=== SWEEP B={B} dtype={dtype} (single-row -> multirow, x=speedup) ===")
    for D in [64, 128, 256, 512]:
        c0, s0 = bench(B, dtype, False, device, D, quiet=True, uniform=uniform)
        c1, s1 = bench(B, dtype, True, device, D, quiet=True, uniform=uniform)
        print(
            f"  D={D:4d}  concat {c0:.3f}->{c1:.3f} ({c0 / c1:.2f}x)   "
            f"split {s0:.3f}->{s1:.3f} ({s0 / s1:.2f}x)",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--multirow", type=int, default=-1, help="-1=both,0=single,1=multi")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--ncu", action="store_true", help="one concat+split, no timing")
    p.add_argument(
        "--uniform", action="store_true", help="uniform lengths (no null CTAs)"
    )
    p.add_argument("-d", type=int, default=128)
    args = p.parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    if args.ncu:
        if args.multirow not in (0, 1):
            p.error(
                "--ncu profiles a single kernel; pass --multirow 0 (single) or 1 (multi)"
            )
        va, vb, oa, ob, cmax, _, _ = build_concat_inputs(
            args.batch, 1086, 3052, 16, args.d, dtype, device, uniform=args.uniform
        )
        set_split_concat_2d_jagged_multirow_kernel(bool(args.multirow))
        with torch.no_grad():
            out = triton_concat_2D_jagged(cmax, va, vb, oa, ob).contiguous()
            triton_split_2D_jagged(out, cmax, oa, ob)
        torch.cuda.synchronize()
        return

    if args.sweep:
        sweep(args.batch, dtype, device, uniform=args.uniform)
        return

    modes = [False, True] if args.multirow == -1 else [bool(args.multirow)]
    for m in modes:
        bench(args.batch, dtype, m, device, args.d, uniform=args.uniform)


if __name__ == "__main__":
    main()
