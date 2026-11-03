#!/usr/bin/env python3
# pyre-unsafe
"""Tune the multirow split/concat kernels on an AMD GPU and print the resulting
autotune caches as base64 on stdout.

The usual way to produce an autotune cache is `buck2 run` on a real AMD host, but
`buck2 run` is local-only and the RE sandbox cannot write back into the repo, so
the cache is carried out through the test log instead:

    buck2 test @fbcode//mode/opt-amd-gpu \\
        fbcode//generative_recommenders/ops/tests:jagged_multirow_autotune_amd_test \\
        -- --print-passing-details

Then feed the two base64 blobs to
scratch/tritoncc-multirow/amd-migration-and-rerun/tools/decode_amd_cache.py.

`waves_per_eu` is dropped from the candidate list before tuning and from the
winning config afterwards: on AMD it lives in `Config.kwargs`, and every kwarg
becomes one more element of the generated `_meta` tuple, which the TritonCC
wrapper unpacks positionally. Filtering before tuning (rather than only stripping
after) keeps the recorded winner the best config that will actually be compiled.
"""

import base64
import pickle
import unittest
from typing import Dict, List, Tuple

import torch

# @manual=//triton:triton
import triton
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.ops.triton.triton_jagged import (
    concat_2D_jagged_multirow,
    split_2D_jagged_multirow,
)

# Mirrors the two generate_autotune_cache.py scripts: D values chosen so that
# next_power_of_2(D) covers every BLOCK_D enumerated in the respective specs.py.
_SPLIT_D_VALUES: Tuple[int, ...] = (64, 128, 256, 512, 768)
_CONCAT_D_VALUES: Tuple[int, ...] = (64, 128, 256, 512, 768, 1664)
_DTYPES: Tuple[torch.dtype, ...] = (
    torch.bfloat16,
    torch.float32,
    torch.float16,
)

_BATCH_SIZE: int = 12
_MAX_SEQ_LEN: int = 32768
_LAST_K: int = 20
_VALUES_BYTES_BUDGET: int = 8 << 30
_SPLIT_FILL: float = 0.65
_CONCAT_FILL: float = 0.59


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _batch_size_for(B: int, max_seq_len: int, D: int) -> int:
    return max(1, min(B, _VALUES_BYTES_BUDGET // (max_seq_len * D * 4)))


def _jagged_offsets(
    B: int, max_seq_len: int, last_k: int, fill: float
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)

    lengths = torch.randint(
        max(1, int((2.0 * fill - 1.0) * max_seq_len)),
        max_seq_len + 1,
        size=(B,),
        generator=g,
        device=device,
    )
    lengths_b = torch.clamp(lengths, max=last_k)
    lengths_a = lengths - lengths_b

    offsets_a = torch.zeros((B + 1,), dtype=torch.int64, device=device)
    offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
    offsets_b = torch.zeros((B + 1,), dtype=torch.int64, device=device)
    offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
    return offsets_a, offsets_b, int(offsets_a[-1].item()), int(offsets_b[-1].item())


def _drop_waves_per_eu(kernel) -> int:
    """Restrict the Autotuner to configs the TritonCC wrapper can consume."""
    before = len(kernel.configs)
    kernel.configs = [c for c in kernel.configs if not c.kwargs.get("waves_per_eu", 0)]
    for cfg in kernel.configs:
        cfg.kwargs.pop("waves_per_eu", None)
    assert kernel.configs, "no candidate config survived the waves_per_eu filter"
    return before


def _emit(name: str, cache: Dict[object, object]) -> None:
    assert cache, f"{name}: autotuner cache is empty - nothing was tuned"
    for key, cfg in cache.items():
        assert "waves_per_eu" not in cfg.kwargs, f"{name}: {key} kept waves_per_eu"
        print(f"    {key} -> {cfg}")
    blob = base64.b64encode(pickle.dumps(cache)).decode("ascii")
    lines: List[str] = [blob[i : i + 76] for i in range(0, len(blob), 76)]
    print(f"===BEGIN {name} {len(cache)} entries===")
    for line in lines:
        print(line)
    print(f"===END {name}===")


def _tune_split() -> Dict[object, object]:
    for dtype in _DTYPES:
        for D in _SPLIT_D_VALUES:
            B = _batch_size_for(B=_BATCH_SIZE, max_seq_len=_MAX_SEQ_LEN, D=D)
            oa, ob, na, nb = _jagged_offsets(B, _MAX_SEQ_LEN, _LAST_K, _SPLIT_FILL)
            values = torch.randn((na + nb, D), dtype=torch.float32, device="cuda").to(
                dtype
            )
            out_a = torch.empty((na, D), dtype=dtype, device="cuda")
            out_b = torch.empty((nb, D), dtype=dtype, device="cuda")
            BLOCK_D = _next_power_of_2(D)
            print(f"  split D={D} (BLOCK_D={BLOCK_D}) dtype={dtype} B={B}")

            def grid(meta, batch_size: int = B):
                return (triton.cdiv(_MAX_SEQ_LEN, meta["BLOCK_N"]), batch_size)

            split_2D_jagged_multirow[grid](
                values,
                0,  # DenseSize
                oa,
                ob,
                out_a,
                out_b,
                D,
                values.stride(0),
                out_a.stride(0),
                out_b.stride(0),
                IS_DENSE_A=False,
                IS_DENSE_B=False,
                BLOCK_D=BLOCK_D,
                IS_REPLACE=False,
                SKIP_LEFT=False,
            )
            torch.cuda.synchronize()
    return dict(split_2D_jagged_multirow.cache)


def _tune_concat() -> Dict[object, object]:
    for dtype in _DTYPES:
        for D in _CONCAT_D_VALUES:
            B = _batch_size_for(B=_BATCH_SIZE, max_seq_len=_MAX_SEQ_LEN, D=D)
            oa, ob, na, nb = _jagged_offsets(B, _MAX_SEQ_LEN, _LAST_K, _CONCAT_FILL)
            values_a = torch.randn((na, D), dtype=torch.float32, device="cuda").to(
                dtype
            )
            values_b = torch.randn((nb, D), dtype=torch.float32, device="cuda").to(
                dtype
            )
            out = torch.empty((na + nb, D), dtype=dtype, device="cuda")
            BLOCK_D = _next_power_of_2(D)
            print(f"  concat D={D} (BLOCK_D={BLOCK_D}) dtype={dtype} B={B}")

            def grid(meta, batch_size: int = B):
                return (triton.cdiv(_MAX_SEQ_LEN, meta["BLOCK_N"]), batch_size)

            concat_2D_jagged_multirow[grid](
                oa,
                values_a,
                ob,
                values_b,
                0,  # DenseSize
                out,
                D,
                values_a.stride(0),
                values_b.stride(0),
                0,  # stride_dense_batch
                out.stride(0),
                IS_DENSE_A=False,
                IS_DENSE_B=False,
                BLOCK_D=BLOCK_D,
                IS_REPLACE=False,
                SKIP_LEFT=False,
            )
            torch.cuda.synchronize()
    return dict(concat_2D_jagged_multirow.cache)


class JaggedMultirowAutotuneAmdTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_emit_amd_autotune_caches(self) -> None:
        self.assertIsNotNone(
            torch.version.hip, "must run on an AMD GPU - use @mode/opt-amd-gpu"
        )
        print(f"Device: {torch.cuda.get_device_name()}")
        for name, kernel in (
            ("split", split_2D_jagged_multirow),
            ("concat", concat_2D_jagged_multirow),
        ):
            before = _drop_waves_per_eu(kernel)
            print(f"{name} candidates: {before} -> {len(kernel.configs)}")
        _emit("split_2D_jagged_multirow", _tune_split())
        _emit("concat_2D_jagged_multirow", _tune_concat())


if __name__ == "__main__":
    unittest.main()
