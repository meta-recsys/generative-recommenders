# HSTU attention on AMD GPUs (Composable Kernel)

This directory contains the HSTU attention forward and backward kernels for AMD
Instinct GPUs, written with [Composable Kernel](https://github.com/ROCm/composable_kernel)
(`ck_tile`).

HSTU attention computes `o = (SiLU(alpha * q @ k^T) * mask * attn_scale) @ v`
(or a scaled softmax instead of SiLU with `use_softmax=True`). The mask
combines causal, local (sliding window), contextual, min-full-attention and
target masking. The kernels support jagged and batched (dense) inputs,
self-attention and cross-attention (separate Q and KV sequence offsets),
grouped-query attention (fewer KV heads than Q heads), head dims 64, 96, 128
and 256, and fp16 / bf16.

| Arch | GPU | Forward | Backward |
|------|-----|---------|----------|
| gfx942 | MI300X / MI325X | yes | yes |
| gfx950 | MI350X / MI355X | yes | yes |
| gfx1250 | MI450X / MI455X | yes | no |

## Layout

- `ck_hstu_attention.py`: the Python API, `ck_hstu_mha(...)`.
- `hstu_attention/`: kernel headers, the generated kernel instances
  (`instances/`, regenerate with `generate_{fwd,bwd}_instances.py`), and the
  PyTorch bindings (`hstu_attention_torch_ops*.{hip,cpp}`). These register
  `torch.ops.ck_hstu_experimental.ck_hstu_mha` (autograd-enabled), plus
  `ck_hstu_mha_fwd` and `ck_hstu_mha_bwd`, with Meta (fake) implementations
  for `torch.compile` / export.
- `setup.py`: builds the PyTorch extension.
- `composable_kernel/`: the Composable Kernel git submodule (headers only).

## Build

Requirements: ROCm 7.0 or newer, and a ROCm build of PyTorch.

```bash
git submodule update --init generative_recommenders/ops/ck/composable_kernel
cd generative_recommenders/ops/ck
# Build only for the GPUs you have; each arch adds to the build time.
GPU_ARCHS="gfx942;gfx950" MAX_JOBS=64 python setup.py install
```

Use `rocminfo | grep gfx` to find your GPU arch. Set `CK_DIR` to use an
existing composable_kernel checkout instead of the submodule.

## Usage

```python
import torch
from generative_recommenders.ops.ck.ck_hstu_attention import ck_hstu_mha

# Jagged self-attention: q, k, v are [total_tokens, num_heads, head_dim].
out = ck_hstu_mha(
    max_seq_len=max_seq_len,
    alpha=1.0 / head_dim,
    q=q,
    k=k,
    v=v,
    seq_offsets_q=seq_offsets,
    seq_offsets_kv=seq_offsets,
    causal=True,
    num_targets=num_targets,
)
out.sum().backward()
```

## Tests and benchmark

From the repository root:

```bash
python -m pytest generative_recommenders/ops/tests/ck_hstu_attention_test.py \
    generative_recommenders/ops/tests/ck_hstu_cross_attention_test.py
python -m generative_recommenders.ops.benchmarks.ck_hstu_attention_bench
```

The tests compare against the PyTorch reference in
`generative_recommenders/ops/pytorch/pt_hstu_attention.py`.

## License

The kernel sources in `hstu_attention/` come from AMD's
[Composable Kernel](https://github.com/ROCm/composable_kernel)
`example/ck_tile/18_hstu_attention`, branch `hstu_attention_fwd_bwd`, commit
[`0ea9398523f9313ad2d89eed86927313a3b46e71`](https://github.com/ROCm/composable_kernel/tree/0ea9398523f9313ad2d89eed86927313a3b46e71/example/ck_tile/18_hstu_attention).
They are licensed under the MIT license in
[`hstu_attention/LICENSE`](hstu_attention/LICENSE), including Meta's later
changes to them. Files that Meta changed beyond formatting carry an additional
`Modifications Copyright (c) Meta Platforms, Inc. and affiliates.` line. The
main changes are the forward and backward pipeline selection on each arch, the
gfx942/gfx950 backward tile settings, the split-KV heuristics and forward
strategy, and the per-dtype kernel entry points.

The PyTorch op registration (`hstu_attention_torch_ops*.{hip,cpp}`), the
instance generators (`generate_{fwd,bwd}_instances.py`) and the rest of this
directory (`ck_hstu_attention.py`, `setup.py`) are licensed under the
Apache-2.0 license of this repository (also in [`LICENSE`](LICENSE)). The
license in each file's header applies.
