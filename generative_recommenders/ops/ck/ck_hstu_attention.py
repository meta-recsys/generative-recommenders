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

import logging
from typing import Optional

import torch

logger: logging.Logger = logging.getLogger(__name__)

_CK_HSTU_EXTENSION_LOADED = False
try:
    # Open-source build (ops/ck/setup.py): importing the extension registers the
    # torch.ops.ck_hstu_experimental ops. It must be imported after torch.
    import ck_hstu_experimental._ck_hstu_experimental_C  # noqa: F401  # pyre-ignore [21]

    _CK_HSTU_EXTENSION_LOADED = True
except (ImportError, OSError):
    pass

if not _CK_HSTU_EXTENSION_LOADED:
    try:
        # Buck build.
        torch.ops.load_library(
            "//generative_recommenders/ops/ck/hstu_attention:hstu_attention_pybind"
        )
    except OSError as e:
        logger.warning(f"Could not load the CK HSTU attention library: {e}")


def is_ck_hstu_available() -> bool:
    """True if running on an AMD GPU with the CK HSTU attention ops registered."""
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and hasattr(torch.ops.ck_hstu_experimental, "ck_hstu_mha")
    )


@torch.fx.wrap
def _min_seq_len_from_offsets(seq_offsets: torch.Tensor) -> int:
    """
    Smallest jagged sequence length behind a prefix-sum offsets tensor, for
    ck_hstu_mha's min_seq_len_q / min_seq_len_kv.

    Must be called from this module. torch.fx.wrap only patches the globals of
    the module that registered it, so a caller reaching this through
    `from ... import` is traced straight through and the host read is constant
    folded into the graph instead of becoming a node.

    Clamped to 1 because 0 is the kernel's "unset" sentinel; an empty sequence
    and a length-1 one both sit far below the invariance threshold anyway.
    """
    seq_lens = seq_offsets[1:] - seq_offsets[:-1]
    if seq_lens.numel() == 0:
        # Empty batch: torch.min would raise. Export-time shape propagation
        # can replay this node on zero-length offsets.
        return 1
    return max(int(torch.min(seq_lens).item()), 1)


@torch.fx.wrap
def ck_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets_q: Optional[torch.Tensor] = None,
    seq_offsets_kv: Optional[torch.Tensor] = None,
    causal: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    min_full_attn_seq_len: int = 0,
    contextual_seq_len: int = 0,
    use_softmax: bool = False,
    is_cross_attention: bool = False,
    max_seq_len_kv: int = 0,
    min_seq_len_q: int = 0,
    min_seq_len_kv: int = 0,
    kv_truncate_len: int = 0,
    kv_keep_last: bool = False,
) -> torch.Tensor:
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
        max_seq_len_kv: max KV sequence length, used by the cross-attention
            mtile / split-kv dispatch heuristic. 0 falls back to max_seq_len
            (correct for self-attention where KV length == Q length).
        min_seq_len_q, min_seq_len_kv: min Q/KV sequence lengths (jagged mode).
            The min/max seqlen ratio selects the grid schedule
            (is_almost_invariant_seqlen). Left at 0 they are read off the
            offsets here; pass values only to override that.
        kv_truncate_len: jagged mode, forward only. > 0 keeps kv_truncate_len
            keys of each KV sequence: the last ones if kv_keep_last, else the
            first ones (truncate_method "keep_last" / "keep_first").
    """
    # Pass attn_scale through as a Tensor (not .item()) so it stays a normal
    # graph value under Dynamo/AOTI export instead of becoming a data-dependent
    # SymFloat; the HIP kernel extracts the float at execution time. Cast to
    # float32 because the kernel reads it as float32 (a mismatched dtype makes
    # the kernel read garbage bits -> NaN).
    if attn_scale is not None:
        attn_scale = attn_scale.to(torch.float32)
    # Under torch.compile / torch.export a data-dependent host read cannot be
    # threaded into an `int` schema argument, so leave the 0 sentinel there.
    if not torch.compiler.is_compiling():
        if seq_offsets_q is not None and min_seq_len_q <= 0:
            min_seq_len_q = _min_seq_len_from_offsets(seq_offsets_q)
        if seq_offsets_kv is not None and min_seq_len_kv <= 0:
            # Self-attention passes one tensor for both; one read covers both.
            min_seq_len_kv = (
                min_seq_len_q
                if seq_offsets_kv is seq_offsets_q
                else _min_seq_len_from_offsets(seq_offsets_kv)
            )
    return torch.ops.ck_hstu_experimental.ck_hstu_mha(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets_q,
        seq_offsets_kv,
        causal,
        num_targets,
        max_attn_len,
        min_full_attn_seq_len,
        contextual_seq_len,
        attn_scale,
        use_softmax,
        is_cross_attention,
        max_seq_len_kv,
        min_seq_len_q,
        min_seq_len_kv,
        kv_truncate_len,
        kv_keep_last,
    )
