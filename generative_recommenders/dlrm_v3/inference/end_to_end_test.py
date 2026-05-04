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
End-to-end smoke test for the HSTU TorchScript + C++ deployment pipeline.

What this binary does, in order:

1. Build a synthetic batch (uih_kjt, candidates_kjt) via :func:`get_random_data`.
2. Build the eager :class:`HSTUSparseScriptModule` and
   :class:`HSTUDenseScriptModule`.
3. Run them eagerly to obtain the reference ``preds_eager``.
4. ``torch.jit.script`` + save:
       - ``sparse.pt`` (CPU)
       - ``dense.pt``  (cuda:0, bf16)
       - ``inputs.pt`` (an :class:`InputsBundle` ScriptModule whose
         ``forward()`` returns ``Tuple[KeyedJaggedTensor, KeyedJaggedTensor]``)
5. ``subprocess.run`` the C++ runner
       ``hstu_runner <sparse.pt> <dense.pt> <inputs.pt> <preds_cpp.pt>``.
6. ``torch.load`` the runner's output and compare against ``preds_eager``
   with :func:`torch.testing.assert_close` (loose tolerance because the
   scripted path uses PyTorch fallbacks instead of Triton + drops autocast).

Usage (manual override of the runner path):

    buck2 run @mode/opt //generative_recommenders/dlrm_v3/inference:end_to_end_test \\
        -- --cpp_runner /path/to/hstu_runner

By default the binary locates the runner via ``libfb.py.parutil`` -- it ships
inside the par as a resource (see BUCK).
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import torch
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import get_random_data
from generative_recommenders.dlrm_v3.inference.dense_predict_module import (
    HSTUDenseScriptModule,
)
from generative_recommenders.dlrm_v3.inference.sparse_predict_module import (
    HSTUSparseScriptModule,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)


_DEFAULT_DATASET = "kuairand-1k"


class InputsBundle(torch.nn.Module):
    """Scripted holder for the test inputs.

    Returns the constituent tensors of the two KJTs as a 4-tuple
    ``(uih_lengths, uih_values, candidates_lengths, candidates_values)`` so
    the traced sparse module can rebuild the KJTs inside its forward (KJT
    instances themselves are not traceable inputs).
    """

    def __init__(
        self,
        uih_kjt: KeyedJaggedTensor,
        candidates_kjt: KeyedJaggedTensor,
    ) -> None:
        super().__init__()
        self.register_buffer("uih_lengths", uih_kjt.lengths())
        self.register_buffer("uih_values", uih_kjt.values())
        self.register_buffer("candidates_lengths", candidates_kjt.lengths())
        self.register_buffer("candidates_values", candidates_kjt.values())

    def forward(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.uih_lengths,
            self.uih_values,
            self.candidates_lengths,
            self.candidates_values,
        )


class _SparseTraceShim(torch.nn.Module):
    """Adapter that takes raw tensors and rebuilds the KJTs inside forward.

    ``torch.jit.trace`` does not accept ``KeyedJaggedTensor`` (or any
    non-Tensor / non-collection-of-Tensor type) as a top-level forward
    input, so we make the traced boundary tensor-only and bake the
    ``List[str]`` of feature keys in as Python constants captured by the
    closure / module attribute.
    """

    def __init__(
        self,
        sparse_module: HSTUSparseScriptModule,
        uih_keys: List[str],
        candidates_keys: List[str],
    ) -> None:
        super().__init__()
        self._sparse_module: HSTUSparseScriptModule = sparse_module
        self._uih_keys: List[str] = uih_keys
        self._candidates_keys: List[str] = candidates_keys

    def forward(
        self,
        uih_lengths: torch.Tensor,
        uih_values: torch.Tensor,
        candidates_lengths: torch.Tensor,
        candidates_values: torch.Tensor,
    ):
        uih_kjt = KeyedJaggedTensor(
            keys=self._uih_keys,
            lengths=uih_lengths,
            values=uih_values,
        )
        candidates_kjt = KeyedJaggedTensor(
            keys=self._candidates_keys,
            lengths=candidates_lengths,
            values=candidates_values,
        )
        return self._sparse_module(
            uih_features=uih_kjt, candidates_features=candidates_kjt
        )


def _find_cpp_runner() -> str:
    """Locate the bundled hstu_runner binary.

    Tries ``importlib.resources`` (the canonical fbcode resource resolver,
    works whether the binary is in a par or unpacked), and falls back to
    looking next to ``sys.argv[0]``.
    """
    try:
        from importlib.resources import files

        path = files("generative_recommenders.dlrm_v3.inference.cpp").joinpath(
            "hstu_runner"
        )
        if path.is_file():
            return str(path)
    except Exception as exc:
        logger.debug("importlib.resources lookup failed: %s", exc)

    candidate = os.path.join(
        os.path.dirname(os.path.abspath(sys.argv[0])), "hstu_runner"
    )
    if os.path.exists(candidate):
        return candidate

    raise RuntimeError(
        "Could not find hstu_runner binary. "
        "Pass --cpp_runner=<path> or build the cpp_binary target first."
    )


def _eager_run(
    sparse_module: HSTUSparseScriptModule,
    dense_module: HSTUDenseScriptModule,
    uih_kjt: KeyedJaggedTensor,
    candidates_kjt: KeyedJaggedTensor,
    device: torch.device,
) -> torch.Tensor:
    """Reference path: sparse → device-move + bf16 → dense, all in Python."""
    with torch.no_grad():
        seq_emb_values, seq_emb_lengths, payload, uih_lens, num_cands = sparse_module(
            uih_features=uih_kjt, candidates_features=candidates_kjt
        )
        seq_emb_values = {
            k: v.to(device).to(torch.bfloat16) for k, v in seq_emb_values.items()
        }
        seq_emb_lengths = {k: v.to(device) for k, v in seq_emb_lengths.items()}
        payload = {k: v.to(device) for k, v in payload.items()}
        uih_lens = uih_lens.to(device)
        num_cands = num_cands.to(device)
        preds = dense_module(
            seq_emb_values, seq_emb_lengths, payload, uih_lens, num_cands
        )
    return preds.detach().to(torch.float32).cpu()


def _build_synthetic_inputs(
    hstu_config,
    table_config: Dict[str, EmbeddingConfig],
    uih_max_seq_len: int,
) -> Tuple[KeyedJaggedTensor, KeyedJaggedTensor]:
    contextual: List[str] = list(hstu_config.contextual_feature_to_max_length.keys())
    # The kuairand-1k dataset has tiny embedding tables for some contextual
    # features (e.g. user_active_degree has num_embeddings=8). Clamp the
    # random value range so every index stays in range for every table.
    min_rows = min(t.num_embeddings for t in table_config.values())
    value_bound = max(2, min_rows)
    logger.info(
        "synthetic value_bound=%d (min table rows=%d across %d tables)",
        value_bound,
        min_rows,
        len(table_config),
    )
    return get_random_data(
        contexual_features=contextual,
        hstu_uih_keys=hstu_config.hstu_uih_feature_names,
        hstu_candidates_keys=hstu_config.hstu_candidate_feature_names,
        uih_max_seq_len=uih_max_seq_len,
        max_num_candidates=hstu_config.max_num_candidates_inference,
        value_bound=value_bound,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpp_runner",
        type=str,
        default=None,
        help="Path to the hstu_runner binary; default: bundled resource.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help="Dataset key for HSTU/embedding configs.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Dense-module device."
    )
    parser.add_argument(
        "--uih_max_seq_len",
        type=int,
        default=128,
        help="Max UIH length for the synthetic batch.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument(
        "--keep_workdir",
        action="store_true",
        help="Do not delete the temp dir holding the saved artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[e2e] %(message)s", force=True)
    logger.setLevel(logging.DEBUG)
    args = _parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA is required; aborting.")
        sys.exit(2)

    runner_path = args.cpp_runner or _find_cpp_runner()
    logger.info("Using C++ runner: %s", runner_path)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    hstu_config = get_hstu_configs(args.dataset)
    table_config = get_embedding_table_config(args.dataset)

    uih_kjt, candidates_kjt = _build_synthetic_inputs(
        hstu_config, table_config, args.uih_max_seq_len
    )

    sparse_module = HSTUSparseScriptModule(
        table_config=table_config,
        hstu_config=hstu_config,
        use_no_copy_embedding_collection=True,
    ).eval()
    dense_module = (
        HSTUDenseScriptModule(hstu_config=hstu_config, table_config=table_config)
        .to(torch.bfloat16)
        .to(device)
        .eval()
    )

    # Pin the HammerKernel to PyTorch on both wrappers. This bypasses the
    # Triton kernels for jagged ops / layer-norm / addmm / hstu attention.
    # Reasons:
    #  - Triton kernels use Python-level dispatch (autotune, constexpr
    #    arguments) that interacts badly with torch.jit.trace's recording
    #    pass.
    #  - The C++ libtorch runtime cannot dispatch into Triton kernels at
    #    deployment time; the scripted graph would have to call the CUDA
    #    C++ ops anyway.
    # The eager reference run (below) uses the same setting so the
    # comparison is apples-to-apples.
    from generative_recommenders.common import HammerKernel

    sparse_module._sparse._hstu_model.set_hammer_kernel(HammerKernel.PYTORCH)
    dense_module._hstu_model.set_hammer_kernel(HammerKernel.PYTORCH)

    # Diagnostic: walk every HammerModule submodule and print its effective
    # kernel selection, so any submodule that didn't pick up the override
    # surfaces immediately. Triton/Triton-CC selections will fail at trace
    # time, so this print is critical for triaging the next iteration if
    # tracing fails.
    from generative_recommenders.common import HammerModule as _HM

    for name, m in list(sparse_module.named_modules()) + list(
        dense_module.named_modules()
    ):
        if isinstance(m, _HM):
            logger.info(
                "kernel-pin %-60s -> %s (is_inference=%s, use_triton_cc=%s)",
                name or "<root>",
                m.hammer_kernel().value,
                m._is_inference,
                m._use_triton_cc,
            )

    # === 1. Eager reference ===
    logger.info("Running eager reference...")
    preds_eager = _eager_run(
        sparse_module, dense_module, uih_kjt, candidates_kjt, device
    )
    logger.info(
        "preds_eager shape=%s sum=%.6f",
        tuple(preds_eager.shape),
        preds_eager.sum().item(),
    )

    # === 2. Trace + save ===
    # We use torch.jit.trace (not torch.jit.script) for the model wrappers.
    # Tracing records the actual tensor ops executed during a forward pass,
    # so it ignores all the source-level dispatch logic (HammerKernel enum,
    # is_fx_tracing(), torch.autocast, IntEnum branches, etc.) that the
    # large HSTU model code base uses extensively. Script is only used for
    # the InputsBundle, which is a small KJT holder with no such issues.
    workdir = tempfile.mkdtemp(prefix="hstu_e2e_")
    sparse_path = os.path.join(workdir, "sparse.pt")
    dense_path = os.path.join(workdir, "dense.pt")
    inputs_path = os.path.join(workdir, "inputs.pt")
    cpp_out_path = os.path.join(workdir, "preds_cpp.pt")
    eager_out_path = os.path.join(workdir, "preds_eager.pt")
    logger.info("workdir: %s", workdir)

    # Re-run sparse eagerly to capture an example output that can drive the
    # dense trace.
    with torch.no_grad():
        sparse_out = sparse_module(
            uih_features=uih_kjt, candidates_features=candidates_kjt
        )
        seq_emb_values = {
            k: v.to(device).to(torch.bfloat16) for k, v in sparse_out[0].items()
        }
        seq_emb_lengths = {k: v.to(device) for k, v in sparse_out[1].items()}
        payload = {k: v.to(device) for k, v in sparse_out[2].items()}
        uih_lens = sparse_out[3].to(device)
        num_cands = sparse_out[4].to(device)

    logger.info("Tracing sparse module via raw-tensor shim (CPU)...")
    sparse_shim = _SparseTraceShim(
        sparse_module=sparse_module,
        uih_keys=list(uih_kjt.keys()),
        candidates_keys=list(candidates_kjt.keys()),
    )
    traced_sparse = torch.jit.trace(
        sparse_shim,
        example_inputs=(
            uih_kjt.lengths(),
            uih_kjt.values(),
            candidates_kjt.lengths(),
            candidates_kjt.values(),
        ),
        strict=False,
        check_trace=False,
    )
    traced_sparse.save(sparse_path)

    logger.info("Tracing dense module (cuda:0, bf16)...")
    traced_dense = torch.jit.trace(
        dense_module,
        example_inputs=(
            seq_emb_values,
            seq_emb_lengths,
            payload,
            uih_lens,
            num_cands,
        ),
        strict=False,
        check_trace=False,
    )
    traced_dense.save(dense_path)

    logger.info("Scripting + saving inputs bundle...")
    torch.jit.script(InputsBundle(uih_kjt, candidates_kjt)).save(inputs_path)
    torch.save(preds_eager, eager_out_path)

    # === 2.5. Python-side roundtrip verification ===
    # Load the saved traced artifacts back in Python and verify they produce
    # the same results as the eager run. This proves the artifacts are correct
    # independently of the C++ runner.
    logger.info("Python roundtrip: loading traced artifacts back...")
    rt_inputs = torch.jit.load(inputs_path)
    rt_sparse = torch.jit.load(sparse_path)
    rt_dense = torch.jit.load(dense_path)

    with torch.no_grad():
        rt_uih_l, rt_uih_v, rt_cand_l, rt_cand_v = rt_inputs()
        logger.info(
            "  rt inputs: uih_l=%s uih_v=%s cand_l=%s cand_v=%s",
            rt_uih_l.shape,
            rt_uih_v.shape,
            rt_cand_l.shape,
            rt_cand_v.shape,
        )

        rt_sparse_out = rt_sparse(rt_uih_l, rt_uih_v, rt_cand_l, rt_cand_v)

        for i, elem in enumerate(rt_sparse_out):
            if isinstance(elem, dict):
                for k, v in elem.items():
                    has_nan = torch.isnan(v).any().item()
                    has_inf = torch.isinf(v).any().item()
                    logger.info(
                        "  sparse_out[%d][%s] shape=%s dtype=%s nan=%s inf=%s",
                        i,
                        k,
                        tuple(v.shape),
                        v.dtype,
                        has_nan,
                        has_inf,
                    )
            elif isinstance(elem, torch.Tensor):
                logger.info(
                    "  sparse_out[%d] shape=%s dtype=%s nan=%s inf=%s",
                    i,
                    tuple(elem.shape),
                    elem.dtype,
                    torch.isnan(elem).any().item(),
                    torch.isinf(elem).any().item(),
                )

        rt_sev = {
            k: v.to(device).to(torch.bfloat16) for k, v in rt_sparse_out[0].items()
        }
        rt_sel = {k: v.to(device) for k, v in rt_sparse_out[1].items()}
        rt_pay = {k: v.to(device) for k, v in rt_sparse_out[2].items()}
        rt_uih = rt_sparse_out[3].to(device)
        rt_nc = rt_sparse_out[4].to(device)

        preds_rt = rt_dense(rt_sev, rt_sel, rt_pay, rt_uih, rt_nc)

    preds_rt_cpu = preds_rt.detach().to(torch.float32).cpu()
    logger.info(
        "preds_roundtrip shape=%s sum=%.6f nan=%s inf=%s",
        tuple(preds_rt_cpu.shape),
        preds_rt_cpu.sum().item(),
        torch.isnan(preds_rt_cpu).any().item(),
        torch.isinf(preds_rt_cpu).any().item(),
    )

    try:
        torch.testing.assert_close(
            preds_eager, preds_rt_cpu, atol=args.atol, rtol=args.rtol
        )
    except AssertionError as e:
        logger.error("PYTHON ROUNDTRIP PARITY FAILED: %s", e)
        if not args.keep_workdir:
            logger.info("(workdir kept for inspection: %s)", workdir)
        sys.exit(1)
    logger.info("PYTHON ROUNDTRIP PASSED (atol=%g rtol=%g)", args.atol, args.rtol)

    # === 3. Invoke C++ runner ===
    cmd = [runner_path, sparse_path, dense_path, inputs_path, cpp_out_path]
    logger.info("Running C++: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.stdout:
        logger.info("--- runner stdout ---\n%s", result.stdout.rstrip())
    if result.stderr:
        logger.info("--- runner stderr ---\n%s", result.stderr.rstrip())
    if result.returncode != 0:
        if result.returncode == -11:
            logger.warning(
                "C++ runner SIGSEGV (exit -11). This is a known issue with "
                "torch-cpp-cuda static initialization on some machines. "
                "Python roundtrip verification passed above. "
                "Artifacts in: %s",
                workdir,
            )
            args.keep_workdir = True
        else:
            logger.error("C++ runner exited with code %d", result.returncode)
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(result.returncode)

    # === 4. Compare ===
    if not os.path.exists(cpp_out_path):
        logger.error("C++ runner did not produce %s", cpp_out_path)
        sys.exit(1)
    preds_cpp = torch.load(cpp_out_path).to(torch.float32).cpu()
    logger.info(
        "preds_cpp   shape=%s sum=%.6f",
        tuple(preds_cpp.shape),
        preds_cpp.sum().item(),
    )

    try:
        torch.testing.assert_close(
            preds_eager, preds_cpp, atol=args.atol, rtol=args.rtol
        )
    except AssertionError as e:
        logger.error("PARITY FAILED: %s", e)
        if not args.keep_workdir:
            logger.info("(workdir kept for inspection: %s)", workdir)
        sys.exit(1)

    logger.info("PASSED: eager and C++ agree (atol=%g rtol=%g)", args.atol, args.rtol)
    if not args.keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
