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
5. Run the C++ runner
       ``hstu_runner [--aott_library <lib.so> ...] <sparse.pt> <dense.pt> <inputs.pt> <preds_cpp.pt>``.
6. ``torch.load`` the runner's output and compare against ``preds_eager``
   with :func:`torch.testing.assert_close` (loose tolerance because the
   scripted path may use either the PyTorch fallback trace or AOT-T-loaded
   Triton inference kernels).

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
import sys
import tempfile
from typing import Any, Dict, List, Tuple

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
from generative_recommenders.dlrm_v3.inference.ts_types import (
    SeqEmbLengths,
    SeqEmbValues,
    unflatten_seq_embeddings,
)
from generative_recommenders.modules.dlrm_hstu import DlrmHSTUConfig
from security.frameworks.python.exec.subprocess import TrustedSubprocessWithList
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
    ) -> Tuple[
        SeqEmbValues,
        SeqEmbLengths,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
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


class _DenseAottTraceShim(torch.nn.Module):
    """FX-traceable dense adapter for the representative AOT-T shape."""

    def __init__(
        self,
        dense_module: HSTUDenseScriptModule,
        max_uih_len: int,
        max_num_candidates: int,
        total_uih_len: int,
        total_targets: int,
    ) -> None:
        super().__init__()
        self._dense_module: HSTUDenseScriptModule = dense_module
        self._max_uih_len: int = max_uih_len
        self._max_num_candidates: int = max_num_candidates
        self._total_uih_len: int = total_uih_len
        self._total_targets: int = total_targets

    def forward(
        self,
        seq_emb_values: SeqEmbValues,
        seq_emb_lengths: SeqEmbLengths,
        payload_features: Dict[str, torch.Tensor],
        uih_seq_lengths: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        seq_embeddings = unflatten_seq_embeddings(seq_emb_values, seq_emb_lengths)

        (
            _,
            _,
            _,
            mt_target_preds,
            _mt_target_labels,
            _mt_target_weights,
        ) = self._dense_module._hstu_model.main_forward(
            seq_embeddings=seq_embeddings,
            payload_features=payload_features,
            max_uih_len=self._max_uih_len,
            uih_seq_lengths=uih_seq_lengths,
            max_num_candidates=self._max_num_candidates,
            num_candidates=num_candidates,
            total_uih_len=self._total_uih_len,
            total_targets=self._total_targets,
        )
        assert mt_target_preds is not None
        return mt_target_preds


def _dense_aott_concrete_args(
    dense_inputs: Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ],
) -> Dict[str, Any]:
    from torch.fx._symbolic_trace import PH

    seq_emb_values, seq_emb_lengths, payload_features, _, _ = dense_inputs
    return {
        "seq_emb_values": {key: PH for key in seq_emb_values},
        "seq_emb_lengths": {key: PH for key in seq_emb_lengths},
        "payload_features": {key: PH for key in payload_features},
    }


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


def _find_aott_libraries() -> List[str]:
    from generative_recommenders.ops.triton_aot.compile.compile_state import (
        get_aott_compile_path,
    )

    compile_path = get_aott_compile_path()
    libraries: List[str] = []
    for root, _, files in os.walk(compile_path):
        for filename in files:
            if filename.endswith(".so"):
                libraries.append(os.path.join(root, filename))
    return sorted(libraries)


def _copy_aott_libraries_to_workdir(
    library_paths: List[str], workdir: str
) -> List[str]:
    copied: List[str] = []
    for index, path in enumerate(library_paths):
        dst = os.path.join(workdir, f"aott_{index}_{os.path.basename(path)}")
        shutil.copy2(path, dst)
        copied.append(dst)
    return copied


def _load_aott_libraries_for_python(library_paths: List[str]) -> None:
    for library_path in library_paths:
        logger.info("Python roundtrip: loading AOT-T library %s", library_path)
        torch.ops.load_library(library_path)


def _save_aott_dense_module(
    dense_module: HSTUDenseScriptModule,
    dense_inputs: Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ],
    dense_path: str,
    workdir: str,
    atol: float,
    rtol: float,
) -> List[str]:
    """Lower the dense module with AOT-T and save a TorchScript artifact.

    This follows the AOT-T example flow:

    1. FX trace the module.
    2. Unwrap outer `aot_triton_kernel_wrapper_*` nodes.
    3. Run representative CUDA inputs under `TritonAOTCompile`.
    4. `transform_kernels` to replace wrappers with `torch.ops.triton_aot.*`.
    5. Script and save the transformed dense module.

    The full HSTU dense wrapper has historically needed tracing rather than FX,
    so failures here are reported with context and the default path remains the
    D102 traced TorchScript fallback.
    """
    from generative_recommenders.ops.triton_aot.compile.triton_aot_compile import (
        TritonAOTCompile,
    )
    from generative_recommenders.ops.triton_aot.preprocess import (
        unwrap_aott_wrapper_nodes,
    )
    from generative_recommenders.ops.triton_aot.transform.transform_kernels import (
        transform_kernels,
    )
    from tgif.fx.tgif_tracer import TGIFTracer

    max_uih_len = int(dense_inputs[3].max().item())
    max_num_candidates = int(dense_inputs[4].max().item())
    total_uih_len = int(dense_inputs[3].sum().item())
    total_targets = int(dense_inputs[4].sum().item())
    trace_shim = _DenseAottTraceShim(
        dense_module=dense_module,
        max_uih_len=max_uih_len,
        max_num_candidates=max_num_candidates,
        total_uih_len=total_uih_len,
        total_targets=total_targets,
    ).eval()

    logger.info(
        "AOT-T dense: FX tracing representative shape "
        "(max_uih_len=%d, max_num_candidates=%d, "
        "total_uih_len=%d, total_targets=%d)...",
        max_uih_len,
        max_num_candidates,
        total_uih_len,
        total_targets,
    )
    try:
        fx_dense = TGIFTracer().symbolic_trace(
            trace_shim,
            concrete_args=_dense_aott_concrete_args(dense_inputs),
        )
        lowered_dense = unwrap_aott_wrapper_nodes(fx_dense, TGIFTracer())
    except Exception as exc:
        raise RuntimeError(
            "AOT-T dense lowering requires an FX-traceable dense entry point. "
            "Use --dense_backend=torchscript to fall back to the D102 traced "
            "TorchScript path."
        ) from exc

    logger.info("AOT-T dense: compiling Triton kernels from sample inputs...")
    with torch.no_grad():
        with TritonAOTCompile():
            ref_output = lowered_dense(*dense_inputs)

    original_code = lowered_dense.code
    lowered_dense = transform_kernels(lowered_dense)
    if lowered_dense.code == original_code:
        logger.warning(
            "AOT-T dense: transform_kernels did not change the FX graph. "
            "This usually means no aot_triton_kernel_wrapper_* nodes were "
            "present in the dense path."
        )

    libraries = _find_aott_libraries()
    if not libraries:
        raise RuntimeError(
            "AOT-T dense lowering produced no .so files. Ensure the dense path "
            "uses HammerKernel.TRITON_INFERENCE branches backed by triton_aot ops."
        )

    with torch.no_grad():
        lowered_output = lowered_dense(*dense_inputs)
    torch.testing.assert_close(ref_output, lowered_output, atol=atol, rtol=rtol)

    logger.info("AOT-T dense: tracing transformed module...")
    torch.jit.trace(
        lowered_dense,
        example_inputs=dense_inputs,
        strict=False,
        check_trace=False,
    ).save(dense_path)
    copied_libraries = _copy_aott_libraries_to_workdir(libraries, workdir)
    logger.info("AOT-T dense: copied %d library file(s)", len(copied_libraries))
    return copied_libraries


def _build_synthetic_inputs(
    hstu_config: DlrmHSTUConfig,
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
        "--dense_backend",
        choices=("torchscript", "aott"),
        default="torchscript",
        help="Dense artifact backend. aott lowers TRITON_INFERENCE wrappers and passes compiled libraries to the C++ runner.",
    )
    parser.add_argument(
        "--aott_library",
        action="append",
        default=[],
        help="Additional prebuilt AOT-T shared library to dlopen before loading dense.pt. May be repeated.",
    )
    parser.add_argument(
        "--keep_workdir",
        action="store_true",
        help="Do not delete the temp dir holding the saved artifacts.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901
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

    from generative_recommenders.common import HammerKernel

    dense_kernel = (
        HammerKernel.TRITON_INFERENCE
        if args.dense_backend == "aott"
        else HammerKernel.PYTORCH
    )
    sparse_module._sparse._hstu_model.set_hammer_kernel(HammerKernel.PYTORCH)
    dense_module._hstu_model.set_hammer_kernel(dense_kernel)

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

    # === 2. Trace/lower + save ===
    # The default path keeps D102's trace-based TorchScript artifact. The
    # AOT-T path follows ModelStore's compile/transform flow and saves a
    # scripted FX module whose Triton kernels dispatch through torch.ops.
    workdir = tempfile.mkdtemp(prefix="hstu_e2e_")
    sparse_path = os.path.join(workdir, "sparse.pt")
    dense_path = os.path.join(workdir, "dense.pt")
    inputs_path = os.path.join(workdir, "inputs.pt")
    cpp_out_path = os.path.join(workdir, "preds_cpp.pt")
    eager_out_path = os.path.join(workdir, "preds_eager.pt")
    aott_library_paths: List[str] = list(args.aott_library)
    python_roundtrip_aott_library_paths: List[str] = list(args.aott_library)
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

    dense_inputs = (
        seq_emb_values,
        seq_emb_lengths,
        payload,
        uih_lens,
        num_cands,
    )
    if args.dense_backend == "aott":
        logger.info("Lowering dense module with AOT-T...")
        generated_aott_library_paths = _save_aott_dense_module(
            dense_module,
            dense_inputs,
            dense_path,
            workdir,
            args.atol,
            args.rtol,
        )
        aott_library_paths.extend(generated_aott_library_paths)
    else:
        logger.info("Tracing dense module (cuda:0, bf16)...")
        traced_dense = torch.jit.trace(
            dense_module,
            example_inputs=dense_inputs,
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
    if python_roundtrip_aott_library_paths:
        _load_aott_libraries_for_python(python_roundtrip_aott_library_paths)
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
    runner_args: List[str] = []
    for library_path in aott_library_paths:
        runner_args.extend(["--aott_library", library_path])
    runner_args.extend([sparse_path, dense_path, inputs_path, cpp_out_path])

    logger.info("Running C++: %s %s", runner_path, " ".join(runner_args))
    # pyre-fixme[6]: TrustedSubprocessWithList requires Literal[str] but this
    # runner is resolved from a built resource or explicit test argument.
    result = TrustedSubprocessWithList.run(
        executable=runner_path,
        cmd_args=runner_args,
        capture_output=True,
        text=True,
        check=False,
    )
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
    preds_cpp = torch.load(cpp_out_path, weights_only=False).to(torch.float32).cpu()
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
