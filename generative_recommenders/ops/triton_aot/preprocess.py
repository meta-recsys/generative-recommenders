# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Preprocessing utilities for triton_aot models before AOT compilation.
"""

import logging

from tgif.fx.tgif_tracer import TGIFTracer
from torch.fx import GraphModule

logger: logging.Logger = logging.getLogger(__name__)

# "aot_triton_kernel_wrapper_" is a pre-defined prefix for
# AOT-T triton kernel wrapper functions. This is required for
# AOT-T backend to recognize and trace correctly for ops transformation.
AOTT_WRAPPER_PREFIX: str = "aot_triton_kernel_wrapper_"


def unwrap_aott_wrapper_nodes(fx_m: GraphModule, tracer: TGIFTracer) -> GraphModule:
    """Mark ``aot_triton_kernel_wrapper_*`` FX nodes as unwrapped and re-trace.

    In the traced FX graph, outer wrapper functions (prefixed with
    ``aot_triton_kernel_wrapper_``) are ``@torch.fx.wrap`` leaves.
    Setting ``node.meta["is_wrapped"] = False`` causes a subsequent
    ``symbolic_trace`` to trace *through* them, exposing the inner
    ``@torch.fx.wrap`` functions (e.g., ``_triton_aot_grouped_gemm``)
    that contain the actual kernel calls.

    Any ``_body_transformer`` hook (e.g. one registered by
    ``early_return_fx_code_transform``) is temporarily removed before
    re-tracing to avoid injecting un-traceable control flow
    (``if Proxy: …``) into the generated ``forward``.  After re-trace
    the hook is restored on the new module.  See P2266562545.

    Args:
        fx_m: The FX GraphModule to modify **in-place** before re-trace.
        tracer: Tracer instance used for the re-trace step.

    Returns:
        The re-traced ``GraphModule`` with AOTT wrappers expanded.
    """
    logger.info("Re-trace to get the AOTT node exposed.")

    # Save and clear the body transformer so that re-trace does not hit
    # ``if Proxy:`` from code-level hooks like early_return_fx_code_transform.
    saved_body_transformer = fx_m.graph._codegen._body_transformer
    fx_m.graph._codegen._body_transformer = None

    unwrap_count = 0
    for node in fx_m.graph.nodes:
        if node.op == "call_function":
            target = node.target
            if hasattr(target, "__name__") and target.__name__.startswith(
                AOTT_WRAPPER_PREFIX
            ):
                logger.info(f"[AOTT] Found inference wrapper node: {node=}")
                node.meta["is_wrapped"] = False
                unwrap_count += 1

    if unwrap_count > 0:
        logger.info(f"[AOTT] Found {unwrap_count} inference wrapper nodes.")
        fx_m.recompile()
    else:
        logger.warning("[AOTT] No inference wrapper node found. Skip re-compile.")

    result = tracer.symbolic_trace(fx_m)

    # Restore the body transformer on the new module.
    if saved_body_transformer is not None:
        result.graph._codegen._body_transformer = saved_body_transformer
        result.recompile()

    return result
