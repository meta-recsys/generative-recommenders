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
torch.package archive for the HSTU model family.

Mirrors :class:`torchrec.inference.model_packager.PredictFactoryPackager` and
the OSS DLRM ``dlrm_packager.main`` entry point. ``main(argv)`` builds a
:class:`HSTUPredictFactory`, scripts the two sub-modules, and writes the
result to the path given by ``--output_path``.
"""

import argparse
import logging
import sys
from typing import List

import torch
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.inference.predict_factory import (
    HSTUModelConfig,
    HSTUPredictFactory,
)
from torch.package import PackageExporter
from torchrec.inference.model_packager import PredictFactoryPackager


logger: logging.Logger = logging.getLogger(__name__)


class HSTUPredictFactoryPackager(PredictFactoryPackager):
    @classmethod
    # pyrefly: ignore[bad-override]
    def set_extern_modules(cls, pe: PackageExporter) -> None:
        # Anything that ships its own .so / has C extensions must be extern;
        # otherwise the package will inline Python wrappers and fail to load
        # on a fresh worker.
        pe.extern(
            [
                "sys",
                "torch",
                "torch.*",
                "torchrec",
                "torchrec.*",
                "fbgemm_gpu",
                "fbgemm_gpu.*",
                # HSTU C++ ops (registered as TORCH_LIBRARY at .so load time).
                "generative_recommenders.ops.cpp.*",
                "generative_recommenders.fb.ultra.ops.*",
                # Triton wrappers — eager-only; the scripted graph never
                # dispatches into them but they are imported transitively.
                "triton",
                "triton.*",
            ]
        )

    @classmethod
    # pyrefly: ignore[bad-override]
    def set_mocked_modules(cls, pe: PackageExporter) -> None:
        pe.mock(["IPython", "IPython.*", "matplotlib", "matplotlib.*"])


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSTU model packager")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the torch.package archive to.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional checkpoint directory to load weights from.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kuairand-1k",
        help="Dataset key (kuairand-1k, movielens-1m, etc.) used to build "
        "the HSTU + table configs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for the dense module.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    hstu_config = get_hstu_configs(args.dataset)
    table_config = get_embedding_table_config(args.dataset)

    model_config = HSTUModelConfig(
        hstu_config=hstu_config,
        table_config=table_config,
        checkpoint_path=args.checkpoint_path,
    )

    script_module = HSTUPredictFactory(model_config).create_predict_module(
        world_size=1, device=args.device
    )

    # Save just the scripted composite (sparse.pt + dense.pt addressable as
    # ``pkg.attr("sparse")`` / ``pkg.attr("dense")``).
    torch.jit.save(torch.jit.script(script_module), args.output_path)
    logger.info("Package written to %s", args.output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
