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


# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}

DTYPE_MAP_FWD_SM8x = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

DTYPE_MAP_BWD = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80, 90]  # Sm kernels support up to
HEAD_DIMENSIONS = [64, 96, 128, 192, 256]

KERNEL_IMPL_TEMPLATE_FWD_SM90 = """
#ifdef OSS_ENV
#include "hstu_attention/flash_fwd_launch_template.h"
#else
#include "flash_fwd_launch_template.h"
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<{ARCH}, {DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
"""

KERNEL_IMPL_TEMPLATE_FWD_SM8x = """
#ifdef OSS_ENV
#include "hstu_attention/flash_fwd_launch_template.h"
#else
#include "flash_fwd_launch_template.h"
#endif

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_fwd_<80, {DTYPE}, {HEAD_DIM}>(Flash_fwd_params &params, cudaStream_t stream);
#endif
#endif
"""

KERNEL_IMPL_TEMPLATE_BWD_SM90 = """
#ifdef OSS_ENV
#include "hstu_attention/flash_bwd_launch_template.h"
#else
#include "flash_bwd_launch_template.h"
#endif

#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_bwd_<{ARCH}, {DTYPE}, {HEAD_DIM}>(Flash_bwd_params &params, cudaStream_t stream);
#endif
"""

KERNEL_IMPL_TEMPLATE_BWD_SM8x = """
#ifdef OSS_ENV
#include "hstu_attention/flash_bwd_launch_template.h"
#else
#include "flash_bwd_launch_template.h"
#endif

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM{HEAD_DIM}
template void run_mha_bwd_<80, {DTYPE}, {HEAD_DIM}>(Flash_bwd_params &params, cudaStream_t stream);
#endif
#endif
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            if self.sm == 90:
                return KERNEL_IMPL_TEMPLATE_FWD_SM90.format(
                    ARCH=str(self.sm),
                    DTYPE=DTYPE_MAP[self.dtype],
                    HEAD_DIM=self.head_dim,
                )
            else:
                # Always enable PackGQA for Sm8x to reduce compilation
                return KERNEL_IMPL_TEMPLATE_FWD_SM8x.format(
                    DTYPE=DTYPE_MAP[self.dtype],
                    HEAD_DIM=self.head_dim,
                )
        else:
            assert self.direction == "bwd"
            if self.sm == 90:
                return KERNEL_IMPL_TEMPLATE_BWD_SM90.format(
                    ARCH=str(self.sm),
                    DTYPE=DTYPE_MAP[self.dtype],
                    HEAD_DIM=self.head_dim,
                )
            else:
                return KERNEL_IMPL_TEMPLATE_BWD_SM8x.format(
                    DTYPE=DTYPE_MAP[self.dtype],
                    HEAD_DIM=self.head_dim,
                )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    kernels: List[Kernel] = []
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        # We always enable PackGQA for Sm8x or Split
        # so we should just pass in packgqa=False to avoid the `_packgqa` in the filename.
        if sm >= 90 or dtype in DTYPE_MAP_FWD_SM8x:
            kernels.append(
                Kernel(
                    sm=sm,
                    dtype=dtype,
                    head_dim=head_dim,
                    direction="fwd",
                )
            )
    for dtype, head_dim, sm in itertools.product(
        DTYPE_MAP_BWD.keys(), HEAD_DIMENSIONS, SM
    ):
        kernels.append(
            Kernel(
                sm=sm,
                dtype=dtype,
                head_dim=head_dim,
                direction="bwd",
            )
        )
    return kernels


def write_kernel(kernel: Union[Kernel], autogen_dir: Path) -> None:
    prelude = """
/* Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */ \n
// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir_name: Optional[str]) -> None:
    output_dir = (
        Path(output_dir_name) if output_dir_name is not None else Path(__file__).parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels_all = list(get_all_kernels())
    for kernel in kernels_all:
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        default="instantiations",
        required=False,
        help="Where to generate the kernels " " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
