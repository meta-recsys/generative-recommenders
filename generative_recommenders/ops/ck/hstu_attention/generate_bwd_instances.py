# @noautodeps
# noqa: C801
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
#
# Copyright (c) 2023-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from pathlib import Path
from typing import List

HSTU_COPYRIGHT_HEADER = """\
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

// The file is automatically generated, don't modify!
// See the generator script
// `{file}`
""".format(
    file=os.path.relpath(os.path.realpath(__file__), start=Path(__file__).parents[4])
)

HSTU_BACKWARD_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"hstu_attention_{mode}_backward_dispatch.hpp\"
#include \"hstu_attention_params.hpp\"
"""

HSTU_BACKWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_backward_dispatch<
    {dtype},
    {has_causal},
    {use_softmax},
    {has_bias},
    {has_dropout},
    {max_k}>(HstuAttention{group_or_not}BwdParams& param, hipStream_t stream);
"""

HSTU_BACKWARD_INSTANCE_FNAME = (
    "hstu_attention_{mode}_backward_{dtype_str}_{has_or_no_causal_str}_{use_softmax_or_not_str}_"
    "{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

HSTU_BACKWARD_INSTANCE_REF_FNAME = (
    "hstu_attention_{mode}_backward_{dtype}_instances_ref.hpp"
)

BOOL_MAP = {True: "true", False: "false"}

BOOL_MAP_CAUSAL = {
    True: "has_causal",
    False: "no_causal",
}

BOOL_MAP_SOFTMAX = {
    True: "softmax_true",
    False: "softmax_false",
}

BOOL_MAP_LSE = {
    True: "lse_true",
    False: "lse_false",
}

BOOL_MAP_BIAS = {
    True: "has_bias",
    False: "no_bias",
}

BOOL_MAP_DROPOUT = {
    True: "has_dropout",
    False: "no_dropout",
}

INT_MAP_MAX_K = {hd: f"maxk_{hd}" for hd in [64, 96, 128, 256]}

TYPE_CTYPE_MAP = {
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
}

TYPE_FNAME_MAP = {
    "fp16": "half",
    "bf16": "bfloat16",
}

MODE_GROUP_OR_NOT_MAP = {
    "batched": "NoGroup",
    "jagged": "NoGroup",
    "group": "Group",
}


def create_backward_instances(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "jagged", "group"]:
        for dtype in ["fp16", "bf16"]:
            for has_causal in [True, False]:
                for use_softmax in [True, False]:
                    for has_bias in [False]:
                        for has_dropout in [True, False]:
                            for max_k in headdims:
                                fname = HSTU_BACKWARD_INSTANCE_FNAME.format(
                                    mode=mode,
                                    dtype_str=dtype,
                                    has_or_no_causal_str=BOOL_MAP_CAUSAL[has_causal],
                                    use_softmax_or_not_str=BOOL_MAP_SOFTMAX[
                                        use_softmax
                                    ],
                                    has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                    has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                    max_k_str=INT_MAP_MAX_K[max_k],
                                )
                                backward_instance_inc = (
                                    HSTU_BACKWARD_INSTANCE_TEMPLATE_INC.format(
                                        mode=mode,
                                        dtype_file=TYPE_FNAME_MAP[dtype],
                                    )
                                )
                                backward_instance = (
                                    HSTU_BACKWARD_INSTANCE_TEMPLATE.format(
                                        extern="",
                                        mode=mode,
                                        dtype=TYPE_CTYPE_MAP[dtype],
                                        has_causal=BOOL_MAP[has_causal],
                                        use_softmax=BOOL_MAP[use_softmax],
                                        has_bias=BOOL_MAP[has_bias],
                                        has_dropout=BOOL_MAP[has_dropout],
                                        max_k=max_k,
                                        group_or_not=MODE_GROUP_OR_NOT_MAP[mode],
                                    )
                                )
                                (instance_dir / fname).write_text(
                                    HSTU_COPYRIGHT_HEADER
                                    + backward_instance_inc
                                    + backward_instance
                                )


def create_backward_instances_ref(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "jagged", "group"]:
        for dtype in ["fp16", "bf16"]:
            ref_fname = HSTU_BACKWARD_INSTANCE_REF_FNAME.format(
                mode=mode,
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            backward_instance_inc = HSTU_BACKWARD_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(HSTU_COPYRIGHT_HEADER)
                file.write(backward_instance_inc)
                for max_k in headdims:
                    for has_bias in [False]:
                        for has_dropout in [True, False]:
                            for has_causal in [True, False]:
                                for use_softmax in [True, False]:
                                    backward_instance = (
                                        HSTU_BACKWARD_INSTANCE_TEMPLATE.format(
                                            extern="extern ",
                                            mode=mode,
                                            dtype=TYPE_CTYPE_MAP[dtype],
                                            has_causal=BOOL_MAP[has_causal],
                                            use_softmax=BOOL_MAP[use_softmax],
                                            has_bias=BOOL_MAP[has_bias],
                                            has_dropout=BOOL_MAP[has_dropout],
                                            max_k=max_k,
                                            group_or_not=MODE_GROUP_OR_NOT_MAP[mode],
                                        )
                                    )
                                    file.write(backward_instance)


if __name__ == "__main__":
    headdims_bwd = [64, 96, 128, 256]

    this_dir = os.path.dirname(__file__)
    output_dir = Path(this_dir) / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove existing backward files in the directory
    for ff in output_dir.glob("*_backward_*"):
        ff.unlink()

    create_backward_instances(output_dir, headdims_bwd)
    create_backward_instances_ref(output_dir, headdims_bwd)
