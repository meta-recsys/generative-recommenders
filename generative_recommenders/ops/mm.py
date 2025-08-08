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

#!/usr/bin/env python3

# pyre-strict

import torch

try:
    from hammer.ops.triton.cc.addmm.triton_cc_addmm import triton_cc_addmm
except ImportError:
    pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.triton.triton_addmm import triton_addmm


def addmm(
    input: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        return triton_addmm(input, mat1, mat2, enable_tma)
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_addmm(input, mat1, mat2)
    else:
        return torch.addmm(input, mat1, mat2)
