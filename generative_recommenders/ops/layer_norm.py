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


from typing import List

import torch
from generative_recommenders.ops.pytorch.pt_layer_norm import (
    pytorch_layer_norm,
    pytorch_rms_norm,
    pytorch_swish_layer_norm,
)
from generative_recommenders.ops.triton.triton_layer_norm import triton_rms_norm

try:
    from hammer.ops.triton.cc.swish_layer_norm.triton_cc_swish_layer_norm import (
        triton_cc_swish_layer_norm,
    )
except ImportError:
    pass
try:
    from hammer.ops.triton.cc.rms_norm.triton_cc_rms_norm import triton_cc_rms_norm
except ImportError:
    pass
from generative_recommenders.common import HammerKernel, HammerModule
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_layer_norm,
    triton_swish_layer_norm,
)
from torch.fx._symbolic_trace import is_fx_tracing

torch.fx.wrap("triton_layer_norm")
torch.fx.wrap("triton_swish_layer_norm")


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        if not is_fx_tracing():
            torch._assert(not x.is_cpu, "x must be device tensor")
            torch._assert(not weight.is_cpu, "weight must be device tensor")
            torch._assert(not bias.is_cpu, "bias must be device tensor")
        return triton_layer_norm(x, weight, bias, eps)
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_swish_layer_norm(
            x,
            weight,
            bias,
            eps,
            is_swish=False,
        )
    else:
        return pytorch_layer_norm(
            x,
            [
                x.shape[-1],
            ],
            weight,
            bias,
            eps,
        )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    kernel: HammerKernel = HammerKernel.PYTORCH,
    silu: bool = False,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        if not is_fx_tracing():
            torch._assert(not x.is_cpu, "x must be device tensor")
            torch._assert(not weight.is_cpu, "weight must be device tensor")
        return triton_rms_norm(x, weight, eps, silu)
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_rms_norm(
            x,
            weight,
            eps,
            silu=silu,
        )
    else:
        return pytorch_rms_norm(
            x,
            [
                x.shape[-1],
            ],
            weight,
            eps,
            silu,
        )


def swish_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        if not is_fx_tracing():
            torch._assert(not x.is_cpu, "x must be device tensor")
            torch._assert(not weight.is_cpu, "weight must be device tensor")
            torch._assert(not bias.is_cpu, "bias must be device tensor")
        return triton_swish_layer_norm(x, [x.shape[-1]], weight, bias, eps)
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_swish_layer_norm(
            x,
            weight,
            bias,
            eps,
            is_swish=True,
        )
    else:
        return pytorch_swish_layer_norm(
            x,
            [
                x.shape[-1],
            ],
            weight,
            bias,
            eps,
        )


class LayerNorm(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self._eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(self._normalized_shape),
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(self._normalized_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(
            x=x,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
            kernel=self.hammer_kernel(),
        )


class RMSNorm(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            x,
            self.weight,
            self._eps,
            silu=False,
            kernel=self.hammer_kernel(),
        )


class RMSNormSilu(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            x,
            self.weight,
            self._eps,
            silu=True,
            kernel=self.hammer_kernel(),
        )


class SwishLayerNorm(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self.weight = torch.nn.Parameter(torch.ones(self._normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape))
        self._eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return swish_layer_norm(
            x=x,
            weight=self.weight,
            bias=self.bias,
            eps=self._eps,
            kernel=self.hammer_kernel(),
        )
