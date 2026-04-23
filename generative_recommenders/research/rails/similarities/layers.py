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

# pyre-unsafe

"""
Defines network architectures used in constructing various learned similarities.

Forked from bailuding/rails @ 664fdb9.
"""

import torch
import torch.nn.functional as F


class GeGLU(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._w = torch.nn.Parameter(
            torch.empty((in_features, out_features * 2)).normal_(mean=0, std=0.02),
        )
        self._b = torch.nn.Parameter(
            torch.zeros((1, out_features * 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[:-1]
        lhs, rhs = torch.split(
            torch.mm(x.reshape(-1, self._in_features), self._w) + self._b,
            [self._out_features, self._out_features],
            dim=-1,
        )
        return (F.gelu(lhs) * rhs).reshape(bs + (self._out_features,))


class SwiGLU(torch.nn.Module):
    """
    SwiGLU from https://arxiv.org/abs/2002.05202.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._w = torch.nn.Parameter(
            torch.empty((in_features, out_features * 2)).normal_(mean=0, std=0.02),
        )
        self._b = torch.nn.Parameter(
            torch.zeros((1, out_features * 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size()[:-1]
        lhs, rhs = torch.split(
            torch.mm(x.reshape(-1, self._in_features), self._w) + self._b,
            [self._out_features, self._out_features],
            dim=-1,
        )
        return (F.silu(lhs) * rhs).reshape(bs + (self._out_features,))
