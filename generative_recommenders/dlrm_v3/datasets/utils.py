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
import json
from typing import List, Tuple


def separate_uih_candidates(
    x: str | int | List[int],
    candidates_max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    if isinstance(x, str):
        if x[0] != "[" and x[-1] != "]":
            x = "[" + x + "]"
        y = json.loads(x)
    else:
        y = x
    y_list = [y] if type(y) == int else list(y)
    candidates, uih = (
        y_list[-candidates_max_seq_len:],
        y_list[:-candidates_max_seq_len],
    )
    return uih, candidates


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y
