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

import functools
import random

import torch


@torch.compiler.assume_constant_result
@functools.lru_cache(maxsize=1)
def is_sm100_plus() -> bool:
    """
    Check if this is a Blackwell Datacenter GPU.
    These are between 100 and 103 for B200-GB300.
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 10 and (props.minor >= 0 and props.minor <= 3)


@torch.compiler.assume_constant_result
@functools.lru_cache(maxsize=1)
def is_sm90() -> bool:
    """
    Check if this is an H100 GPU (SM 9.0).
    """
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 0


@torch.compiler.assume_constant_result
@functools.lru_cache(maxsize=1)
def is_sm90_plus() -> bool:
    """
    Check if this is an H100 GPU (SM 9.0) or newer (Blackwell).
    """
    return is_sm100_plus() or is_sm90()


@torch.compiler.assume_constant_result
@functools.lru_cache(maxsize=1)
def get_sm_count() -> int:
    """
    Get the number of streaming multiprocessors (SMs) on the current GPU.
    """
    return torch.cuda.get_device_properties(0).multi_processor_count


def generate_dropout_seed() -> int:
    """Generate a random seed for dropout operations."""
    return random.randrange(0, 2**62)
