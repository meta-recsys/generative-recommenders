# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# pyre-strict

"""
This module bundles all necessary C++ dependencies (ATen, torch, etc.) for the
local triton_aot copy so Python code can simply import this module and do:
torch.ops.load_library.

Usage:
    import generative_recommenders.ops.triton_aot.build.runtime_deps  # noqa: F401
"""
