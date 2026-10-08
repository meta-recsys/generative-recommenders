# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from pathlib import Path

from .core import FunctionTool, RunContext


def _resolve(context: RunContext, value: str, *, write: bool) -> Path:
    return context.authorize_path(value, write=write)


def read_file(context: RunContext, path: str) -> str:
    """Read a UTF-8 file below the configured read root."""
    with context.open_path(path) as stream:
        return stream.read()


def write_file(context: RunContext, path: str, content: str) -> str:
    """Write a UTF-8 file below the configured write root."""
    target = _resolve(context, path, write=True)
    with context.open_path(path, write=True) as stream:
        stream.write(content)
    return str(target)


def file_tools() -> list[FunctionTool]:
    return [FunctionTool(read_file), FunctionTool(write_file)]
