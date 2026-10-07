# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from pathlib import Path

from .core import FunctionTool, RunContext


def _resolve(context: RunContext, value: str, *, write: bool) -> Path:
    root_key = "write_root" if write else "read_root"
    root = Path(context.values[root_key]).resolve()
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root):
        raise PermissionError(f"path escapes {root_key}")
    return candidate


def read_file(context: RunContext, path: str) -> str:
    """Read a UTF-8 file below the configured read root."""
    return _resolve(context, path, write=False).read_text(encoding="utf-8")


def write_file(context: RunContext, path: str, content: str) -> str:
    """Write a UTF-8 file below the configured write root."""
    target = _resolve(context, path, write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return str(target)


def file_tools() -> list[FunctionTool]:
    return [FunctionTool(read_file), FunctionTool(write_file)]
