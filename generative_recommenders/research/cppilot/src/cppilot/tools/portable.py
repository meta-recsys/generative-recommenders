# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import fnmatch
import re
import urllib.request
from typing import Any, cast

from .core import FunctionTool, RunContext
from .process import run_process, validate_command
from .state import state_tools


def edit_file(
    context: RunContext, path: str, old: str, new: str, count: int = 1
) -> str:
    """Replace an exact string in a file with both read and write grants."""
    if not old or count < 1:
        raise ValueError("old text must be nonempty and count positive")
    target = context.authorize_path(path, write=True)
    with context.open_path(target) as stream:
        content = stream.read()
    if content.count(old) < count:
        raise ValueError("old text has fewer occurrences than requested")
    with context.open_path(target, write=True) as stream:
        stream.write(content.replace(old, new, count))
    return str(target)


def glob_files(context: RunContext, pattern: str) -> list[str]:
    """List granted files; skip denied symlinks without exposing their targets."""
    root = context.authorize_path(".")
    result = []
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if not fnmatch.fnmatch(relative, pattern):
            continue
        try:
            target = context.authorize_path(path)
        except PermissionError:
            continue
        if target.is_file():
            result.append(relative)
    return sorted(result)


def grep_files(
    context: RunContext, pattern: str, glob: str = "*"
) -> list[dict[str, Any]]:
    """Search granted text files with a regular expression and line numbers."""
    expression = re.compile(pattern)
    root = context.authorize_path(".")
    matches: list[dict[str, Any]] = []
    for relative in glob_files(context, glob):
        try:
            with context.open_path(root / relative) as stream:
                lines = stream.read().splitlines()
        except (UnicodeDecodeError, PermissionError, FileNotFoundError):
            continue
        matches.extend(
            {"path": relative, "line": number, "text": line}
            for number, line in enumerate(lines, 1)
            if expression.search(line)
        )
    return matches


async def shell(
    context: RunContext,
    command: list[str],
    cwd: str = ".",
    timeout: float = 30,
) -> dict[str, Any]:
    """Run argv in a sandbox, or locally without isolation when none is configured.

    A local cwd grant is not a restriction on the command's filesystem access.
    """
    validate_command(command)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if context.cancelled.is_set():
        raise asyncio.CancelledError
    directory = context.authorize_path(cwd)
    if context.sandbox is not None:
        lock = context.session.setdefault("sandbox_lock", asyncio.Lock())
        async with lock:
            sandbox_id = context.session.get("sandbox_id")
            if sandbox_id is None:
                sandbox_id = await context.sandbox.start()
                context.session["sandbox_id"] = sandbox_id
        worker = asyncio.create_task(
            context.sandbox.execute(sandbox_id, command, directory)
        )
        watcher = asyncio.create_task(context.cancelled.wait())
        try:
            done, _ = await asyncio.wait(
                {worker, watcher}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if watcher in done:
                raise asyncio.CancelledError
            if worker not in done:
                raise TimeoutError("sandbox command timed out")
            result = await worker
        except (asyncio.CancelledError, TimeoutError):
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await context.sandbox.stop(sandbox_id)
            context.session.pop("sandbox_id", None)
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
    else:
        result = await run_process(
            command, cwd=directory, timeout=timeout, cancelled=context.cancelled
        )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


async def code_execution(
    context: RunContext, code: str, language: str = "python"
) -> dict[str, Any]:
    """Execute a small program through the shell tool, preferably in a sandbox."""
    commands = {"python": ["python3", "-c", code], "bash": ["bash", "-c", code]}
    if language not in commands:
        raise ValueError(f"unsupported language: {language}")
    return await shell(context, commands[language])


async def web_fetch(url: str, timeout: float = 20) -> str:
    """Fetch HTTP(S) with a size bound. This is not an SSRF/network sandbox."""
    if not url.startswith(("http://", "https://")):
        raise ValueError("only HTTP(S) URLs are supported")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    def fetch() -> str:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content = response.read(2_000_001)
            if len(content) > 2_000_000:
                raise ValueError("response size limit exceeded")
            return cast(bytes, content).decode("utf-8", errors="replace")

    return await asyncio.to_thread(fetch)


def portable_tools() -> list[FunctionTool]:
    return [
        FunctionTool(fn)
        for fn in (edit_file, glob_files, grep_files, shell, code_execution, web_fetch)
    ] + state_tools()
