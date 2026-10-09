# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Sequence
from pathlib import Path

from ..interfaces import ExecutionResult


def validate_command(command: Sequence[str]) -> None:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a nonempty argv sequence")
    if (
        any(not isinstance(arg, str) or "\0" in arg for arg in command)
        or not command[0]
    ):
        raise ValueError("command arguments must be strings without NUL bytes")


async def _terminate(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            # The parent may have exited while descendants still hold pipes.
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


async def run_process(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    cancelled: asyncio.Event | None = None,
    output_limit: int = 2_000_000,
) -> ExecutionResult:
    """Execute argv without interpolation; bound output and reap on cancellation.

    This portable command tool deliberately permits arbitrary executables. It is
    not a security sandbox; use ContainerSandbox for isolation. No shell parser
    is involved, and Meta-specific exec dependencies are intentionally absent.
    """
    validate_command(command)
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive")
    if output_limit < 1:
        raise ValueError("output_limit must be positive")
    if cancelled is not None and cancelled.is_set():
        raise asyncio.CancelledError
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=os.name == "posix",
    )

    async def read(stream: asyncio.StreamReader | None) -> bytes:
        if stream is None:
            return b""
        result = bytearray()
        while chunk := await stream.read(65536):
            result.extend(chunk)
            if len(result) > output_limit:
                raise ValueError("command output limit exceeded")
        return bytes(result)

    async def collect() -> ExecutionResult:
        readers = [
            asyncio.create_task(read(stream))
            for stream in (process.stdout, process.stderr)
        ]
        try:
            stdout, stderr = await asyncio.gather(*readers)
            returncode = await process.wait()
            return ExecutionResult(
                returncode,
                stdout.decode(errors="replace"),
                stderr.decode(errors="replace"),
            )
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    worker = asyncio.create_task(collect())
    watcher = asyncio.create_task(cancelled.wait()) if cancelled is not None else None
    try:
        pending = {worker, watcher} if watcher is not None else {worker}
        done, _ = await asyncio.wait(
            pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
        )
        if watcher is not None and watcher in done:
            raise asyncio.CancelledError
        if worker not in done:
            raise TimeoutError("command timed out")
        return await worker
    finally:
        if watcher is not None:
            watcher.cancel()
        worker.cancel()
        await _terminate(process)
        await asyncio.gather(
            worker, *([watcher] if watcher is not None else []), return_exceptions=True
        )
