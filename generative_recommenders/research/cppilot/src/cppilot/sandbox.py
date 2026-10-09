# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import math
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Sequence

from .interfaces import ExecutionResult, Sandbox
from .tools.process import run_process, validate_command


class ContainerSandbox(Sandbox):
    """Async Docker/Podman CLI sandbox. Network is off unless explicitly enabled.

    Mount only trusted, narrow directories. Bind mounts expose host files and
    containers are not a boundary against a compromised engine or host kernel.
    """

    def __init__(
        self,
        image: str,
        *,
        engine: str = "podman",
        mounts: Sequence[tuple[Path, str, bool]] = (),
        cpus: float | None = None,
        memory: str | None = None,
        network: bool = False,
        pids_limit: int = 128,
        timeout: float = 30,
        read_only: bool = True,
    ) -> None:
        if engine not in {"docker", "podman"}:
            raise ValueError("engine must be 'docker' or 'podman'")
        if (
            not image
            or image.startswith("-")
            or any(char.isspace() or char == "\0" for char in image)
        ):
            raise ValueError("image must be a non-option container image reference")
        if cpus is not None and (not math.isfinite(cpus) or cpus <= 0):
            raise ValueError("cpus must be finite and positive")
        if memory is not None and not re.fullmatch(
            r"[1-9][0-9]*(?:[bkmgBKMG])?", memory
        ):
            raise ValueError("memory must be a positive size such as '512m'")
        if pids_limit < 1 or timeout <= 0:
            raise ValueError("pids_limit and timeout must be positive")
        checked = []
        destinations = set()
        for source, destination, writable in mounts:
            source = Path(source).resolve(strict=True)
            target = PurePosixPath(destination)
            if not target.is_absolute() or ".." in target.parts or str(target) == "/":
                raise ValueError(
                    "mount destination must be an absolute non-root container path"
                )
            if any(char in str(source) + destination for char in (",", "\0", "\n")):
                raise ValueError(
                    "mount paths must not contain commas, NULs, or newlines"
                )
            if str(source) == source.anchor or source == Path.home().resolve():
                raise ValueError(
                    "mounting the host root or home directory is not allowed"
                )
            if str(target) in destinations:
                raise ValueError("duplicate mount destination")
            destinations.add(str(target))
            checked.append((source, str(target), writable))
        self.image, self.engine, self.mounts = image, engine, tuple(checked)
        self.cpus, self.memory, self.network = cpus, memory, network
        self.pids_limit, self.timeout, self.read_only = pids_limit, timeout, read_only
        self._containers: set[str] = set()

    async def _run(self, *arguments: str) -> ExecutionResult:
        return await run_process([self.engine, *arguments], timeout=self.timeout)

    async def start(self) -> str:
        name = f"cppilot-{uuid.uuid4().hex}"
        arguments = [
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "bridge" if self.network else "none",
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
        ]
        if self.read_only:
            arguments.append("--read-only")
        if self.cpus is not None:
            arguments += ["--cpus", str(self.cpus)]
        if self.memory is not None:
            arguments += ["--memory", self.memory, "--memory-swap", self.memory]
        for source, destination, writable in self.mounts:
            mount = f"type=bind,src={source},dst={destination}"
            if not writable:
                mount += ",readonly"
            arguments += ["--mount", mount]
        self._containers.add(name)
        try:
            result = await self._run(
                *arguments, "--entrypoint", "sleep", self.image, "infinity"
            )
            if result.returncode:
                raise RuntimeError(result.stderr or "container failed to start")
        except (
            asyncio.CancelledError,
            RuntimeError,
            OSError,
            ValueError,
        ):
            await self.stop(name)
            raise
        return name

    def _container_cwd(self, cwd: Path) -> str:
        resolved = cwd.resolve()
        # Most-specific host mount wins, matching nested bind mount behavior.
        for source, destination, _ in sorted(
            self.mounts, key=lambda item: len(item[0].parts), reverse=True
        ):
            if resolved.is_relative_to(source):
                return str(
                    PurePosixPath(destination) / resolved.relative_to(source).as_posix()
                )
        raise PermissionError("working directory is not exposed by a configured mount")

    async def execute(
        self, sandbox_id: str, command: Sequence[str], cwd: Path | None = None
    ) -> ExecutionResult:
        if sandbox_id not in self._containers:
            raise KeyError(sandbox_id)
        validate_command(command)
        arguments = ["exec"]
        if cwd is not None:
            arguments += ["--workdir", self._container_cwd(cwd)]
        try:
            return await self._run(*arguments, sandbox_id, *command)
        except (asyncio.CancelledError, TimeoutError, ValueError):
            # Killing the CLI alone does not kill the in-container exec process.
            await self.stop(sandbox_id)
            raise

    async def stop(self, sandbox_id: str) -> None:
        if sandbox_id not in self._containers:
            return
        result = await self._run("rm", "-f", sandbox_id)
        if result.returncode:
            raise RuntimeError(result.stderr or "container removal failed")
        self._containers.discard(sandbox_id)

    async def close(self) -> None:
        for sandbox_id in tuple(self._containers):
            await self.stop(sandbox_id)
