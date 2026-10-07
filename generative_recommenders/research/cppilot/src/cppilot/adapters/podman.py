# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..interfaces import ExecutionResult, Sandbox


class PodmanSandbox(Sandbox):
    def __init__(self, image: str, *, client: Any | None = None) -> None:
        if client is None:
            import podman

            client = podman.PodmanClient(
                base_url="unix:///run/user/1000/podman/podman.sock"
            )
        self.client = client
        self.image = image
        self._containers: dict[str, Any] = {}

    async def start(self) -> str:
        container = self.client.containers.create(
            self.image, command=["sleep", "infinity"], network_mode="none"
        )
        container.start()
        self._containers[container.id] = container
        return container.id

    async def execute(
        self, sandbox_id: str, command: Sequence[str], cwd: Path | None = None
    ) -> ExecutionResult:
        result = self._containers[sandbox_id].exec_run(
            list(command), workdir=str(cwd) if cwd else None, demux=True
        )
        stdout, stderr = result.output
        return ExecutionResult(
            result.exit_code, (stdout or b"").decode(), (stderr or b"").decode()
        )

    async def stop(self, sandbox_id: str) -> None:
        container = self._containers.pop(sandbox_id)
        container.remove(force=True)
