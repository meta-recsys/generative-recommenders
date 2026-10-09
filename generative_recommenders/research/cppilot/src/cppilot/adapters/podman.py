# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..sandbox import ContainerSandbox


class PodmanSandbox(ContainerSandbox):
    """Backward-compatible Podman specialization of ContainerSandbox."""

    def __init__(
        self,
        image: str,
        *,
        mounts: Sequence[tuple[Path, str, bool]] = (),
        cpus: float | None = None,
        memory: str | None = None,
        network: bool = False,
    ) -> None:
        super().__init__(
            image,
            engine="podman",
            mounts=mounts,
            cpus=cpus,
            memory=memory,
            network=network,
        )
