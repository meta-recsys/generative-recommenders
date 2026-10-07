# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Workflow(ABC, Generic[InputT, OutputT]):
    @abstractmethod
    async def run(self, value: InputT) -> OutputT: ...
