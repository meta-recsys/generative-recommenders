# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .items import RunItem, Usage


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: Sequence[RunItem]
    tools: Sequence[ToolDefinition] = ()
    output_schema: dict[str, Any] | None = None
    system: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    items: Sequence[RunItem]
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    streaming: bool = False
    tools: bool = True
    structured_output: bool = False


class ModelProvider(ABC):
    capabilities = ModelCapabilities()

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse: ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[RunItem]:
        response = await self.generate(request)
        for item in response.items:
            yield item
        yield replace(
            response.usage, metadata={**response.usage.metadata, **response.metadata}
        )


class RetryableModelError(RuntimeError):
    """A transient provider error that the runner may retry."""


Compactor = Callable[[list[RunItem]], list[RunItem] | Awaitable[list[RunItem]]]


class MemoryStore(ABC):
    async def get_active_agent(self, conversation_id: str) -> str | None:
        """Optional durable state for stores predating agent handoffs."""
        return None

    async def set_active_agent(
        self, conversation_id: str, agent_name: str | None
    ) -> None:
        return None

    async def append_turn(
        self, conversation_id: str, items: Sequence[RunItem], agent_name: str
    ) -> None:
        """Stores may override this to atomically commit history and active agent."""
        await self.append(conversation_id, items)
        await self.set_active_agent(conversation_id, agent_name)

    @abstractmethod
    async def load(self, conversation_id: str) -> list[RunItem]: ...

    @abstractmethod
    async def append(self, conversation_id: str, items: Sequence[RunItem]) -> None: ...

    @abstractmethod
    async def clear(self, conversation_id: str) -> None: ...

    @abstractmethod
    async def compact(
        self, conversation_id: str, compactor: Compactor
    ) -> list[RunItem]: ...


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    agent_name: str
    conversation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    active_agent: str | None = None


class SessionStore(ABC):
    @abstractmethod
    async def create(
        self,
        agent_name: str,
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session: ...

    @abstractmethod
    async def load(self, session_id: str) -> Session | None: ...

    @abstractmethod
    async def update(self, session: Session) -> None: ...

    @abstractmethod
    async def list(self) -> list[Session]: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str


class Sandbox(ABC):
    @abstractmethod
    async def start(self) -> str: ...

    @abstractmethod
    async def execute(
        self, sandbox_id: str, command: Sequence[str], cwd: Path | None = None
    ) -> ExecutionResult: ...

    @abstractmethod
    async def stop(self, sandbox_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class Event:
    name: str
    data: dict[str, Any] = field(default_factory=dict)


class EventSink(ABC):
    @abstractmethod
    async def emit(self, event: Event) -> None: ...


class NoopEventSink(EventSink):
    async def emit(self, event: Event) -> None:
        pass
