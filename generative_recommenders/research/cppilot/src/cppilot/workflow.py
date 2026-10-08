# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Generic, TypeVar

from .agent import Agent
from .interfaces import MemoryStore, ModelProvider, Sandbox, Session, SessionStore
from .items import RunItem
from .registry import load_entry_points
from .runner import AgentRunner, RunResult
from .tools.core import RunContext

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Workflow(ABC, Generic[InputT, OutputT]):
    @abstractmethod
    async def run(self, value: InputT) -> OutputT: ...

    async def stream(self, value: InputT) -> AsyncIterator[Any]:
        yield await self.run(value)

    async def close(self) -> None:
        """Override to release resources owned by a custom workflow."""

    async def __aenter__(self) -> Workflow[InputT, OutputT]:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class WorkflowContext:
    """Per-session tool grants and sandbox lifecycle; injected stores are borrowed."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        conversation_id: str | None = None,
        sandbox: Sandbox | None = None,
        read_roots: tuple[Path, ...] = (),
        write_roots: tuple[Path, ...] = (),
        values: dict[str, Any] | None = None,
    ) -> None:
        self.memory = memory
        self.conversation_id = conversation_id or uuid.uuid4().hex
        self.context = RunContext(
            dict(values or {}),
            read_roots=read_roots,
            write_roots=write_roots,
            sandbox=sandbox,
        )
        self.context.values.update(memory=memory, conversation_id=self.conversation_id)
        self._sandbox_id: str | None = None
        self._entered = False

    async def __aenter__(self) -> RunContext:
        if self._entered:
            raise RuntimeError("workflow context is already entered")
        self.context.cancelled.clear()
        if self.context.sandbox is not None:
            self._sandbox_id = await self.context.sandbox.start()
            self.context.values["sandbox_id"] = self._sandbox_id
            self.context.session["sandbox_id"] = self._sandbox_id
        self._entered = True
        return self.context

    async def __aexit__(self, *exc: Any) -> None:
        self.context.cancelled.set()
        tasks = list(self.context.session.pop("background_tasks", {}).values())
        for task in tasks:
            task.cancel()
        try:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                if self._sandbox_id is not None:
                    sandbox = self.context.sandbox
                    assert sandbox is not None
                    await sandbox.stop(self._sandbox_id)
            finally:
                self._sandbox_id = None
                self._entered = False
                self.context.values.pop("sandbox_id", None)
                self.context.session.pop("sandbox_id", None)


def _agents(root: Agent) -> dict[str, Agent]:
    result: dict[str, Agent] = {}
    pending = [root]
    seen: set[int] = set()
    while pending:
        agent = pending.pop()
        if id(agent) in seen:
            continue
        seen.add(id(agent))
        if agent.name in result:
            raise ValueError(f"duplicate agent name {agent.name!r}")
        result[agent.name] = agent
        pending.extend(agent.handoffs)
    return result


class AgentWorkflow(Workflow[str, RunResult]):
    def __init__(
        self,
        agent: Agent,
        *,
        memory: MemoryStore | None = None,
        sessions: SessionStore | None = None,
        session: Session | None = None,
        conversation_id: str | None = None,
        runner: AgentRunner | None = None,
        context: WorkflowContext | None = None,
    ) -> None:
        self.agent = agent
        self.memory = memory
        if self.memory is None:
            self.memory = (
                runner.memory if runner else (context.memory if context else None)
            )
        if runner is not None and self.memory is not runner.memory:
            raise ValueError("workflow and runner must use the same memory store")
        self.sessions, self.session = sessions, session
        self.conversation_id = (
            session.conversation_id
            if session
            else (
                conversation_id
                or (context.conversation_id if context else uuid.uuid4().hex)
            )
        )
        self.runner = runner or AgentRunner(memory=self.memory)
        self.context = context
        if context is not None:
            context.memory = self.memory
            context.conversation_id = self.conversation_id
            context.context.values.update(
                memory=self.memory, conversation_id=self.conversation_id
            )
        self._lock = asyncio.Lock()

    def _active(self) -> Agent:
        name = self.session.active_agent if self.session else None
        agents = _agents(self.agent)
        if name and name not in agents:
            raise ValueError(f"saved active agent {name!r} is unavailable")
        return agents[name] if name else self.agent

    @asynccontextmanager
    async def _execution(self) -> AsyncIterator[Agent]:
        async with self._lock:
            agents = _agents(self.agent)
            active = self._active()
            if self.memory is not None:
                saved = await self.memory.get_active_agent(self.conversation_id)
                if saved is None and self.session is not None:
                    await self.memory.set_active_agent(
                        self.conversation_id, active.name
                    )
                active = self.agent
            if self.context:
                async with self.context as context:
                    clones = {
                        name: replace(agent, context=context)
                        for name, agent in agents.items()
                    }
                    for name, agent in agents.items():
                        clones[name].handoffs = tuple(
                            clones[target.name] for target in agent.handoffs
                        )
                    yield clones[active.name]
            else:
                yield active

    async def _save(self, active: str | None) -> None:
        if self.session and self.sessions:
            self.session = replace(
                self.session, active_agent=active or self.session.active_agent
            )
            await self.sessions.update(self.session)

    async def run(self, value: str) -> RunResult:
        async with self._execution() as agent:
            result = await self.runner.run(
                agent, value, conversation_id=self.conversation_id
            )
            await self._save(result.agent_name)
            return result

    async def stream(self, value: str) -> AsyncIterator[RunItem]:
        async with self._execution() as agent:
            stream = self.runner.stream(
                agent, value, conversation_id=self.conversation_id
            )
            try:
                async for item in stream:
                    yield item
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
                if self.memory:
                    await self._save(
                        await self.memory.get_active_agent(self.conversation_id)
                    )

    async def set_model(
        self, provider: ModelProvider, selection: dict[str, Any]
    ) -> None:
        """Replace providers on a cloned graph, preserving session and handoff state."""
        async with self._lock:
            agents = _agents(self.agent)
            clones = {
                name: replace(agent, provider=provider)
                for name, agent in agents.items()
            }
            for name, agent in agents.items():
                clones[name].handoffs = tuple(
                    clones[target.name] for target in agent.handoffs
                )
            if self.session is not None:
                session = replace(
                    self.session, metadata={**self.session.metadata, **selection}
                )
                if self.sessions is not None:
                    await self.sessions.update(session)
                self.session = session
            self.agent = clones[self.agent.name]

    async def clear(self) -> None:
        async with self._lock:
            if self.memory:
                await self.memory.clear(self.conversation_id)
            if self.session and self.sessions:
                self.session = replace(self.session, active_agent=self.agent.name)
                await self.sessions.update(self.session)


class WorkflowRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Workflow[Any, Any]]] = {}

    def register(self, name: str, factory: Callable[..., Workflow[Any, Any]]) -> None:
        if not name:
            raise ValueError("workflow name must not be empty")
        if name in self._factories:
            raise ValueError(f"workflow {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, **configuration: Any) -> Workflow[Any, Any]:
        try:
            factory = self._factories[name]
        except KeyError as error:
            raise KeyError(f"unknown workflow {name!r}") from error
        return factory(**configuration)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def load_entry_points(self, group: str = "cppilot.workflows") -> list[str]:
        return load_entry_points(self.register, group)
