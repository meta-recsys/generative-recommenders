# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, cast

from .core import FunctionTool, RunContext


@dataclass(slots=True)
class TodoList:
    items: dict[str, bool] = field(default_factory=dict)

    def add(self, label: str) -> str:
        if not label.strip():
            raise ValueError("todo label must not be empty")
        self.items.setdefault(label, False)
        return label

    def complete(self, label: str) -> str:
        if label not in self.items:
            raise KeyError(label)
        self.items[label] = True
        return label

    def tools(self) -> list[FunctionTool]:
        def todo_add(label: str) -> str:
            """Add an unfinished todo item without resetting existing items."""
            return self.add(label)

        def todo_complete(label: str) -> str:
            """Mark an existing todo item complete."""
            return self.complete(label)

        def todo_list() -> dict[str, bool]:
            """List todo items and completion state."""
            return dict(self.items)

        return [
            FunctionTool(todo_add),
            FunctionTool(todo_complete),
            FunctionTool(todo_list),
        ]


@dataclass(slots=True)
class TaskManager:
    tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    def start(self, name: str, awaitable: Coroutine[Any, Any, Any]) -> None:
        if name in self.tasks and not self.tasks[name].done():
            awaitable.close()
            raise ValueError(f"task {name!r} already exists")
        task = asyncio.create_task(awaitable, name=name)
        self.tasks[name] = task
        task.add_done_callback(self._observe)

    @staticmethod
    def _observe(task: asyncio.Task[Any]) -> None:
        # Retrieving the exception prevents unobserved-task warnings. result()
        # still raises the original error for the consumer.
        if not task.cancelled():
            task.exception()

    def status(self, name: str) -> dict[str, Any]:
        task = self.tasks[name]
        state = "running"
        result: dict[str, Any] = {"name": name}
        if task.cancelled():
            state = "cancelled"
        elif task.done():
            error = task.exception()
            output = None if error else task.result()
            failed = error is not None or bool(getattr(output, "failed", False))
            state = "failed" if failed else "completed"
            if error:
                result["error"] = str(error) or type(error).__name__
            elif failed and output is not None:
                result["error"] = str(output.output)
        result["status"] = state
        return result

    async def result(self, name: str, timeout: float | None = None) -> Any:
        # A timed-out waiter must not cancel a shared job.
        waiting = asyncio.shield(self.tasks[name])
        return (
            await asyncio.wait_for(waiting, timeout)
            if timeout is not None
            else await waiting
        )

    async def cancel(self, name: str) -> dict[str, Any]:
        task = self.tasks[name]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return self.status(name)

    async def close(self) -> None:
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    def tools(self) -> list[FunctionTool]:
        async def task_result(name: str, timeout: float | None = None) -> Any:
            """Wait for a named task without cancelling it on waiter timeout."""
            return await self.result(name, timeout)

        async def task_status(name: str) -> dict[str, Any]:
            """Inspect the state of a named task."""
            return self.status(name)

        async def task_cancel(name: str) -> dict[str, Any]:
            """Cancel and reap a named task."""
            return await self.cancel(name)

        async def task_list() -> list[dict[str, Any]]:
            """List known tasks and their states."""
            return [self.status(name) for name in sorted(self.tasks)]

        return [
            FunctionTool(fn)
            for fn in (task_result, task_status, task_cancel, task_list)
        ]


@dataclass(slots=True)
class JobManager:
    """Local registered coroutine jobs, or an explicitly supplied remote backend."""

    backend: Any | None = None
    handlers: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, Any]]] = field(
        default_factory=dict
    )
    tasks: TaskManager = field(default_factory=TaskManager)

    def fresh(self) -> JobManager:
        return JobManager(self.backend, dict(self.handlers))

    async def submit(self, name: str, payload: dict[str, Any]) -> str:
        if self.backend is not None:
            return cast(str, await self.backend.submit(name, payload))
        if name not in self.handlers:
            raise KeyError(f"unknown job handler: {name}")
        job_id = uuid.uuid4().hex
        self.tasks.start(job_id, self.handlers[name](dict(payload)))
        return job_id

    async def status(self, job_id: str) -> dict[str, Any]:
        if self.backend is not None:
            return dict(await self.backend.status(job_id))
        return self.tasks.status(job_id)

    async def result(self, job_id: str, timeout: float | None = None) -> Any:
        if self.backend is not None:
            if not hasattr(self.backend, "result"):
                raise NotImplementedError("backend does not support job results")
            return await self.backend.result(job_id)
        return await self.tasks.result(job_id, timeout)

    async def cancel(self, job_id: str) -> Any:
        if self.backend is not None:
            if not hasattr(self.backend, "cancel"):
                raise NotImplementedError("backend does not support cancellation")
            return await self.backend.cancel(job_id)
        return await self.tasks.cancel(job_id)

    async def close(self) -> None:
        await self.tasks.close()


@dataclass(slots=True)
class TeamMailbox:
    messages: dict[str, list[str]] = field(default_factory=dict)
    _events: dict[str, asyncio.Event] = field(
        default_factory=dict, init=False, repr=False
    )

    def send(self, sender: str, recipient: str, message: str) -> None:
        if not recipient.strip():
            raise ValueError("recipient must not be empty")
        self.messages.setdefault(recipient, []).append(f"{sender}: {message}")
        self._events.setdefault(recipient, asyncio.Event()).set()

    async def receive(self, name: str, timeout: float = 0) -> list[str]:
        event = self._events.setdefault(name, asyncio.Event())
        if not self.messages.get(name) and timeout > 0:
            try:
                await asyncio.wait_for(event.wait(), timeout)
            except TimeoutError:
                return []
        messages = self.messages.pop(name, [])
        event.clear()
        return messages

    def tools(self) -> list[FunctionTool]:
        async def team_send(context: RunContext, recipient: str, message: str) -> str:
            """Send a message to a named teammate mailbox."""
            self.send(
                str(context.values.get("agent_name", "agent")), recipient, message
            )
            return "sent"

        async def team_receive(context: RunContext, timeout: float = 0) -> list[str]:
            """Receive and clear messages; optionally wait a bounded number of seconds."""
            return await self.receive(
                str(context.values.get("agent_name", "agent")), timeout
            )

        return [FunctionTool(team_send), FunctionTool(team_receive)]


def _task_manager(context: RunContext) -> TaskManager:
    manager = cast(TaskManager, context.tasks)
    # Runner-owned tasks return ToolResult, not plain strings. Preserve that
    # object so FunctionTool serializes output and failed state together.
    manager.tasks.update(context.session.get("background_tasks", {}))
    return manager


def state_tools() -> list[FunctionTool]:
    """Context-local tools. Unlike manager.tools(), these capture no mutable state."""

    async def todo_add(context: RunContext, label: str) -> str:
        """Add an unfinished todo item."""
        return cast(TodoList, context.todos).add(label)

    async def todo_complete(context: RunContext, label: str) -> str:
        """Mark an existing todo item complete."""
        return cast(TodoList, context.todos).complete(label)

    async def todo_list(context: RunContext) -> dict[str, bool]:
        """List todo items and completion state."""
        return dict(context.todos.items)

    async def task_list(context: RunContext) -> list[dict[str, Any]]:
        """List local and runner background tasks."""
        manager = _task_manager(context)
        return [manager.status(name) for name in sorted(manager.tasks)]

    async def task_status(context: RunContext, name: str) -> dict[str, Any]:
        """Inspect a local or runner background task."""
        return _task_manager(context).status(name)

    async def task_result(
        context: RunContext, name: str, timeout: float | None = None
    ) -> Any:
        """Wait for a task result, preserving a runner ToolResult's failure state."""
        return await _task_manager(context).result(name, timeout)

    async def task_cancel(context: RunContext, name: str) -> dict[str, Any]:
        """Cancel and reap a task."""
        return await _task_manager(context).cancel(name)

    async def job_submit(
        context: RunContext, name: str, payload: dict[str, Any]
    ) -> str:
        """Submit a registered local job or use the configured job backend."""
        return await cast(JobManager, context.jobs).submit(name, payload)

    async def job_status(context: RunContext, job_id: str) -> dict[str, Any]:
        """Inspect a job's state."""
        return await cast(JobManager, context.jobs).status(job_id)

    async def job_result(
        context: RunContext, job_id: str, timeout: float | None = None
    ) -> Any:
        """Wait for a job result when the backend supports it."""
        return await context.jobs.result(job_id, timeout)

    async def job_cancel(context: RunContext, job_id: str) -> Any:
        """Cancel a job when the backend supports it."""
        return await context.jobs.cancel(job_id)

    async def team_send(context: RunContext, recipient: str, message: str) -> str:
        """Send a message through the context's team mailbox."""
        context.teams.send(
            str(context.values.get("agent_name", "agent")), recipient, message
        )
        return "sent"

    async def team_receive(context: RunContext, timeout: float = 0) -> list[str]:
        """Receive messages from the context's team mailbox."""
        return await cast(TeamMailbox, context.teams).receive(
            str(context.values.get("agent_name", "agent")), timeout
        )

    return [
        FunctionTool(fn)
        for fn in (
            todo_add,
            todo_complete,
            todo_list,
            task_list,
            task_status,
            task_result,
            task_cancel,
            job_submit,
            job_status,
            job_result,
            job_cancel,
            team_send,
            team_receive,
        )
    ]
