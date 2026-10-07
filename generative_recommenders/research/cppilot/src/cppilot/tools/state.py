# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable

from .core import FunctionTool, RunContext


@dataclass(slots=True)
class TodoList:
    items: dict[str, bool] = field(default_factory=dict)

    def tools(self) -> list[FunctionTool]:
        def todo_add(label: str) -> str:
            """Add an unfinished todo item."""
            self.items[label] = False
            return label

        def todo_complete(label: str) -> str:
            """Mark an existing todo item complete."""
            if label not in self.items:
                raise KeyError(label)
            self.items[label] = True
            return label

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

    def start(self, name: str, awaitable: Awaitable[Any]) -> None:
        if name in self.tasks and not self.tasks[name].done():
            raise ValueError(f"task {name!r} already exists")
        self.tasks[name] = asyncio.create_task(awaitable, name=name)

    async def result(self, name: str) -> Any:
        return await self.tasks[name]


@dataclass(slots=True)
class TeamMailbox:
    messages: dict[str, list[str]] = field(default_factory=dict)

    def tools(self) -> list[FunctionTool]:
        def team_send(context: RunContext, recipient: str, message: str) -> str:
            """Send a message to a named teammate mailbox."""
            sender = str(context.values.get("agent_name", "agent"))
            self.messages.setdefault(recipient, []).append(f"{sender}: {message}")
            return "sent"

        def team_receive(context: RunContext) -> list[str]:
            """Receive and clear messages for the current agent."""
            name = str(context.values.get("agent_name", "agent"))
            return self.messages.pop(name, [])

        return [FunctionTool(team_send), FunctionTool(team_receive)]
