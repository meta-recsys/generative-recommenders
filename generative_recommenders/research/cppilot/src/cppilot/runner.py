# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from .agent import Agent, Guardrail
from .interfaces import (
    Event,
    EventSink,
    MemoryStore,
    ModelRequest,
    NoopEventSink,
    ToolDefinition,
)
from .items import Message, RunItem, ToolCall, ToolResult, Usage


class GuardrailTriggered(RuntimeError):
    pass


class MaxTurnsExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class RunResult:
    output: str
    items: list[RunItem]
    usage: Usage = field(default_factory=Usage)
    agent_name: str = ""


async def _guarded(guardrails: Sequence[Guardrail], value: str, agent: Agent) -> None:
    for guardrail in guardrails:
        result = guardrail(value, agent.context)
        if inspect.isawaitable(result):
            result = await result
        if result:
            raise GuardrailTriggered(getattr(guardrail, "__name__", "guardrail"))


class AgentRunner:
    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        events: EventSink | None = None,
        max_turns: int = 15,
        retries: int = 3,
    ) -> None:
        self.memory = memory
        self.events = events or NoopEventSink()
        self.max_turns = max_turns
        self.retries = retries

    async def _generate(self, agent: Agent, request: ModelRequest):
        for attempt in range(self.retries + 1):
            try:
                await self.events.emit(
                    Event("model.start", {"agent": agent.name, "attempt": attempt + 1})
                )
                return await agent.provider.generate(request)
            except Exception as error:
                await self.events.emit(
                    Event("model.retry", {"error": str(error), "attempt": attempt + 1})
                )
                if attempt == self.retries:
                    raise
                await asyncio.sleep(0.1 * (2**attempt))

    async def run(
        self, agent: Agent, prompt: str, *, conversation_id: str | None = None
    ) -> RunResult:
        await _guarded(agent.input_guardrails, prompt, agent)
        history = (
            await self.memory.load(conversation_id)
            if self.memory and conversation_id
            else []
        )
        items: list[RunItem] = [
            *history,
            Message("system", agent.instructions),
            Message("user", prompt),
        ]
        current = agent
        total_usage = Usage()
        for _ in range(self.max_turns):
            tools = list(current.tools)
            handoff_map = {
                f"transfer_to_{target.name}": target for target in current.handoffs
            }
            definitions = [tool.definition for tool in tools] + [
                ToolDefinition(
                    name=name,
                    description=f"Transfer this run to {target.name}",
                    parameters={"type": "object", "properties": {}},
                )
                for name, target in handoff_map.items()
            ]
            response = await self._generate(
                current, ModelRequest(items, definitions, current.output_schema)
            )
            items.extend(response.items)
            total_usage = Usage(
                total_usage.input_tokens + response.usage.input_tokens,
                total_usage.output_tokens + response.usage.output_tokens,
            )
            calls = [item for item in response.items if isinstance(item, ToolCall)]
            messages = [
                item
                for item in response.items
                if isinstance(item, Message) and item.role == "assistant"
            ]
            if not calls:
                output = messages[-1].content if messages else ""
                if current.output_schema is not None:
                    json.loads(output)
                await _guarded(current.output_guardrails, output, current)
                if self.memory and conversation_id:
                    await self.memory.append(conversation_id, items[len(history) :])
                await self.events.emit(Event("run.complete", {"agent": current.name}))
                return RunResult(
                    output, items[len(history) :], total_usage, current.name
                )
            tool_map = {tool.definition.name: tool for tool in tools}
            for call in calls:
                if call.name in handoff_map:
                    current = handoff_map[call.name]
                    items.append(
                        ToolResult(
                            call.call_id, call.name, json.dumps({"agent": current.name})
                        )
                    )
                    await self.events.emit(Event("handoff", {"agent": current.name}))
                    continue
                tool = tool_map.get(call.name)
                if tool is None:
                    result = ToolResult(
                        call.call_id, call.name, f"unknown tool: {call.name}", True
                    )
                else:
                    try:
                        await self.events.emit(Event("tool.start", {"tool": call.name}))
                        result = ToolResult(
                            call.call_id,
                            call.name,
                            await tool.invoke(call.arguments, current.context),
                        )
                    except Exception as error:
                        result = ToolResult(call.call_id, call.name, str(error), True)
                items.append(result)
        raise MaxTurnsExceeded(f"run exceeded {self.max_turns} turns")

    async def stream(
        self, agent: Agent, prompt: str, *, conversation_id: str | None = None
    ) -> AsyncIterator[RunItem]:
        result = await self.run(agent, prompt, conversation_id=conversation_id)
        for item in result.items:
            yield item
        yield result.usage
