# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .agent import Agent, Guardrail
from .interfaces import (
    Event,
    EventSink,
    MemoryStore,
    ModelRequest,
    ModelResponse,
    NoopEventSink,
    RetryableModelError,
    ToolDefinition,
)
from .items import Message, RunItem, ToolCall, ToolResult, Usage


class GuardrailTriggered(RuntimeError):
    pass


class MaxTurnsExceeded(RuntimeError):
    pass


class StructuredOutputError(ValueError):
    pass


class ConfigurationError(ValueError):
    pass


@dataclass(slots=True)
class RunResult:
    output: str
    items: list[RunItem]
    usage: Usage = field(default_factory=Usage)
    agent_name: str = ""
    usage_by_agent: dict[str, Usage] = field(default_factory=dict)
    run_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parsed_output: Any = None


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _guarded(guardrails: Sequence[Guardrail], value: str, agent: Agent) -> None:
    for guardrail in guardrails:
        if await _resolve(guardrail(value, agent.context)):
            raise GuardrailTriggered(getattr(guardrail, "__name__", "guardrail"))


def _add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        left.input_tokens + right.input_tokens,
        left.output_tokens + right.output_tokens,
        provider=right.provider
        if not left.provider or left.provider == right.provider
        else "",
        model=right.model if not left.model or left.model == right.model else "",
        metadata={**left.metadata, **right.metadata},
    )


def _schema_validator(schema: dict[str, Any]) -> Any:
    try:
        import jsonschema
    except ImportError as error:
        raise ConfigurationError(
            "structured output validation requires the 'schema' extra"
        ) from error
    validator_type = jsonschema.validators.validator_for(schema)
    try:
        validator_type.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise ConfigurationError(f"invalid output schema: {error.message}") from error
    return validator_type(schema)


def _validate_schema(output: str, schema: dict[str, Any]) -> Any:
    validator = _schema_validator(schema)
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise StructuredOutputError(
            f"model output is not valid JSON: {error}"
        ) from error
    errors = list(validator.iter_errors(value))
    if errors:
        raise StructuredOutputError(errors[0].message)
    return value


def _copy_state(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Copy mutable state containers, retaining application service handles."""
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, dict):
        result: Any = {}
        memo[id(value)] = result
        result.update({key: _copy_state(item, memo) for key, item in value.items()})
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(_copy_state(item, memo) for item in value)
        return result
    if isinstance(value, set):
        return set(value)
    if isinstance(value, tuple):
        return tuple(_copy_state(item, memo) for item in value)
    return value


def _agents(root: Agent) -> dict[str, Agent]:
    """Clone run-local state while preserving graph cycles and provider objects."""
    agents: dict[str, Agent] = {}
    originals: dict[str, Agent] = {}
    shared_teams = type(root.context.teams)()
    pending = [root]
    while pending:
        agent = pending.pop()
        if agent.name in originals:
            if originals[agent.name] is not agent:
                raise ConfigurationError(f"duplicate agent name: {agent.name}")
            continue
        originals[agent.name] = agent
        context = replace(
            agent.context,
            values=_copy_state(agent.context.values),
            session=_copy_state(agent.context.session),
            cancelled=asyncio.Event(),
            **agent.context.fresh_managers(),
        )
        context.teams = shared_teams
        context.session["background_tasks"] = {}
        agents[agent.name] = replace(agent, context=context)
        pending.extend(agent.handoffs)
    for name, original in originals.items():
        agents[name].handoffs = tuple(
            agents[target.name] for target in original.handoffs
        )
    return agents


class AgentRunner:
    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        events: EventSink | None = None,
        max_turns: int = 15,
        retries: int = 3,
    ) -> None:
        if max_turns < 1 or retries < 0:
            raise ConfigurationError(
                "max_turns must be positive and retries nonnegative"
            )
        self.memory = memory
        self.events = events or NoopEventSink()
        self.max_turns = max_turns
        self.retries = retries

    async def _emit(self, name: str, run_id: str, **data: Any) -> None:
        await self.events.emit(Event(name, {"run_id": run_id, **data}))

    async def _model(
        self, agent: Agent, request: ModelRequest, run_id: str, *, streaming: bool
    ) -> AsyncGenerator[RunItem | ModelResponse, None]:
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            published = False
            await self._emit(
                "model.start", run_id, agent=agent.name, attempt=attempt + 1
            )
            try:
                if streaming:
                    items: list[RunItem] = []
                    usage = Usage()
                    iterator = agent.provider.stream(request)
                    try:
                        async for item in iterator:
                            if isinstance(item, Usage):
                                usage = item
                            else:
                                items.append(item)
                                published = True
                                yield item
                    finally:
                        close = getattr(iterator, "aclose", None)
                        if close is not None:
                            await close()
                    response = ModelResponse(items, usage, usage.metadata)
                else:
                    response = await agent.provider.generate(request)
                    for item in response.items:
                        yield item
                await self._emit(
                    "model.complete",
                    run_id,
                    agent=agent.name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency=time.monotonic() - started,
                )
                yield response
                return
            except RetryableModelError as error:
                await self._emit(
                    "model.error", run_id, agent=agent.name, error=str(error)
                )
                if published or attempt == self.retries:
                    raise
                await self._emit(
                    "model.retry", run_id, agent=agent.name, attempt=attempt + 1
                )
                await asyncio.sleep(0.1 * (2**attempt))
            except GeneratorExit:
                raise
            except BaseException as error:
                await self._emit(
                    "model.error", run_id, agent=agent.name, error=type(error).__name__
                )
                raise

    async def _instructions(self, agent: Agent) -> str:
        value = (
            agent.instructions(agent.context)
            if callable(agent.instructions)
            else agent.instructions
        )
        result = str(await _resolve(value))
        if agent.skill_loader is not None:
            descriptions = "\n".join(
                f"- {skill.name}: {skill.description}"
                for skill in agent.skill_loader.available()
            )
            result += f"\n\nAvailable skills:\n{descriptions}"
        prompts = [tool.system_prompt for tool in agent.tools if tool.system_prompt]
        return "\n\n".join([result, *prompts])

    async def _finish(
        self,
        agent: Agent,
        output: str,
        items: list[RunItem],
        usage: Usage,
        usage_by_agent: dict[str, Usage],
        conversation_id: str | None,
        run_id: str,
        metadata: dict[str, Any],
    ) -> RunResult:
        parsed = (
            _validate_schema(output, agent.output_schema)
            if agent.output_schema is not None
            else None
        )
        output_type = getattr(agent, "output_type", None)
        if output_type is not None:
            try:
                parsed = output_type.model_validate_json(output)
            except ValueError as error:
                raise StructuredOutputError(str(error)) from error
        await _guarded(agent.output_guardrails, output, agent)
        if self.memory and conversation_id:
            await self.memory.append_turn(conversation_id, items, agent.name)
        await self._emit(
            "run.complete",
            run_id,
            agent=agent.name,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        return RunResult(
            output, items, usage, agent.name, usage_by_agent, run_id, metadata, parsed
        )

    async def _invoke(
        self, tool: Any, call: ToolCall, agent: Agent, run_id: str
    ) -> ToolResult:
        started = time.monotonic()
        await self._emit(
            "tool.start", run_id, tool=call.name, call_id=call.call_id, agent=agent.name
        )
        try:
            invocation = tool.invoke(call.arguments, agent.context)
            output = (
                await asyncio.wait_for(invocation, tool.timeout)
                if tool.timeout is not None
                else await invocation
            )
        except Exception as error:
            await self._emit(
                "tool.error",
                run_id,
                tool=call.name,
                call_id=call.call_id,
                agent=agent.name,
                error=str(error),
            )
            return ToolResult(
                call.call_id, call.name, str(error) or type(error).__name__, True
            )
        await self._emit(
            "tool.complete",
            run_id,
            tool=call.name,
            call_id=call.call_id,
            agent=agent.name,
            latency=time.monotonic() - started,
        )
        return ToolResult(call.call_id, call.name, output)

    async def _execute(  # noqa: C901
        self, root: Agent, prompt: str, conversation_id: str | None, *, streaming: bool
    ) -> AsyncGenerator[RunItem | RunResult, None]:
        run_id = uuid.uuid4().hex
        graph = _agents(root)
        current = graph[root.name]
        if self.memory and conversation_id:
            active = await self.memory.get_active_agent(conversation_id)
            if active is not None:
                if active not in graph:
                    raise ConfigurationError(
                        f"saved active agent {active!r} is not configured"
                    )
                current = graph[active]
        for agent in graph.values():
            if agent.output_schema is not None:
                _schema_validator(agent.output_schema)
        history = (
            await self.memory.load(conversation_id)
            if self.memory and conversation_id
            else []
        )
        items: list[RunItem] = [
            item
            for item in history
            if not isinstance(item, Message) or item.role != "system"
        ]
        newly_added: list[RunItem] = [Message("user", prompt)]
        items.extend(newly_added)
        total_usage = Usage()
        usage_by_agent: dict[str, Usage] = {}
        metadata: dict[str, Any] = {}
        background: list[asyncio.Task[Any]] = []
        await self._emit(
            "run.start", run_id, agent=current.name, conversation_id=conversation_id
        )
        try:
            await _guarded(current.input_guardrails, prompt, current)
            for turn in range(self.max_turns):
                current.context.values.update(agent_name=current.name, run_id=run_id)
                instructions = await self._instructions(current)
                tools = list(current.tools)
                handoffs = {
                    f"transfer_to_{target.name}": target for target in current.handoffs
                }
                definitions = [tool.definition for tool in tools] + [
                    ToolDefinition(
                        name,
                        f"Transfer this run to {target.name}",
                        {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    )
                    for name, target in handoffs.items()
                ]
                if len({definition.name for definition in definitions}) != len(
                    definitions
                ):
                    raise ConfigurationError("tool and handoff names must be unique")
                request = ModelRequest(
                    [Message("system", instructions), *items],
                    definitions,
                    current.output_schema,
                    instructions,
                    {"run_id": run_id, "turn": turn},
                )
                response: ModelResponse | None = None
                model_stream = self._model(
                    current, request, run_id, streaming=streaming
                )
                try:
                    async for event in model_stream:
                        if isinstance(event, ModelResponse):
                            response = event
                        else:
                            yield event
                finally:
                    await model_stream.aclose()
                if response is None:
                    raise RuntimeError("provider did not return a response")
                items.extend(response.items)
                newly_added.extend(response.items)
                metadata[current.name] = response.metadata
                total_usage = _add_usage(total_usage, response.usage)
                usage_by_agent[current.name] = _add_usage(
                    usage_by_agent.get(current.name, Usage()), response.usage
                )
                calls = [item for item in response.items if isinstance(item, ToolCall)]
                if not calls:
                    output = "".join(
                        item.content
                        for item in response.items
                        if isinstance(item, Message) and item.role == "assistant"
                    )
                    yield await self._finish(
                        current,
                        output,
                        newly_added,
                        total_usage,
                        usage_by_agent,
                        conversation_id,
                        run_id,
                        metadata,
                    )
                    return
                tool_map = {tool.definition.name: tool for tool in tools}
                calling_agent = current
                final_output: str | None = None
                for call in calls:
                    if call.name in handoffs:
                        destination = handoffs[call.name]
                        await _guarded(
                            destination.input_guardrails, prompt, destination
                        )
                        current = destination
                        result = ToolResult(
                            call.call_id, call.name, json.dumps({"agent": current.name})
                        )
                        await self._emit(
                            "handoff",
                            run_id,
                            agent=current.name,
                            source=calling_agent.name,
                        )
                    else:
                        tool = tool_map.get(call.name)
                        if tool is None:
                            result = ToolResult(
                                call.call_id,
                                call.name,
                                f"unknown tool: {call.name}",
                                True,
                            )
                        elif tool.background:
                            task = asyncio.create_task(
                                self._invoke(tool, call, calling_agent, run_id)
                            )
                            background.append(task)
                            calling_agent.context.session["background_tasks"][
                                call.call_id
                            ] = task
                            result = ToolResult(
                                call.call_id,
                                call.name,
                                json.dumps({"background_task": call.call_id}),
                            )
                        else:
                            result = await self._invoke(
                                tool, call, calling_agent, run_id
                            )
                            if tool.final_result and not result.failed:
                                final_output = result.output
                    items.append(result)
                    newly_added.append(result)
                    yield result
                if final_output is not None:
                    yield await self._finish(
                        calling_agent,
                        final_output,
                        newly_added,
                        total_usage,
                        usage_by_agent,
                        conversation_id,
                        run_id,
                        metadata,
                    )
                    return
            raise MaxTurnsExceeded(f"run exceeded {self.max_turns} turns")
        except GeneratorExit:
            raise
        except BaseException as error:
            await self._emit(
                "run.error", run_id, agent=current.name, error=type(error).__name__
            )
            raise
        finally:
            for agent in graph.values():
                agent.context.cancelled.set()
            for task in background:
                if not task.done():
                    task.cancel()
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            for agent in graph.values():
                await agent.context.tasks.close()
                await agent.context.jobs.close()

    async def run(
        self, agent: Agent, prompt: str, *, conversation_id: str | None = None
    ) -> RunResult:
        iterator = self._execute(agent, prompt, conversation_id, streaming=False)
        try:
            async for item in iterator:
                if isinstance(item, RunResult):
                    await iterator.aclose()
                    return item
        finally:
            await iterator.aclose()
        raise RuntimeError("run did not produce a result")

    async def stream(
        self, agent: Agent, prompt: str, *, conversation_id: str | None = None
    ) -> AsyncGenerator[RunItem, None]:
        """Yield live text/tool items, then aggregate usage; failures match run()."""
        iterator = self._execute(agent, prompt, conversation_id, streaming=True)
        try:
            async for item in iterator:
                if isinstance(item, RunResult):
                    await iterator.aclose()
                    yield item.usage
                    return
                yield item
        finally:
            await iterator.aclose()

    async def parallel(
        self, runs: Sequence[tuple[Agent, str]], *, return_exceptions: bool = False
    ) -> list[RunResult | BaseException]:
        tasks = [asyncio.create_task(self.run(agent, prompt)) for agent, prompt in runs]
        try:
            return list(
                await asyncio.gather(*tasks, return_exceptions=return_exceptions)
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
