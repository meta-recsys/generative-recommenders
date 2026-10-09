# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from unittest.mock import patch

from cppilot import (
    Agent,
    AgentRunner,
    FunctionTool,
    Message,
    ModelProvider,
    ToolCall,
    Usage,
)
from cppilot.interfaces import Event, EventSink, ModelRequest, ModelResponse
from cppilot.runner import ConfigurationError, GuardrailTriggered, MaxTurnsExceeded
from cppilot.tools import RunContext


class RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class SequenceProvider(ModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class RuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_stream_is_live_and_ordered(self) -> None:
        gate = asyncio.Event()

        class LiveProvider(ModelProvider):
            async def generate(self, request: ModelRequest) -> ModelResponse:
                raise AssertionError("native stream must be used")

            async def stream(
                self, request: ModelRequest
            ) -> AsyncIterator[Message | Usage]:
                yield Message("assistant", "first")
                await gate.wait()
                yield Message("assistant", " second")
                yield Usage(3, 2)

        stream = AgentRunner().stream(Agent("live", "Answer.", LiveProvider()), "go")
        self.assertEqual(await anext(stream), Message("assistant", "first"))
        gate.set()
        self.assertEqual(await anext(stream), Message("assistant", " second"))
        self.assertEqual(await anext(stream), Usage(3, 2))
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_stream_close_cancels_provider(self) -> None:
        closed = asyncio.Event()

        class LiveProvider(ModelProvider):
            async def generate(self, request: ModelRequest) -> ModelResponse:
                raise AssertionError("native stream must be used")

            async def stream(self, request: ModelRequest) -> AsyncIterator[Message]:
                try:
                    yield Message("assistant", "hello")
                    await asyncio.Event().wait()
                finally:
                    closed.set()

        stream = AgentRunner().stream(Agent("live", "Answer.", LiveProvider()), "go")
        await anext(stream)
        await stream.aclose()
        self.assertTrue(closed.is_set())

    async def test_max_turns_matches_stream_and_run(self) -> None:
        for streaming in (False, True):
            provider = SequenceProvider([ModelResponse([ToolCall("1", "missing", {})])])
            agent = Agent("loop", "Loop.", provider)
            runner = AgentRunner(max_turns=1)
            with self.assertRaises(MaxTurnsExceeded):
                if streaming:
                    async for _ in runner.stream(agent, "go"):
                        pass
                else:
                    await runner.run(agent, "go")

    async def test_final_tools_apply_guardrails_and_complete_all_results(self) -> None:
        def final() -> str:
            return "blocked"

        def extra() -> str:
            return "extra"

        provider = SequenceProvider(
            [ModelResponse([ToolCall("1", "final", {}), ToolCall("2", "extra", {})])]
        )
        agent = Agent(
            "final",
            "Use tools.",
            provider,
            [FunctionTool(final, final_result=True), FunctionTool(extra)],
            output_guardrails=[lambda output, context: output == "blocked"],
        )
        with self.assertRaises(GuardrailTriggered):
            await AgentRunner().run(agent, "go")

    async def test_multi_call_handoff_uses_source_context(self) -> None:
        def who(context: RunContext) -> str:
            return context.values["identity"]

        destination_provider = SequenceProvider(
            [ModelResponse([Message("assistant", "done")])]
        )
        destination = Agent(
            "destination",
            "Destination.",
            destination_provider,
            context=RunContext({"identity": "destination"}),
        )
        source = Agent(
            "source",
            "Source.",
            SequenceProvider(
                [
                    ModelResponse(
                        [
                            ToolCall("1", "transfer_to_destination", {}),
                            ToolCall("2", "who", {}),
                        ]
                    )
                ]
            ),
            [FunctionTool(who)],
            handoffs=[destination],
            context=RunContext({"identity": "source"}),
        )
        result = await AgentRunner().run(source, "go")
        from cppilot.items import ToolResult

        item = result.items[-2]
        assert isinstance(item, ToolResult)
        self.assertEqual(item.output, "source")
        self.assertEqual(result.agent_name, "destination")

    async def test_correlated_events_and_usage(self) -> None:
        sink = RecordingSink()
        provider = SequenceProvider(
            [ModelResponse([Message("assistant", "ok")], Usage(2, 4))]
        )
        result = await AgentRunner(events=sink).run(
            Agent("helper", "Answer.", provider), "go"
        )
        self.assertEqual(result.usage_by_agent["helper"], Usage(2, 4))
        self.assertEqual(
            {event.data["run_id"] for event in sink.events}, {result.run_id}
        )
        self.assertEqual(
            [event.name for event in sink.events],
            ["run.start", "model.start", "model.complete", "run.complete"],
        )

    async def test_parallel_cancels_siblings_on_failure(self) -> None:
        cancelled = asyncio.Event()
        started = asyncio.Event()

        class WaitingProvider(ModelProvider):
            async def generate(self, request: ModelRequest) -> ModelResponse:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
                raise AssertionError("unreachable")

        class FailedProvider(ModelProvider):
            async def generate(self, request: ModelRequest) -> ModelResponse:
                await started.wait()
                raise ValueError("failed")

        with self.assertRaisesRegex(ValueError, "failed"):
            await AgentRunner().parallel(
                [
                    (Agent("waiting", "Wait.", WaitingProvider()), "go"),
                    (Agent("failed", "Fail.", FailedProvider()), "go"),
                ]
            )
        self.assertTrue(cancelled.is_set())

    async def test_run_does_not_mutate_application_context(self) -> None:
        context = RunContext({"identity": "root"})
        provider = SequenceProvider([ModelResponse([Message("assistant", "ok")])])
        await AgentRunner().run(
            Agent("helper", "Answer.", provider, context=context), "go"
        )
        self.assertEqual(context.values, {"identity": "root"})
        self.assertFalse(context.cancelled.is_set())

    async def test_nested_context_is_run_local(self) -> None:
        def mutate(context: RunContext) -> str:
            context.values["nested"]["items"].append("changed")
            return "ok"

        context = RunContext({"nested": {"items": []}})
        provider = SequenceProvider([ModelResponse([ToolCall("1", "mutate", {})])])
        await AgentRunner().run(
            Agent(
                "local",
                "Use tools.",
                provider,
                [FunctionTool(mutate, final_result=True)],
                context=context,
            ),
            "go",
        )
        self.assertEqual(context.values, {"nested": {"items": []}})

    async def test_final_tool_handoff_uses_source_guardrails(self) -> None:
        def final() -> str:
            return "blocked"

        destination = Agent("destination", "Answer.", SequenceProvider([]))
        source = Agent(
            "source",
            "Use tools.",
            SequenceProvider(
                [
                    ModelResponse(
                        [
                            ToolCall("1", "transfer_to_destination", {}),
                            ToolCall("2", "final", {}),
                        ]
                    )
                ]
            ),
            [FunctionTool(final, final_result=True)],
            handoffs=[destination],
            output_guardrails=[lambda output, context: output == "blocked"],
        )
        with self.assertRaises(GuardrailTriggered):
            await AgentRunner().run(source, "go")

    async def test_background_cleanup_precedes_terminal_usage(self) -> None:
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def background() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return "unreachable"

        class BackgroundProvider(ModelProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, request: ModelRequest) -> ModelResponse:
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse([ToolCall("1", "background", {})])
                await started.wait()
                return ModelResponse([Message("assistant", "done")])

        stream = AgentRunner().stream(
            Agent(
                "background",
                "Use tools.",
                BackgroundProvider(),
                [FunctionTool(background, background=True)],
            ),
            "go",
        )
        while not isinstance(await anext(stream), Usage):
            pass
        self.assertTrue(cancelled.is_set())
        await stream.aclose()

    async def test_schema_extra_missing_fails_before_model_call(self) -> None:
        provider = SequenceProvider([ModelResponse([Message("assistant", "{}")])])
        with patch.dict("sys.modules", {"jsonschema": None}):
            with self.assertRaisesRegex(ConfigurationError, "schema.*extra"):
                await AgentRunner().run(
                    Agent("typed", "JSON.", provider, output_schema={"type": "object"}),
                    "go",
                )
        self.assertEqual(provider.requests, [])

    async def test_structured_output_validation(self) -> None:
        try:
            import jsonschema
        except ImportError:
            self.skipTest("schema extra not installed")
        self.assertIsNotNone(jsonschema)
        from cppilot.runner import StructuredOutputError

        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        for output in ("not JSON", '{"count": "wrong"}'):
            with self.assertRaises(StructuredOutputError):
                await AgentRunner().run(
                    Agent(
                        "typed",
                        "JSON.",
                        SequenceProvider(
                            [ModelResponse([Message("assistant", output)])]
                        ),
                        output_schema=schema,
                    ),
                    "go",
                )
        result = await AgentRunner().run(
            Agent(
                "typed",
                "JSON.",
                SequenceProvider(
                    [ModelResponse([Message("assistant", '{"count": 2}')])]
                ),
                output_schema=schema,
            ),
            "go",
        )
        self.assertEqual(result.parsed_output, {"count": 2})

    async def test_invalid_configuration(self) -> None:
        with self.assertRaises(ConfigurationError):
            AgentRunner(max_turns=0)
        with self.assertRaises(ConfigurationError):
            AgentRunner(retries=-1)
