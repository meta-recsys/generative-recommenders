# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from cppilot import (
    Agent,
    AgentRunner,
    FunctionTool,
    Message,
    ModelProvider,
    RetryableModelError,
    ToolCall,
)
from cppilot.interfaces import ModelRequest, ModelResponse
from cppilot.local import FileMemoryStore, FileSessionStore
from cppilot.sqlite import SQLiteMemoryStore
from cppilot.tools import file_tools, RunContext


class FakeProvider(ModelProvider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class CPPilotTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_and_memory(self) -> None:
        def add(left: int, right: int) -> int:
            """Add two integers."""
            return left + right

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provider = FakeProvider(
                [
                    ModelResponse([ToolCall("1", "add", {"left": 2, "right": 3})]),
                    ModelResponse([Message("assistant", "five")]),
                ]
            )
            result = await AgentRunner(memory=FileMemoryStore(root)).run(
                Agent("math", "Use tools.", provider, [FunctionTool(add)]),
                "2+3",
                conversation_id="demo",
            )
            self.assertEqual(result.output, "five")
            self.assertIn('"tool_result"', (root / "demo.jsonl").read_text())
            self.assertEqual(
                provider.requests[0].tools[0].parameters["properties"]["left"]["type"],
                "integer",
            )

    async def test_handoff(self) -> None:
        specialist = Agent(
            "specialist",
            "Handle it.",
            FakeProvider([ModelResponse([Message("assistant", "handled")])]),
        )
        router = Agent(
            "router",
            "Route it.",
            FakeProvider(
                [ModelResponse([ToolCall("h", "transfer_to_specialist", {})])]
            ),
            handoffs=[specialist],
        )
        result = await AgentRunner().run(router, "help")
        self.assertEqual((result.output, result.agent_name), ("handled", "specialist"))

    async def test_sessions_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FileSessionStore(Path(directory))
            created = await store.create(
                "agent", "conversation", {"topic": "synthetic"}
            )
            self.assertEqual(await store.load(created.id), created)
            self.assertEqual(await store.list(), [created])
            await store.delete(created.id)
            self.assertIsNone(await store.load(created.id))

    async def test_file_tools_reject_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = RunContext({"read_root": root, "write_root": root})
            read, write = file_tools()
            await write.invoke({"path": "safe/note.txt", "content": "ok"}, context)
            self.assertEqual(
                await read.invoke({"path": "safe/note.txt"}, context), "ok"
            )
            with self.assertRaises(PermissionError):
                await read.invoke({"path": "../secret"}, context)

    async def test_handoff_applies_dynamic_destination_instructions(self) -> None:
        specialist_provider = FakeProvider(
            [ModelResponse([Message("assistant", "ok")])]
        )

        async def instructions(context: RunContext) -> str:
            return f"Specialist for {context.values['topic']}"

        specialist = Agent(
            "specialist",
            instructions,
            specialist_provider,
            context=RunContext({"topic": "tools"}),
        )
        router = Agent(
            "router",
            "Route.",
            FakeProvider(
                [ModelResponse([ToolCall("h", "transfer_to_specialist", {})])]
            ),
            handoffs=[specialist],
        )
        result = await AgentRunner().run(router, "help")
        self.assertEqual(result.agent_name, "specialist")
        self.assertEqual(specialist_provider.requests[0].system, "Specialist for tools")

    async def test_retries_only_retryable_provider_errors(self) -> None:
        class FlakyProvider(ModelProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, request: ModelRequest) -> ModelResponse:
                self.calls += 1
                if self.calls == 1:
                    raise RetryableModelError("temporary")
                return ModelResponse([Message("assistant", "recovered")])

        provider = FlakyProvider()
        result = await AgentRunner(retries=1).run(
            Agent("retry", "Retry.", provider), "go"
        )
        self.assertEqual(result.output, "recovered")
        self.assertEqual(provider.calls, 2)

    async def test_parallel_runs_preserve_order(self) -> None:
        agents = [
            Agent(
                str(index),
                "Answer.",
                FakeProvider([ModelResponse([Message("assistant", str(index))])]),
            )
            for index in range(3)
        ]
        results = await AgentRunner().parallel([(agent, "go") for agent in agents])
        self.assertEqual(
            [
                result.output
                for result in results
                if not isinstance(result, BaseException)
            ],
            ["0", "1", "2"],
        )

    async def test_tool_timeout_is_returned_to_model(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(1)
            return "late"

        provider = FakeProvider(
            [
                ModelResponse([ToolCall("1", "slow", {})]),
                ModelResponse([Message("assistant", "handled")]),
            ]
        )
        result = await AgentRunner().run(
            Agent(
                "timeout", "Use tools.", provider, [FunctionTool(slow, timeout=0.001)]
            ),
            "go",
        )
        self.assertEqual(result.output, "handled")
        self.assertTrue(any(getattr(item, "failed", False) for item in result.items))

    async def test_sqlite_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "cppilot.db")
            await store.append("conversation", [Message("user", "hello")])
            loaded = await store.load("conversation")
            self.assertEqual(loaded, [Message("user", "hello")])
