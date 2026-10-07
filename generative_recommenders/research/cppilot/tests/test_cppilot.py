# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cppilot import Agent, AgentRunner, FunctionTool, Message, ModelProvider, ToolCall
from cppilot.interfaces import ModelRequest, ModelResponse
from cppilot.local import FileMemoryStore, FileSessionStore
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


if __name__ == "__main__":
    unittest.main()
