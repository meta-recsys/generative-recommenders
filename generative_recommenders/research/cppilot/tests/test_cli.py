# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cppilot.cli import _command, _tail_compactor, execute, main, parser, Terminal
from cppilot.interfaces import ModelProvider, ModelRequest, ModelResponse
from cppilot.items import Message, ToolCall, ToolResult
from cppilot.local import FileMemoryStore, FileSessionStore
from cppilot.registry import ModelRegistry
from cppilot.sqlite import SQLiteMemoryStore, SQLiteSessionStore
from cppilot.workflow import Workflow, WorkflowRegistry


class EchoProvider(ModelProvider):
    def __init__(self):
        self.requests = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse([Message("assistant", f"reply {len(self.requests)}")])


class ScriptedTerminal:
    def __init__(self, prompts=()):
        self.prompts = iter(prompts)
        self.output = []

    def write(self, text, *, end="\n"):
        self.output.append(text + end)

    async def read(self):
        return next(self.prompts)


class CLITest(unittest.IsolatedAsyncioTestCase):
    async def test_interactive_retains_history_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "--storage",
                    "file",
                    "--memory-root",
                    str(root / "memory"),
                    "--session-root",
                    str(root / "sessions"),
                    "--provider",
                    "fake",
                ]
            )
            provider = EchoProvider()
            models = ModelRegistry()
            models.register("fake", provider)
            terminal = ScriptedTerminal(["hello", "again", "/session", "/exit"])
            await execute(args, models=models, terminal=terminal)
            sessions = FileSessionStore(root / "sessions")
            saved = (await sessions.list())[0]
            self.assertEqual(len(provider.requests), 2)
            self.assertTrue(
                any(
                    isinstance(item, Message) and item.content == "hello"
                    for item in provider.requests[1].messages
                )
            )
            self.assertIn(saved.id + "\n", terminal.output)
            resume = parser().parse_args(
                [
                    "resume",
                    saved.id,
                    "next",
                    "--storage",
                    "file",
                    "--memory-root",
                    str(root / "memory"),
                    "--session-root",
                    str(root / "sessions"),
                ]
            )
            with redirect_stderr(io.StringIO()):
                await execute(resume, models=models, terminal=terminal)
            self.assertEqual(len(provider.requests), 3)
            self.assertEqual(
                len(await FileMemoryStore(root / "memory").load(saved.conversation_id)),
                6,
            )

    async def test_memory_commands_clear_and_compact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parser().parse_args(
                [
                    "--storage",
                    "file",
                    "--memory-root",
                    str(root / "memory"),
                    "--session-root",
                    str(root / "sessions"),
                    "--provider",
                    "fake",
                ]
            )
            models = ModelRegistry()
            provider = EchoProvider()
            models.register("fake", provider)
            terminal = ScriptedTerminal(
                [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "/compact",
                    "/memory",
                    "/clear",
                    "fresh",
                    "/exit",
                ]
            )
            await execute(args, models=models, terminal=terminal)
            self.assertIn("Memory compacted: 8 items retained.\n", terminal.output)
            self.assertIn("Memory cleared.\n", terminal.output)
            self.assertFalse(
                any(
                    isinstance(i, Message) and i.content == "one"
                    for i in provider.requests[-1].messages
                )
            )

    async def test_info_does_not_construct_provider_or_storage(self) -> None:
        models = ModelRegistry()

        def fail():
            raise AssertionError("provider should not load")

        models.register("fake", fail)
        terminal = ScriptedTerminal()
        await execute(
            parser().parse_args(["info", "--storage", "postgres"]),
            models=models,
            terminal=terminal,
        )
        self.assertIn("providers: fake", terminal.output[0])

    async def test_custom_workflow_cleanup_and_resume_configuration(self) -> None:
        class Custom(Workflow[str, str]):
            closed = False

            async def run(self, value):
                return "custom: " + value

            async def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            models, workflows = ModelRegistry(), WorkflowRegistry()
            models.register("fake", EchoProvider())
            custom = Custom()
            workflows.register("custom", lambda **configuration: custom)
            args = parser().parse_args(
                [
                    "run",
                    "hello",
                    "--provider",
                    "fake",
                    "--workflow",
                    "custom",
                    "--storage",
                    "file",
                    "--memory-root",
                    directory,
                    "--session-root",
                    directory,
                ]
            )
            terminal = ScriptedTerminal()
            with redirect_stderr(io.StringIO()):
                await execute(
                    args, models=models, workflows=workflows, terminal=terminal
                )
            self.assertEqual(terminal.output, ["custom: hello\n"])
            self.assertTrue(custom.closed)
            session = (await FileSessionStore(Path(directory)).list())[0]
            args = parser().parse_args(
                [
                    "resume",
                    session.id,
                    "--provider",
                    "wrong",
                    "--storage",
                    "file",
                    "--session-root",
                    directory,
                ]
            )
            with self.assertRaisesRegex(ValueError, "cannot override saved provider"):
                await execute(args, models=models, terminal=terminal)

    async def test_default_sqlite_and_saved_model_switch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "sessions.db")
            models = ModelRegistry()
            first, second = EchoProvider(), EchoProvider()
            models.register("first", first)
            models.register("second", second)
            terminal = ScriptedTerminal(
                [
                    "hello",
                    "/models",
                    "/model missing",
                    "/model second",
                    "next",
                    "/model",
                    "/sessions",
                    "/exit",
                ]
            )
            args = parser().parse_args(["--provider", "first", "--database", database])
            await execute(args, models=models, terminal=terminal)
            sessions = SQLiteSessionStore(database)
            memory = SQLiteMemoryStore(database)
            try:
                saved = (await sessions.list())[0]
                self.assertEqual(saved.metadata["configured_model"], "second")
                self.assertEqual(saved.metadata["provider"], "second")
                self.assertEqual(len(first.requests), 1)
                self.assertEqual(len(second.requests), 1)
                self.assertTrue(
                    any(
                        isinstance(i, Message) and i.content == "hello"
                        for i in second.requests[0].messages
                    )
                )
                self.assertIn("Model switched to second.\n", terminal.output)
                self.assertTrue(
                    any("unknown model" in text for text in terminal.output)
                )
                self.assertTrue(
                    any(
                        saved.id in text and "model=second" in text
                        for text in terminal.output
                    )
                )
                with redirect_stderr(io.StringIO()):
                    await execute(
                        parser().parse_args(
                            ["resume", saved.id, "again", "--database", database]
                        ),
                        models=models,
                        terminal=terminal,
                    )
                self.assertEqual(len(second.requests), 2)
                self.assertEqual(len(await memory.load(saved.conversation_id)), 6)
            finally:
                await sessions.close()
                await memory.close()

    async def test_config_alias_and_explicit_storage_override(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "storage": "sqlite",
                        "database": str(root / "unused.db"),
                        "model": "fast",
                        "models": {"fast": {"provider": "fake", "model": "small"}},
                    }
                )
            )
            models = ModelRegistry()
            provider = EchoProvider()
            received = []

            def factory(**configuration):
                received.append(configuration)
                return provider

            models.register("fake", factory)
            terminal = ScriptedTerminal()
            args = parser().parse_args(
                [
                    "hello",
                    "--config",
                    str(config),
                    "--storage",
                    "file",
                    "--memory-root",
                    str(root / "memory"),
                    "--session-root",
                    str(root / "sessions"),
                ]
            )
            with redirect_stderr(io.StringIO()):
                await execute(args, models=models, terminal=terminal)
            saved = (await FileSessionStore(root / "sessions").list())[0]
            self.assertEqual(received, [{"model": "small"}])
            self.assertEqual(saved.metadata["configured_model"], "fast")
            self.assertFalse((root / "unused.db").exists())
            # Resume uses the saved configuration, even without the alias config file.
            resume_models = ModelRegistry()
            resume_models.register("fake", factory)
            with redirect_stderr(io.StringIO()):
                await execute(
                    parser().parse_args(
                        [
                            "resume",
                            saved.id,
                            "again",
                            "--storage",
                            "file",
                            "--session-root",
                            str(root / "sessions"),
                            "--memory-root",
                            str(root / "memory"),
                        ]
                    ),
                    models=resume_models,
                    terminal=terminal,
                )
            self.assertEqual(received, [{"model": "small"}, {"model": "small"}])

    async def test_stream_prints_assistant_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            models = ModelRegistry()
            models.register("fake", EchoProvider())
            terminal = ScriptedTerminal()
            args = parser().parse_args(
                [
                    "hello",
                    "--provider",
                    "fake",
                    "--stream",
                    "--storage",
                    "file",
                    "--memory-root",
                    directory,
                    "--session-root",
                    directory,
                ]
            )
            with redirect_stderr(io.StringIO()):
                await execute(args, models=models, terminal=terminal)
            self.assertEqual("".join(terminal.output), "reply 1\n")


class CLIParsingTest(unittest.TestCase):
    def test_commands_and_legacy_prompt(self) -> None:
        self.assertEqual(
            _command(parser().parse_args(["hello"])), ("run", None, "hello")
        )
        self.assertEqual(
            _command(parser().parse_args(["resume", "session", "hello", "world"])),
            ("resume", "session", "hello world"),
        )
        with self.assertRaisesRegex(ValueError, "session id"):
            _command(parser().parse_args(["resume"]))

    def test_plain_terminal_and_main_info(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            Terminal(plain=True).write("[literal]")
            main(["info", "--plain"])
        self.assertIn("[literal]\n", output.getvalue())
        self.assertIn("providers: anthropic, gemini, openai", output.getvalue())

    def test_optional_ui_missing_falls_back_on_tty(self) -> None:
        import builtins

        original_import = builtins.__import__

        def missing_ui(name, *args, **kwargs):
            if name in {"rich.console", "prompt_toolkit"}:
                raise ImportError("optional UI unavailable")
            return original_import(name, *args, **kwargs)

        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.__import__", side_effect=missing_ui),
        ):
            terminal = Terminal()
        self.assertIsNone(terminal.console)
        self.assertIsNone(terminal.prompt_session)

    def test_compaction_keeps_complete_tool_turns(self) -> None:
        items = []
        for i in range(5):
            items.extend(
                [
                    Message("user", str(i)),
                    ToolCall(str(i), "tool", {}),
                    ToolResult(str(i), "tool", "ok"),
                ]
            )
        compacted = _tail_compactor(items)
        self.assertEqual(compacted, items[3:])
        self.assertEqual(len(compacted), 12)

    def test_entrypoints_explicit_and_sorted(self) -> None:
        models = ModelRegistry()
        provider = EchoProvider()
        entry = SimpleNamespace(name="fake", load=lambda: provider)
        with patch(
            "cppilot.registry.metadata.entry_points", return_value=[entry]
        ) as discover:
            self.assertEqual(models.names(), [])
            discover.assert_not_called()
            self.assertEqual(models.load_entry_points(), ["fake"])
            self.assertIs(models.get("fake"), provider)
            discover.assert_called_once_with(group="cppilot.models")
            with self.assertRaisesRegex(ValueError, "already registered"):
                models.load_entry_points()
