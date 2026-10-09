# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cppilot.agent import Agent
from cppilot.interfaces import (
    ExecutionResult,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    Sandbox,
)
from cppilot.items import item_to_dict, Message, ToolCall, ToolResult, Usage
from cppilot.local import (
    _atomic_write,
    ConcurrentModificationError,
    FileMemoryStore,
    FileSessionStore,
)
from cppilot.settings import Settings
from cppilot.sqlite import SQLiteMemoryStore, SQLiteSessionStore
from cppilot.storage import PostgresMemoryStore, PostgresSessionStore
from cppilot.workflow import AgentWorkflow, WorkflowContext, WorkflowRegistry


class PersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_memory_contracts_and_concurrent_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for factory in (
                lambda: FileMemoryStore(root / "files"),
                lambda: SQLiteMemoryStore(root / "db"),
                lambda: SQLiteMemoryStore(":memory:"),
            ):
                store = factory()
                with self.subTest(store=type(store).__name__):
                    items = [
                        Message("user", "hi", metadata={"source": "test"}),
                        ToolCall("c", "tool", {"x": 1}),
                        ToolResult("c", "tool", "ok"),
                        Usage(1, 2),
                    ]
                    await store.append("a", items)
                    self.assertEqual(await store.load("a"), items)
                    await store.set_active_agent("a", "specialist")
                    await asyncio.gather(
                        *(
                            store.append("a", [Message("user", str(i))])
                            for i in range(30)
                        )
                    )
                    loaded = await store.load("a")
                    self.assertEqual(loaded[:4], items)
                    self.assertCountEqual(
                        # pyrefly: ignore [missing-attribute]
                        [i.content for i in loaded[4:]],
                        [str(i) for i in range(30)],
                    )
                    self.assertEqual(await store.get_active_agent("a"), "specialist")
                    self.assertEqual(
                        await store.compact("a", lambda history: history[:4]), items
                    )
                    self.assertEqual(await store.get_active_agent("a"), "specialist")
                    await store.clear("a")
                    self.assertEqual(await store.load("a"), [])
                    self.assertIsNone(await store.get_active_agent("a"))
                close = getattr(store, "close", None)
                if close:
                    await close()

    async def test_append_turn_atomic_and_concurrent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for store in (
                FileMemoryStore(Path(directory) / "files"),
                SQLiteMemoryStore(Path(directory) / "db"),
            ):
                with (
                    patch.object(
                        store, "append", side_effect=AssertionError("split append")
                    ),
                    patch.object(
                        store,
                        "set_active_agent",
                        side_effect=AssertionError("split state"),
                    ),
                ):
                    await asyncio.gather(
                        *(
                            store.append_turn(
                                "a", [Message("assistant", str(i))], str(i)
                            )
                            for i in range(16)
                        )
                    )
                loaded = await store.load("a")
                self.assertCountEqual(
                    # pyrefly: ignore [missing-attribute]
                    [item.content for item in loaded],
                    [str(i) for i in range(16)],
                )
                # pyrefly: ignore [missing-attribute]
                self.assertEqual(await store.get_active_agent("a"), loaded[-1].content)
                if isinstance(store, SQLiteMemoryStore):
                    await store.close()

    async def test_append_turn_failure_preserves_history_and_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Message("assistant", "original")
            added = Message("assistant", "new")
            files = FileMemoryStore(Path(directory) / "files")
            await files.append_turn("a", [original], "original")
            with patch(
                "cppilot.local.os.replace", side_effect=OSError("replace failed")
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    await files.append_turn("a", [added], "new")
            self.assertEqual(await files.load("a"), [original])
            self.assertEqual(await files.get_active_agent("a"), "original")
            database = SQLiteMemoryStore(Path(directory) / "db")
            await database.append_turn("a", [original], "original")
            # Fail after the item insert, proving the transaction rolls it back.
            with database.connect() as connection:
                connection.execute(
                    "CREATE TRIGGER fail_state BEFORE UPDATE OF active_agent ON memory_state BEGIN SELECT RAISE(ABORT, 'state failed'); END"
                )
            with self.assertRaisesRegex(sqlite3.IntegrityError, "state failed"):
                await database.append_turn("a", [added], "new")
            self.assertEqual(await database.load("a"), [original])
            self.assertEqual(await database.get_active_agent("a"), "original")
            await database.close()

    async def test_separate_store_instances_do_not_lose_appends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for factory in (
                lambda: FileMemoryStore(Path(directory) / "files"),
                lambda: SQLiteMemoryStore(Path(directory) / "db"),
            ):
                stores = [factory() for _ in range(8)]
                await asyncio.gather(
                    *(
                        store.append("a", [Message("user", str(i))])
                        for i, store in enumerate(stores)
                    )
                )
                self.assertCountEqual(
                    # pyrefly: ignore [missing-attribute]
                    [i.content for i in await stores[0].load("a")],
                    [str(i) for i in range(8)],
                )
                for store in stores:
                    if isinstance(store, SQLiteMemoryStore):
                        await store.close()

    async def test_compaction_conflict_and_failure_preserve_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for store in (
                FileMemoryStore(Path(directory) / "files"),
                SQLiteMemoryStore(Path(directory) / "db"),
            ):
                original = Message("user", "original")
                added = Message("user", "concurrent")
                await store.append("a", [original])
                entered, release = asyncio.Event(), asyncio.Event()

                async def compact(history, entered=entered, release=release):
                    entered.set()
                    await release.wait()
                    return []

                task = asyncio.create_task(store.compact("a", compact))
                await entered.wait()
                await store.append("a", [added])
                release.set()
                with self.assertRaises(ConcurrentModificationError):
                    await task
                self.assertEqual(await store.load("a"), [original, added])

                def fail(history):
                    raise ValueError("compactor failed")

                with self.assertRaisesRegex(ValueError, "compactor failed"):
                    await store.compact("a", fail)
                self.assertEqual(await store.load("a"), [original, added])
                if isinstance(store, SQLiteMemoryStore):
                    await store.close()

    async def test_session_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for store in (
                FileSessionStore(Path(directory) / "sessions"),
                SQLiteSessionStore(Path(directory) / "db"),
                SQLiteSessionStore(":memory:"),
            ):
                session = await store.create("router", "a", {"workflow": "test"})
                self.assertEqual(session.active_agent, "router")
                updated = replace(session, active_agent="specialist")
                await store.update(updated)
                self.assertEqual(await store.load(session.id), updated)
                self.assertEqual(await store.list(), [updated])
                await store.delete(session.id)
                self.assertIsNone(await store.load(session.id))
                if isinstance(store, SQLiteSessionStore):
                    await store.close()

    async def test_legacy_files_and_invalid_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = Message("user", "legacy")
            (root / "old.jsonl").write_text(json.dumps(item_to_dict(original)) + "\n")
            store = FileMemoryStore(root)
            self.assertEqual(await store.load("old"), [original])
            await store.set_active_agent("old", "helper")
            self.assertEqual(await store.load("old"), [original])
            for value in ("", "../outside", "a/b"):
                with self.assertRaises(ValueError):
                    await store.append(value, [])
            with self.assertRaises(ValueError):
                await FileSessionStore(root).delete("../outside")

    async def test_filesystem_write_does_not_block_event_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entered, release = threading.Event(), threading.Event()

            def slow_write(path, content):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test did not release writer")
                _atomic_write(path, content)

            store = FileMemoryStore(Path(directory))
            with patch("cppilot.local._atomic_write", side_effect=slow_write):
                task = asyncio.create_task(
                    store.append("a", [Message("user", "written")])
                )
                try:
                    await asyncio.wait_for(asyncio.to_thread(entered.wait, 5), 6)
                    self.assertFalse(task.done())
                    self.assertEqual(await store.load("a"), [])
                finally:
                    release.set()
                    await task
            self.assertEqual(await store.load("a"), [Message("user", "written")])

    async def test_sqlite_closed_store(self) -> None:
        store = SQLiteMemoryStore(":memory:")
        await store.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await store.load("a")

    def test_postgres_dependency_error_is_lazy(self) -> None:
        with self.assertRaises(ValueError):
            PostgresMemoryStore("sqlite://")
        if importlib.util.find_spec("sqlalchemy") is None:
            with self.assertRaisesRegex(RuntimeError, "SQLAlchemy"):
                PostgresMemoryStore("postgresql+asyncpg://localhost/test")


class SettingsTest(unittest.TestCase):
    def test_toml_json_and_environment_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            toml = root / "config.toml"
            toml.write_text(
                '[cppilot]\nstorage = "file"\nprovider = "fake"\nplain = false\n[cppilot.models.fast]\nprovider = "fake"\nmodel = "small"\n'
            )
            loaded = Settings.load(
                toml,
                environ={
                    "CPPILOT_STORAGE": "sqlite",
                    "CPPILOT_DATABASE": "override.db",
                    "CPPILOT_PLAIN": "yes",
                },
            )
            self.assertEqual(loaded.storage, "sqlite")
            self.assertEqual(loaded.database, "override.db")
            self.assertTrue(loaded.plain)
            self.assertEqual(
                loaded.models, {"fast": {"provider": "fake", "model": "small"}}
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {"storage": "file", "memory_root": "custom", "model": "fast"}
                )
            )
            loaded = Settings.load(
                environ={
                    "CPPILOT_CONFIG": str(config),
                    "CPPILOT_MODELS": '{"fast": {"provider": "fake"}}',
                }
            )
            self.assertEqual(loaded.memory_root, Path("custom"))
            self.assertEqual(loaded.model, "fast")
            self.assertEqual(Settings.load(environ={}).storage, "sqlite")

    def test_invalid_settings_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for value in (
                [],
                {"unknown": 1},
                {"plain": "false"},
                {"database": 12},
                {"models": {"x": {"provider": "x"}}},
                {"cppilot": []},
            ):
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(ValueError):
                    Settings.load(path, environ={})
            with self.assertRaisesRegex(ValueError, "true or false"):
                Settings.load(environ={"CPPILOT_STREAM": "invalid"})
            with self.assertRaises(FileNotFoundError):
                Settings.load(Path(directory) / "missing.toml", environ={})


class EngineResult:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def first(self):
        return self.rows[0] if self.rows else None

    def one(self):
        if len(self.rows) != 1:
            raise AssertionError("expected one database row")
        return self.rows[0]

    def __iter__(self):
        return iter(self.rows)


class ContractEngine:
    """SQLAlchemy-shaped engine backed by real SQLite transactions.

    Only PostgreSQL locking syntax is translated; CRUD and revision SQL execute.
    This tests store behavior without an installed SDK or a network database.
    """

    def __init__(self):
        self.database = sqlite3.connect(":memory:")
        self.lock = asyncio.Lock()
        self.sql = []
        self.rollbacks = 0
        self.disposed = False
        self.fail_update = False

    @asynccontextmanager
    async def begin(self):
        async with self.lock:
            self.database.execute("BEGIN")
            try:
                yield self
                self.database.commit()
            except BaseException:
                self.database.rollback()
                self.rollbacks += 1
                raise

    @asynccontextmanager
    async def connect(self):
        async with self.lock:
            yield self

    async def execute(self, statement, parameters=None):
        sql = str(statement)
        self.sql.append(sql)
        if "pg_advisory_xact_lock" in sql:
            return EngineResult()
        if self.fail_update and sql.startswith("UPDATE cppilot_memory"):
            raise RuntimeError("injected update failure")
        cursor = self.database.execute(
            sql.removesuffix(" FOR UPDATE"), parameters or {}
        )
        return EngineResult(cursor.fetchall())

    async def commit(self):
        self.database.commit()

    async def rollback(self):
        self.database.rollback()
        self.rollbacks += 1

    async def dispose(self):
        self.disposed = True
        self.database.close()


class PostgresContractTest(unittest.IsolatedAsyncioTestCase):
    def stores(self, engine, url="postgresql+asyncpg://test"):
        def module(name):
            if name == "sqlalchemy":
                return SimpleNamespace(text=lambda sql: sql)
            if name == "sqlalchemy.ext.asyncio":
                return SimpleNamespace(create_async_engine=lambda url: engine)
            raise ImportError(name)

        with patch("cppilot.storage.importlib.import_module", side_effect=module):
            return PostgresMemoryStore(url, engine=engine), PostgresSessionStore(
                url, engine=engine
            )

    async def test_append_turn_commit_and_rollback_both_dialects(self) -> None:
        for url in ("postgresql+asyncpg://test", "sqlite+aiosqlite:///:memory:"):
            engine = ContractEngine()
            memory, sessions = self.stores(engine, url)
            try:
                original, added = (
                    Message("assistant", "original"),
                    Message("assistant", "new"),
                )
                with (
                    patch.object(
                        memory, "append", side_effect=AssertionError("split append")
                    ),
                    patch.object(
                        memory,
                        "set_active_agent",
                        side_effect=AssertionError("split state"),
                    ),
                ):
                    await memory.append_turn("a", [original], "original")
                self.assertEqual(await memory.load("a"), [original])
                self.assertEqual(await memory.get_active_agent("a"), "original")
                engine.fail_update = True
                with self.assertRaisesRegex(RuntimeError, "injected update failure"):
                    await memory.append_turn("a", [added], "new")
                engine.fail_update = False
                self.assertEqual(await memory.load("a"), [original])
                self.assertEqual(await memory.get_active_agent("a"), "original")
                await asyncio.gather(
                    *(
                        memory.append_turn("a", [Message("assistant", str(i))], str(i))
                        for i in range(8)
                    )
                )
                loaded = await memory.load("a")
                self.assertEqual(await memory.get_active_agent("a"), loaded[-1].content)
                self.assertEqual(len(loaded), 9)
                session = await sessions.create("a", "a")
                self.assertEqual(await sessions.load(session.id), session)
                await sessions.delete(session.id)
                self.assertEqual(await sessions.list(), [])
                if url.startswith("sqlite"):
                    self.assertIn("BEGIN IMMEDIATE", engine.sql)
                    self.assertFalse(
                        any(
                            "FOR UPDATE" in sql or "pg_advisory" in sql
                            for sql in engine.sql
                        )
                    )
                else:
                    self.assertTrue(any("FOR UPDATE" in sql for sql in engine.sql))
                self.assertEqual(engine.rollbacks, 1)
            finally:
                await engine.dispose()

    @unittest.skipUnless(
        importlib.util.find_spec("sqlalchemy") is not None
        and importlib.util.find_spec("aiosqlite") is not None,
        "storage extra not installed",
    )
    async def test_real_sqlalchemy_sqlite_engine(self) -> None:
        from cppilot.storage import SQLAlchemyMemoryStore, SQLAlchemySessionStore

        with tempfile.TemporaryDirectory() as directory:
            url = "sqlite+aiosqlite:///" + str(Path(directory) / "async.db")
            memory, sessions = SQLAlchemyMemoryStore(url), SQLAlchemySessionStore(url)
            try:
                await asyncio.gather(
                    *(
                        memory.append_turn("a", [Message("assistant", str(i))], str(i))
                        for i in range(16)
                    )
                )
                loaded = await memory.load("a")
                self.assertEqual(len(loaded), 16)
                # pyrefly: ignore [missing-attribute]
                self.assertEqual(await memory.get_active_agent("a"), loaded[-1].content)
                self.assertEqual(
                    await memory.compact("a", lambda history: history[-1:]), loaded[-1:]
                )
                session = await sessions.create("agent", "a")
                self.assertEqual(await sessions.load(session.id), session)
                await sessions.delete(session.id)
                self.assertEqual(await sessions.list(), [])
            finally:
                await memory.close()
                await sessions.close()

    async def test_memory_and_session_contracts(self) -> None:
        engine = ContractEngine()
        memory, sessions = self.stores(engine)
        try:
            self.assertEqual(await memory.load("a"), [])
            items = [
                Message("user", "hello"),
                ToolCall("c", "tool", {}),
                ToolResult("c", "tool", "ok"),
                Usage(1, 2),
            ]
            await memory.append("a", items)
            await memory.set_active_agent("a", "specialist")
            await asyncio.gather(
                *(memory.append("a", [Message("user", str(i))]) for i in range(8))
            )
            loaded = await memory.load("a")
            self.assertEqual(loaded[:4], items)
            self.assertCountEqual(
                [i.content for i in loaded[4:]], [str(i) for i in range(8)]
            )

            async def compact(history):
                return history[:4]

            self.assertEqual(await memory.compact("a", compact), items)
            self.assertEqual(await memory.get_active_agent("a"), "specialist")
            session = await sessions.create("router", "a", {"model": "small"})
            updated = replace(session, active_agent="specialist")
            await sessions.update(updated)
            self.assertEqual(await sessions.load(session.id), updated)
            self.assertEqual(await sessions.list(), [updated])
            await sessions.delete(session.id)
            self.assertIsNone(await sessions.load(session.id))
            await memory.clear("a")
            self.assertEqual(await memory.load("a"), [])
            self.assertIsNone(await memory.get_active_agent("a"))
            self.assertTrue(any("FOR UPDATE" in sql for sql in engine.sql))
            self.assertEqual(
                sum("pg_advisory_xact_lock" in sql for sql in engine.sql), 2
            )
            await memory.close()
            await sessions.close()
            self.assertFalse(engine.disposed)
        finally:
            await engine.dispose()

    async def test_conflicts_and_update_failure_rollback(self) -> None:
        engine = ContractEngine()
        memory, _ = self.stores(engine)
        try:
            original = Message("user", "original")
            added = Message("user", "new")
            await memory.append("a", [original])
            entered, release = asyncio.Event(), asyncio.Event()

            async def compact(history):
                entered.set()
                await release.wait()
                return []

            task = asyncio.create_task(memory.compact("a", compact))
            await entered.wait()
            await memory.append("a", [added])
            release.set()
            with self.assertRaises(ConcurrentModificationError):
                await task
            self.assertEqual(await memory.load("a"), [original, added])
            engine.fail_update = True
            with self.assertRaisesRegex(RuntimeError, "injected update failure"):
                await memory.clear("a")
            engine.fail_update = False
            self.assertEqual(await memory.load("a"), [original, added])
            self.assertEqual(engine.rollbacks, 2)
        finally:
            await engine.dispose()


class Responses(ModelProvider):
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeSandbox(Sandbox):
    def __init__(self):
        self.started = 0
        self.stopped = []

    async def start(self):
        self.started += 1
        return "sandbox"

    async def execute(self, sandbox_id, command, cwd=None):
        return ExecutionResult(0, "", "")

    async def stop(self, sandbox_id):
        self.stopped.append(sandbox_id)


class WorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_resume_saved_destination_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = FileMemoryStore(Path(directory) / "memory")
            sessions = FileSessionStore(Path(directory) / "sessions")
            specialist_provider = Responses(
                [
                    ModelResponse([Message("assistant", "first")]),
                    ModelResponse([Message("assistant", "resumed")]),
                ]
            )
            specialist = Agent("specialist", "Specialist.", specialist_provider)
            router = Agent(
                "router",
                "Route.",
                Responses(
                    [ModelResponse([ToolCall("h", "transfer_to_specialist", {})])]
                ),
                handoffs=[specialist],
            )
            session = await sessions.create("router", "a")
            flow = AgentWorkflow(
                router, memory=memory, sessions=sessions, session=session
            )
            self.assertEqual((await flow.run("first")).output, "first")
            saved = await sessions.load(session.id)
            # pyrefly: ignore [missing-attribute]
            self.assertEqual(saved.active_agent, "specialist")
            resumed = AgentWorkflow(
                router, memory=memory, sessions=sessions, session=saved
            )
            self.assertEqual((await resumed.run("next")).output, "resumed")
            self.assertEqual(len(specialist_provider.requests), 2)
            await resumed.clear()
            self.assertIsNone(await memory.get_active_agent("a"))
            # pyrefly: ignore [missing-attribute]
            self.assertEqual((await sessions.load(session.id)).active_agent, "router")

    async def test_switch_model_preserves_destination_and_original_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = FileMemoryStore(Path(directory) / "memory")
            sessions = FileSessionStore(Path(directory) / "sessions")
            original = Responses([])
            replacement = Responses([ModelResponse([Message("assistant", "switched")])])
            specialist = Agent("specialist", "Specialist.", original)
            root = Agent("router", "Route.", original, handoffs=[specialist])
            session = await sessions.create("router", "a")
            session = replace(session, active_agent="specialist")
            await sessions.update(session)
            await memory.set_active_agent("a", "specialist")
            flow = AgentWorkflow(
                root, memory=memory, sessions=sessions, session=session
            )
            await flow.set_model(
                replacement, {"provider": "replacement", "model": "new"}
            )
            result = await flow.run("next")
            self.assertEqual(result.agent_name, "specialist")
            self.assertEqual(result.output, "switched")
            self.assertIs(root.provider, original)
            self.assertIs(specialist.provider, original)
            saved = await sessions.load(session.id)
            # pyrefly: ignore [missing-attribute]
            self.assertEqual(saved.metadata["model"], "new")
            # pyrefly: ignore [missing-attribute]
            self.assertEqual(saved.active_agent, "specialist")

    async def test_context_cleanup_on_failure_and_reentry(self) -> None:
        sandbox = FakeSandbox()
        manager = WorkflowContext(sandbox=sandbox, read_roots=(Path("/tmp"),))
        task = None
        with self.assertRaisesRegex(ValueError, "failed"):
            async with manager as context:
                self.assertEqual(context.values["sandbox_id"], "sandbox")
                self.assertEqual(context.session["sandbox_id"], "sandbox")
                task = asyncio.create_task(asyncio.sleep(100))
                context.session["background_tasks"] = {"task": task}
                raise ValueError("failed")
        self.assertTrue(task.cancelled())
        self.assertEqual(sandbox.stopped, ["sandbox"])
        async with manager as context:
            self.assertFalse(context.cancelled.is_set())
        self.assertEqual(sandbox.started, 2)

    async def test_stream_close_cleans_sandbox_and_restores_agent(self) -> None:
        sandbox = FakeSandbox()
        agent = Agent(
            "a", "Answer.", Responses([ModelResponse([Message("assistant", "answer")])])
        )
        previous = agent.context
        flow = AgentWorkflow(agent, context=WorkflowContext(sandbox=sandbox))
        stream = flow.stream("go")
        self.assertIsNotNone(await anext(stream))
        # pyrefly: ignore [missing-attribute]
        await stream.aclose()
        self.assertEqual(sandbox.stopped, ["sandbox"])
        self.assertIs(agent.context, previous)

    def test_factory_key_error_is_preserved(self) -> None:
        registry = WorkflowRegistry()

        def factory(**configuration):
            raise KeyError("missing configuration")

        registry.register("test", factory)
        with self.assertRaisesRegex(KeyError, "missing configuration"):
            registry.create("test")
        self.assertEqual(registry.names(), ["test"])
