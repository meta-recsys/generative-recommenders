# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from .interfaces import Compactor, MemoryStore, Session, SessionStore
from .items import item_from_dict, item_to_dict, RunItem
from .local import ConcurrentModificationError


class _SQLiteDatabase:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._closed = False
        # Keep a private connection alive for :memory: stores across worker threads.
        self._keeper = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if self.path == ":memory:"
            else None
        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("store is closed")
            if self._keeper is None:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            connection = self._keeper or sqlite3.connect(self.path, timeout=30)
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS memory (conversation TEXT, position INTEGER, item TEXT, PRIMARY KEY(conversation, position))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, value TEXT)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS memory_state (conversation TEXT PRIMARY KEY, revision INTEGER NOT NULL, active_agent TEXT)"
                )
                connection.commit()
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                if connection is not self._keeper:
                    connection.close()

    def execute(self, sql: str, parameters: tuple[Any, ...]) -> None:
        with self.connect() as database:
            database.execute(sql, parameters)

    async def close(self) -> None:
        def operation() -> None:
            with self._lock:
                self._closed = True
                if self._keeper is not None:
                    self._keeper.close()
                    self._keeper = None

        await asyncio.to_thread(operation)


class SQLiteMemoryStore(_SQLiteDatabase, MemoryStore):
    def _snapshot(self, conversation_id: str) -> tuple[list[RunItem], int]:
        with self.connect() as database:
            database.execute("BEGIN")
            state = database.execute(
                "SELECT revision FROM memory_state WHERE conversation = ?",
                (conversation_id,),
            ).fetchone()
            rows = database.execute(
                "SELECT item FROM memory WHERE conversation = ? ORDER BY position",
                (conversation_id,),
            ).fetchall()
            return [item_from_dict(json.loads(row[0])) for row in rows], state[
                0
            ] if state else 0

    @staticmethod
    def _bump(database: sqlite3.Connection, conversation_id: str) -> None:
        database.execute(
            "INSERT INTO memory_state VALUES (?, 1, NULL) ON CONFLICT(conversation) DO UPDATE SET revision = revision + 1",
            (conversation_id,),
        )

    async def load(self, conversation_id: str) -> list[RunItem]:
        return (await asyncio.to_thread(self._snapshot, conversation_id))[0]

    async def append(self, conversation_id: str, items: Sequence[RunItem]) -> None:
        await self._append(conversation_id, items)

    async def append_turn(
        self, conversation_id: str, items: Sequence[RunItem], agent_name: str
    ) -> None:
        await self._append(conversation_id, items, agent_name)

    async def _append(
        self,
        conversation_id: str,
        items: Sequence[RunItem],
        agent_name: str | None = None,
    ) -> None:
        serialized = [json.dumps(item_to_dict(item)) for item in items]

        def operation() -> None:
            with self.connect() as database:
                database.execute("BEGIN IMMEDIATE")
                row = database.execute(
                    "SELECT COALESCE(MAX(position) + 1, 0) FROM memory WHERE conversation = ?",
                    (conversation_id,),
                ).fetchone()
                start = row[0]
                database.executemany(
                    "INSERT INTO memory VALUES (?, ?, ?)",
                    [
                        (conversation_id, start + i, item)
                        for i, item in enumerate(serialized)
                    ],
                )
                self._bump(database, conversation_id)
                if agent_name is not None:
                    database.execute(
                        "UPDATE memory_state SET active_agent = ? WHERE conversation = ?",
                        (agent_name, conversation_id),
                    )

        await asyncio.to_thread(operation)

    async def clear(self, conversation_id: str) -> None:
        def operation() -> None:
            with self.connect() as database:
                database.execute("BEGIN IMMEDIATE")
                database.execute(
                    "DELETE FROM memory WHERE conversation = ?", (conversation_id,)
                )
                self._bump(database, conversation_id)
                database.execute(
                    "UPDATE memory_state SET active_agent = NULL WHERE conversation = ?",
                    (conversation_id,),
                )

        await asyncio.to_thread(operation)

    async def get_active_agent(self, conversation_id: str) -> str | None:
        def operation() -> str | None:
            with self.connect() as database:
                row = database.execute(
                    "SELECT active_agent FROM memory_state WHERE conversation = ?",
                    (conversation_id,),
                ).fetchone()
                return row[0] if row else None

        return await asyncio.to_thread(operation)

    async def set_active_agent(
        self, conversation_id: str, agent_name: str | None
    ) -> None:
        def operation() -> None:
            with self.connect() as database:
                database.execute("BEGIN IMMEDIATE")
                self._bump(database, conversation_id)
                database.execute(
                    "UPDATE memory_state SET active_agent = ? WHERE conversation = ?",
                    (agent_name, conversation_id),
                )

        await asyncio.to_thread(operation)

    async def compact(
        self, conversation_id: str, compactor: Compactor
    ) -> list[RunItem]:
        history, revision = await asyncio.to_thread(self._snapshot, conversation_id)
        result = compactor(history)
        if inspect.isawaitable(result):
            result = await result
        serialized = [json.dumps(item_to_dict(item)) for item in result]

        def operation() -> None:
            with self.connect() as database:
                database.execute("BEGIN IMMEDIATE")
                row = database.execute(
                    "SELECT revision FROM memory_state WHERE conversation = ?",
                    (conversation_id,),
                ).fetchone()
                if (row[0] if row else 0) != revision:
                    raise ConcurrentModificationError(
                        "memory changed during compaction"
                    )
                database.execute(
                    "DELETE FROM memory WHERE conversation = ?", (conversation_id,)
                )
                database.executemany(
                    "INSERT INTO memory VALUES (?, ?, ?)",
                    [(conversation_id, i, item) for i, item in enumerate(serialized)],
                )
                self._bump(database, conversation_id)

        await asyncio.to_thread(operation)
        return result


class SQLiteSessionStore(_SQLiteDatabase, SessionStore):
    async def create(
        self,
        agent_name: str,
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = Session(
            str(uuid.uuid4()),
            agent_name,
            conversation_id,
            dict(metadata or {}),
            agent_name,
        )
        await self.update(session)
        return session

    async def load(self, session_id: str) -> Session | None:
        def operation() -> Session | None:
            with self.connect() as database:
                row = database.execute(
                    "SELECT value FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                return Session(**json.loads(row[0])) if row else None

        return await asyncio.to_thread(operation)

    async def update(self, session: Session) -> None:
        await asyncio.to_thread(
            self.execute,
            "INSERT INTO sessions VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET value = excluded.value",
            (session.id, json.dumps(asdict(session))),
        )

    async def list(self) -> list[Session]:
        def operation() -> list[Session]:
            with self.connect() as database:
                return [
                    Session(**json.loads(row[0]))
                    for row in database.execute(
                        "SELECT value FROM sessions ORDER BY id"
                    )
                ]

        return await asyncio.to_thread(operation)

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(
            self.execute, "DELETE FROM sessions WHERE id = ?", (session_id,)
        )
