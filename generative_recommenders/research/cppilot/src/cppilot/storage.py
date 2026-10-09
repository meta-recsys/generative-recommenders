# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, cast, Sequence

from .interfaces import Compactor, MemoryStore, Session, SessionStore
from .items import item_from_dict, item_to_dict, RunItem
from .local import ConcurrentModificationError


class _SQLAlchemyDatabase:
    """Optional async PostgreSQL/SQLite storage; no SDK loaded on import."""

    def __init__(self, url: str, *, engine: Any = None) -> None:
        self._sqlite = url.startswith("sqlite+aiosqlite://")
        if not self._sqlite and not url.startswith(
            ("postgresql+asyncpg://", "postgresql+psycopg://")
        ):
            raise ValueError(
                "use postgresql+asyncpg, postgresql+psycopg, or sqlite+aiosqlite URL"
            )
        try:
            sqlalchemy = importlib.import_module("sqlalchemy")
            asynchronous = importlib.import_module("sqlalchemy.ext.asyncio")
        except ImportError as error:
            raise RuntimeError(
                "async storage requires SQLAlchemy[asyncio] and the selected asyncpg, psycopg, or aiosqlite driver"
            ) from error
        self._text = sqlalchemy.text
        self._owned = engine is None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self.engine = (
            engine if engine is not None else asynchronous.create_async_engine(url)
        )

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[Any]:
        if not self._sqlite:
            async with self.engine.begin() as connection:
                yield connection
            return
        # SQLite has no row locks. Reserve the writer before reading the revision.
        async with self._write_lock:
            async with self.engine.connect() as connection:
                try:
                    await connection.execute(self._text("PRAGMA busy_timeout=30000"))
                    await connection.execute(self._text("BEGIN IMMEDIATE"))
                    yield connection
                    await connection.commit()
                except BaseException:
                    await connection.rollback()
                    raise

    @asynccontextmanager
    async def _read_connection(self) -> AsyncIterator[Any]:
        # In-memory SQLite pools can reuse one physical connection for reads/writes.
        if self._sqlite:
            async with self._write_lock:
                async with self.engine.connect() as connection:
                    yield connection
        else:
            async with self.engine.connect() as connection:
                yield connection

    async def initialize(self) -> None:
        async with self._initialization_lock:
            if self._initialized:
                return
            await self._initialize_schema()
            self._initialized = True

    async def _initialize_schema(self) -> None:
        async with self._transaction() as connection:
            # Serialize first-use DDL across processes and both store types.
            if not self._sqlite:
                await connection.execute(
                    self._text("SELECT pg_advisory_xact_lock(1129336916)")
                )
            await connection.execute(
                self._text(
                    "CREATE TABLE IF NOT EXISTS cppilot_memory (conversation TEXT PRIMARY KEY, revision BIGINT NOT NULL, value TEXT NOT NULL)"
                )
            )
            await connection.execute(
                self._text(
                    "CREATE TABLE IF NOT EXISTS cppilot_sessions (id TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
            )

    async def close(self) -> None:
        if self._owned:
            await self.engine.dispose()
        self._initialized = False

    async def __aenter__(self) -> _SQLAlchemyDatabase:
        await self.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class SQLAlchemyMemoryStore(_SQLAlchemyDatabase, MemoryStore):
    async def _snapshot(self, conversation_id: str) -> tuple[dict[str, Any], int]:
        await self.initialize()
        async with self._read_connection() as connection:
            result = await connection.execute(
                self._text(
                    "SELECT value, revision FROM cppilot_memory WHERE conversation = :id"
                ),
                {"id": conversation_id},
            )
            row = result.first()
            return (
                (json.loads(row[0]), row[1])
                if row
                else ({"items": [], "active_agent": None}, 0)
            )

    async def _change(
        self, conversation_id: str, change: Any, *, revision: int | None = None
    ) -> None:
        await self.initialize()
        async with self._transaction() as connection:
            await connection.execute(
                self._text(
                    "INSERT INTO cppilot_memory VALUES (:id, 0, :value) ON CONFLICT(conversation) DO NOTHING"
                ),
                {
                    "id": conversation_id,
                    "value": json.dumps({"items": [], "active_agent": None}),
                },
            )
            result = await connection.execute(
                self._text(
                    "SELECT value, revision FROM cppilot_memory WHERE conversation = :id"
                    + ("" if self._sqlite else " FOR UPDATE")
                ),
                {"id": conversation_id},
            )
            row = result.one()
            if revision is not None and row[1] != revision:
                raise ConcurrentModificationError("memory changed during compaction")
            state = json.loads(row[0])
            change(state)
            await connection.execute(
                self._text(
                    "UPDATE cppilot_memory SET value = :value, revision = revision + 1 WHERE conversation = :id"
                ),
                {"id": conversation_id, "value": json.dumps(state)},
            )

    async def load(self, conversation_id: str) -> list[RunItem]:
        state, _ = await self._snapshot(conversation_id)
        return [item_from_dict(item) for item in state["items"]]

    async def append(self, conversation_id: str, items: Sequence[RunItem]) -> None:
        additions = [item_to_dict(item) for item in items]
        await self._change(
            conversation_id, lambda state: state["items"].extend(additions)
        )

    async def append_turn(
        self, conversation_id: str, items: Sequence[RunItem], agent_name: str
    ) -> None:
        additions = [item_to_dict(item) for item in items]

        def change(state: dict[str, Any]) -> None:
            state["items"].extend(additions)
            state["active_agent"] = agent_name

        await self._change(conversation_id, change)

    async def clear(self, conversation_id: str) -> None:
        await self._change(
            conversation_id, lambda state: state.update(items=[], active_agent=None)
        )

    async def get_active_agent(self, conversation_id: str) -> str | None:
        return cast(
            str | None, (await self._snapshot(conversation_id))[0]["active_agent"]
        )

    async def set_active_agent(
        self, conversation_id: str, agent_name: str | None
    ) -> None:
        await self._change(
            conversation_id, lambda state: state.update(active_agent=agent_name)
        )

    async def compact(
        self, conversation_id: str, compactor: Compactor
    ) -> list[RunItem]:
        state, revision = await self._snapshot(conversation_id)
        result = compactor([item_from_dict(item) for item in state["items"]])
        if inspect.isawaitable(result):
            result = await result
        serialized = [item_to_dict(item) for item in result]
        await self._change(
            conversation_id,
            lambda state: state.update(items=serialized),
            revision=revision,
        )
        return result


class SQLAlchemySessionStore(_SQLAlchemyDatabase, SessionStore):
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
        await self.initialize()
        async with self._read_connection() as connection:
            result = await connection.execute(
                self._text("SELECT value FROM cppilot_sessions WHERE id = :id"),
                {"id": session_id},
            )
            row = result.first()
            return Session(**json.loads(row[0])) if row else None

    async def update(self, session: Session) -> None:
        await self.initialize()
        async with self._transaction() as connection:
            await connection.execute(
                self._text(
                    "INSERT INTO cppilot_sessions VALUES (:id, :value) ON CONFLICT(id) DO UPDATE SET value = EXCLUDED.value"
                ),
                {"id": session.id, "value": json.dumps(asdict(session))},
            )

    async def list(self) -> list[Session]:
        await self.initialize()
        async with self._read_connection() as connection:
            result = await connection.execute(
                self._text("SELECT value FROM cppilot_sessions ORDER BY id")
            )
            return [Session(**json.loads(row[0])) for row in result]

    async def delete(self, session_id: str) -> None:
        await self.initialize()
        async with self._transaction() as connection:
            await connection.execute(
                self._text("DELETE FROM cppilot_sessions WHERE id = :id"),
                {"id": session_id},
            )


PostgresMemoryStore = SQLAlchemyMemoryStore
PostgresSessionStore = SQLAlchemySessionStore
