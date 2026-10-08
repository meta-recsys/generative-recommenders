# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import inspect
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .interfaces import Compactor, MemoryStore, Session, SessionStore
from .items import item_from_dict, item_to_dict, RunItem


class FileMemoryStore(MemoryStore):
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, conversation_id: str) -> Path:
        if not conversation_id or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in conversation_id
        ):
            raise ValueError(
                "conversation_id must contain only letters, digits, '-' and '_'"
            )
        return self.root / f"{conversation_id}.jsonl"

    async def load(self, conversation_id: str) -> list[RunItem]:
        path = self._path(conversation_id)
        if not path.exists():
            return []
        return [
            item_from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    async def append(self, conversation_id: str, items: Sequence[RunItem]) -> None:
        path = self._path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            for item in items:
                output.write(json.dumps(item_to_dict(item)) + "\n")

    async def clear(self, conversation_id: str) -> None:
        self._path(conversation_id).unlink(missing_ok=True)

    async def compact(
        self, conversation_id: str, compactor: Compactor
    ) -> list[RunItem]:
        result = compactor(await self.load(conversation_id))
        if inspect.isawaitable(result):
            result = await result
        path = self._path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(item_to_dict(item)) + "\n" for item in result),
            encoding="utf-8",
        )
        return result


class FileSessionStore(SessionStore):
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, session_id: str) -> Path:
        try:
            uuid.UUID(session_id)
        except ValueError as error:
            raise ValueError("invalid session id") from error
        return self.root / f"{session_id}.json"

    async def create(
        self,
        agent_name: str,
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        session = Session(
            str(uuid.uuid4()), agent_name, conversation_id, metadata or {}
        )
        await self.update(session)
        return session

    async def load(self, session_id: str) -> Session | None:
        path = self._path(session_id)
        return (
            Session(**json.loads(path.read_text(encoding="utf-8")))
            if path.exists()
            else None
        )

    async def update(self, session: Session) -> None:
        path = self._path(session.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(session), indent=2), encoding="utf-8")

    async def list(self) -> list[Session]:
        if not self.root.exists():
            return []
        sessions = [
            Session(**json.loads(path.read_text(encoding="utf-8")))
            for path in self.root.glob("*.json")
        ]
        return sorted(sessions, key=lambda session: session.id)

    async def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
