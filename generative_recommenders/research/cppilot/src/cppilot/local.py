# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from .interfaces import Compactor, MemoryStore, Session, SessionStore
from .items import item_from_dict, item_to_dict, RunItem


class ConcurrentModificationError(RuntimeError):
    """History changed while a compactor was running; retry with fresh history."""


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    # Lock a stable sibling inode: locking the data inode fails after replace().
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _decode(text: str) -> tuple[list[RunItem], str | None]:
    items: list[RunItem] = []
    active = None
    for line in text.splitlines():
        if line:
            value = json.loads(line)
            if "_cppilot_memory" in value:
                active = value.get("active_agent")
            else:
                items.append(item_from_dict(value))
    return items, active


def _encode(items: Sequence[RunItem], active: str | None) -> str:
    # The header keeps history, active agent, and revision in one atomic file.
    header = {"_cppilot_memory": uuid.uuid4().hex, "active_agent": active}
    return (
        "\n".join(
            json.dumps(value) for value in [header, *(item_to_dict(i) for i in items)]
        )
        + "\n"
    )


class FileMemoryStore(MemoryStore):
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

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
        return _decode(await asyncio.to_thread(_read, self._path(conversation_id)))[0]

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
        path = self._path(conversation_id)
        additions = list(items)

        def operation() -> None:
            with _file_lock(path):
                history, active = _decode(_read(path))
                _atomic_write(
                    path,
                    _encode(
                        [*history, *additions],
                        active if agent_name is None else agent_name,
                    ),
                )

        await asyncio.to_thread(operation)

    async def clear(self, conversation_id: str) -> None:
        path = self._path(conversation_id)

        def operation() -> None:
            with _file_lock(path):
                _atomic_write(path, _encode([], None))

        await asyncio.to_thread(operation)

    async def get_active_agent(self, conversation_id: str) -> str | None:
        return _decode(await asyncio.to_thread(_read, self._path(conversation_id)))[1]

    async def set_active_agent(
        self, conversation_id: str, agent_name: str | None
    ) -> None:
        path = self._path(conversation_id)

        def operation() -> None:
            with _file_lock(path):
                history, _ = _decode(_read(path))
                _atomic_write(path, _encode(history, agent_name))

        await asyncio.to_thread(operation)

    async def compact(
        self, conversation_id: str, compactor: Compactor
    ) -> list[RunItem]:
        path = self._path(conversation_id)
        snapshot = await asyncio.to_thread(_read, path)
        history, active = _decode(snapshot)
        result = compactor(history)
        if inspect.isawaitable(result):
            result = await result
        content = _encode(result, active)

        def operation() -> None:
            with _file_lock(path):
                if _read(path) != snapshot:
                    raise ConcurrentModificationError(
                        "memory changed during compaction"
                    )
                _atomic_write(path, content)

        await asyncio.to_thread(operation)
        return result


class FileSessionStore(SessionStore):
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, session_id: str) -> Path:
        try:
            canonical = str(uuid.UUID(session_id))
        except (ValueError, AttributeError) as error:
            raise ValueError("invalid session id") from error
        return self.root / f"{canonical}.json"

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
        text = await asyncio.to_thread(_read, self._path(session_id))
        return Session(**json.loads(text)) if text else None

    async def update(self, session: Session) -> None:
        path = self._path(session.id)
        content = json.dumps(asdict(session), indent=2)

        def operation() -> None:
            with _file_lock(path):
                _atomic_write(path, content)

        await asyncio.to_thread(operation)

    async def list(self) -> list[Session]:
        def operation() -> list[Session]:
            sessions = []
            for path in self.root.glob("*.json"):
                text = _read(path)
                if text:
                    sessions.append(Session(**json.loads(text)))
            return sorted(sessions, key=lambda session: session.id)

        return await asyncio.to_thread(operation)

    async def delete(self, session_id: str) -> None:
        path = self._path(session_id)

        def operation() -> None:
            with _file_lock(path):
                path.unlink(missing_ok=True)

        await asyncio.to_thread(operation)
