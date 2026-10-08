# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .interfaces import MemoryStore, SessionStore
from .local import FileMemoryStore, FileSessionStore
from .sqlite import SQLiteMemoryStore, SQLiteSessionStore


def _load_config(path: Path | str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.suffix.lower() == ".toml":
        values = tomllib.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        values = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("config must be TOML or JSON")
    if not isinstance(values, dict):
        raise ValueError("config must contain an object")
    values = values.get("cppilot", values)
    if not isinstance(values, dict):
        raise ValueError("cppilot config section must contain an object")
    return values


def _environment_value(name: str, value: str) -> Any:
    if name == "models":
        return json.loads(value)
    if name in {"plain", "stream", "load_entry_points"}:
        normalized = value.lower()
        if normalized not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"CPPILOT_{name.upper()} must be true or false")
        return normalized in {"true", "1", "yes"}
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    memory_root: Path = Path(".cppilot/memory")
    session_root: Path = Path(".cppilot/sessions")
    storage: str = "sqlite"
    database: str = ".cppilot/cppilot.db"
    provider: str | None = None
    model: str | None = None
    workflow: str | None = None
    plain: bool = False
    stream: bool = False
    load_entry_points: bool = False
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(
        cls, path: Path | str | None = None, *, environ: Mapping[str, str] | None = None
    ) -> Settings:
        """Load a flat config (or [cppilot] section), then CPPILOT_* overrides.

        No implicit filesystem search: pass a path or set CPPILOT_CONFIG.
        CPPILOT_MODELS is a JSON object of named provider configurations.
        """
        environment = os.environ if environ is None else environ
        selected = path if path is not None else environment.get("CPPILOT_CONFIG")
        values = _load_config(selected) if selected is not None else {}
        names = {entry.name for entry in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        values = dict(values)
        for name in names:
            key = "CPPILOT_" + name.upper()
            if key in environment:
                values[name] = _environment_value(name, environment[key])
        for name in ("memory_root", "session_root"):
            if name in values:
                if not isinstance(values[name], str):
                    raise ValueError(f"{name} must be a string")
                values[name] = Path(values[name]).expanduser()
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.storage, str) or self.storage not in {
            "file",
            "sqlite",
            "postgres",
        }:
            raise ValueError(f"unknown storage backend {self.storage!r}")
        if not isinstance(self.database, str) or not self.database:
            raise ValueError("database must be a nonempty string")
        for name in ("provider", "model", "workflow"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("plain", "stream", "load_entry_points"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.models, dict):
            raise ValueError("models must be an object")
        for name, configuration in self.models.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(configuration, dict)
            ):
                raise ValueError("models must map nonempty names to objects")
            provider = configuration.get("provider")
            if not isinstance(provider, str) or not provider or provider in self.models:
                raise ValueError(
                    f"model {name!r} must reference a provider, not another alias"
                )

    def stores(self) -> tuple[MemoryStore, SessionStore]:
        self.validate()
        if self.storage == "file":
            return FileMemoryStore(self.memory_root), FileSessionStore(
                self.session_root
            )
        if self.storage == "sqlite":
            if self.database == ":memory:":
                raise ValueError(
                    "CLI sessions require a durable database, not :memory:"
                )
            return SQLiteMemoryStore(self.database), SQLiteSessionStore(self.database)
        from .storage import PostgresMemoryStore, PostgresSessionStore

        return PostgresMemoryStore(self.database), PostgresSessionStore(self.database)
