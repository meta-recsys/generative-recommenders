# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str
    root: Path


class SkillLoader:
    def __init__(self, roots: list[Path], allowed: set[str] | None = None) -> None:
        self.skills: dict[str, Skill] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("SKILL.md")):
                skill = self._parse(path)
                if skill and (allowed is None or skill.name in allowed):
                    self.skills[skill.name] = skill

    @staticmethod
    def _parse(path: Path) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            return None
        try:
            header, body = text[4:].split("\n---", 1)
        except ValueError:
            return None
        fields = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
        name = fields.get("name", "").strip()
        description = fields.get("description", "").strip()
        return (
            Skill(name, description, body.strip(), path.parent)
            if name and description
            else None
        )

    def load(self, name: str) -> Skill:
        try:
            return self.skills[name]
        except KeyError as error:
            raise KeyError(f"unknown skill {name!r}") from error
