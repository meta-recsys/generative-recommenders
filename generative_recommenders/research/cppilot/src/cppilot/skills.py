# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import importlib
import json
import tomllib
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tools.core import FunctionTool, RunContext


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str
    root: Path


def _unique_pairs(pairs: Sequence[tuple[Any, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("frontmatter keys must be strings")
        if key in result:
            raise ValueError(f"duplicate frontmatter field: {key}")
        result[key] = value
    return result


def _yaml_metadata(header: str) -> Any:
    try:
        yaml = importlib.import_module("yaml")
    except ImportError as error:
        raise RuntimeError(
            "YAML skill frontmatter requires optional PyYAML; use JSON (---) or TOML (+++) without dependencies"
        ) from error

    from yaml import SafeLoader
    from yaml.nodes import MappingNode

    class UniqueSafeLoader(SafeLoader):
        def construct_mapping(
            self, node: MappingNode, deep: bool = False
        ) -> dict[Hashable, Any]:
            self.flatten_mapping(node)
            pairs = [
                (
                    self.construct_object(key, deep=deep),
                    self.construct_object(value, deep=deep),
                )
                for key, value in node.value
            ]
            try:
                mapping: dict[Hashable, Any] = {}
                mapping.update(_unique_pairs(pairs))
                return mapping
            except TypeError as error:
                raise ValueError("frontmatter keys must be scalar strings") from error

    try:
        return yaml.load(header, Loader=UniqueSafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(str(error)) from error


class SkillLoader:
    """Load JSON/YAML frontmatter between --- lines or TOML between +++ lines.

    JSON and TOML use stdlib parsers. YAML requires optional PyYAML and uses
    SafeLoader, with duplicate keys rejected. No handwritten YAML fallback is
    used: install the optional dependency or convert metadata to JSON/TOML.
    """

    def __init__(self, roots: list[Path], allowed: set[str] | None = None) -> None:
        self.skills: dict[str, Skill] = {}
        for configured in roots:
            root = Path(configured).resolve()
            if not root.exists():
                continue
            for path in sorted(root.rglob("SKILL.md")):
                resolved = path.resolve()
                if not resolved.is_relative_to(root) or any(
                    parent.is_symlink()
                    for parent in (path, *path.parents)
                    if parent.is_relative_to(root)
                ):
                    continue
                with RunContext(read_roots=(root,)).open_path(path) as stream:
                    skill = self._parse(resolved, text=stream.read())
                if skill and (allowed is None or skill.name in allowed):
                    if skill.name in self.skills:
                        raise ValueError(f"duplicate skill name: {skill.name}")
                    self.skills[skill.name] = skill

    @classmethod
    def from_skills(cls, skills: Sequence[Skill]) -> SkillLoader:
        loader = cls([])
        for skill in skills:
            if not isinstance(skill, Skill):
                raise TypeError("skills must contain Skill instances")
            if skill.name in loader.skills:
                raise ValueError(f"duplicate skill name: {skill.name}")
            loader.skills[skill.name] = skill
        return loader

    @staticmethod
    def _parse(path: Path, *, text: str | None = None) -> Skill | None:
        if text is None:
            text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0] not in {"---", "+++"}:
            return None
        delimiter = lines[0]
        try:
            closing = lines.index(delimiter, 1)
        except ValueError as error:
            raise ValueError(f"{path}: unterminated skill frontmatter") from error
        header = "\n".join(lines[1:closing])
        body = "\n".join(lines[closing + 1 :])
        try:
            if delimiter == "+++":
                metadata = tomllib.loads(header)
            elif header.lstrip().startswith(("{", "[")):
                metadata = json.loads(header, object_pairs_hook=_unique_pairs)
            else:
                metadata = _yaml_metadata(header)
        except ValueError as error:
            raise ValueError(f"{path}: invalid skill frontmatter: {error}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: skill frontmatter must be an object")
        name, description = metadata.get("name"), metadata.get("description")
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not name.strip()
            or not description.strip()
        ):
            raise ValueError(f"{path}: name and description must be nonempty strings")
        return Skill(name.strip(), description.strip(), body.strip(), path.parent)

    def load(self, name: str) -> Skill:
        try:
            return self.skills[name]
        except KeyError as error:
            raise KeyError(f"unknown skill {name!r}") from error

    def tool(self) -> FunctionTool:
        """Return the single allowlisted tool exposed to an agent."""

        def load_skill(name: str) -> str:
            """Load the complete instructions for an available skill by name."""
            return self.load(name).instructions

        tool = FunctionTool(
            load_skill,
            name="load_skill",
            system_prompt="Use load_skill to read available skill instructions before applying them.",
        )
        tool.skill_loader = self
        return tool

    def available(self) -> list[Skill]:
        return [self.skills[name] for name in sorted(self.skills)]
