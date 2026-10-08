# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .interfaces import ModelProvider
from .skills import SkillLoader
from .tools.core import RunContext, Tool

Guardrail = Callable[[str, RunContext], bool | Awaitable[bool]]
Instructions: TypeAlias = str | Callable[[RunContext], str | Awaitable[str]]


@dataclass(slots=True)
class Agent:
    name: str
    instructions: Instructions
    provider: ModelProvider
    tools: Sequence[Tool] = ()
    handoffs: Sequence["Agent"] = ()
    input_guardrails: Sequence[Guardrail] = ()
    output_guardrails: Sequence[Guardrail] = ()
    output_schema: dict[str, Any] | None = None
    context: RunContext = field(default_factory=RunContext)
    skills: Sequence[Any] | SkillLoader = ()
    skill_loader: SkillLoader | None = None
    output_type: Any | None = None

    def __post_init__(self) -> None:
        if self.output_type is not None:
            if not hasattr(self.output_type, "model_json_schema"):
                raise TypeError("output_type must be a Pydantic v2 model class")
            if self.output_schema is None:
                self.output_schema = self.output_type.model_json_schema()
        loader = self.skill_loader
        if isinstance(self.skills, SkillLoader):
            if loader is not None and loader is not self.skills:
                raise ValueError("configure only one skill loader")
            loader = self.skills
        if loader is None and self.skills and not isinstance(self.skills, SkillLoader):
            loader = SkillLoader.from_skills(self.skills)
        if loader is None:
            return
        self.skill_loader = loader
        self.skills = tuple(loader.available())
        existing = [tool for tool in self.tools if tool.definition.name == "load_skill"]
        if existing and not any(
            getattr(tool, "skill_loader", None) is loader for tool in existing
        ):
            raise ValueError(
                "load_skill tool name is reserved for the configured loader"
            )
        if not existing:
            self.tools = (*self.tools, loader.tool())
