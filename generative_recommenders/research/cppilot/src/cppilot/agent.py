# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .interfaces import ModelProvider
from .tools.core import RunContext, Tool

Guardrail = Callable[[str, RunContext], bool | Awaitable[bool]]


@dataclass(slots=True)
class Agent:
    name: str
    instructions: str
    provider: ModelProvider
    tools: Sequence[Tool] = ()
    handoffs: Sequence["Agent"] = ()
    input_guardrails: Sequence[Guardrail] = ()
    output_guardrails: Sequence[Guardrail] = ()
    output_schema: dict[str, Any] | None = None
    context: RunContext = field(default_factory=RunContext)
