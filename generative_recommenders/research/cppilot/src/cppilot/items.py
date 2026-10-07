# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    type: Literal["message"] = "message"


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    failed: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    type: Literal["usage"] = "usage"


RunItem: TypeAlias = Message | ToolCall | ToolResult | Usage


def item_to_dict(item: RunItem) -> dict[str, Any]:
    return asdict(item)


def item_from_dict(value: dict[str, Any]) -> RunItem:
    kind = value.get("type")
    classes = {
        "message": Message,
        "tool_call": ToolCall,
        "tool_result": ToolResult,
        "usage": Usage,
    }
    try:
        cls = classes[kind]
    except KeyError as error:
        raise ValueError(f"unknown run item type: {kind!r}") from error
    return cls(**value)
