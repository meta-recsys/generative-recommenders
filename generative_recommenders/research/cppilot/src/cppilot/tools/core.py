# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints

from ..interfaces import ToolDefinition


@dataclass(slots=True)
class RunContext:
    values: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any], context: RunContext) -> str: ...


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}
    return {
        "type": {str: "string", int: "integer", float: "number", bool: "boolean"}.get(
            annotation, "string"
        )
    }


class FunctionTool(Tool):
    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.function = function
        self.signature = inspect.signature(function)
        hints = get_type_hints(function)
        first = next(iter(self.signature.parameters.items()), None)
        self._takes_context = (
            first is not None and hints.get(first[0], first[1].annotation) is RunContext
        )
        properties: dict[str, Any] = {}
        required: list[str] = []
        for index, (param_name, param) in enumerate(self.signature.parameters.items()):
            if index == 0 and self._takes_context:
                continue
            properties[param_name] = _json_type(hints.get(param_name, str))
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        self._definition = ToolDefinition(
            name=name or function.__name__,
            description=description or inspect.getdoc(function) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def invoke(self, arguments: dict[str, Any], context: RunContext) -> str:
        args = (context,) if self._takes_context else ()
        result = self.function(*args, **arguments)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)


def function_tool(function: Callable[..., Any]) -> FunctionTool:
    return FunctionTool(function)
