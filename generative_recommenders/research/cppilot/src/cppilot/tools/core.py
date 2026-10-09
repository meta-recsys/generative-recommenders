# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import enum
import importlib
import inspect
import json
import os
import re
import stat
import types
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    Literal,
    TextIO,
    TYPE_CHECKING,
    Union,
)

from ..interfaces import ToolDefinition

if TYPE_CHECKING:
    from ..skills import SkillLoader


def _manager(name: str) -> Any:
    # Lazy construction avoids a core/state import cycle and keeps state run-local.
    return getattr(importlib.import_module(".state", __package__), name)()


@dataclass(slots=True)
class RunContext:
    values: dict[str, Any] = field(default_factory=dict)
    read_roots: tuple[Path, ...] = ()
    write_roots: tuple[Path, ...] = ()
    sandbox: Any | None = None
    session: dict[str, Any] = field(default_factory=dict)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    allow_symlinks: bool = False
    todos: Any = field(default_factory=lambda: _manager("TodoList"))
    tasks: Any = field(default_factory=lambda: _manager("TaskManager"))
    teams: Any = field(default_factory=lambda: _manager("TeamMailbox"))
    jobs: Any = field(default_factory=lambda: _manager("JobManager"))

    def _grant(self, value: str | Path, write: bool) -> tuple[Path, Path]:
        roots = self.write_roots if write else self.read_roots
        legacy = self.values.get("write_root" if write else "read_root")
        if not roots and legacy is not None:
            roots = (Path(legacy),)
        raw = Path(value)
        if ".." in raw.parts:
            raise PermissionError("parent traversal is not allowed")
        for configured in roots:
            root = Path(configured).resolve(strict=True)
            candidate = raw if raw.is_absolute() else root / raw
            if not candidate.is_relative_to(root):
                continue
            if not self.allow_symlinks:
                cursor = root
                for part in candidate.relative_to(root).parts:
                    cursor /= part
                    if cursor.is_symlink():
                        raise PermissionError("symlink traversal is not allowed")
            resolved = candidate.resolve()
            if resolved.is_relative_to(root):
                return root, resolved
        raise PermissionError("path is outside authorized roots or no grant configured")

    def authorize_path(self, value: str | Path, *, write: bool = False) -> Path:
        """Check a grant. File I/O must use open_path to avoid check/open races."""
        return self._grant(value, write)[1]

    @contextmanager
    def open_path(self, value: str | Path, *, write: bool = False) -> Iterator[TextIO]:
        """Open a regular file via no-follow directory descriptors (POSIX).

        Symlinks are denied by default. Opt-in links must resolve inside the same
        grant; the canonical path is still opened without following replacement
        links. Hosts without dir_fd/O_NOFOLLOW support fail closed.
        """
        root, target = self._grant(value, write)
        if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            raise NotImplementedError(
                "secure file tools require POSIX dir_fd and O_NOFOLLOW"
            )
        directory = os.open(target.anchor, os.O_RDONLY | os.O_DIRECTORY)
        descriptor = None
        try:
            parts = target.parts[1:]
            if not parts:
                raise PermissionError("a regular file is required")
            for index, part in enumerate(parts[:-1]):
                if write and index >= len(root.parts) - 1:
                    try:
                        os.mkdir(part, dir_fd=directory)
                    except FileExistsError:
                        pass
                child = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = child
            flags = (os.O_WRONLY | os.O_CREAT) if write else os.O_RDONLY
            descriptor = os.open(
                parts[-1],
                flags | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=directory,
            )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or (write and info.st_nlink > 1):
                raise PermissionError(
                    "regular, non-hardlinked writable files are required"
                )
            if write:
                os.ftruncate(descriptor, 0)
            stream = os.fdopen(descriptor, "w" if write else "r", encoding="utf-8")
            descriptor = None
            with stream:
                yield stream
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory)

    def fresh_managers(self) -> dict[str, Any]:
        """Keyword arguments for dataclasses.replace when cloning per-run state."""
        return {
            "todos": _manager("TodoList"),
            "tasks": _manager("TaskManager"),
            "teams": _manager("TeamMailbox"),
            "jobs": self.jobs.fresh(),
        }


class Tool(ABC):
    timeout: float | None = None
    final_result: bool = False
    background: bool = False
    system_prompt: str | None = None

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition: ...

    @abstractmethod
    async def invoke(self, arguments: dict[str, Any], context: RunContext) -> str: ...


def _json_default(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_type(annotation: Any) -> dict[str, Any]:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        if len(args) > 1:
            raise TypeError("Annotated constraints require use_pydantic=True")
        return _json_type(args[0])
    if annotation in {Any, inspect.Parameter.empty}:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if hasattr(annotation, "model_json_schema"):
        return dict(annotation.model_json_schema())
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        # pyrefly: ignore [invalid-literal]
        return _json_type(Literal[tuple(member.value for member in annotation)])
    if origin is Literal:
        schemas = {_json_type(type(value)).get("type") for value in args}
        return {
            **({"type": next(iter(schemas))} if len(schemas) == 1 else {}),
            "enum": list(args),
        }
    if origin in {types.UnionType, Union}:
        return {"anyOf": [_json_type(value) for value in args]}
    if origin in {list, tuple} or annotation in {list, tuple}:
        if origin is tuple and args and args[-1] is not Ellipsis:
            return {
                "type": "array",
                "prefixItems": [_json_type(arg) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }
        return {"type": "array", "items": _json_type(args[0]) if args else {}}
    if origin is dict or annotation is dict:
        if args and args[0] not in {str, Any}:
            raise TypeError("JSON object keys must be strings")
        return {
            "type": "object",
            "additionalProperties": _json_type(args[1]) if args else {},
        }
    if inspect.isclass(annotation) and is_dataclass(annotation):
        hints = get_type_hints(annotation, include_extras=True)
        properties, required = {}, []
        for item in fields(annotation):
            if not item.init:
                continue
            properties[item.name] = _json_type(hints[item.name])
            if item.default is MISSING and item.default_factory is MISSING:
                required.append(item.name)
            elif item.default is not MISSING:
                properties[item.name]["default"] = json.loads(
                    json.dumps(item.default, default=_json_default)
                )
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    primitive = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if annotation in primitive:
        return {"type": primitive[annotation]}
    raise TypeError(
        f"unsupported annotation {annotation!r}; use the optional Pydantic tools schema"
    )


def _validate(value: Any, annotation: Any) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        if len(args) > 1:
            raise TypeError("Annotated constraints require use_pydantic=True")
        return _validate(value, args[0])
    if annotation in {Any, inspect.Parameter.empty}:
        return value
    if origin in {types.UnionType, Union}:
        for arg in args:
            try:
                return _validate(value, arg)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"value does not match {annotation!r}")
    if origin is Literal:
        if not any(type(value) is type(arg) and value == arg for arg in args):
            raise ValueError(f"expected one of {args!r}")
        return value
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        if isinstance(value, annotation):
            return value
        for member in annotation:
            if type(value) is type(member.value) and value == member.value:
                return member
        raise ValueError(f"invalid {annotation.__name__} value")
    if hasattr(annotation, "model_validate_json"):
        return annotation.model_validate_json(json.dumps(value), strict=True)
    if inspect.isclass(annotation) and is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise TypeError("expected an object")
        hints = get_type_hints(annotation, include_extras=True)
        valid = {item.name for item in fields(annotation) if item.init}
        if set(value) - valid:
            raise TypeError("unexpected nested fields")
        return annotation(
            **{key: _validate(item, hints[key]) for key, item in value.items()}
        )
    if origin in {list, tuple} or annotation in {list, tuple}:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected an array")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError("wrong tuple length")
            return tuple(_validate(item, arg) for item, arg in zip(value, args))
        values = [_validate(item, args[0]) if args else item for item in value]
        return tuple(values) if origin is tuple or annotation is tuple else values
    if origin is dict or annotation is dict:
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise TypeError("expected an object with string keys")
        return {
            key: _validate(item, args[1]) if args else item
            for key, item in value.items()
        }
    if annotation is float and type(value) in {int, float}:
        return float(value)
    if type(value) is not annotation:
        raise TypeError(f"expected {getattr(annotation, '__name__', annotation)}")
    return value


def _doc_parameters(doc: str, *, use_griffe: bool, style: str) -> dict[str, str]:
    if use_griffe:
        try:
            griffe = importlib.import_module("griffe")
        except ImportError as error:
            raise RuntimeError(
                "docstring parsing requires the 'tools' extra (Griffe)"
            ) from error
        result: dict[str, str] = {}
        for section in griffe.Docstring(doc).parse(parser=style):
            if getattr(section.kind, "value", section.kind) == "parameters":
                result.update({item.name: item.description for item in section.value})
        return result
    result = {}
    active = False
    current = None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            active, current = True, None
            continue
        sphinx = re.match(r":param\s+(\w+):\s*(.*)", stripped)
        google = (
            re.match(r"(\w+)(?:\s*\([^)]*\))?:\s*(.*)", stripped)
            if active and line.startswith(" ")
            else None
        )
        match = sphinx or google
        if match:
            current = match[1]
            result[current] = match[2]
        elif active and current and line.startswith(" ") and stripped:
            result[current] += " " + stripped
        elif stripped:
            active, current = False, None
    return result


def _hoist_definitions(schema: Any, definitions: dict[str, Any]) -> None:
    if isinstance(schema, dict):
        for name, value in schema.pop("$defs", {}).items():
            if name in definitions and definitions[name] != value:
                raise ValueError(f"conflicting schema definition: {name}")
            definitions[name] = value
        for value in list(schema.values()):
            _hoist_definitions(value, definitions)
    elif isinstance(schema, list):
        for value in schema:
            _hoist_definitions(value, definitions)


class FunctionTool(Tool):
    """Sync functions run in worker threads; async functions stay on the loop.

    A timeout cancels waiting, not a running Python thread: that function may
    still finish or mutate state. Use a subprocess sandbox for hard termination.
    Sync functions must be thread-safe; use async tools for asyncio-owned state.
    """

    skill_loader: SkillLoader | None = None

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        timeout: float | None = None,
        final_result: bool = False,
        background: bool = False,
        system_prompt: str | None = None,
        use_pydantic: bool = False,
        use_griffe: bool = False,
        docstring_style: str = "google",
    ) -> None:
        self.function = function
        self.timeout, self.final_result = timeout, final_result
        self.background, self.system_prompt = background, system_prompt
        self.signature = inspect.signature(function)
        self._hints = get_type_hints(function, include_extras=True)
        first = next(iter(self.signature.parameters.items()), None)
        self._context_name = (
            first[0]
            if first and self._hints.get(first[0], first[1].annotation) is RunContext
            else None
        )
        self._takes_context = self._context_name is not None
        self._parameters = {
            key: param
            for key, param in self.signature.parameters.items()
            if key != self._context_name
        }
        if any(
            param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}
            for param in self.signature.parameters.values()
        ):
            raise TypeError("variadic tool parameters are not supported")
        doc = inspect.getdoc(function) or ""
        descriptions = _doc_parameters(
            doc, use_griffe=use_griffe, style=docstring_style
        )
        self._model: Any = None
        if use_pydantic:
            schema = self._pydantic_schema(descriptions)
        else:
            properties, required = {}, []
            for key, param in self._parameters.items():
                properties[key] = _json_type(self._hints.get(key, Any))
                if key in descriptions:
                    properties[key]["description"] = descriptions[key]
                if param.default is inspect.Parameter.empty:
                    required.append(key)
                else:
                    properties[key]["default"] = json.loads(
                        json.dumps(param.default, default=_json_default)
                    )
            schema = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
            definitions: dict[str, Any] = {}
            _hoist_definitions(schema, definitions)
            if definitions:
                schema["$defs"] = definitions
        self._definition = ToolDefinition(
            name or function.__name__,
            description if description is not None else doc,
            schema,
        )

    def _pydantic_schema(self, descriptions: dict[str, str]) -> dict[str, Any]:
        try:
            pydantic = importlib.import_module("pydantic")
        except ImportError as error:
            raise RuntimeError(
                "tool schema validation requires the 'tools' extra (Pydantic v2)"
            ) from error
        if not hasattr(pydantic.BaseModel, "model_json_schema"):
            raise RuntimeError("tool schema validation requires Pydantic v2")
        model_fields = {}
        for key, param in self._parameters.items():
            default = ... if param.default is inspect.Parameter.empty else param.default
            model_fields[key] = (
                self._hints.get(key, Any),
                pydantic.Field(default, description=descriptions.get(key)),
            )
        self._model = pydantic.create_model(
            f"{self.function.__name__}Arguments",
            __config__=pydantic.ConfigDict(
                extra="forbid", strict=True, validate_default=True
            ),
            **model_fields,
        )
        return cast(dict[str, Any], self._model.model_json_schema())

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def invoke(self, arguments: dict[str, Any], context: RunContext) -> str:
        if context.cancelled.is_set():
            raise asyncio.CancelledError
        unknown = set(arguments) - set(self._parameters)
        if unknown:
            raise TypeError(f"unexpected arguments: {', '.join(sorted(unknown))}")
        if self._model is not None:
            model = self._model.model_validate_json(json.dumps(arguments), strict=True)
            values = {key: getattr(model, key) for key in self._parameters}
        else:
            values = {}
            for key, param in self._parameters.items():
                if key not in arguments and param.default is inspect.Parameter.empty:
                    raise TypeError(f"missing required argument: {key}")
                value = arguments[key] if key in arguments else param.default
                values[key] = _validate(value, self._hints.get(key, Any))
        bound = self.signature.bind_partial()
        if self._context_name:
            bound.arguments[self._context_name] = context
        bound.arguments.update(values)

        async def call() -> Any:
            if inspect.iscoroutinefunction(self.function):
                result = self.function(*bound.args, **bound.kwargs)
            else:
                result = await asyncio.to_thread(
                    self.function, *bound.args, **bound.kwargs
                )
            return await result if inspect.isawaitable(result) else result

        result = (
            await asyncio.wait_for(call(), self.timeout)
            if self.timeout is not None
            else await call()
        )
        return (
            result
            if isinstance(result, str)
            else json.dumps(result, default=_json_default)
        )


def function_tool(function: Callable[..., Any]) -> FunctionTool:
    return FunctionTool(function)
