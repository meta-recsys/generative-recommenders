# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import base64
import importlib
import inspect
import json
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from ..interfaces import (
    ModelCapabilities,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    RetryableModelError,
)
from ..items import Message, RunItem, ToolCall, ToolResult, Usage


def _dump(value: Any) -> Any:
    """Keep SDK metadata serializable without requiring an SDK at import time."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _dump(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(part) for part in value]
    if hasattr(value, "__dict__"):
        return _dump(vars(value))
    return value


def _raw_dump(value: Any) -> Any:
    """Encode arbitrary native parts, retaining byte values through JSON storage."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python", exclude_none=True)
    if isinstance(value, Enum):
        return _raw_dump(value.value)
    if isinstance(value, bytes):
        return {"__cppilot_bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _raw_dump(part) for key, part in value.items()}
    if isinstance(value, (list, tuple)):
        return [_raw_dump(part) for part in value]
    if hasattr(value, "__dict__"):
        return _raw_dump(vars(value))
    return value


def _raw_load(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__cppilot_bytes__"}:
            return base64.b64decode(value["__cppilot_bytes__"])
        return {key: _raw_load(part) for key, part in value.items()}
    if isinstance(value, list):
        return [_raw_load(part) for part in value]
    return value


def _supports_parameter(method: Any, name: str) -> bool:
    parameters = inspect.signature(method).parameters
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _supports_field(module_name: str, model_name: str, field: str) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        # Injected clients need not install the official SDK.
        return True
    model = getattr(module, model_name)
    return field in model.model_fields


def _schema_prompt(schema: dict[str, Any]) -> str:
    return (
        "Return only JSON matching this JSON Schema, without markdown fences:\n"
        + json.dumps(schema)
    )


def _translate_error(error: Exception, sdk: str) -> None:
    try:
        module = importlib.import_module(sdk)
    except ImportError:
        module = None
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "code", None)
    transient = status in (408, 409, 429) or (
        isinstance(status, int) and 500 <= status < 600
    )
    api_types = tuple(
        cls
        for name in ("APIError", "APIConnectionError", "APITimeoutError")
        if isinstance(cls := getattr(module, name, None), type)
    )
    connection_types = tuple(
        cls
        for name in ("APIConnectionError", "APITimeoutError")
        if isinstance(cls := getattr(module, name, None), type)
    )
    if isinstance(error, connection_types) or (
        isinstance(error, api_types) and transient
    ):
        raise RetryableModelError(str(error)) from error
    if sdk == "google.genai.errors":
        try:
            httpx = importlib.import_module("httpx")
        except ImportError:
            return
        if isinstance(error, httpx.TransportError):
            raise RetryableModelError(str(error)) from error


async def _close_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await result


class OpenAIProvider(ModelProvider):
    capabilities = ModelCapabilities(streaming=True, structured_output=True)

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        client: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(base_url=base_url)
        self.client = client
        self.model = model

    def _kwargs(self, request: ModelRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for index, item in enumerate(request.messages):
            # The runner mirrors canonical instructions in its leading message.
            if (
                index == 0
                and request.system
                and isinstance(item, Message)
                and item.role == "system"
                and item.content == request.system
            ):
                continue
            if isinstance(item, Message):
                if (
                    item.role == "assistant"
                    and messages
                    and messages[-1]["role"] == "assistant"
                ):
                    messages[-1]["content"] = (
                        messages[-1].get("content") or ""
                    ) + item.content
                else:
                    messages.append({"role": item.role, "content": item.content})
            elif isinstance(item, ToolCall):
                if not messages or messages[-1]["role"] != "assistant":
                    messages.append({"role": "assistant", "content": None})
                messages[-1].setdefault("tool_calls", []).append(
                    {
                        "id": item.call_id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": json.dumps(item.arguments),
                        },
                    }
                )
            elif isinstance(item, ToolResult):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": item.output,
                    }
                )
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if request.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
        if request.output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.output_schema},
            }
        return kwargs

    def _metadata(self, response: Any) -> dict[str, Any]:
        return {
            key: _dump(value)
            for key in (
                "id",
                "_request_id",
                "model",
                "created",
                "system_fingerprint",
                "service_tier",
            )
            if (value := getattr(response, key, None)) is not None
        }

    def _usage(self, usage: Any, metadata: dict[str, Any]) -> Usage:
        return Usage(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            provider="openai",
            model=metadata.get("model", self.model),
            metadata={**metadata, "usage": _dump(usage)},
        )

    @staticmethod
    def _call(
        call_id: str, name: str, arguments: str, metadata: dict[str, Any]
    ) -> ToolCall:
        parsed = json.loads(arguments or "{}")
        if not isinstance(parsed, dict):
            raise ValueError("OpenAI tool arguments must be a JSON object")
        return ToolCall(call_id, name, parsed, metadata=dict(metadata))

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = await self.client.chat.completions.create(
                **self._kwargs(request)
            )
        except Exception as error:
            _translate_error(error, "openai")
            raise
        metadata = self._metadata(response)
        items: list[RunItem] = []
        if response.choices:
            choice = response.choices[0]
            metadata["finish_reason"] = choice.finish_reason
            message = choice.message
            if getattr(message, "refusal", None) is not None:
                metadata["refusal"] = message.refusal
            if message.content:
                items.append(
                    Message("assistant", message.content, metadata=dict(metadata))
                )
            for call in getattr(message, "tool_calls", None) or ():
                items.append(
                    self._call(
                        call.id, call.function.name, call.function.arguments, metadata
                    )
                )
        return ModelResponse(
            items, self._usage(getattr(response, "usage", None), metadata), metadata
        )

    async def stream(  # noqa: C901
        self, request: ModelRequest
    ) -> AsyncIterator[RunItem]:
        stream = None
        metadata: dict[str, Any] = {}
        usage = None
        calls: dict[int, dict[str, str]] = {}
        try:
            stream = await self.client.chat.completions.create(
                **self._kwargs(request),
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                metadata.update(self._metadata(chunk))
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                for choice in chunk.choices or ():
                    if getattr(choice, "index", 0) != 0:
                        continue
                    if choice.finish_reason is not None:
                        metadata["finish_reason"] = choice.finish_reason
                    delta = choice.delta
                    if getattr(delta, "refusal", None):
                        metadata["refusal"] = (
                            metadata.get("refusal", "") + delta.refusal
                        )
                    if getattr(delta, "content", None):
                        yield Message(
                            "assistant", delta.content, metadata=dict(metadata)
                        )
                    for part in getattr(delta, "tool_calls", None) or ():
                        call = calls.setdefault(
                            part.index, {"id": "", "name": "", "arguments": ""}
                        )
                        if getattr(part, "id", None):
                            call["id"] = part.id
                        function = getattr(part, "function", None)
                        if function is not None:
                            call["name"] += getattr(function, "name", None) or ""
                            call["arguments"] += (
                                getattr(function, "arguments", None) or ""
                            )
            for index in sorted(calls):
                call = calls[index]
                if not call["id"] or not call["name"]:
                    raise ValueError("OpenAI stream returned an incomplete tool call")
                yield self._call(call["id"], call["name"], call["arguments"], metadata)
            yield self._usage(usage, metadata)
        except Exception as error:
            _translate_error(error, "openai")
            raise
        finally:
            if stream is not None:
                await _close_stream(stream)
