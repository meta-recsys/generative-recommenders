# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any, cast
from uuid import uuid4

from ..interfaces import ModelCapabilities, ModelProvider, ModelRequest, ModelResponse
from ..items import Message, RunItem, ToolCall, ToolResult, Usage
from .openai import (
    _close_stream,
    _dump,
    _raw_dump,
    _raw_load,
    _schema_prompt,
    _supports_field,
    _translate_error,
)


class GeminiProvider(ModelProvider):
    capabilities = ModelCapabilities(streaming=True, structured_output=True)

    def __init__(
        self, model: str = "gemini-2.5-flash", *, client: Any | None = None
    ) -> None:
        if client is None:
            from google import genai

            client = genai.Client()
        self.client = client
        self.model = model

    def _kwargs(self, request: ModelRequest) -> dict[str, Any]:  # noqa: C901
        contents: list[dict[str, Any]] = []
        system = [request.system] if request.system else []
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
            if isinstance(item, Message) and item.role == "system":
                system.append(item.content)
                continue
            if isinstance(item, Message):
                role = "model" if item.role == "assistant" else "user"
                part: dict[str, Any] = {"text": item.content}
            elif isinstance(item, ToolCall):
                role, part = (
                    "model",
                    {
                        "function_call": {
                            "name": item.name,
                            "args": item.arguments,
                        }
                    },
                )
                if not item.metadata.get("gemini_synthetic_id", False):
                    part["function_call"]["id"] = item.call_id
            elif isinstance(item, ToolResult):
                role, part = (
                    "user",
                    {
                        "function_response": {
                            "name": item.name,
                            "response": {"output": item.output, "failed": item.failed},
                        }
                    },
                )
                original = next(
                    (
                        call
                        for call in request.messages
                        if isinstance(call, ToolCall) and call.call_id == item.call_id
                    ),
                    None,
                )
                if original is None or not original.metadata.get(
                    "gemini_synthetic_id", False
                ):
                    part["function_response"]["id"] = item.call_id
            else:
                continue
            if signature := item.metadata.get("thought_signature"):
                # pyrefly: ignore [unsupported-operation]
                part["thought_signature"] = base64.b64decode(signature)
            if not contents or contents[-1]["role"] != role:
                contents.append({"role": role, "parts": []})
            contents[-1]["parts"].extend(
                self._replay_parts(item.metadata.get("gemini_parts_before", []))
            )
            if not item.metadata.get("replay_only"):
                contents[-1]["parts"].append(part)
            contents[-1]["parts"].extend(
                self._replay_parts(item.metadata.get("gemini_parts_after", []))
            )
        config: dict[str, Any] = {"system_instruction": "\n".join(system)}
        if request.tools:
            if not _supports_field(
                "google.genai.types", "FunctionDeclaration", "parameters_json_schema"
            ):
                raise ValueError(
                    "Gemini tool schemas require a google-genai SDK with "
                    "FunctionDeclaration.parameters_json_schema; upgrade google-genai"
                )
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters_json_schema": tool.parameters,
                        }
                        for tool in request.tools
                    ]
                }
            ]
        if request.output_schema is not None:
            config["response_mime_type"] = "application/json"
            if _supports_field(
                "google.genai.types", "GenerateContentConfig", "response_json_schema"
            ):
                config["response_json_schema"] = request.output_schema
            else:
                config["system_instruction"] += "\n\n" + _schema_prompt(
                    request.output_schema
                )
        return {"model": self.model, "contents": contents, "config": config}

    @staticmethod
    def _replay_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        restored = cast(list[dict[str, Any]], _raw_load(parts))
        for part in restored:
            signature = part.get("thought_signature")
            if isinstance(signature, str):
                part["thought_signature"] = base64.b64decode(signature)
        return restored

    def _metadata(self, response: Any) -> dict[str, Any]:
        metadata = {
            key: _dump(value)
            for key in (
                "response_id",
                "model_version",
                "prompt_feedback",
            )
            if (value := getattr(response, key, None)) is not None
        }
        candidates = getattr(response, "candidates", None) or ()
        if candidates:
            for key in (
                "finish_reason",
                "safety_ratings",
                "citation_metadata",
                "grounding_metadata",
            ):
                if (value := getattr(candidates[0], key, None)) is not None:
                    metadata[key] = _dump(value)
        return metadata

    def _items(  # noqa: C901
        self, response: Any, metadata: dict[str, Any], *, streaming: bool = False
    ) -> list[RunItem]:
        items: list[RunItem] = []
        candidates = getattr(response, "candidates", None) or ()
        if not candidates:
            return items
        preceding: list[dict[str, Any]] = []
        for part in (
            getattr(getattr(candidates[0], "content", None), "parts", None) or ()
        ):
            item_metadata = dict(metadata)
            signature = getattr(part, "thought_signature", None)
            if signature:
                item_metadata["thought_signature"] = base64.b64encode(signature).decode(
                    "ascii"
                )
            if preceding:
                item_metadata["gemini_parts_before"] = list(preceding)
            if getattr(part, "thought", False):
                # Thought parts are replay state, not user-visible assistant text.
                raw = _raw_dump(part)
                if signature:
                    raw["thought_signature"] = base64.b64encode(signature).decode(
                        "ascii"
                    )
                preceding.append(raw)
                continue
            call = getattr(part, "function_call", None)
            if call is not None:
                call_id = getattr(call, "id", None)
                if not call_id:
                    call_id = f"gemini-{uuid4().hex}"
                    item_metadata["gemini_synthetic_id"] = True
                items.append(
                    ToolCall(
                        call_id,
                        call.name,
                        dict(call.args or {}),
                        metadata=item_metadata,
                    )
                )
            elif getattr(part, "text", None):
                items.append(Message("assistant", part.text, metadata=item_metadata))
            else:
                preceding.append(_raw_dump(part))
                continue
            preceding.clear()
        if preceding:
            if streaming:
                metadata["unrepresented_parts"] = preceding
            elif items:
                items[-1].metadata["gemini_parts_after"] = preceding
            else:
                items.append(
                    Message(
                        "assistant",
                        "",
                        metadata={
                            **metadata,
                            "replay_only": True,
                            "gemini_parts_after": preceding,
                        },
                    )
                )
        return items

    def _usage(self, usage: Any, metadata: dict[str, Any]) -> Usage:
        return Usage(
            getattr(usage, "prompt_token_count", 0) or 0,
            (getattr(usage, "candidates_token_count", 0) or 0)
            + (getattr(usage, "thoughts_token_count", 0) or 0),
            provider="gemini",
            model=metadata.get("model_version", self.model),
            metadata={**metadata, "usage": _dump(usage)},
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = await self.client.aio.models.generate_content(
                **self._kwargs(request)
            )
        except Exception as error:
            _translate_error(error, "google.genai.errors")
            raise
        metadata = self._metadata(response)
        items = self._items(response, metadata)
        return ModelResponse(
            items,
            self._usage(getattr(response, "usage_metadata", None), metadata),
            metadata,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[RunItem]:
        stream = None
        metadata: dict[str, Any] = {}
        usage = None
        preceding: list[dict[str, Any]] = []
        try:
            stream = await self.client.aio.models.generate_content_stream(
                **self._kwargs(request)
            )
            async for response in stream:
                metadata.update(self._metadata(response))
                if getattr(response, "usage_metadata", None) is not None:
                    usage = response.usage_metadata
                chunk_metadata = dict(metadata)
                for item in self._items(response, chunk_metadata, streaming=True):
                    if preceding:
                        item.metadata["gemini_parts_before"] = (
                            preceding + item.metadata.get("gemini_parts_before", [])
                        )
                        preceding = []
                    yield item
                preceding.extend(chunk_metadata.get("unrepresented_parts", []))
            if preceding:
                yield Message(
                    "assistant",
                    "",
                    metadata={
                        **metadata,
                        "replay_only": True,
                        "gemini_parts_after": preceding,
                    },
                )
            yield self._usage(usage, metadata)
        except Exception as error:
            _translate_error(error, "google.genai.errors")
            raise
        finally:
            if stream is not None:
                await _close_stream(stream)
