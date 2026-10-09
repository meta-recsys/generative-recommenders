# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..interfaces import (
    ModelCapabilities,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    RetryableModelError,
)
from ..items import Message, RunItem, ToolCall, ToolResult, Usage
from .openai import (
    _close_stream,
    _dump,
    _raw_dump,
    _raw_load,
    _schema_prompt,
    _supports_parameter,
    _translate_error,
)


class AnthropicProvider(ModelProvider):
    capabilities = ModelCapabilities(streaming=True, structured_output=True)

    def __init__(
        self, model: str = "claude-sonnet-4-5", *, client: Any | None = None
    ) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
        self.client = client
        self.model = model

    def _kwargs(self, request: ModelRequest) -> dict[str, Any]:
        system = [request.system] if request.system else []
        messages: list[dict[str, Any]] = []
        block: dict[str, Any]
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
                role, block = item.role, {"type": "text", "text": item.content}
            elif isinstance(item, ToolCall):
                role, block = (
                    "assistant",
                    {
                        "type": "tool_use",
                        "id": item.call_id,
                        "name": item.name,
                        "input": item.arguments,
                    },
                )
            elif isinstance(item, ToolResult):
                role, block = (
                    "user",
                    {
                        "type": "tool_result",
                        "tool_use_id": item.call_id,
                        "content": item.output,
                        "is_error": item.failed,
                    },
                )
            else:
                continue
            if not messages or messages[-1]["role"] != role:
                messages.append({"role": role, "content": []})
            messages[-1]["content"].extend(
                _raw_load(item.metadata.get("anthropic_blocks_before", []))
            )
            if not item.metadata.get("replay_only"):
                messages[-1]["content"].append(block)
            messages[-1]["content"].extend(
                _raw_load(item.metadata.get("anthropic_blocks_after", []))
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": "\n".join(system),
            "messages": messages,
        }
        if request.tools:
            kwargs["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
        if request.output_schema is not None:
            if _supports_parameter(self.client.messages.create, "output_config"):
                kwargs["output_config"] = {
                    "format": {"type": "json_schema", "schema": request.output_schema},
                }
            else:
                kwargs["system"] += "\n\n" + _schema_prompt(request.output_schema)
        return kwargs

    def _metadata(self, response: Any) -> dict[str, Any]:
        return {
            key: _dump(value)
            for key in (
                "id",
                "_request_id",
                "model",
                "stop_reason",
                "stop_sequence",
            )
            if (value := getattr(response, key, None)) is not None
        }

    def _usage(self, usage: dict[str, Any], metadata: dict[str, Any]) -> Usage:
        return Usage(
            sum(
                usage.get(key, 0) or 0
                for key in (
                    "input_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                )
            ),
            usage.get("output_tokens", 0) or 0,
            provider="anthropic",
            model=metadata.get("model", self.model),
            metadata={**metadata, "usage": dict(usage)},
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = await self.client.messages.create(**self._kwargs(request))
        except Exception as error:
            _translate_error(error, "anthropic")
            raise
        metadata = self._metadata(response)
        items: list[RunItem] = []
        preceding: list[dict[str, Any]] = []
        for block in response.content:
            item_metadata = dict(metadata)
            if preceding:
                item_metadata["anthropic_blocks_before"] = list(preceding)
            if block.type == "text":
                items.append(Message("assistant", block.text, metadata=item_metadata))
            elif block.type == "tool_use":
                items.append(
                    ToolCall(
                        block.id, block.name, dict(block.input), metadata=item_metadata
                    )
                )
            else:
                preceding.append(_raw_dump(block))
                continue
            preceding.clear()
        if preceding:
            if items:
                items[-1].metadata["anthropic_blocks_after"] = preceding
            else:
                items.append(
                    Message(
                        "assistant",
                        "",
                        metadata={
                            **metadata,
                            "replay_only": True,
                            "anthropic_blocks_after": preceding,
                        },
                    )
                )
        return ModelResponse(
            items, self._usage(_dump(response.usage), metadata), metadata
        )

    async def stream(  # noqa: C901
        self, request: ModelRequest
    ) -> AsyncIterator[RunItem]:
        stream = None
        metadata: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        blocks: dict[int, dict[str, Any]] = {}
        preceding: list[dict[str, Any]] = []
        try:
            stream = await self.client.messages.create(
                **self._kwargs(request), stream=True
            )
            async for event in stream:
                if event.type == "message_start":
                    metadata.update(self._metadata(event.message))
                    usage.update(_dump(event.message.usage))
                elif event.type == "message_delta":
                    metadata.update(self._metadata(event.delta))
                    usage.update(_dump(event.usage) or {})
                elif event.type == "content_block_start":
                    block = _raw_dump(event.content_block)
                    blocks[event.index] = {"block": block, "json": ""}
                    if block["type"] == "text" and block.get("text"):
                        item_metadata = dict(metadata)
                        if preceding:
                            item_metadata["anthropic_blocks_before"] = list(preceding)
                            preceding.clear()
                        yield Message(
                            "assistant", block["text"], metadata=item_metadata
                        )
                elif event.type == "content_block_delta":
                    state = blocks[event.index]
                    delta = event.delta
                    if delta.type == "text_delta":
                        item_metadata = dict(metadata)
                        if preceding:
                            item_metadata["anthropic_blocks_before"] = list(preceding)
                            preceding.clear()
                        yield Message("assistant", delta.text, metadata=item_metadata)
                    elif delta.type == "input_json_delta":
                        state["json"] += delta.partial_json
                    elif delta.type == "thinking_delta":
                        state["block"]["thinking"] = (
                            state["block"].get("thinking", "") + delta.thinking
                        )
                    elif delta.type == "signature_delta":
                        state["block"]["signature"] = (
                            state["block"].get("signature", "") + delta.signature
                        )
                elif event.type == "content_block_stop":
                    state = blocks.pop(event.index)
                    block = state["block"]
                    if block["type"] == "tool_use":
                        arguments = (
                            json.loads(state["json"])
                            if state["json"]
                            else block.get("input", {})
                        )
                        if not isinstance(arguments, dict):
                            raise ValueError(
                                "Anthropic tool arguments must be a JSON object"
                            )
                        item_metadata = dict(metadata)
                        if preceding:
                            item_metadata["anthropic_blocks_before"] = list(preceding)
                            preceding.clear()
                        yield ToolCall(
                            block["id"],
                            block["name"],
                            arguments,
                            metadata=item_metadata,
                        )
                    elif block["type"] != "text":
                        preceding.append(block)
                elif event.type == "error":
                    error = _dump(event.error)
                    if error.get("type") in (
                        "overloaded_error",
                        "rate_limit_error",
                        "api_error",
                    ):
                        raise RetryableModelError(error.get("message", str(error)))
                    raise RuntimeError(error.get("message", str(error)))
            if blocks:
                raise ValueError(
                    "Anthropic stream ended with incomplete content blocks"
                )
            if preceding:
                yield Message(
                    "assistant",
                    "",
                    metadata={
                        **metadata,
                        "replay_only": True,
                        "anthropic_blocks_after": preceding,
                    },
                )
            yield self._usage(usage, metadata)
        except Exception as error:
            _translate_error(error, "anthropic")
            raise
        finally:
            if stream is not None:
                await _close_stream(stream)
