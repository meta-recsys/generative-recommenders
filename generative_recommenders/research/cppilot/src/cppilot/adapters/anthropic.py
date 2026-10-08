# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from typing import Any

from ..interfaces import ModelProvider, ModelRequest, ModelResponse
from ..items import Message, ToolCall, Usage


class AnthropicProvider(ModelProvider):
    def __init__(
        self, model: str = "claude-sonnet-4-5", *, client: Any | None = None
    ) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic()
        self.client = client
        self.model = model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        system = "\n".join(
            item.content
            for item in request.messages
            if isinstance(item, Message) and item.role == "system"
        )
        messages = [
            {"role": item.role, "content": item.content}
            for item in request.messages
            if isinstance(item, Message) and item.role != "system"
        ]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=[
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ],
        )
        items: list[Message | ToolCall] = []
        for block in response.content:
            if block.type == "text":
                items.append(Message("assistant", block.text))
            elif block.type == "tool_use":
                items.append(ToolCall(block.id, block.name, dict(block.input)))
        return ModelResponse(
            items,
            Usage(response.usage.input_tokens, response.usage.output_tokens),
        )
