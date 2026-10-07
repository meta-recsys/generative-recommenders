# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from typing import Any

from ..interfaces import ModelProvider, ModelRequest, ModelResponse
from ..items import Message, Usage


class GeminiProvider(ModelProvider):
    def __init__(
        self, model: str = "gemini-2.5-flash", *, client: Any | None = None
    ) -> None:
        if client is None:
            from google import genai

            client = genai.Client()
        self.client = client
        self.model = model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        prompt = "\n".join(
            f"{item.role}: {item.content}"
            for item in request.messages
            if isinstance(item, Message)
        )
        response = await self.client.aio.models.generate_content(
            model=self.model, contents=prompt
        )
        metadata = getattr(response, "usage_metadata", None)
        return ModelResponse(
            [Message("assistant", response.text or "")],
            Usage(
                getattr(metadata, "prompt_token_count", 0) or 0,
                getattr(metadata, "candidates_token_count", 0) or 0,
            ),
        )
