# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

import asyncio

from cppilot import Agent, AgentRunner, Message, ModelProvider
from cppilot.interfaces import ModelRequest, ModelResponse


class EchoProvider(ModelProvider):
    async def generate(self, request: ModelRequest) -> ModelResponse:
        prompt = next(
            item.content
            for item in reversed(request.messages)
            if isinstance(item, Message) and item.role == "user"
        )
        return ModelResponse([Message("assistant", f"Echo: {prompt}")])


async def main() -> None:
    result = await AgentRunner().run(
        Agent("echo", "Echo the user.", EchoProvider()), "hello"
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
