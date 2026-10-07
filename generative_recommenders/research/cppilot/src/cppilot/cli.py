# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
import asyncio

from .agent import Agent
from .runner import AgentRunner


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="cppilot", description="Run a provider-neutral CPPilot agent"
    )
    result.add_argument("prompt", nargs="?", help="prompt to send")
    result.add_argument(
        "--provider", choices=("anthropic", "gemini"), default="anthropic"
    )
    result.add_argument("--model")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.prompt is None:
        parser().print_help()
        return
    if args.provider == "anthropic":
        from .adapters.anthropic import AnthropicProvider

        provider = AnthropicProvider(**({"model": args.model} if args.model else {}))
    else:
        from .adapters.gemini import GeminiProvider

        provider = GeminiProvider(**({"model": args.model} if args.model else {}))
    result = asyncio.run(
        AgentRunner().run(
            Agent("assistant", "Be concise and helpful.", provider), args.prompt
        )
    )
    print(result.output)


if __name__ == "__main__":
    main()
