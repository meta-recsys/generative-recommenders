# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

from .interfaces import ModelProvider


def load_entry_points(register: Callable[[str, Any], None], group: str) -> list[str]:
    """Explicit discovery only. Entry points export factories, not import side effects."""
    loaded = []
    for entry in sorted(
        metadata.entry_points(group=group), key=lambda entry: entry.name
    ):
        register(entry.name, entry.load())
        loaded.append(entry.name)
    return loaded


class ModelRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, Callable[..., ModelProvider] | ModelProvider] = {}
        self._configurations: dict[str, dict[str, Any]] = {}

    def register(
        self, name: str, provider: Callable[..., ModelProvider] | ModelProvider
    ) -> None:
        if not name:
            raise ValueError("model name must not be empty")
        if name in self._providers:
            raise ValueError(f"model {name!r} is already registered")
        self._providers[name] = provider

    def get(self, name: str, **configuration: Any) -> ModelProvider:
        try:
            provider = self._providers[name]
        except KeyError as error:
            raise KeyError(f"unknown model {name!r}") from error
        if callable(provider):
            return provider(**configuration)
        if configuration:
            raise ValueError("configuration cannot be supplied to a model instance")
        return provider

    def register_configured(
        self, name: str, provider: str, **configuration: Any
    ) -> None:
        if provider not in self._providers:
            raise KeyError(f"unknown provider {provider!r}")

        def factory(**overrides: Any) -> ModelProvider:
            return self.get(provider, **{**configuration, **overrides})

        self.register(name, factory)
        self._configurations[name] = {"provider": provider, **configuration}

    def configuration(self, name: str) -> dict[str, Any]:
        if name not in self._providers:
            raise KeyError(f"unknown model {name!r}")
        return dict(self._configurations.get(name, {"provider": name}))

    def names(self) -> list[str]:
        return sorted(self._providers)

    def load_entry_points(self, group: str = "cppilot.models") -> list[str]:
        return load_entry_points(self.register, group)

    def register_builtin_providers(self) -> None:
        # Lazy factories keep SDK dependencies out of import and info paths.
        def anthropic(**configuration: Any) -> ModelProvider:
            from .adapters.anthropic import AnthropicProvider

            return AnthropicProvider(**configuration)

        def gemini(**configuration: Any) -> ModelProvider:
            from .adapters.gemini import GeminiProvider

            return GeminiProvider(**configuration)

        def openai(**configuration: Any) -> ModelProvider:
            from .adapters.openai import OpenAIProvider

            return OpenAIProvider(**configuration)

        for name, factory in (
            ("anthropic", anthropic),
            ("gemini", gemini),
            ("openai", openai),
        ):
            self.register(name, factory)
