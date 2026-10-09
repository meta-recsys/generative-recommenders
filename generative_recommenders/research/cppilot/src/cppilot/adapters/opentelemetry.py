# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..interfaces import Event, EventSink


@dataclass(slots=True)
class _ActiveSpan:
    span: Any
    started: float


class OpenTelemetryEventSink(EventSink):
    """Correlate runner lifecycle events without ambient task-local spans.

    Model spans are per attempt, tool spans per call_id, and retries/handoffs
    are short child spans. Call close() when abandoning incomplete runs.
    This sink does not configure exporters or capture prompts/tool arguments.
    """

    def __init__(self, tracer: Any | None = None) -> None:
        try:
            from opentelemetry import trace
        except ImportError as error:
            raise RuntimeError("tracing requires OpenTelemetry") from error
        self.trace = trace
        self.tracer = tracer if tracer is not None else trace.get_tracer("cppilot")
        self._spans: dict[tuple[str, str, str], _ActiveSpan] = {}

    def _key(self, event: Event, kind: str) -> tuple[str, str, str]:
        identity = ""
        if kind == "tool":
            identity = str(event.data.get("call_id", event.data.get("tool", "")))
        elif kind == "model":
            identity = str(event.data.get("agent", ""))
        return str(event.data.get("run_id", "")), kind, identity

    def _start(self, key: tuple[str, str, str], event: Event) -> _ActiveSpan:
        root = self._spans.get((key[0], "run", ""))
        context = self.trace.set_span_in_context(root.span) if root else None
        active = _ActiveSpan(
            self.tracer.start_span(f"cppilot.{key[1]}", context=context),
            time.monotonic(),
        )
        self._attributes(active.span, event.data)
        return active

    @staticmethod
    def _attributes(span: Any, data: dict[str, Any]) -> None:
        # Only operational fields are exported, never arbitrary application data.
        fields = {
            "run_id",
            "agent",
            "call_id",
            "tool",
            "attempt",
            "source",
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "latency",
            "error",
        }
        for key in fields & data.keys():
            value = data[key]
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(f"cppilot.{key}", value)

    def _finish(
        self, active: _ActiveSpan, event: Event, *, failed: bool = False
    ) -> None:
        self._attributes(active.span, event.data)
        if "latency" not in event.data:
            active.span.set_attribute(
                "cppilot.latency", time.monotonic() - active.started
            )
        if failed:
            from opentelemetry.trace import Status, StatusCode

            description = str(event.data.get("error", "operation failed"))
            active.span.set_status(Status(StatusCode.ERROR, description))
            active.span.add_event("exception", {"exception.message": description})
        active.span.end()

    async def emit(self, event: Event) -> None:
        kind, _, phase = event.name.partition(".")
        if kind in {"run", "model", "tool"} and phase in {
            "start",
            "complete",
            "end",
            "error",
        }:
            key = self._key(event, kind)
            if phase == "start":
                if kind != "run" and key[0] and (key[0], "run", "") not in self._spans:
                    return
                if key in self._spans:
                    self._finish(
                        self._spans.pop(key),
                        Event("superseded", {"error": "missing terminal event"}),
                        failed=True,
                    )
                self._spans[key] = self._start(key, event)
                return
            active = self._spans.pop(key, None)
            if active is None:
                # A late background terminal must not recreate a finalized span.
                return
            self._finish(active, event, failed=phase == "error")
            if kind == "run":
                for child_key in list(self._spans):
                    if child_key[0] == key[0]:
                        self._finish(
                            self._spans.pop(child_key),
                            Event("abandoned", {"error": "run ended before operation"}),
                            failed=True,
                        )
            return
        key = self._key(event, event.name)
        if key[0] and (key[0], "run", "") not in self._spans:
            return
        active = self._start(key, event)
        active.span.add_event(event.name)
        self._finish(active, event)

    def close(self) -> None:
        for active in self._spans.values():
            self._finish(active, Event("closed", {"error": "sink closed"}), failed=True)
        self._spans.clear()
