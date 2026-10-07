# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from typing import Any

from ..interfaces import Event, EventSink


class OpenTelemetryEventSink(EventSink):
    def __init__(self, tracer: Any | None = None) -> None:
        if tracer is None:
            from opentelemetry import trace

            tracer = trace.get_tracer("cppilot")
        self.tracer = tracer

    async def emit(self, event: Event) -> None:
        with self.tracer.start_as_current_span(event.name) as span:
            for key, value in event.data.items():
                span.set_attribute(key, str(value))
