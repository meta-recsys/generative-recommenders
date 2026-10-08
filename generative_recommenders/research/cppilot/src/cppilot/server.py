# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import inspect
import json
import math
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from copy import copy
from dataclasses import replace
from typing import Any, TypeVar

from .agent import Agent
from .integrations import AuthProvider
from .items import item_to_dict
from .runner import (
    AgentRunner,
    GuardrailTriggered,
    MaxTurnsExceeded,
    StructuredOutputError,
)

T = TypeVar("T")


class ClientDisconnected(Exception):
    pass


async def _connected(awaitable: Awaitable[T], request: Any, interval: float) -> T:
    stopping = asyncio.Event()

    async def disconnected() -> None:
        while not stopping.is_set():
            if await request.is_disconnected():
                return
            if not stopping.is_set():
                await asyncio.sleep(interval)

    work = asyncio.ensure_future(awaitable)
    monitor = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait(
            (work, monitor), return_when=asyncio.FIRST_COMPLETED
        )
        if monitor in done:
            monitor.result()
            raise ClientDisconnected()
        return work.result()
    finally:
        stopping.set()
        for task in (work, monitor):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, monitor, return_exceptions=True)


def _request_agent(
    root: Agent, principal: Any, application_state: Mapping[str, Any]
) -> Agent:
    clones: dict[int, Agent] = {}
    originals: dict[int, Agent] = {}
    pending = [root]
    while pending:
        agent = pending.pop()
        identity = id(agent)
        if identity in clones:
            continue
        originals[identity] = agent
        clone = copy(agent)
        clone.context = replace(
            agent.context,
            values={
                **agent.context.values,
                "principal": principal,
                "application_state": dict(application_state),
            },
            session=dict(agent.context.session),
            cancelled=asyncio.Event(),
            **agent.context.fresh_managers(),
        )
        clones[identity] = clone
        pending.extend(agent.handoffs)
    for identity, agent in originals.items():
        clones[identity].handoffs = tuple(
            clones[id(target)] for target in agent.handoffs
        )
    return clones[id(root)]


def _stream_error(error: Exception) -> dict[str, Any]:
    code = "run_failed"
    if isinstance(error, GuardrailTriggered):
        code = "guardrail_triggered"
    elif isinstance(error, MaxTurnsExceeded):
        code = "max_turns_exceeded"
    elif isinstance(error, StructuredOutputError):
        code = "invalid_output"
    return {"type": "error", "code": code, "message": "Run failed", "terminal": True}


def create_app(  # noqa: C901
    resolve_agent: Callable[[str], Agent],
    *,
    runner: AgentRunner | None = None,
    authenticate: Callable[[str | None], Awaitable[Any]] | None = None,
    auth_provider: AuthProvider | None = None,
    disconnect_interval: float = 0.1,
    application_dependency: Callable[..., Any] | None = None,
    resolve_authorized_agent: (
        Callable[[str, Any, str | None, Mapping[str, Any]], Agent | Awaitable[Agent]]
        | None
    ) = None,
) -> Any:
    """Optional JSON/SSE hosting. Authentication applies to every app route.

    resolve_agent performs name lookup only, not per-user authorization.
    resolve_authorized_agent may authorize/select using (name, principal,
    conversation_id, application_state), raising PermissionError on denial.
    application_dependency is an optional FastAPI dependency returning a mapping;
    it can read request.state.principal and use nested Depends. Both principal
    and application_state are placed in each request-local graph context.values.
    SDK imports and application state are created only when this factory is used.
    """
    try:
        import anyio
        from fastapi import Depends, FastAPI, HTTPException, Request
        from fastapi.responses import StreamingResponse
    except ImportError as error:
        raise RuntimeError("server support requires FastAPI") from error
    if not math.isfinite(disconnect_interval) or disconnect_interval <= 0:
        raise ValueError("disconnect_interval must be positive")
    if authenticate is not None and auth_provider is not None:
        raise ValueError("configure only one authentication provider")

    async def authorize(request: Any) -> None:
        credential = request.headers.get("authorization")
        try:
            if auth_provider is not None:
                principal = await auth_provider.authenticate(credential)
                if not principal:
                    raise PermissionError("empty principal")
            elif authenticate is not None:
                principal = await authenticate(credential)
                if principal is False:
                    raise PermissionError("rejected credential")
            else:
                principal = None
        except Exception as error:
            raise HTTPException(
                status_code=401,
                detail="unauthorized",
                headers={"WWW-Authenticate": "Bearer"},
            ) from error
        request.state.principal = principal

    # FastAPI resolves annotations in module globals; Request is a lazy import.
    authorize.__annotations__["request"] = Request
    dependencies = [Depends(authorize)]
    if application_dependency is not None:
        application_state_dependency = Depends(application_dependency)

        async def application(
            request: Any, state: Any = application_state_dependency
        ) -> None:
            if not isinstance(state, Mapping):
                raise TypeError("application dependency must return a mapping")
            request.state.application_state = dict(state)

        application.__annotations__["request"] = Request
        dependencies.append(Depends(application))
    app = FastAPI(
        title="CPPilot",
        dependencies=dependencies,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.authorize = authorize
    active_runner = runner if runner is not None else AgentRunner()

    async def inputs(
        body: dict[str, Any], request: Any
    ) -> tuple[Agent, str, str | None]:
        if not isinstance(body.get("agent"), str) or not body["agent"]:
            raise HTTPException(
                status_code=422, detail="agent must be a nonempty string"
            )
        if not isinstance(body.get("prompt"), str):
            raise HTTPException(status_code=422, detail="prompt must be a string")
        conversation = body.get("conversation_id")
        if conversation is not None and not isinstance(conversation, str):
            raise HTTPException(
                status_code=422, detail="conversation_id must be a string"
            )
        principal = getattr(request.state, "principal", None)
        state = getattr(request.state, "application_state", {})
        try:
            agent = (
                resolve_authorized_agent(body["agent"], principal, conversation, state)
                if resolve_authorized_agent is not None
                else resolve_agent(body["agent"])
            )
            if inspect.isawaitable(agent):
                agent = await agent
        except PermissionError as error:
            raise HTTPException(status_code=403, detail="forbidden") from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail="unknown agent") from error
        return _request_agent(agent, principal, state), body["prompt"], conversation

    async def run(body: dict[str, Any], request: Any) -> dict[str, Any]:
        agent, prompt, conversation = await inputs(body, request)
        try:
            result = await _connected(
                active_runner.run(agent, prompt, conversation_id=conversation),
                request,
                disconnect_interval,
            )
        except ClientDisconnected as error:
            raise HTTPException(
                status_code=499, detail="client disconnected"
            ) from error
        return {
            "output": result.output,
            "agent": result.agent_name,
            "run_id": getattr(result, "run_id", ""),
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
        }

    class ClosingStreamingResponse(StreamingResponse):
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                # ASGI write failure can leave the generator suspended at yield.
                with anyio.CancelScope(shield=True):
                    close = getattr(self.body_iterator, "aclose", None)
                    if callable(close):
                        await close()

    async def stream(body: dict[str, Any], request: Any) -> Any:
        agent, prompt, conversation = await inputs(body, request)

        async def events() -> AsyncGenerator[str, None]:
            try:
                iterator = active_runner.stream(
                    agent, prompt, conversation_id=conversation
                )
            except Exception as error:
                yield f"event: error\ndata: {json.dumps(_stream_error(error))}\n\n"
                return
            async with aclosing(iterator):
                while True:
                    try:
                        item = await _connected(
                            anext(iterator), request, disconnect_interval
                        )
                    except (StopAsyncIteration, ClientDisconnected):
                        return
                    except Exception as error:
                        yield f"event: error\ndata: {json.dumps(_stream_error(error))}\n\n"
                        return
                    yield f"data: {json.dumps(item_to_dict(item))}\n\n"

        return ClosingStreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    run.__annotations__["request"] = Request
    stream.__annotations__["request"] = Request
    app.post("/v1/run")(run)
    app.post("/v1/stream")(stream)
    return app
