# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from .interfaces import ToolDefinition
from .tools.core import RunContext, Tool


class MCPToolError(RuntimeError):
    """The remote tool returned an MCP error result."""


class MCPTool(Tool):
    def __init__(self, session: Any, definition: ToolDefinition) -> None:
        self.session, self._definition = session, definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def invoke(self, arguments: dict[str, Any], context: RunContext) -> str:
        if context.cancelled.is_set():
            raise MCPToolError("run cancelled")
        result = await self.session.call_tool(self._definition.name, arguments)
        content = []
        for block in result.content:
            if getattr(block, "type", None) == "text":
                content.append(block.text)
            elif hasattr(block, "model_dump"):
                content.append(json.dumps(block.model_dump(mode="json")))
            else:
                content.append(str(block))
        output = "\n".join(content)
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            output = (
                json.dumps({"content": content, "structuredContent": structured})
                if content
                else json.dumps(structured)
            )
        if getattr(result, "isError", False):
            raise MCPToolError(output or "remote tool failed")
        return output


class MCPClient:
    """Wrap a borrowed session, or own one with the transport factories.

    Tools returned by a managed client are valid only inside its context.
    The official SDK owns initialization, HTTP auth, and transport shutdown.
    """

    def __init__(self, session: Any) -> None:
        self.session = session

    @classmethod
    @asynccontextmanager
    async def stdio(
        cls,
        command: str,
        args: Sequence[str] = (),
        *,
        env: Mapping[str, str] | None = None,
    ) -> AsyncIterator[MCPClient]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as error:
            raise RuntimeError("MCP support requires the official MCP SDK") from error
        parameters = StdioServerParameters(
            command=command, args=list(args), env=dict(env) if env is not None else None
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield cls(session)

    @classmethod
    @asynccontextmanager
    async def streamable_http(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        auth: Any = None,
    ) -> AsyncIterator[MCPClient]:
        """Connect using SDK StreamableHTTP and an optional httpx Auth provider.

        Use HTTPS outside trusted local deployments. Headers may contain a
        static bearer credential; auth may implement the SDK OAuth provider.
        """
        try:
            from mcp import ClientSession

            transport_module = importlib.import_module("mcp.client.streamable_http")
        except ImportError as error:
            raise RuntimeError(
                "MCP HTTP support requires the official MCP SDK"
            ) from error
        options = dict(headers) if headers is not None else None
        modern_transport = getattr(transport_module, "streamable_http_client", None)
        if modern_transport is not None:
            async with transport_module.create_mcp_http_client(
                headers=options, auth=auth
            ) as http_client:
                async with modern_transport(url, http_client=http_client) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        yield cls(session)
        else:
            async with transport_module.streamablehttp_client(
                url, headers=options, auth=auth
            ) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield cls(session)

    async def tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            result = (
                await self.session.list_tools(cursor=cursor)
                if cursor is not None
                else await self.session.list_tools()
            )
            tools.extend(
                MCPTool(
                    self.session,
                    ToolDefinition(
                        tool.name, tool.description or "", dict(tool.inputSchema)
                    ),
                )
                for tool in result.tools
            )
            cursor = getattr(result, "nextCursor", None)
            if cursor is None:
                return tools
            if cursor in seen:
                raise RuntimeError("MCP server repeated a pagination cursor")
            seen.add(cursor)
