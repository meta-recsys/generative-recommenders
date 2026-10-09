# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import base64
import json
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from cppilot.adapters.anthropic import AnthropicProvider
from cppilot.adapters.gemini import GeminiProvider
from cppilot.adapters.openai import OpenAIProvider
from cppilot.interfaces import ModelRequest, RetryableModelError, ToolDefinition
from cppilot.items import (
    item_from_dict,
    item_to_dict,
    Message,
    ToolCall,
    ToolResult,
    Usage,
)


class FakeStream:
    def __init__(self, chunks=(), error=None, wait=False):
        self.chunks = iter(chunks)
        self.error = error
        self.wait = wait
        self.entered = asyncio.Event()
        self.aclose = AsyncMock()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.entered.set()
        if self.wait:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None


def openai_response(content="ok", calls=()):
    return NS(
        id="o1",
        model="actual-openai",
        created=1,
        system_fingerprint="fp",
        choices=[
            NS(
                finish_reason="tool_calls" if calls else "stop",
                message=NS(content=content, tool_calls=calls, refusal=None),
            )
        ],
        usage=NS(
            prompt_tokens=10,
            completion_tokens=4,
            prompt_tokens_details=NS(cached_tokens=3),
        ),
    )


def openai_chunk(text=None, calls=(), finish=None, usage=None, index=0):
    return NS(
        id="o1",
        model="actual-openai",
        choices=[
            NS(
                index=index,
                finish_reason=finish,
                delta=NS(content=text, tool_calls=calls),
            )
        ],
        usage=usage,
    )


def anthropic_response(blocks=None):
    return NS(
        id="a1",
        model="actual-anthropic",
        stop_reason="end_turn",
        stop_sequence=None,
        content=blocks if blocks is not None else [NS(type="text", text="ok")],
        usage=NS(input_tokens=12, output_tokens=5, cache_read_input_tokens=7),
    )


def gemini_response(parts=(), usage=None, finish="STOP"):
    return NS(
        response_id="g1",
        model_version="actual-gemini",
        prompt_feedback=None,
        candidates=[
            NS(content=NS(parts=parts), finish_reason=finish, safety_ratings=[])
        ],
        usage_metadata=usage,
    )


def client_for(provider, result=None):
    method = AsyncMock(return_value=result)
    if provider is OpenAIProvider:
        return NS(chat=NS(completions=NS(create=method))), method
    if provider is AnthropicProvider:
        return NS(messages=NS(create=method)), method
    return (
        NS(aio=NS(models=NS(generate_content=method, generate_content_stream=method))),
        method,
    )


class ProviderTest(unittest.IsolatedAsyncioTestCase):
    def request(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        return ModelRequest(
            [
                Message("system", "history instructions"),
                Message("user", "go"),
                Message("assistant", "checking"),
                ToolCall("one", "lookup", {"x": 1}),
                ToolCall("two", "lookup", {"x": 2}),
                ToolResult("one", "lookup", "first"),
                ToolResult("two", "lookup", "bad", failed=True),
            ],
            tools=[ToolDefinition("lookup", "Look up", schema)],
            output_schema=schema,
            system="current instructions",
        )

    async def test_native_request_mapping_and_grouped_roundtrip(self):
        request = self.request()
        for provider, result in (
            (OpenAIProvider, openai_response()),
            (AnthropicProvider, anthropic_response()),
            (GeminiProvider, gemini_response([NS(text="ok", function_call=None)])),
        ):
            with self.subTest(provider=provider.__name__):
                client, method = client_for(provider, result)
                await provider(client=client).generate(request)
                kwargs = method.call_args.kwargs
                self.assertTrue(provider.capabilities.streaming)
                self.assertTrue(provider.capabilities.structured_output)
                if provider is OpenAIProvider:
                    self.assertEqual(kwargs["messages"][0]["content"], request.system)
                    assistant = kwargs["messages"][3]
                    self.assertEqual(assistant["content"], "checking")
                    self.assertEqual(
                        [c["id"] for c in assistant["tool_calls"]], ["one", "two"]
                    )
                    self.assertEqual(kwargs["messages"][5]["tool_call_id"], "two")
                    self.assertEqual(
                        kwargs["response_format"]["json_schema"]["schema"],
                        request.output_schema,
                    )
                    self.assertEqual(
                        kwargs["tools"][0]["function"]["parameters"],
                        request.tools[0].parameters,
                    )
                elif provider is AnthropicProvider:
                    self.assertEqual(
                        kwargs["system"], "current instructions\nhistory instructions"
                    )
                    self.assertEqual(len(kwargs["messages"][1]["content"]), 3)
                    self.assertEqual(
                        [b["tool_use_id"] for b in kwargs["messages"][2]["content"]],
                        ["one", "two"],
                    )
                    self.assertTrue(kwargs["messages"][2]["content"][1]["is_error"])
                    self.assertEqual(
                        kwargs["output_config"]["format"]["schema"],
                        request.output_schema,
                    )
                    self.assertEqual(
                        kwargs["tools"][0]["input_schema"], request.tools[0].parameters
                    )
                else:
                    self.assertEqual(
                        kwargs["config"]["system_instruction"],
                        "current instructions\nhistory instructions",
                    )
                    self.assertEqual(len(kwargs["contents"][1]["parts"]), 3)
                    responses = kwargs["contents"][2]["parts"]
                    self.assertEqual(
                        [p["function_response"]["id"] for p in responses],
                        ["one", "two"],
                    )
                    self.assertTrue(
                        responses[1]["function_response"]["response"]["failed"]
                    )
                    self.assertEqual(
                        kwargs["config"]["response_json_schema"], request.output_schema
                    )
                    self.assertEqual(
                        kwargs["config"]["tools"][0]["function_declarations"][0][
                            "parameters_json_schema"
                        ],
                        request.tools[0].parameters,
                    )

    async def test_runner_dual_system_representation_and_fallback(self):
        instructions = "Use tools.\nKeep answers brief."
        cases = (
            (
                ModelRequest(
                    [Message("system", instructions), Message("user", "go")],
                    system=instructions,
                ),
                [instructions],
            ),
            (
                ModelRequest(
                    [
                        Message("system", instructions),
                        Message("system", "Independent rule"),
                        Message("user", "go"),
                    ],
                    system=instructions,
                ),
                [instructions, "Independent rule"],
            ),
            (
                ModelRequest([Message("user", "go")], system=instructions),
                [instructions],
            ),
            (
                ModelRequest([Message("system", instructions), Message("user", "go")]),
                [instructions],
            ),
            (
                ModelRequest(
                    [
                        Message("system", "Independent rule"),
                        Message("system", "Independent rule"),
                        Message("user", "go"),
                    ],
                    system=instructions,
                ),
                [instructions, "Independent rule", "Independent rule"],
            ),
        )
        for provider, response in (
            (OpenAIProvider, openai_response()),
            (AnthropicProvider, anthropic_response()),
            (GeminiProvider, gemini_response()),
        ):
            for request, expected in cases:
                with self.subTest(provider=provider.__name__, expected=expected):
                    client, method = client_for(provider, response)
                    await provider(client=client).generate(request)
                    kwargs = method.call_args.kwargs
                    if provider is OpenAIProvider:
                        actual = [
                            message["content"]
                            for message in kwargs["messages"]
                            if message["role"] == "system"
                        ]
                        self.assertEqual(actual, expected)
                    elif provider is AnthropicProvider:
                        self.assertEqual(kwargs["system"], "\n".join(expected))
                    else:
                        self.assertEqual(
                            kwargs["config"]["system_instruction"], "\n".join(expected)
                        )

    async def test_signed_text_parts_preserve_native_replay_order(self):
        parts = [
            NS(text="first", function_call=None, thought_signature=b"first-signature"),
            NS(text="private", thought=True, thought_signature=b"thinking-signature"),
            NS(
                text="second", function_call=None, thought_signature=b"second-signature"
            ),
        ]
        client, _ = client_for(GeminiProvider, gemini_response(parts))
        provider = GeminiProvider(client=client)
        response = await provider.generate(ModelRequest([]))
        stored = [
            item_from_dict(json.loads(json.dumps(item_to_dict(item))))
            for item in response.items
        ]
        replay = provider._kwargs(ModelRequest(stored))["contents"][0]["parts"]
        self.assertEqual(
            replay,
            [
                {"text": "first", "thought_signature": b"first-signature"},
                {
                    "text": "private",
                    "thought": True,
                    "thought_signature": b"thinking-signature",
                },
                {"text": "second", "thought_signature": b"second-signature"},
            ],
        )
        client, _ = client_for(
            AnthropicProvider,
            anthropic_response(
                [
                    NS(type="text", text="first"),
                    NS(type="thinking", thinking="private", signature="signed"),
                    NS(type="text", text="second"),
                ]
            ),
        )
        provider = AnthropicProvider(client=client)
        response = await provider.generate(ModelRequest([]))
        replay = provider._kwargs(ModelRequest(response.items))["messages"][0][
            "content"
        ]
        self.assertEqual(
            replay,
            [
                {"type": "text", "text": "first"},
                {"type": "thinking", "thinking": "private", "signature": "signed"},
                {"type": "text", "text": "second"},
            ],
        )

    async def test_empty_tools_and_schema_not_sent(self):
        for provider, result in (
            (OpenAIProvider, openai_response()),
            (AnthropicProvider, anthropic_response()),
            (GeminiProvider, gemini_response()),
        ):
            with self.subTest(provider=provider.__name__):
                client, method = client_for(provider, result)
                await provider(client=client).generate(
                    ModelRequest([Message("user", "go")])
                )
                kwargs = method.call_args.kwargs
                config = kwargs.get("config", kwargs)
                self.assertNotIn("tools", config)
                self.assertNotIn("response_format", config)
                self.assertNotIn("output_config", config)
                self.assertNotIn("response_json_schema", config)

    async def test_generate_preserves_metadata_usage_and_complete_calls(self):
        responses = (
            (
                OpenAIProvider,
                openai_response(
                    "text",
                    [NS(id="c", function=NS(name="lookup", arguments='{"x":1}'))],
                ),
                10,
                4,
                "id",
            ),
            (
                AnthropicProvider,
                anthropic_response(
                    [
                        NS(type="text", text="text"),
                        NS(type="tool_use", id="c", name="lookup", input={"x": 1}),
                    ]
                ),
                19,
                5,
                "id",
            ),
            (
                GeminiProvider,
                gemini_response(
                    [
                        NS(text="text", function_call=None),
                        NS(
                            text=None,
                            function_call=NS(id="c", name="lookup", args={"x": 1}),
                        ),
                    ],
                    NS(
                        prompt_token_count=10,
                        candidates_token_count=4,
                        thoughts_token_count=2,
                        cached_content_token_count=3,
                    ),
                ),
                10,
                6,
                "response_id",
            ),
        )
        for provider, result, input_tokens, output_tokens, id_key in responses:
            with self.subTest(provider=provider.__name__):
                client, _ = client_for(provider, result)
                response = await provider(client=client).generate(ModelRequest([]))
                self.assertEqual(response.items[0].content, "text")
                self.assertEqual(response.items[1].arguments, {"x": 1})
                self.assertEqual(response.items[1].call_id, "c")
                self.assertIn(id_key, response.items[0].metadata)
                self.assertEqual(response.usage.input_tokens, input_tokens)
                self.assertEqual(response.usage.output_tokens, output_tokens)
                self.assertIn(id_key, response.usage.metadata)
                self.assertEqual(
                    response.usage.model, "actual-" + response.usage.provider
                )
                self.assertIn("usage", response.usage.metadata)
                json.dumps(item_to_dict(response.usage))

    async def test_openai_stream_fragmented_parallel_calls_and_final_usage(self):
        def call(index, call_id=None, name=None, args=None):
            return NS(index=index, id=call_id, function=NS(name=name, arguments=args))

        stream = FakeStream(
            [
                openai_chunk("hel", [call(1, "two", "look", '{"x":')]),
                openai_chunk(
                    "lo",
                    [call(0, "one", "lookup", "{}"), call(1, name="up", args="2}")],
                ),
                openai_chunk(finish="tool_calls"),
                NS(
                    id="o1",
                    model="actual-openai",
                    choices=[],
                    usage=NS(prompt_tokens=10, completion_tokens=4),
                ),
            ]
        )
        client, method = client_for(OpenAIProvider, stream)
        items = [
            item
            async for item in OpenAIProvider(client=client).stream(ModelRequest([]))
        ]
        self.assertEqual(
            [i.content for i in items if isinstance(i, Message)], ["hel", "lo"]
        )
        calls = [i for i in items if isinstance(i, ToolCall)]
        self.assertEqual(
            [(c.call_id, c.name, c.arguments) for c in calls],
            [("one", "lookup", {}), ("two", "lookup", {"x": 2})],
        )
        self.assertIsInstance(items[-1], Usage)
        self.assertEqual(items[-1].output_tokens, 4)
        self.assertEqual(items[-1].metadata["finish_reason"], "tool_calls")
        self.assertTrue(method.call_args.kwargs["stream"])
        self.assertEqual(
            method.call_args.kwargs["stream_options"], {"include_usage": True}
        )
        stream.aclose.assert_awaited_once()

    async def test_anthropic_stream_blocks_cache_usage_and_thinking_replay(self):
        stream = FakeStream(
            [
                NS(type="message_start", message=anthropic_response([])),
                NS(
                    type="content_block_start",
                    index=0,
                    content_block=NS(type="thinking", thinking="", signature=""),
                ),
                NS(
                    type="content_block_delta",
                    index=0,
                    delta=NS(type="thinking_delta", thinking="private"),
                ),
                NS(
                    type="content_block_delta",
                    index=0,
                    delta=NS(type="signature_delta", signature="signed"),
                ),
                NS(type="content_block_stop", index=0),
                NS(
                    type="content_block_start",
                    index=1,
                    content_block=NS(type="text", text=""),
                ),
                NS(
                    type="content_block_delta",
                    index=1,
                    delta=NS(type="text_delta", text="checking"),
                ),
                NS(type="content_block_stop", index=1),
                NS(
                    type="content_block_start",
                    index=2,
                    content_block=NS(type="tool_use", id="c", name="lookup", input={}),
                ),
                NS(
                    type="content_block_delta",
                    index=2,
                    delta=NS(type="input_json_delta", partial_json='{"x":'),
                ),
                NS(
                    type="content_block_delta",
                    index=2,
                    delta=NS(type="input_json_delta", partial_json="1}"),
                ),
                NS(type="content_block_stop", index=2),
                NS(
                    type="message_delta",
                    delta=NS(stop_reason="tool_use", stop_sequence=None),
                    usage=NS(output_tokens=9),
                ),
            ]
        )
        client, method = client_for(AnthropicProvider, stream)
        provider = AnthropicProvider(client=client)
        items = [item async for item in provider.stream(ModelRequest([]))]
        self.assertEqual(
            [i.content for i in items if isinstance(i, Message)], ["checking"]
        )
        self.assertEqual(items[1].arguments, {"x": 1})
        self.assertEqual(items[-1].input_tokens, 19)
        self.assertEqual(items[-1].output_tokens, 9)
        self.assertEqual(items[-1].metadata["usage"]["cache_read_input_tokens"], 7)
        self.assertEqual(items[-1].metadata["stop_reason"], "tool_use")
        replay = provider._kwargs(ModelRequest(items))
        self.assertEqual(
            replay["messages"][0]["content"][0],
            {"type": "thinking", "thinking": "private", "signature": "signed"},
        )
        self.assertTrue(method.call_args.kwargs["stream"])
        stream.aclose.assert_awaited_once()

    async def test_gemini_stream_usage_signatures_candidate_selection_and_ids(self):
        first = gemini_response([NS(text="hi", function_call=None)])
        first.candidates.append(NS(content=NS(parts=[NS(text="wrong candidate")])))
        second = gemini_response(
            [
                NS(
                    text=None,
                    function_call=NS(id=None, name="lookup", args={"x": 1}),
                    thought_signature=b"signature",
                )
            ],
            NS(prompt_token_count=8, candidates_token_count=3, thoughts_token_count=2),
        )
        final = gemini_response(
            [],
            NS(prompt_token_count=8, candidates_token_count=4, thoughts_token_count=2),
        )
        stream = FakeStream([first, second, final])
        client, method = client_for(GeminiProvider, stream)
        provider = GeminiProvider(client=client)
        items = [item async for item in provider.stream(ModelRequest([]))]
        self.assertEqual(items[0].content, "hi")
        self.assertIsInstance(items[1], ToolCall)
        self.assertTrue(items[1].call_id.startswith("gemini-"))
        self.assertEqual(items[-1].input_tokens, 8)
        self.assertEqual(items[-1].output_tokens, 6)
        self.assertEqual(items[-1].metadata["finish_reason"], "STOP")
        stored = item_from_dict(json.loads(json.dumps(item_to_dict(items[1]))))
        replay = provider._kwargs(
            ModelRequest([stored, ToolResult(stored.call_id, stored.name, "ok")])
        )
        call_part = replay["contents"][0]["parts"][0]
        self.assertEqual(call_part["thought_signature"], b"signature")
        self.assertNotIn("id", call_part["function_call"])
        self.assertNotIn("id", replay["contents"][1]["parts"][0]["function_response"])
        method.assert_awaited_once()
        stream.aclose.assert_awaited_once()

    async def test_gemini_blocked_null_candidates(self):
        response = NS(
            candidates=None,
            prompt_feedback=NS(block_reason="SAFETY"),
            usage_metadata=None,
        )
        client, _ = client_for(GeminiProvider, response)
        result = await GeminiProvider(client=client).generate(ModelRequest([]))
        self.assertEqual(result.items, [])
        self.assertEqual(result.metadata["prompt_feedback"]["block_reason"], "SAFETY")
        self.assertEqual(result.usage.output_tokens, 0)

    async def test_all_streams_close_on_early_exit_and_cancel(self):
        for provider in (OpenAIProvider, AnthropicProvider, GeminiProvider):
            for cancel in (False, True):
                with self.subTest(provider=provider.__name__, cancel=cancel):
                    chunks = {
                        OpenAIProvider: [openai_chunk("hi")],
                        AnthropicProvider: [
                            NS(
                                type="content_block_start",
                                index=0,
                                content_block=NS(type="text", text="hi"),
                            )
                        ],
                        GeminiProvider: [
                            gemini_response([NS(text="hi", function_call=None)])
                        ],
                    }
                    sdk_stream = FakeStream(chunks[provider], wait=cancel)
                    client, _ = client_for(provider, sdk_stream)
                    iterator = provider(client=client).stream(ModelRequest([]))
                    if cancel:
                        task = asyncio.create_task(anext(iterator))
                        await sdk_stream.entered.wait()
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                    else:
                        item = await anext(iterator)
                        self.assertEqual(item.content, "hi")
                        await iterator.aclose()
                    sdk_stream.aclose.assert_awaited_once()

    async def test_sdk_retryable_errors_and_nonretryable_passthrough(self):
        class APIError(Exception):
            def __init__(self, status):
                super().__init__(str(status))
                self.status_code = status
                self.code = status

        class APIConnectionError(APIError):
            pass

        module = NS(APIError=APIError, APIConnectionError=APIConnectionError)
        for provider, module_name in (
            (OpenAIProvider, "openai"),
            (AnthropicProvider, "anthropic"),
            (GeminiProvider, "google.genai.errors"),
        ):
            for status in (408, 429, 500, 503, 400, 401, 403, 404):
                for streaming in (False, True):
                    with self.subTest(
                        provider=provider.__name__, status=status, streaming=streaming
                    ):
                        error = APIError(status)
                        sdk_stream = FakeStream(error=error)
                        client, method = client_for(provider, sdk_stream)
                        if not streaming:
                            method.side_effect = error
                        adapter = provider(client=client)
                        expected = (
                            RetryableModelError
                            if status in (408, 429, 500, 503)
                            else APIError
                        )
                        with patch.dict(sys.modules, {module_name: module}):
                            with self.assertRaises(expected) as caught:
                                if streaming:
                                    _ = [
                                        i
                                        async for i in adapter.stream(ModelRequest([]))
                                    ]
                                else:
                                    await adapter.generate(ModelRequest([]))
                        if expected is RetryableModelError:
                            self.assertIs(caught.exception.__cause__, error)
                        else:
                            self.assertIs(caught.exception, error)
                        if streaming:
                            sdk_stream.aclose.assert_awaited_once()
            client, method = client_for(provider)
            error = APIConnectionError(None)
            method.side_effect = error
            with patch.dict(sys.modules, {module_name: module}):
                with self.assertRaises(RetryableModelError) as caught:
                    await provider(client=client).generate(ModelRequest([]))
            self.assertIs(caught.exception.__cause__, error)

    async def test_gemini_thought_only_chunks_replay_without_leaking_text(self):
        stream = FakeStream(
            [
                gemini_response(
                    [NS(text="private", thought=True, thought_signature=b"signed")]
                ),
                gemini_response(
                    [NS(text=None, function_call=NS(id="c", name="lookup", args={}))]
                ),
            ]
        )
        client, _ = client_for(GeminiProvider, stream)
        provider = GeminiProvider(client=client)
        items = [item async for item in provider.stream(ModelRequest([]))]
        self.assertEqual(len(items), 2)
        self.assertIsInstance(items[0], ToolCall)
        stored = item_from_dict(json.loads(json.dumps(item_to_dict(items[0]))))
        replay = provider._kwargs(ModelRequest([stored]))
        self.assertEqual(
            replay["contents"][0]["parts"][0],
            {
                "text": "private",
                "thought": True,
                "thought_signature": b"signed",
            },
        )
        self.assertEqual(
            stored.metadata["gemini_parts_before"][0]["thought_signature"],
            base64.b64encode(b"signed").decode("ascii"),
        )

    async def test_stream_open_failure_translates_and_preserves_cause(self):
        class APIError(Exception):
            status_code = 503
            code = 503

        for provider, name in (
            (OpenAIProvider, "openai"),
            (AnthropicProvider, "anthropic"),
            (GeminiProvider, "google.genai.errors"),
        ):
            with self.subTest(provider=provider.__name__):
                client, method = client_for(provider)
                error = APIError("unavailable")
                method.side_effect = error
                with patch.dict(sys.modules, {name: NS(APIError=APIError)}):
                    with self.assertRaises(RetryableModelError) as caught:
                        _ = [
                            item
                            async for item in provider(client=client).stream(
                                ModelRequest([])
                            )
                        ]
                self.assertIs(caught.exception.__cause__, error)

    async def test_gemini_httpx_transport_failure(self):
        class TransportError(Exception):
            pass

        client, method = client_for(GeminiProvider)
        error = TransportError("network")
        method.side_effect = error
        with patch.dict(
            sys.modules,
            {
                "google.genai.errors": NS(),
                "httpx": NS(TransportError=TransportError),
            },
        ):
            with self.assertRaises(RetryableModelError) as caught:
                await GeminiProvider(client=client).generate(ModelRequest([]))
        self.assertIs(caught.exception.__cause__, error)

    async def test_anthropic_error_events_and_incomplete_blocks_close(self):
        for chunks, expected in (
            (
                [NS(type="error", error=NS(type="overloaded_error", message="busy"))],
                RetryableModelError,
            ),
            (
                [
                    NS(
                        type="error",
                        error=NS(type="invalid_request_error", message="bad"),
                    )
                ],
                RuntimeError,
            ),
            (
                [
                    NS(
                        type="content_block_start",
                        index=0,
                        content_block=NS(
                            type="tool_use", id="c", name="lookup", input={}
                        ),
                    )
                ],
                ValueError,
            ),
        ):
            with self.subTest(expected=expected.__name__):
                stream = FakeStream(chunks)
                client, _ = client_for(AnthropicProvider, stream)
                with self.assertRaises(expected):
                    _ = [
                        item
                        async for item in AnthropicProvider(client=client).stream(
                            ModelRequest([])
                        )
                    ]
                stream.aclose.assert_awaited_once()

    async def test_openai_refusal_and_nonselected_stream_candidate(self):
        response = openai_response(None)
        response.choices[0].message.refusal = "cannot answer"
        client, _ = client_for(OpenAIProvider, response)
        result = await OpenAIProvider(client=client).generate(ModelRequest([]))
        self.assertEqual(result.items, [])
        self.assertEqual(result.usage.metadata["refusal"], "cannot answer")
        stream = FakeStream(
            [openai_chunk("wrong", index=1), openai_chunk("right", finish="stop")]
        )
        client, _ = client_for(OpenAIProvider, stream)
        items = [
            item
            async for item in OpenAIProvider(client=client).stream(ModelRequest([]))
        ]
        self.assertEqual(
            [i.content for i in items if isinstance(i, Message)], ["right"]
        )
        self.assertEqual(items[-1].metadata["finish_reason"], "stop")

    async def test_sdk_model_dump_preserves_detailed_usage(self):
        class SDKUsage:
            prompt_tokens = 9
            completion_tokens = 3

            def model_dump(self, *, mode, exclude_none):
                return {
                    "prompt_tokens": 9,
                    "completion_tokens": 3,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }

        response = openai_response()
        response.usage = SDKUsage()
        client, _ = client_for(OpenAIProvider, response)
        result = await OpenAIProvider(client=client).generate(ModelRequest([]))
        self.assertEqual(
            result.usage.metadata["usage"]["completion_tokens_details"],
            {"reasoning_tokens": 2},
        )

    async def test_trailing_raw_blocks_generate_stream_and_json_roundtrip(self):
        for provider in (AnthropicProvider, GeminiProvider):
            for represented in ("text", "tool", None):
                for streaming in (False, True):
                    with self.subTest(
                        provider=provider.__name__,
                        represented=represented,
                        streaming=streaming,
                    ):
                        if provider is AnthropicProvider:
                            raw = NS(
                                type="image",
                                source=NS(
                                    type="base64",
                                    media_type="image/png",
                                    data=b"\x00\xffimage",
                                ),
                            )
                            first = (
                                NS(type="text", text="visible")
                                if represented == "text"
                                else NS(
                                    type="tool_use", id="c", name="lookup", input={}
                                )
                            )
                            blocks = ([first] if represented else []) + [
                                NS(
                                    type="thinking",
                                    thinking="private",
                                    signature="signed",
                                ),
                                raw,
                            ]
                            result = anthropic_response(blocks)
                            chunks = [
                                NS(type="message_start", message=anthropic_response([]))
                            ]
                            for index, block in enumerate(blocks):
                                chunks.extend(
                                    [
                                        NS(
                                            type="content_block_start",
                                            index=index,
                                            content_block=block,
                                        ),
                                        NS(type="content_block_stop", index=index),
                                    ]
                                )
                            key, container, field = (
                                "anthropic_blocks_after",
                                "messages",
                                "content",
                            )
                            expected_raw = [
                                {
                                    "type": "thinking",
                                    "thinking": "private",
                                    "signature": "signed",
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b"\x00\xffimage",
                                    },
                                },
                            ]
                        else:
                            raw = NS(
                                inline_data=NS(
                                    mime_type="image/png", data=b"\x00\xffimage"
                                )
                            )
                            first = (
                                NS(text="visible", function_call=None)
                                if represented == "text"
                                else NS(
                                    text=None,
                                    function_call=NS(id="c", name="lookup", args={}),
                                )
                            )
                            blocks = ([first] if represented else []) + [
                                NS(
                                    text="private",
                                    thought=True,
                                    thought_signature=b"signed",
                                ),
                                raw,
                            ]
                            result = gemini_response(blocks)
                            chunks = [gemini_response([block]) for block in blocks]
                            key, container, field = (
                                "gemini_parts_after",
                                "contents",
                                "parts",
                            )
                            expected_raw = [
                                {
                                    "text": "private",
                                    "thought": True,
                                    "thought_signature": b"signed",
                                },
                                {
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": b"\x00\xffimage",
                                    }
                                },
                            ]
                        sdk_stream = FakeStream(chunks)
                        client, _ = client_for(
                            provider, sdk_stream if streaming else result
                        )
                        adapter = provider(client=client)
                        if streaming:
                            items = [
                                item async for item in adapter.stream(ModelRequest([]))
                            ]
                            self.assertIsInstance(items[-1], Usage)
                            items = items[:-1]
                            sdk_stream.aclose.assert_awaited_once()
                        else:
                            items = list(
                                (await adapter.generate(ModelRequest([]))).items
                            )
                        self.assertIn(key, items[-1].metadata)
                        if streaming or represented is None:
                            self.assertEqual(items[-1].content, "")
                            self.assertTrue(items[-1].metadata["replay_only"])
                        stored = [
                            item_from_dict(json.loads(json.dumps(item_to_dict(item))))
                            for item in items
                        ]
                        replay = adapter._kwargs(ModelRequest(stored))[container][0][
                            field
                        ]
                        self.assertEqual(replay[-2:], expected_raw)
                        self.assertEqual(len(replay), 3 if represented else 2)
                        self.assertEqual(
                            "".join(
                                item.content
                                for item in items
                                if isinstance(item, Message)
                            ),
                            "visible" if represented == "text" else "",
                        )

    async def test_older_gemini_tool_schema_feature_guard(self):
        client, method = client_for(GeminiProvider, gemini_response())
        types = NS(FunctionDeclaration=NS(model_fields={"parameters": None}))
        with patch.dict(sys.modules, {"google.genai.types": types}):
            with self.assertRaisesRegex(ValueError, "parameters_json_schema; upgrade"):
                await GeminiProvider(client=client).generate(self.request())
        method.assert_not_awaited()

    async def test_native_sdk_raw_dump_preserves_bytes_and_enums(self):
        from enum import Enum

        class MediaKind(str, Enum):
            IMAGE = "image/png"

        class NativePart:
            def model_dump(self, *, mode, exclude_none):
                self.mode = mode
                return {
                    "inline_data": {"mime_type": MediaKind.IMAGE, "data": b"\xff\x00"}
                }

        part = NativePart()
        client, _ = client_for(GeminiProvider, gemini_response([part]))
        provider = GeminiProvider(client=client)
        response = await provider.generate(ModelRequest([]))
        stored = item_from_dict(json.loads(json.dumps(item_to_dict(response.items[0]))))
        self.assertEqual(part.mode, "python")
        self.assertEqual(
            provider._kwargs(ModelRequest([stored]))["contents"][0]["parts"],
            [{"inline_data": {"mime_type": "image/png", "data": b"\xff\x00"}}],
        )

    async def test_older_sdk_structured_output_prompt_fallback(self):
        async def old_create(
            *, model, max_tokens, system, messages, stream=False, tools=None
        ):
            return anthropic_response()

        client = NS(messages=NS(create=old_create))
        adapter = AnthropicProvider(client=client)
        request = self.request()
        kwargs = adapter._kwargs(request)
        self.assertNotIn("output_config", kwargs)
        self.assertIn(json.dumps(request.output_schema), kwargs["system"])
        response = await adapter.generate(request)
        self.assertEqual(response.items[0].content, "ok")
        client, method = client_for(GeminiProvider, gemini_response())
        types = NS(
            GenerateContentConfig=NS(model_fields={"response_mime_type": None}),
            FunctionDeclaration=NS(model_fields={"parameters_json_schema": None}),
        )
        with patch.dict(sys.modules, {"google.genai.types": types}):
            await GeminiProvider(client=client).generate(request)
        config = method.call_args.kwargs["config"]
        self.assertNotIn("response_json_schema", config)
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertIn(json.dumps(request.output_schema), config["system_instruction"])

    async def test_invalid_tool_json_does_not_become_retryable(self):
        for arguments in ('{"broken":', "[]"):
            client, _ = client_for(
                OpenAIProvider,
                openai_response(
                    None, [NS(id="c", function=NS(name="lookup", arguments=arguments))]
                ),
            )
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    await OpenAIProvider(client=client).generate(ModelRequest([]))
