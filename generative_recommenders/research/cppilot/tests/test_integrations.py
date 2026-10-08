# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from cppilot.adapters.opentelemetry import OpenTelemetryEventSink
from cppilot.agent import Agent
from cppilot.eval import (
    accuracy,
    DatasetRunner,
    EvaluationExample,
    EvaluationResult,
    PromptRunner,
    summarize,
    token_f1,
    token_precision,
    token_recall,
)
from cppilot.integrations import (
    BearerTokenAuth,
    FsspecObjectStore,
    KubernetesJobBackend,
    MLflowMetricsBackend,
    PrometheusMetricsBackend,
    SQLiteReadOnlyBackend,
)
from cppilot.interfaces import Event, ToolDefinition
from cppilot.items import Message, Usage
from cppilot.mcp import MCPClient, MCPTool, MCPToolError
from cppilot.runner import GuardrailTriggered, MaxTurnsExceeded, StructuredOutputError
from cppilot.server import _connected, ClientDisconnected, create_app
from cppilot.tools.core import RunContext


def module(name: str, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class BackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_bearer_auth(self):
        auth = BearerTokenAuth("secret", principal="alice")
        self.assertEqual(await auth.authenticate("bEaReR secret"), "alice")
        for credential in (
            None,
            "",
            "secret",
            "Basic secret",
            "Bearer wrong",
            "Bearer secret extra",
        ):
            with (
                self.subTest(credential=credential),
                self.assertRaises(PermissionError),
            ):
                await auth.authenticate(credential)
        with self.assertRaises(ValueError):
            BearerTokenAuth("")

    async def test_sql_readonly_bound_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data #?.db"
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE data (value TEXT)")
            connection.executemany(
                "INSERT INTO data VALUES (?)",
                [("a",), ("b",), ("'; DROP TABLE data; --",)],
            )
            connection.commit()
            connection.close()
            backend = SQLiteReadOnlyBackend(path, max_rows=2)
            self.assertEqual(
                await backend.query("SELECT value FROM data ORDER BY rowid"),
                [{"value": "a"}, {"value": "b"}],
            )
            dangerous = "'; DROP TABLE data; --"
            self.assertEqual(
                await backend.query(
                    "SELECT value FROM data WHERE value = ?", [dangerous]
                ),
                [{"value": dangerous}],
            )
            self.assertEqual(
                await backend.query(
                    "WITH t AS (SELECT value FROM data) SELECT count(*) AS n FROM t"
                ),
                [{"n": 3}],
            )
            for sql in (
                "DELETE FROM data",
                "UPDATE data SET value='bad'",
                "DROP TABLE data",
                "CREATE TABLE other (x)",
                "PRAGMA query_only=OFF",
                "PRAGMA journal_mode=WAL",
                "ATTACH DATABASE ':memory:' AS other",
                "BEGIN",
                "VACUUM",
                "SELECT load_extension('bad')",
                "SELECT 1; DELETE FROM data",
            ):
                with self.subTest(sql=sql), self.assertRaises(sqlite3.Error):
                    await backend.query(sql)
            self.assertEqual(
                await backend.query("SELECT count(*) AS n FROM data"), [{"n": 3}]
            )

    async def test_sql_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.db"
            sqlite3.connect(path).close()
            backend = SQLiteReadOnlyBackend(path, timeout=0.001)
            with self.assertRaises(sqlite3.OperationalError):
                await backend.query(
                    "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<100000000) SELECT sum(x) FROM n"
                )
            self.assertEqual(await backend.query("SELECT 1 AS n"), [{"n": 1}])

    async def test_fsspec_storage_closes_streams(self):
        filesystem = MagicMock()
        filesystem._strip_protocol.side_effect = lambda uri: uri.removeprefix(
            "memory://"
        )
        filesystem.unstrip_protocol.side_effect = lambda path: "memory://" + path
        stream = io.BytesIO(b"value")
        filesystem.open.return_value = stream
        store = FsspecObjectStore(filesystem=filesystem)
        self.assertEqual(await store.read("memory://object"), b"value")
        self.assertTrue(stream.closed)
        filesystem.open.assert_called_once_with("object", "rb")
        output = MagicMock()
        filesystem.open.return_value = output
        await store.write("memory://object", b"new")
        output.__enter__.return_value.write.assert_called_once_with(b"new")
        output.__exit__.assert_called_once()
        filesystem.ls.return_value = ["a", "b"]
        self.assertEqual(
            [entry async for entry in store.list("memory://")],
            ["memory://a", "memory://b"],
        )
        filesystem.ls.assert_called_once_with("", detail=False)

    async def test_fsspec_lazy_sdk(self):
        filesystem = MagicMock()
        filesystem.open.return_value = io.BytesIO(b"ok")
        resolve = MagicMock(return_value=(filesystem, "key"))
        with patch.dict(
            "sys.modules", {"fsspec.core": module("fsspec.core", url_to_fs=resolve)}
        ):
            self.assertEqual(
                await FsspecObjectStore(storage_options={"token": "credential"}).read(
                    "scheme://key"
                ),
                b"ok",
            )
        resolve.assert_called_once_with("scheme://key", token="credential")

    async def test_kubernetes_manifest_and_status(self):
        api = MagicMock()
        api.read_namespaced_job_status.return_value.to_dict.return_value = {
            "status": {"succeeded": 1}
        }
        backend = KubernetesJobBackend("eval", api=api)
        payload = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": "old", "namespace": "other"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "worker", "image": "image", "args": ["a;b"]}
                        ]
                    }
                }
            },
        }
        self.assertEqual(await backend.submit("run-1", payload), "run-1")
        manifest = api.create_namespaced_job.call_args.kwargs["body"]
        self.assertEqual(manifest["metadata"], {"name": "run-1", "namespace": "eval"})
        self.assertEqual(payload["metadata"]["name"], "old")
        self.assertEqual(
            manifest["spec"]["template"]["spec"]["containers"][0]["args"], ["a;b"]
        )
        self.assertEqual(await backend.status("run-1"), {"status": {"succeeded": 1}})
        api.read_namespaced_job_status.assert_called_once_with(
            name="run-1", namespace="eval"
        )
        with self.assertRaises(ValueError):
            await backend.submit("../bad", payload)
        with self.assertRaises(ValueError):
            await backend.submit("good", {"kind": "Pod"})

    async def test_prometheus_counters_and_label_contract(self):
        sdk = MagicMock()
        registry = object()
        backend = PrometheusMetricsBackend(sdk=sdk, registry=registry)
        await backend.increment("calls", 2, labels={"agent": "one"})
        await backend.increment("calls", labels={"agent": "two"})
        sdk.Counter.assert_called_once_with(
            "calls", "CPPilot counter: calls", labelnames=("agent",), registry=registry
        )
        self.assertEqual(
            sdk.Counter.return_value.labels.return_value.inc.call_args_list[0].args,
            (2.0,),
        )
        self.assertEqual(
            sdk.Counter.return_value.labels.call_args.kwargs, {"agent": "two"}
        )
        with self.assertRaises(ValueError):
            await backend.increment("calls", labels={"tool": "other"})
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                await backend.increment("calls", value)

    async def test_mlflow_explicit_run_cumulative_and_failure(self):
        client = MagicMock()
        backend = MLflowMetricsBackend("run", client=client)
        await backend.increment("calls", 2)
        client.log_metric.side_effect = OSError("unavailable")
        with self.assertRaises(OSError):
            await backend.increment("calls", 3)
        client.log_metric.side_effect = None
        await backend.increment("calls", 1)
        kwargs = client.log_metric.call_args.kwargs
        self.assertEqual(
            (kwargs["run_id"], kwargs["value"], kwargs["step"]), ("run", 3.0, 1)
        )
        with self.assertRaises(ValueError):
            await backend.increment("calls", labels={"agent": "one"})


class EvaluationTest(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_and_empty_semantics(self):
        self.assertEqual(accuracy("A", "a"), 0)
        self.assertEqual(token_precision("a a b", "a b b c"), 0.5)
        self.assertAlmostEqual(token_recall("a a b", "a b b c"), 2 / 3)
        self.assertAlmostEqual(token_f1("a a b", "a b b c"), 4 / 7)
        for metric in (token_precision, token_recall, token_f1):
            self.assertEqual(metric("", ""), 1)
            self.assertEqual(metric("a", ""), 0)
            self.assertEqual(metric("", "a"), 0)

    async def test_dataset_order_concurrency_async_metrics(self):
        active = 0
        peak = 0
        first_pair = asyncio.Event()

        async def predict(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                first_pair.set()
            await first_pair.wait()
            active -= 1
            return str(value)

        async def score(expected, actual):
            return accuracy(expected, actual)

        results = await DatasetRunner(predict, score, metrics={"f1": token_f1}).run(
            [EvaluationExample(i, str(i)) for i in range(4)], concurrency=2
        )
        self.assertEqual([result.actual for result in results], ["0", "1", "2", "3"])
        self.assertEqual(peak, 2)
        summary = summarize(results)
        self.assertEqual(
            (summary.count, summary.mean_score, summary.metrics), (4, 1.0, {"f1": 1.0})
        )
        self.assertEqual(summarize([]).mean_score, 0)
        for concurrency in (0, -1, True, 1.5):
            with (
                self.subTest(concurrency=concurrency),
                self.assertRaises((ValueError, TypeError)),
            ):
                await DatasetRunner(predict, score).run([], concurrency=concurrency)
        with self.assertRaises(ValueError):
            EvaluationResult("a", "a", float("nan"))

    async def test_dataset_failure_cancels_siblings(self):
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def predict(value):
            if value == "fail":
                await started.wait()
                raise ValueError("prediction failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with self.assertRaisesRegex(ValueError, "prediction failed"):
            await DatasetRunner(predict, accuracy).run(
                [EvaluationExample("fail", ""), EvaluationExample("wait", "")],
                concurrency=2,
            )
        self.assertTrue(stopped.is_set())

    async def test_prompt_comparison(self):
        predict = AsyncMock(
            side_effect=lambda prompt: "yes" if prompt.startswith("good") else "no"
        )
        results = await PromptRunner(
            predict, {"first": "good {question}", "second": "bad {question}"}
        ).run([EvaluationExample({"question": "answer?"}, "yes")])
        self.assertEqual(results["first"][0].score, 1)
        self.assertEqual(results["second"][0].score, 0)
        self.assertEqual(predict.await_args_list[0].args, ("good answer?",))
        self.assertEqual(results["first"][0].metrics["token_f1"], 1)


class MCPTest(unittest.IsolatedAsyncioTestCase):
    async def test_tool_errors_and_structured_content(self):
        session = SimpleNamespace(
            call_tool=AsyncMock(
                return_value=SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="bad")], isError=True
                )
            )
        )
        tool = MCPTool(session, ToolDefinition("tool", "", {}))
        with self.assertRaisesRegex(MCPToolError, "bad"):
            await tool.invoke({"x": 1}, RunContext())
        session.call_tool.assert_awaited_once_with("tool", {"x": 1})
        session.call_tool.return_value = SimpleNamespace(
            content=[], structuredContent={"ok": True}, isError=False
        )
        self.assertEqual(json.loads(await tool.invoke({}, RunContext())), {"ok": True})
        session.call_tool.return_value.content = [
            SimpleNamespace(type="text", text="done")
        ]
        self.assertEqual(
            json.loads(await tool.invoke({}, RunContext())),
            {"content": ["done"], "structuredContent": {"ok": True}},
        )

    async def test_tools_pagination(self):
        remote = SimpleNamespace(
            name="a", description=None, inputSchema={"type": "object"}
        )
        session = SimpleNamespace(
            list_tools=AsyncMock(
                side_effect=[
                    SimpleNamespace(tools=[remote], nextCursor=""),
                    SimpleNamespace(tools=[], nextCursor=None),
                ]
            )
        )
        tools = await MCPClient(session).tools()
        self.assertEqual(
            tools[0].definition, ToolDefinition("a", "", {"type": "object"})
        )
        self.assertEqual(session.list_tools.await_args_list[1].kwargs, {"cursor": ""})

    async def test_modern_http_client_lifecycle(self):
        actions = []
        session = SimpleNamespace(initialize=AsyncMock())
        auth = object()

        @asynccontextmanager
        async def http_factory(*, headers, auth):
            actions.append((headers, auth))
            try:
                yield "http-client"
            finally:
                actions.append("http closed")

        @asynccontextmanager
        async def transport(url, *, http_client):
            self.assertEqual(
                (url, http_client), ("https://example.test/mcp", "http-client")
            )
            try:
                yield "read", "write"
            finally:
                actions.append("transport closed")

        @asynccontextmanager
        async def managed_session(read, write):
            self.assertEqual((read, write), ("read", "write"))
            try:
                yield session
            finally:
                actions.append("session closed")

        with patch.dict(
            "sys.modules",
            {
                "mcp": module("mcp", ClientSession=managed_session),
                "mcp.client.streamable_http": module(
                    "mcp.client.streamable_http",
                    streamable_http_client=transport,
                    create_mcp_http_client=http_factory,
                ),
            },
        ):
            async with MCPClient.streamable_http(
                "https://example.test/mcp",
                headers={"Authorization": "Bearer token"},
                auth=auth,
            ) as client:
                self.assertIs(client.session, session)
                session.initialize.assert_awaited_once()
        self.assertEqual(actions[0], ({"Authorization": "Bearer token"}, auth))
        self.assertEqual(
            actions[1:], ["session closed", "transport closed", "http closed"]
        )

    async def test_managed_transport_lifecycle_and_auth(self):
        actions = []
        session = SimpleNamespace(initialize=AsyncMock())

        @asynccontextmanager
        async def transport(*args, **kwargs):
            actions.append(("transport", args, kwargs))
            try:
                yield (
                    ("read", "write", lambda: "id")
                    if args and isinstance(args[0], str)
                    else ("read", "write")
                )
            finally:
                actions.append("transport closed")

        @asynccontextmanager
        async def managed_session(read, write):
            self.assertEqual((read, write), ("read", "write"))
            try:
                yield session
            finally:
                actions.append("session closed")

        parameters = MagicMock(return_value=object())
        modules = {
            "mcp": module(
                "mcp", ClientSession=managed_session, StdioServerParameters=parameters
            ),
            "mcp.client.stdio": module("mcp.client.stdio", stdio_client=transport),
            "mcp.client.streamable_http": module(
                "mcp.client.streamable_http", streamablehttp_client=transport
            ),
        }
        with patch.dict("sys.modules", modules):
            async with MCPClient.stdio(
                "python", ["worker.py"], env={"KEY": "value"}
            ) as client:
                self.assertIs(client.session, session)
            parameters.assert_called_once_with(
                command="python", args=["worker.py"], env={"KEY": "value"}
            )
            self.assertEqual(actions[-2:], ["session closed", "transport closed"])
            auth = object()
            with self.assertRaisesRegex(ValueError, "consumer"):
                async with MCPClient.streamable_http(
                    "https://example.test/mcp",
                    headers={"Authorization": "Bearer token"},
                    auth=auth,
                ):
                    raise ValueError("consumer")
            self.assertEqual(
                actions[-3][2],
                {"headers": {"Authorization": "Bearer token"}, "auth": auth},
            )
            self.assertEqual(actions[-2:], ["session closed", "transport closed"])
            session.initialize.side_effect = ValueError("initialize")
            with self.assertRaisesRegex(ValueError, "initialize"):
                async with MCPClient.stdio("python"):
                    self.fail("initialization must fail before yielding")
            self.assertEqual(actions[-2:], ["session closed", "transport closed"])
            with self.assertRaisesRegex(ValueError, "initialize"):
                async with MCPClient.streamable_http("https://example.test/mcp"):
                    self.fail("HTTP initialization must fail before yielding")
            self.assertEqual(actions[-2:], ["session closed", "transport closed"])


class TelemetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_span_correlation_errors_and_cleanup(self):
        spans = []

        class Span:
            def __init__(self, name, context):
                self.name, self.parent = name, context
                self.attributes, self.events = {}, []
                self.ended = False
                self.status = None

            def set_attribute(self, key, value):
                self.attributes[key] = value

            def add_event(self, name, attributes=None):
                self.events.append((name, attributes))

            def set_status(self, status):
                self.status = status

            def end(self):
                self.ended = True

        def start(name, context=None):
            span = Span(name, context)
            spans.append(span)
            return span

        trace = module(
            "opentelemetry.trace",
            set_span_in_context=lambda span: span,
            Status=lambda code, description: (code, description),
            StatusCode=SimpleNamespace(ERROR="ERROR"),
        )
        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": module("opentelemetry", trace=trace),
                "opentelemetry.trace": trace,
            },
        ):
            sink = OpenTelemetryEventSink(SimpleNamespace(start_span=start))
            await sink.emit(
                Event("run.start", {"run_id": "one", "agent": "a", "prompt": "secret"})
            )
            await sink.emit(Event("run.start", {"run_id": "two", "agent": "a"}))
            await sink.emit(
                Event("model.start", {"run_id": "one", "agent": "a", "attempt": 1})
            )
            await sink.emit(
                Event(
                    "model.error", {"run_id": "one", "agent": "a", "error": "transient"}
                )
            )
            await sink.emit(
                Event("model.retry", {"run_id": "one", "agent": "a", "attempt": 1})
            )
            await sink.emit(
                Event("model.start", {"run_id": "one", "agent": "a", "attempt": 2})
            )
            await sink.emit(
                Event(
                    "model.complete",
                    {
                        "run_id": "one",
                        "agent": "a",
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "latency": 0.2,
                    },
                )
            )
            await sink.emit(
                Event(
                    "tool.start",
                    {"run_id": "two", "agent": "a", "call_id": "x", "tool": "read"},
                )
            )
            await sink.emit(
                Event("handoff", {"run_id": "one", "agent": "b", "source": "a"})
            )
            await sink.emit(Event("run.complete", {"run_id": "one", "agent": "b"}))
            self.assertIs(spans[2].parent, spans[0])
            self.assertIs(spans[5].parent, spans[1])
            self.assertEqual(spans[2].status, ("ERROR", "transient"))
            self.assertEqual(spans[4].attributes["cppilot.input_tokens"], 7)
            self.assertEqual(spans[4].attributes["cppilot.latency"], 0.2)
            self.assertNotIn("cppilot.prompt", spans[0].attributes)
            self.assertFalse(spans[1].ended)
            await sink.emit(
                Event(
                    "run.error", {"run_id": "two", "agent": "a", "error": "cancelled"}
                )
            )
            self.assertTrue(spans[5].ended)
            self.assertEqual(spans[5].status[0], "ERROR")
            span_count = len(spans)
            await sink.emit(
                Event(
                    "tool.error",
                    {"run_id": "two", "agent": "a", "call_id": "x", "error": "late"},
                )
            )
            self.assertEqual(len(spans), span_count)
            await sink.emit(Event("run.start", {"run_id": "abandoned", "agent": "a"}))
            sink.close()
            self.assertTrue(all(span.ended for span in spans))
            self.assertFalse(sink._spans)


class RealServerTest(unittest.TestCase):
    def setUp(self):
        try:
            from fastapi import Depends, HTTPException, Request
            from fastapi.testclient import TestClient
        except ImportError as error:
            self.skipTest(
                f"optional FastAPI/TestClient dependencies unavailable: {error.name}"
            )
        self.Depends, self.HTTPException, self.Request, self.TestClient = (
            Depends,
            HTTPException,
            Request,
            TestClient,
        )

    def test_real_request_cancel_scope_stops_disconnect_monitor(self):
        async def check():
            receiving = asyncio.Event()

            async def receive():
                receiving.set()
                await asyncio.Event().wait()
                return {"type": "http.disconnect"}

            request = self.Request({"type": "http"}, receive)

            async def work():
                await receiving.wait()
                return "completed"

            result = await asyncio.wait_for(
                _connected(work(), request, 0.001), timeout=1
            )
            self.assertEqual(result, "completed")
            self.assertFalse(
                [
                    task
                    for task in asyncio.all_tasks()
                    if task is not asyncio.current_task() and not task.done()
                ]
            )

        asyncio.run(check())

    def test_real_dispatch_dependencies_principal_and_sse(self):
        root = Agent("root", "", MagicMock())
        child = Agent("child", "", MagicMock(), handoffs=[root])
        root.handoffs = [child]
        seen = []

        class Runner:
            async def run(self, agent, prompt, *, conversation_id=None):
                seen.append(agent)
                return SimpleNamespace(
                    output=agent.context.values["principal"],
                    agent_name=agent.name,
                    usage=Usage(),
                    run_id="id",
                )

            async def stream(self, agent, prompt, *, conversation_id=None):
                seen.append(agent)
                yield Message("assistant", agent.context.values["principal"])
                raise MaxTurnsExceeded("sensitive provider detail")

        async def authenticate(credential):
            if credential not in {"Bearer alice", "Bearer bob"}:
                raise PermissionError("bad")
            return credential.split()[1]

        def tenant():
            return "tenant"

        tenant_dependency = self.Depends(tenant)

        async def application(request, name=tenant_dependency):
            if request.state.principal == "bob" and request.headers.get("x-deny"):
                raise self.HTTPException(status_code=403, detail="forbidden")
            return {"tenant": name}

        application.__annotations__["request"] = self.Request

        async def resolve(name, principal, conversation, state):
            if conversation != principal:
                raise PermissionError("private conversation")
            return root

        app = create_app(
            lambda name: root,
            runner=Runner(),
            authenticate=authenticate,
            application_dependency=application,
            resolve_authorized_agent=resolve,
        )
        with self.TestClient(app) as client:
            body = {"agent": "root", "prompt": "p", "conversation_id": "alice"}
            self.assertEqual(client.post("/v1/run", json=body).status_code, 401)
            response = client.post(
                "/v1/run", json=body, headers={"authorization": "Bearer alice"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["output"], "alice")
            self.assertEqual(
                seen[-1].handoffs[0].context.values["application_state"],
                {"tenant": "tenant"},
            )
            self.assertEqual(
                client.post(
                    "/v1/run", json=body, headers={"authorization": "Bearer bob"}
                ).status_code,
                403,
            )
            body["conversation_id"] = "bob"
            response = client.post(
                "/v1/stream", json=body, headers={"authorization": "Bearer bob"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("event: error\n", response.text)
            self.assertIn('"code": "max_turns_exceeded"', response.text)
            self.assertNotIn("sensitive", response.text)
            self.assertEqual(seen[-1].handoffs[0].context.values["principal"], "bob")
            self.assertEqual(
                client.post(
                    "/v1/run",
                    json=body,
                    headers={"authorization": "Bearer bob", "x-deny": "1"},
                ).status_code,
                403,
            )
            self.assertEqual(
                client.post(
                    "/v1/run",
                    json={"agent": "root", "prompt": 3},
                    headers={"authorization": "Bearer alice"},
                ).status_code,
                422,
            )
        self.assertNotIn("principal", root.context.values)
        self.assertNotIn("principal", child.context.values)
        self.assertIsNot(seen[0].context, seen[-1].context)


class ServerTest(unittest.IsolatedAsyncioTestCase):
    def sdk(self):
        class HTTPException(Exception):
            def __init__(self, status_code, detail, headers=None):
                self.status_code, self.detail, self.headers = (
                    status_code,
                    detail,
                    headers,
                )

        class App:
            def __init__(self, **kwargs):
                self.dependencies = kwargs["dependencies"]
                self.options = kwargs
                self.routes = {}
                self.state = SimpleNamespace()

            def post(self, path):
                def register(function):
                    self.routes[path] = function
                    return function

                return register

        class StreamingResponse:
            def __init__(self, content, **kwargs):
                self.body_iterator, self.options = content, kwargs

            async def __call__(self, scope, receive, send):
                async for frame in self.body_iterator:
                    await send(frame)

        class CancelScope:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return {
            "anyio": module("anyio", CancelScope=CancelScope),
            "fastapi": module(
                "fastapi",
                FastAPI=App,
                Depends=lambda fn: fn,
                HTTPException=HTTPException,
                Request=type("Request", (), {}),
            ),
            "fastapi.responses": module(
                "fastapi.responses", StreamingResponse=StreamingResponse
            ),
        }, HTTPException

    async def test_auth_dependency_validation_json_sse(self):
        modules, exception = self.sdk()
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(
                    output="answer", agent_name="agent", usage=Usage(2, 3), run_id="id"
                )
            )
        )

        async def stream(*args, **kwargs):
            yield Message("assistant", 'line\nquote"')
            yield Usage(2, 3)

        runner.stream = stream
        resolve = MagicMock(return_value=Agent("agent", "", MagicMock()))
        request = SimpleNamespace(
            headers={"authorization": "Bearer token"},
            state=SimpleNamespace(),
            is_disconnected=AsyncMock(return_value=False),
        )
        with patch.dict("sys.modules", modules):
            app = create_app(
                resolve,
                runner=runner,
                auth_provider=BearerTokenAuth("token", principal="alice"),
            )
            self.assertEqual(len(app.dependencies), 1)
            self.assertIsNone(app.options["docs_url"])
            self.assertIsNone(app.options["openapi_url"])
            self.assertIs(app.dependencies[0], app.state.authorize)
            await app.state.authorize(request)
            self.assertEqual(request.state.principal, "alice")
            result = await app.routes["/v1/run"](
                {"agent": "agent", "prompt": "go"}, request
            )
            self.assertEqual(result["usage"], {"input_tokens": 2, "output_tokens": 3})
            self.assertEqual(result["run_id"], "id")
            response = await app.routes["/v1/stream"](
                {"agent": "agent", "prompt": "go"}, request
            )
            frames = [frame async for frame in response.body_iterator]
            self.assertEqual(
                json.loads(frames[0][6:].strip())["content"], 'line\nquote"'
            )
            self.assertEqual(json.loads(frames[1][6:].strip())["type"], "usage")
            request.headers = {}
            with self.assertRaises(exception) as failure:
                await app.state.authorize(request)
            self.assertEqual(failure.exception.status_code, 401)
            for body in (
                {},
                {"agent": "a", "prompt": 1},
                {"agent": "a", "prompt": "p", "conversation_id": []},
            ):
                with self.assertRaises(exception) as failure:
                    await app.routes["/v1/run"](body, request)
                self.assertEqual(failure.exception.status_code, 422)

    async def test_principal_graph_isolation_and_authorized_resolver(self):
        modules, exception = self.sdk()
        child = Agent("child", "", MagicMock())
        root = Agent("root", "", MagicMock(), handoffs=[child])
        child.handoffs = [root]
        runner = SimpleNamespace(
            run=AsyncMock(
                return_value=SimpleNamespace(
                    output="ok", agent_name="root", usage=Usage()
                )
            )
        )
        resolver = AsyncMock(return_value=root)
        with patch.dict("sys.modules", modules):
            app = create_app(
                lambda name: root, runner=runner, resolve_authorized_agent=resolver
            )
            for principal in ("alice", "bob"):
                request = SimpleNamespace(
                    state=SimpleNamespace(
                        principal=principal, application_state={"tenant": principal}
                    ),
                    is_disconnected=AsyncMock(return_value=False),
                )
                await app.routes["/v1/run"](
                    {"agent": "root", "prompt": "p", "conversation_id": principal},
                    request,
                )
            first, second = [call.args[0] for call in runner.run.await_args_list]
            self.assertEqual(first.context.values["principal"], "alice")
            self.assertEqual(second.handoffs[0].context.values["principal"], "bob")
            self.assertEqual(
                first.handoffs[0].context.values["application_state"],
                {"tenant": "alice"},
            )
            self.assertIs(first.handoffs[0].handoffs[0], first)
            self.assertIsNot(first.context.cancelled, second.context.cancelled)
            self.assertIsNot(first.context.tasks, root.context.tasks)
            first.context.values["new"] = "local"
            self.assertNotIn("principal", root.context.values)
            self.assertNotIn("principal", child.context.values)
            resolver.assert_awaited_with("root", "bob", "bob", {"tenant": "bob"})
            resolver.side_effect = PermissionError("sensitive access detail")
            with self.assertRaises(exception) as failure:
                await app.routes["/v1/run"]({"agent": "root", "prompt": "p"}, request)
            self.assertEqual(
                (failure.exception.status_code, failure.exception.detail),
                (403, "forbidden"),
            )

    async def test_terminal_sse_errors_sanitized(self):
        modules, _ = self.sdk()
        for error, code in (
            (GuardrailTriggered("secret"), "guardrail_triggered"),
            (MaxTurnsExceeded("secret"), "max_turns_exceeded"),
            (StructuredOutputError("secret"), "invalid_output"),
            (RuntimeError("secret"), "run_failed"),
        ):
            closed = asyncio.Event()

            async def failing(*args, error=error, closed=closed, **kwargs):
                try:
                    yield Message("assistant", "partial")
                    raise error
                finally:
                    closed.set()

            request = SimpleNamespace(
                state=SimpleNamespace(), is_disconnected=AsyncMock(return_value=False)
            )
            with patch.dict("sys.modules", modules):
                app = create_app(
                    lambda name: Agent(name, "", MagicMock()),
                    runner=SimpleNamespace(stream=failing),
                )
                response = await app.routes["/v1/stream"](
                    {"agent": "a", "prompt": "p"}, request
                )
                frames = [frame async for frame in response.body_iterator]
            self.assertEqual(len(frames), 2)
            self.assertTrue(frames[-1].startswith("event: error\n"))
            self.assertEqual(
                json.loads(frames[-1].split("data: ")[1]),
                {
                    "type": "error",
                    "code": code,
                    "message": "Run failed",
                    "terminal": True,
                },
            )
            self.assertNotIn("secret", frames[-1])
            self.assertTrue(closed.is_set())

    async def test_disconnect_cancels_pending_work(self):
        stopped = asyncio.Event()

        async def slow():
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        with self.assertRaises(ClientDisconnected):
            await _connected(slow(), request, 0.001)
        self.assertTrue(stopped.is_set())

    async def test_sse_disconnect_closes_generator(self):
        modules, _ = self.sdk()
        stopped = asyncio.Event()

        async def slow(*args, **kwargs):
            try:
                await asyncio.Event().wait()
                yield Message("assistant", "late")
            finally:
                stopped.set()

        request = SimpleNamespace(
            state=SimpleNamespace(), is_disconnected=AsyncMock(return_value=True)
        )
        with patch.dict("sys.modules", modules):
            app = create_app(
                lambda name: Agent(name, "", MagicMock()),
                runner=SimpleNamespace(stream=slow),
            )
            response = await app.routes["/v1/stream"](
                {"agent": "a", "prompt": "p"}, request
            )
            self.assertEqual([frame async for frame in response.body_iterator], [])
        self.assertTrue(stopped.is_set())

    async def test_sse_write_failure_and_cancellation_close_stream(self):
        modules, _ = self.sdk()
        for cancelled in (False, True):
            stopped, sending = asyncio.Event(), asyncio.Event()

            async def stream(*args, stopped=stopped, **kwargs):
                try:
                    yield Message("assistant", "first")
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

            async def send(frame, sending=sending, cancelled=cancelled):
                sending.set()
                if cancelled:
                    await asyncio.Event().wait()
                raise OSError("write failed")

            request = SimpleNamespace(
                state=SimpleNamespace(), is_disconnected=AsyncMock(return_value=False)
            )
            with patch.dict("sys.modules", modules):
                app = create_app(
                    lambda name: Agent(name, "", MagicMock()),
                    runner=SimpleNamespace(stream=stream),
                )
                response = await app.routes["/v1/stream"](
                    {"agent": "a", "prompt": "p"}, request
                )
                task = asyncio.create_task(response({}, None, send))
                await sending.wait()
                if cancelled:
                    task.cancel()
                with self.assertRaises(
                    asyncio.CancelledError if cancelled else OSError
                ):
                    await task
            self.assertTrue(stopped.is_set())

    async def test_disconnect_poll_swallowed_cancel_still_stops(self):
        polling = asyncio.Event()
        cancelled = asyncio.Event()

        async def is_disconnected():
            polling.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Starlette's CancelScope-based poll may consume cancellation.
                cancelled.set()
                return False

        async def work():
            await polling.wait()
            return "done"

        result = await asyncio.wait_for(
            _connected(work(), SimpleNamespace(is_disconnected=is_disconnected), 0.001),
            timeout=1,
        )
        self.assertEqual(result, "done")
        self.assertTrue(cancelled.is_set())

    async def test_outer_cancellation_cleans_work(self):
        started, stopped = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        task = asyncio.create_task(
            _connected(
                slow(),
                SimpleNamespace(is_disconnected=AsyncMock(return_value=False)),
                0.001,
            )
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stopped.is_set())
