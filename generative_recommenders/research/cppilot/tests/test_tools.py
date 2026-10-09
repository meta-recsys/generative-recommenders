# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import dataclasses
import enum
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

from cppilot.agent import Agent
from cppilot.interfaces import ExecutionResult, ModelProvider, ModelResponse
from cppilot.items import Message, ToolResult
from cppilot.runner import AgentRunner
from cppilot.sandbox import ContainerSandbox
from cppilot.skills import SkillLoader
from cppilot.tools import (
    FunctionTool,
    JobManager,
    portable_tools,
    RunContext,
    state_tools,
    TaskManager,
    TeamMailbox,
)
from cppilot.tools.files import read_file, write_file
from cppilot.tools.portable import edit_file, glob_files, grep_files, shell, web_fetch
from cppilot.tools.process import run_process


class Mode(enum.Enum):
    FAST = "fast"
    SAFE = "safe"


@dataclasses.dataclass
class Options:
    mode: Mode
    counts: list[int]
    note: str | None = None


def nested(
    options: Options,
    tags: dict[str, list[Literal["a", "b"]]],
    choice: int | str = "auto",
) -> dict:
    """Configure nested inputs.

    Args:
        options: Nested execution options.
        tags: Labels grouped by name.
        choice: An integer or automatic selection.
    """
    return {
        "mode": options.mode.value,
        "counts": options.counts,
        "tags": tags,
        "choice": choice,
    }


class FakeProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse([Message("assistant", "ok")])


class ToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_nested_schema_validation_defaults_and_docstrings(self) -> None:
        tool = FunctionTool(nested)
        schema = tool.definition.parameters
        self.assertEqual(
            schema["properties"]["options"]["properties"]["mode"]["enum"],
            ["fast", "safe"],
        )
        self.assertEqual(
            schema["properties"]["options"]["properties"]["counts"]["items"],
            {"type": "integer"},
        )
        self.assertEqual(schema["properties"]["choice"]["default"], "auto")
        self.assertEqual(
            schema["properties"]["tags"]["description"], "Labels grouped by name."
        )
        self.assertEqual(len(schema["properties"]["choice"]["anyOf"]), 2)
        output = json.loads(
            await tool.invoke(
                {"options": {"mode": "fast", "counts": [1]}, "tags": {"group": ["a"]}},
                RunContext(),
            )
        )
        self.assertEqual(
            output,
            {"mode": "fast", "counts": [1], "tags": {"group": ["a"]}, "choice": "auto"},
        )
        for arguments in (
            {"options": {"mode": "bad", "counts": [1]}, "tags": {}},
            {"options": {"mode": "fast", "counts": [True]}, "tags": {}},
            {"options": {"mode": "fast", "counts": [1], "unknown": 1}, "tags": {}},
            {"options": {"mode": "fast", "counts": [1]}, "tags": {"x": ["c"]}},
            {"options": {"mode": "fast", "counts": [1]}, "tags": {}, "unknown": 1},
            {"tags": {}},
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaises((TypeError, ValueError)),
            ):
                await tool.invoke(arguments, RunContext())

    async def test_context_cannot_be_supplied_and_positional_only(self) -> None:
        def tool_fn(ctx: RunContext, number: int, /, *, label: str = "x") -> str:
            return f"{ctx.values['prefix']}:{number}:{label}"

        tool = FunctionTool(tool_fn)
        self.assertNotIn("ctx", tool.definition.parameters["properties"])
        self.assertEqual(
            await tool.invoke({"number": 2}, RunContext({"prefix": "p"})), "p:2:x"
        )
        with self.assertRaises(TypeError):
            await tool.invoke({"ctx": {}, "number": 2}, RunContext())
        with self.assertRaises(TypeError):
            FunctionTool(lambda **kwargs: kwargs)

    async def test_async_timeout_and_cancelled_context(self) -> None:
        stopped = asyncio.Event()

        async def slow() -> str:
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()
            return "late"

        with self.assertRaises(TimeoutError):
            await FunctionTool(slow, timeout=0.01).invoke({}, RunContext())
        self.assertTrue(stopped.is_set())
        context = RunContext()
        context.cancelled.set()
        with self.assertRaises(asyncio.CancelledError):
            await FunctionTool(slow).invoke({}, context)

    async def test_sync_timeout_keeps_loop_responsive_thread_finishes(self) -> None:
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def blocking() -> str:
            entered.set()
            try:
                release.wait(2)
                return "finished"
            finally:
                finished.set()

        worker = asyncio.create_task(
            FunctionTool(blocking, timeout=0.1).invoke({}, RunContext())
        )
        try:
            while not entered.is_set() and not worker.done():
                await asyncio.sleep(0.001)
            self.assertTrue(entered.is_set())
            with self.assertRaises(TimeoutError):
                await worker
            self.assertFalse(finished.is_set())
        finally:
            release.set()
            await asyncio.to_thread(finished.wait, 2)
        self.assertTrue(finished.is_set())

        async def eventual() -> str:
            return "awaited"

        def returns_awaitable():
            return eventual()

        self.assertEqual(
            await FunctionTool(returns_awaitable).invoke({}, RunContext()), "awaited"
        )

    async def test_structured_frontmatter_json_toml_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            description = 'Quoted: # literal "content"'
            for name, text in (
                (
                    "json",
                    "---\n"
                    + json.dumps({"name": "json", "description": description})
                    + "\n---\nbody",
                ),
                (
                    "toml",
                    '+++\nname = "toml"\ndescription = '
                    + json.dumps(description)
                    + "\n+++\nbody",
                ),
            ):
                folder = root / name
                folder.mkdir()
                (folder / "SKILL.md").write_text(text)
            loader = SkillLoader([root])
            self.assertEqual(loader.load("json").description, description)
            self.assertEqual(loader.load("toml").description, description)
            for text in (
                '---\n{"name": "a", "name": "b", "description": "x"}\n---',
                '---\n{"name": "a", "description": 2}\n---',
                '---\n{"name": "a"}\n---',
                '+++\nname = "a"\nname = "b"\n+++',
                '---\n{"name":',
            ):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    SkillLoader._parse(root / "invalid", text=text)
            with patch(
                "cppilot.skills.importlib.import_module",
                side_effect=ImportError("yaml"),
            ):
                with self.assertRaisesRegex(RuntimeError, "PyYAML"):
                    SkillLoader._parse(
                        root / "yaml",
                        text='---\nname: safe\ndescription: "Quoted: # literal"\n---\nbody',
                    )

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML extra not installed")
    async def test_yaml_quoted_multiline_duplicate_and_unsafe(self) -> None:
        path = Path(tempfile.gettempdir()) / "skill-parser-test"
        skill = SkillLoader._parse(
            path, text='---\nname: safe\ndescription: "Quoted: # literal"\n---\nbody'
        )
        # pyrefly: ignore [missing-attribute]
        self.assertEqual(skill.description, "Quoted: # literal")
        folded = SkillLoader._parse(
            path,
            text="---\nname: safe\ndescription: >\n  first: text\n  second # text\n---\nbody",
        )
        # pyrefly: ignore [missing-attribute]
        self.assertEqual(folded.description, "first: text second # text")
        for metadata in (
            "name: a\nname: b\ndescription: x",
            "name: a\ndescription: !!python/object/apply:os.system [echo]",
        ):
            with self.assertRaises(ValueError):
                SkillLoader._parse(path, text="---\n" + metadata + "\n---\nbody")

    async def test_manager_close_reaps_pending_and_preserves_completed(self) -> None:
        context = RunContext()
        stopped = asyncio.Event()
        started = asyncio.Event()

        async def pending() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            return "unreachable"

        async def completed() -> str:
            return "done"

        context.tasks.start("pending", pending())
        context.tasks.start("completed", completed())
        await started.wait()
        self.assertEqual(await context.tasks.result("completed"), "done")

        async def job_handler(payload: dict):
            await asyncio.Event().wait()

        context.jobs.handlers["pending"] = job_handler
        job_id = await context.jobs.submit("pending", {})
        await asyncio.sleep(0)
        await context.tasks.close()
        await context.jobs.close()
        self.assertTrue(stopped.is_set())
        self.assertEqual(context.tasks.status("pending")["status"], "cancelled")
        self.assertEqual(context.tasks.status("completed")["status"], "completed")
        self.assertEqual((await context.jobs.status(job_id))["status"], "cancelled")
        self.assertTrue(all(task.done() for task in context.tasks.tasks.values()))

    async def test_optional_dependency_errors(self) -> None:
        original = __import__("importlib").import_module

        def missing(name, *args, **kwargs):
            if name in {"pydantic", "griffe"}:
                raise ImportError(name)
            return original(name, *args, **kwargs)

        with patch("cppilot.tools.core.importlib.import_module", side_effect=missing):
            with self.assertRaisesRegex(RuntimeError, "Pydantic"):
                FunctionTool(nested, use_pydantic=True)
            with self.assertRaisesRegex(RuntimeError, "Griffe"):
                FunctionTool(nested, use_griffe=True)

    @unittest.skipUnless(
        importlib.util.find_spec("pydantic"), "Pydantic extra not installed"
    )
    async def test_pydantic_nested_models_refs_constraints(self) -> None:
        import pydantic

        if not hasattr(pydantic.BaseModel, "model_json_schema"):
            self.skipTest("Pydantic v2 required")

        class Child(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(extra="forbid")
            mode: Mode = Mode.SAFE
            count: int = pydantic.Field(ge=1)

        class Parent(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(extra="forbid")
            children: list[Child]

        Parent.model_rebuild(_types_namespace={"Child": Child})

        def typed(payload, choice: int | str = "auto") -> dict:
            return {
                "count": payload.children[0].count,
                "mode": payload.children[0].mode.value,
                "choice": choice,
            }

        typed.__annotations__["payload"] = Parent
        tool = FunctionTool(typed, use_pydantic=True)
        json.dumps(tool.definition.parameters)
        self.assertIn("Child", tool.definition.parameters["$defs"])
        output = json.loads(
            await tool.invoke({"payload": {"children": [{"count": 2}]}}, RunContext())
        )
        self.assertEqual(output, {"count": 2, "mode": "safe", "choice": "auto"})
        for payload in (
            {"children": [{"count": 0}]},
            {"children": [{"count": "2"}]},
            {"children": [{"count": 2, "extra": True}]},
        ):
            with self.assertRaises(ValueError):
                await tool.invoke({"payload": payload}, RunContext())
        fallback = FunctionTool(typed)
        self.assertIn("Child", fallback.definition.parameters["$defs"])
        self.assertNotIn(
            "$defs", fallback.definition.parameters["properties"]["payload"]
        )

    @unittest.skipUnless(
        importlib.util.find_spec("griffe"), "Griffe extra not installed"
    )
    async def test_griffe_parameter_docs(self) -> None:
        tool = FunctionTool(nested, use_griffe=True)
        self.assertEqual(
            tool.definition.parameters["properties"]["options"]["description"],
            "Nested execution options.",
        )

    async def test_secure_files_grants_symlinks_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            context = RunContext(read_roots=(root,), write_roots=(root,))
            write_file(context, "sub/note.txt", "first\nmatch\n")
            self.assertEqual(read_file(context, "sub/note.txt"), "first\nmatch\n")
            edit_file(context, "sub/note.txt", "first", "changed")
            self.assertEqual(
                grep_files(context, "match"),
                [{"path": "sub/note.txt", "line": 2, "text": "match"}],
            )
            secret = base / "secret.txt"
            secret.write_text("secret match")
            (root / "escape").symlink_to(secret)
            (root / "inside").symlink_to(root / "sub/note.txt")
            (root / "folder").symlink_to(base, target_is_directory=True)
            for path in (
                "../secret.txt",
                str(secret),
                "escape",
                "inside",
                "folder/secret.txt",
            ):
                with self.subTest(path=path), self.assertRaises(PermissionError):
                    read_file(context, path)
            self.assertEqual(glob_files(context, "*"), ["sub/note.txt"])
            self.assertEqual(len(grep_files(context, "match")), 1)
            allowed = dataclasses.replace(context, allow_symlinks=True)
            self.assertEqual(read_file(allowed, "inside"), "changed\nmatch\n")
            with self.assertRaises(PermissionError):
                read_file(allowed, "escape")
            with self.assertRaises(PermissionError):
                edit_file(RunContext(write_roots=(root,)), "sub/note.txt", "match", "x")
            with self.assertRaises(PermissionError):
                write_file(RunContext(), "any", "x")
            with self.assertRaises(ValueError):
                edit_file(context, "sub/note.txt", "match", "x", 2)
            os.link(secret, root / "hardlink")
            with self.assertRaises(PermissionError):
                write_file(context, "hardlink", "bad")
            self.assertEqual(secret.read_text(), "secret match")

    async def test_secure_open_rejects_replaced_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, outside = base / "root", base / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "sub").mkdir()
            context = RunContext(write_roots=(root,))
            original = RunContext._grant

            def race(instance, value, write):
                grant = original(instance, value, write)
                (root / "sub").rmdir()
                (root / "sub").symlink_to(outside, target_is_directory=True)
                return grant

            with patch.object(RunContext, "_grant", race), self.assertRaises(OSError):
                with context.open_path("sub/new", write=True):
                    self.fail("must not open through replacement symlink")
            self.assertFalse((outside / "new").exists())

    async def test_skills_injected_once_and_runner_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = root / "safe"
            skill.mkdir()
            (skill / "SKILL.md").write_text(
                "---\n"
                + json.dumps(
                    {"name": "safe", "description": "Safe instructions for tools"}
                )
                + "\n---\nFull instructions."
            )
            loader = SkillLoader([root], {"safe"})
            provider = FakeProvider()
            agent = Agent("skills", "Base", provider, skill_loader=loader)
            clone = dataclasses.replace(agent, context=RunContext())
            self.assertEqual(
                [tool.definition.name for tool in clone.tools], ["load_skill"]
            )
            self.assertEqual(
                await clone.tools[0].invoke({"name": "safe"}, clone.context),
                "Full instructions.",
            )
            with self.assertRaises(KeyError):
                await clone.tools[0].invoke({"name": "forbidden"}, clone.context)
            await AgentRunner().run(clone, "go")
            self.assertIn(
                "safe: Safe instructions for tools", provider.requests[0].system
            )
            self.assertNotIn("Full instructions.", provider.requests[0].system)
            self.assertEqual(provider.requests[0].tools[0].name, "load_skill")
            self.assertEqual(
                Agent("legacy", "Base", provider, skills=loader.available())
                .tools[0]
                .definition.name,
                "load_skill",
            )
            self.assertEqual(
                Agent("direct", "Base", provider, skills=loader).skills,
                tuple(loader.available()),
            )
            outside = root / "outside"
            outside.mkdir()
            (outside / "SKILL.md").write_text(
                '+++\nname = "other"\ndescription = "no"\n+++\nsecret'
            )
            (skill / "link").symlink_to(outside, target_is_directory=True)
            (skill / "escape").mkdir()
            (skill / "escape/SKILL.md").symlink_to(outside / "SKILL.md")
            self.assertEqual(
                [item.name for item in SkillLoader([skill]).available()], ["safe"]
            )

    async def test_context_state_isolation_jobs_and_mailbox(self) -> None:
        tools = {tool.definition.name: tool for tool in state_tools()}
        left, right = RunContext(), RunContext()
        await tools["todo_add"].invoke({"label": "work"}, left)
        await tools["todo_complete"].invoke({"label": "work"}, left)
        await tools["todo_add"].invoke({"label": "work"}, left)
        self.assertEqual(
            json.loads(await tools["todo_list"].invoke({}, left)), {"work": True}
        )
        self.assertEqual(json.loads(await tools["todo_list"].invoke({}, right)), {})
        fresh = dataclasses.replace(left, **left.fresh_managers())
        self.assertIsNot(fresh.tasks, left.tasks)
        self.assertEqual(fresh.todos.items, {})

        async def handler(payload: dict) -> dict:
            return {"value": payload["value"] + 1}

        left.jobs.handlers["increment"] = handler
        job_id = await tools["job_submit"].invoke(
            {"name": "increment", "payload": {"value": 2}}, left
        )
        self.assertEqual(
            json.loads(await tools["job_result"].invoke({"job_id": job_id}, left)),
            {"value": 3},
        )
        self.assertEqual((await left.jobs.status(job_id))["status"], "completed")
        self.assertIn("increment", left.fresh_managers()["jobs"].handlers)
        mailbox = TeamMailbox()
        receiver = asyncio.create_task(mailbox.receive("b", 1))
        await asyncio.sleep(0)
        mailbox.send("a", "b", "hello")
        self.assertEqual(await receiver, ["a: hello"])
        self.assertEqual(await mailbox.receive("b", 0.001), [])
        self.assertIn("job_submit", {tool.definition.name for tool in portable_tools()})
        await left.jobs.close()

    async def test_task_wait_timeout_failures_cancel_and_runner_results(self) -> None:
        manager = TaskManager()
        event = asyncio.Event()

        async def slow() -> int:
            await event.wait()
            return 7

        manager.start("slow", slow())
        with self.assertRaises(ValueError):
            manager.start("slow", slow())
        with self.assertRaises(TimeoutError):
            await manager.result("slow", 0.001)
        self.assertEqual(manager.status("slow")["status"], "running")
        event.set()
        self.assertEqual(await manager.result("slow"), 7)

        async def failed() -> None:
            raise ValueError("broken")

        manager.start("bad", failed())
        with self.assertRaisesRegex(ValueError, "broken"):
            await manager.result("bad")
        self.assertEqual(manager.status("bad")["status"], "failed")
        event.clear()
        manager.start("cancel", slow())
        self.assertEqual((await manager.cancel("cancel"))["status"], "cancelled")

        async def runner_result() -> ToolResult:
            return ToolResult("call", "tool", "bad output", True)

        task = asyncio.create_task(runner_result())
        context = RunContext(session={"background_tasks": {"call": task}})
        result_tool = next(
            tool for tool in state_tools() if tool.definition.name == "task_result"
        )
        output = json.loads(await result_tool.invoke({"name": "call"}, context))
        self.assertEqual(output["output"], "bad output")
        self.assertTrue(output["failed"])
        self.assertEqual(context.tasks.status("call")["status"], "failed")
        await manager.close()
        await context.tasks.close()

    async def test_job_backend_delegation(self) -> None:
        backend = AsyncMock()
        backend.submit.return_value = "remote"
        backend.status.return_value = {"status": "running"}
        manager = JobManager(backend)
        self.assertEqual(await manager.submit("name", {"a": 1}), "remote")
        self.assertEqual(await manager.status("remote"), {"status": "running"})
        backend.submit.assert_awaited_once_with("name", {"a": 1})
        self.assertIs(manager.fresh().backend, backend)

    async def test_process_execution_output_limits_timeout_and_cancel(self) -> None:
        result = await run_process(
            [
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); print('err', file=sys.stderr)",
                "a;$(bad)",
            ]
        )
        self.assertEqual(result.stdout.strip(), "a;$(bad)")
        self.assertEqual(result.stderr.strip(), "err")
        self.assertEqual(result.returncode, 0)
        with self.assertRaisesRegex(ValueError, "output limit"):
            await run_process(
                [sys.executable, "-c", "print('x' * 1000)"], output_limit=100
            )
        with self.assertRaises(TimeoutError):
            await run_process(
                [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.01
            )
        cancelled = asyncio.Event()
        worker = asyncio.create_task(
            run_process(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cancelled=cancelled,
            )
        )
        await asyncio.sleep(0.02)
        cancelled.set()
        with self.assertRaises(asyncio.CancelledError):
            await worker
        with self.assertRaises(ValueError):
            await run_process("echo bad")

    async def test_output_failure_reaps_process_and_readers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "pid"
            code = (
                "import os, pathlib, sys, time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "print('x' * 10000, flush=True); time.sleep(10)"
            )
            with self.assertRaisesRegex(ValueError, "output limit"):
                await run_process(
                    [sys.executable, "-c", code, str(pid_path)], output_limit=100
                )
            pid = int(pid_path.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            live = [
                task
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()
            ]
            self.assertEqual(live, [])

    async def test_shell_local_and_sandbox_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = await shell(
                RunContext(read_roots=(root,)), [sys.executable, "-c", "print('ok')"]
            )
            self.assertEqual(result["stdout"], "ok\n")
            sandbox = AsyncMock()
            sandbox.start.return_value = "box"

            async def wait(*args):
                await asyncio.sleep(10)

            sandbox.execute.side_effect = wait
            context = RunContext(read_roots=(root,), sandbox=sandbox)
            with self.assertRaises(TimeoutError):
                await shell(context, ["true"], timeout=0.01)
            sandbox.stop.assert_awaited_once_with("box")
            self.assertNotIn("sandbox_id", context.session)

    async def test_container_flags_mount_mapping_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sandbox = ContainerSandbox(
                "image",
                engine="docker",
                mounts=[(root, "/work", False)],
                cpus=1.5,
                memory="512m",
            )
            calls = []

            async def run(*args):
                calls.append(args)
                return ExecutionResult(0, "ok", "")

            with patch.object(sandbox, "_run", side_effect=run):
                name = await sandbox.start()
                await sandbox.execute(name, ["echo", "a;b"], root / "sub")
                await sandbox.stop(name)
                await sandbox.stop(name)
            start = calls[0]
            self.assertEqual(start[start.index("--network") + 1], "none")
            self.assertEqual(start[start.index("--memory") + 1], "512m")
            self.assertIn("--read-only", start)
            self.assertIn("--pids-limit", start)
            self.assertIn(f"type=bind,src={root},dst=/work,readonly", start)
            self.assertEqual(
                calls[1], ("exec", "--workdir", "/work/sub", name, "echo", "a;b")
            )
            self.assertEqual(calls[2], ("rm", "-f", name))
            self.assertEqual(len(calls), 3)
            with self.assertRaises(KeyError):
                await sandbox.execute(name, ["true"])
            with self.assertRaises(PermissionError):
                sandbox._container_cwd(root.parent)
            for kwargs in (
                {"engine": "bad"},
                {"cpus": -1},
                {"memory": "bad"},
                {"mounts": [(root, "relative", True)]},
                {"mounts": [(root, "/", True)]},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    # pyrefly: ignore [bad-argument-type]
                    ContainerSandbox("image", **kwargs)

    async def test_container_start_failure_cleanup(self) -> None:
        sandbox = ContainerSandbox("image")
        calls = []

        async def run(*args):
            calls.append(args)
            if args[0] == "run":
                raise asyncio.CancelledError
            return ExecutionResult(0, "", "")

        with (
            patch.object(sandbox, "_run", side_effect=run),
            self.assertRaises(asyncio.CancelledError),
        ):
            await sandbox.start()
        self.assertEqual(calls[-1][0:2], ("rm", "-f"))
        self.assertEqual(sandbox._containers, set())

    async def test_container_cancellation_removes_container_and_failed_stop_retries(
        self,
    ) -> None:
        sandbox = ContainerSandbox("image")
        sandbox._containers.add("owned")
        calls = []

        async def run(*args):
            calls.append(args)
            if args[0] == "exec":
                raise asyncio.CancelledError
            return ExecutionResult(0, "", "")

        with (
            patch.object(sandbox, "_run", side_effect=run),
            self.assertRaises(asyncio.CancelledError),
        ):
            await sandbox.execute("owned", ["sleep", "10"])
        self.assertEqual(calls[-1], ("rm", "-f", "owned"))
        self.assertNotIn("owned", sandbox._containers)
        sandbox._containers.add("retry")
        with patch.object(
            sandbox, "_run", return_value=ExecutionResult(1, "", "failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await sandbox.stop("retry")
        self.assertIn("retry", sandbox._containers)
        with patch.object(sandbox, "_run", return_value=ExecutionResult(0, "", "")):
            await sandbox.close()
        self.assertEqual(sandbox._containers, set())

    async def test_web_fetch_scheme_and_size(self) -> None:
        with self.assertRaises(ValueError):
            await web_fetch("file:///secret")
        with patch("cppilot.tools.portable.urllib.request.urlopen") as open_url:
            open_url.return_value.__enter__.return_value.read.return_value = (
                b"x" * 2_000_001
            )
            with self.assertRaisesRegex(ValueError, "size limit"):
                await web_fetch("https://example.com")
