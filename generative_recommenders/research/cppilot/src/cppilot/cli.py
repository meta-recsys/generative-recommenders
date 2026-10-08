# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast, Sequence

from .agent import Agent
from .interfaces import MemoryStore, SessionStore
from .items import Message, RunItem
from .registry import ModelRegistry
from .settings import Settings
from .workflow import AgentWorkflow, Workflow, WorkflowRegistry


class Terminal:
    def __init__(self, *, plain: bool = False) -> None:
        self.console: Any = None
        self.prompt_session: Any = None
        if not plain and sys.stdout.isatty():
            try:
                from rich.console import Console

                self.console = Console()
            except ImportError:
                pass
        if not plain and sys.stdin.isatty():
            try:
                from prompt_toolkit import PromptSession

                self.prompt_session = PromptSession()
            except ImportError:
                pass

    def write(self, text: str, *, end: str = "\n") -> None:
        if self.console is not None:
            self.console.print(text, end=end, markup=False, highlight=False)
        else:
            sys.stdout.write(text + end)
            sys.stdout.flush()

    async def read(self) -> str:
        if self.prompt_session is not None:
            return cast(str, await self.prompt_session.prompt_async("cppilot> "))
        return await asyncio.to_thread(input, "cppilot> ")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="cppilot", description="Run or resume a CPPilot session"
    )
    result.add_argument(
        "arguments", nargs="*", help="[run] [prompt], resume SESSION [prompt], or info"
    )
    result.add_argument("--config", type=Path)
    result.add_argument("--provider")
    result.add_argument("--model")
    result.add_argument("--workflow")
    result.add_argument("--conversation-id")
    result.add_argument("--memory-root", type=Path)
    result.add_argument("--session-root", type=Path)
    result.add_argument(
        "--storage",
        choices=("file", "sqlite", "postgres"),
        help="storage backend (default: sqlite)",
    )
    result.add_argument("--database")
    result.add_argument("--info", action="store_true")
    result.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--plain", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument(
        "--load-entry-points", action=argparse.BooleanOptionalAction, default=None
    )
    return result


def _command(args: argparse.Namespace) -> tuple[str, str | None, str | None]:
    parts = list(args.arguments)
    command = parts.pop(0) if parts and parts[0] in {"run", "resume", "info"} else "run"
    if args.info:
        command = "info"
    session_id = None
    if command == "resume":
        if not parts:
            raise ValueError("resume requires a session id")
        session_id = parts.pop(0)
    return command, session_id, " ".join(parts) if parts else None


def _tail_compactor(items: list[RunItem]) -> list[RunItem]:
    # Keep complete recent user turns, never split a tool call/result pair.
    starts = [
        i
        for i, item in enumerate(items)
        if isinstance(item, Message) and item.role == "user"
    ]
    return items[starts[-4] :] if len(starts) > 4 else items


async def _output(
    workflow: Workflow[Any, Any], prompt: str, terminal: Terminal, stream: bool
) -> None:
    if not stream:
        result = await workflow.run(prompt)
        terminal.write(str(result.output if hasattr(result, "output") else result))
        return
    iterator = workflow.stream(prompt)
    wrote_text = False
    try:
        async for item in iterator:
            if isinstance(item, Message) and item.role == "assistant":
                terminal.write(item.content, end="")
                wrote_text = True
            elif isinstance(item, str):
                terminal.write(item, end="")
                wrote_text = True
            elif hasattr(item, "output") and not hasattr(item, "call_id"):
                terminal.write(str(item.output))
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
        if wrote_text:
            terminal.write("")


async def _interactive(  # noqa: C901
    workflow: Workflow[Any, Any],
    memory: MemoryStore,
    conversation_id: str,
    session_id: str,
    terminal: Terminal,
    stream: bool,
    models: ModelRegistry,
    sessions: SessionStore,
) -> None:
    terminal.write(
        f"session: {session_id}\n/exit /session /sessions /models /model NAME /memory /clear /compact /help"
    )
    while True:
        try:
            prompt = (await terminal.read()).strip()
        except (EOFError, KeyboardInterrupt):
            terminal.write("")
            return
        if prompt in {"/exit", "/quit"}:
            return
        if prompt == "/help":
            terminal.write(
                "/session: session id; /sessions: saved sessions; /models: configured models; /model NAME: switch model; /memory: history; /clear: reset; /compact: keep four recent turns; /exit: quit"
            )
        elif prompt == "/session":
            terminal.write(session_id)
        elif prompt == "/models":
            terminal.write("\n".join(models.names()))
        elif prompt == "/sessions":
            terminal.write(
                "\n".join(
                    f"{saved.id} {saved.agent_name} model={saved.metadata.get('configured_model') or saved.metadata.get('provider', '')}"
                    for saved in await sessions.list()
                )
                or "No saved sessions."
            )
        elif prompt == "/model" or prompt.startswith("/model "):
            name = prompt.partition(" ")[2].strip()
            if not name:
                saved = await sessions.load(session_id)
                terminal.write(
                    str(
                        saved.metadata.get("configured_model")
                        or saved.metadata.get("provider", "")
                    )
                    if saved
                    else "No saved model."
                )
            elif not isinstance(workflow, AgentWorkflow):
                terminal.write("Model switching requires an AgentWorkflow.")
            else:
                try:
                    configuration = models.configuration(name)
                    provider = models.get(name)
                    await workflow.set_model(
                        provider,
                        {
                            "configured_model": name,
                            "provider": configuration["provider"],
                            "model": configuration.get("model"),
                            "model_configuration": configuration,
                        },
                    )
                    terminal.write(f"Model switched to {name}.")
                except (KeyError, ValueError, RuntimeError, ImportError) as error:
                    terminal.write(f"Cannot switch model: {error}")
        elif prompt == "/memory":
            terminal.write(
                json.dumps(
                    [asdict(item) for item in await memory.load(conversation_id)],
                    indent=2,
                )
            )
        elif prompt == "/clear":
            if isinstance(workflow, AgentWorkflow):
                await workflow.clear()
            else:
                await memory.clear(conversation_id)
            terminal.write("Memory cleared.")
        elif prompt == "/compact":
            retained = await memory.compact(conversation_id, _tail_compactor)
            terminal.write(f"Memory compacted: {len(retained)} items retained.")
        elif prompt.startswith("/"):
            terminal.write("Unknown command. Use /help.")
        elif prompt:
            await _output(workflow, prompt, terminal, stream)


async def execute(
    args: argparse.Namespace,
    *,
    models: ModelRegistry | None = None,
    workflows: WorkflowRegistry | None = None,
    terminal: Terminal | None = None,
) -> None:
    command, session_id, prompt = _command(args)
    args = argparse.Namespace(**vars(args))
    settings = await asyncio.to_thread(Settings.load, args.config)
    for name in (
        "memory_root",
        "session_root",
        "storage",
        "database",
        "plain",
        "stream",
        "load_entry_points",
    ):
        if getattr(args, name) is None:
            setattr(args, name, getattr(settings, name))
    if command != "resume":
        for name in ("provider", "model", "workflow"):
            if getattr(args, name) is None:
                setattr(args, name, getattr(settings, name))
    terminal = terminal or Terminal(plain=args.plain)
    if models is None:
        models = ModelRegistry()
        models.register_builtin_providers()
    workflows = workflows or WorkflowRegistry()
    if args.load_entry_points:
        models.load_entry_points()
        workflows.load_entry_points()
    for name, configuration in settings.models.items():
        configuration = dict(configuration)
        if name in models.names() and models.configuration(name) == configuration:
            continue
        models.register_configured(name, configuration.pop("provider"), **configuration)
    if command == "info":
        from . import __version__

        terminal.write(
            f"CPPilot {__version__}\nproviders: {', '.join(models.names())}\nworkflows: {', '.join(workflows.names()) or '(none)'}"
        )
        return
    memory, sessions = Settings(
        args.memory_root, args.session_root, args.storage, args.database
    ).stores()
    try:
        await _session(
            args, session_id, prompt, memory, sessions, models, workflows, terminal
        )
    finally:
        for store in (memory, sessions):
            close = getattr(store, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result


async def _session(
    args: argparse.Namespace,
    session_id: str | None,
    prompt: str | None,
    memory: MemoryStore,
    sessions: SessionStore,
    models: ModelRegistry,
    workflows: WorkflowRegistry,
    terminal: Terminal,
) -> None:
    session = await sessions.load(session_id) if session_id else None
    if session_id and session is None:
        raise ValueError(f"session {session_id!r} not found")
    saved = session.metadata if session else {}
    if session:
        for key in ("provider", "model", "workflow", "conversation_id"):
            requested = getattr(args, key)
            expected = (
                session.conversation_id if key == "conversation_id" else saved.get(key)
            )
            if requested is not None and requested != expected:
                raise ValueError(f"cannot override saved {key} when resuming")
    provider_name = args.provider or saved.get("provider") or "anthropic"
    model = args.model if args.model is not None else saved.get("model")
    workflow_name = (
        args.workflow if args.workflow is not None else saved.get("workflow")
    )
    conversation_id = (
        session.conversation_id
        if session
        else (args.conversation_id or uuid.uuid4().hex)
    )
    configured = saved.get("model_configuration") if session else None
    if configured:
        configuration = dict(configured)
        provider = models.get(configuration.pop("provider"), **configuration)
    elif model and model in models.names() and args.provider is None:
        configuration = models.configuration(model)
        provider = models.get(model)
        saved = {
            **saved,
            "configured_model": model,
            "model_configuration": configuration,
        }
        provider_name, model = configuration["provider"], configuration.get("model")
    else:
        provider = models.get(provider_name, **({"model": model} if model else {}))
    if workflow_name:
        workflow = workflows.create(
            workflow_name,
            provider=provider,
            memory=memory,
            sessions=sessions,
            session=session,
            conversation_id=conversation_id,
        )
    else:
        workflow = AgentWorkflow(
            Agent("assistant", "Be concise and helpful.", provider),
            memory=memory,
            sessions=sessions,
            session=session,
            conversation_id=conversation_id,
        )
    async with workflow:
        if session is None:
            session = await sessions.create(
                "assistant" if not workflow_name else workflow_name,
                conversation_id,
                {
                    **saved,
                    "provider": provider_name,
                    "model": model,
                    "workflow": workflow_name,
                },
            )
            if isinstance(workflow, AgentWorkflow):
                # Custom workflows may use their own agent graph.
                session = replace(
                    session,
                    agent_name=workflow.agent.name,
                    active_agent=workflow.agent.name,
                )
                await sessions.update(session)
                workflow.session = session
        if prompt is not None:
            sys.stderr.write(f"session: {session.id}\n")
            await _output(workflow, prompt, terminal, args.stream)
        else:
            await _interactive(
                workflow,
                memory,
                conversation_id,
                session.id,
                terminal,
                args.stream,
                models,
                sessions,
            )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        asyncio.run(execute(args))
    except (ValueError, KeyError, RuntimeError, ImportError, OSError) as error:
        arguments.exit(2, f"cppilot: {error}\n")
    except KeyboardInterrupt:
        arguments.exit(130, "\n")


if __name__ == "__main__":
    main()
