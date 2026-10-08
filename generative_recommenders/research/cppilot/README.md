# CPPilot

CPPilot is a compact Python framework for provider-neutral, tool-using agents. Its
core has no runtime dependencies and does not import provider, telemetry, or sandbox
adapters.

The API is young. Minor releases may change interfaces until version 1.0.

## Install

CPPilot supports Python 3.12 and 3.14.

```bash
python -m pip install .
python -m pip install '.[anthropic]'  # or .[gemini], .[telemetry], .[podman]
cppilot --help
```

Provider credentials use each provider SDK's standard environment variables. For
example, set `ANTHROPIC_API_KEY` before selecting the Anthropic adapter, or
`GEMINI_API_KEY` for Gemini.

## Quick start

```python
import asyncio
from cppilot import Agent, AgentRunner
from cppilot.adapters.anthropic import AnthropicProvider

async def main():
    agent = Agent("helper", "Answer clearly.", AnthropicProvider())
    print((await AgentRunner().run(agent, "Give me one idea.")).output)

asyncio.run(main())
```

`examples/custom_provider.py` implements `ModelProvider` without an SDK.
`examples/function_tools.py` shows schema-derived function tools. Local conversation
and session persistence are available as `FileMemoryStore` and `FileSessionStore` in
`cppilot.local`:

```python
from pathlib import Path
from cppilot.local import FileMemoryStore, FileSessionStore

memory = FileMemoryStore(Path(".cppilot/memory"))
sessions = FileSessionStore(Path(".cppilot/sessions"))
# Pass memory to AgentRunner and reuse a conversation_id to resume context.
```

Extend CPPilot by implementing `ModelProvider`, `MemoryStore`, `SessionStore`,
`Sandbox`, or `EventSink`. Provider adapters live in separate modules and load their
optional SDK only when instantiated.

## Add a custom extension

A useful pattern for a product extension is to keep domain operations in tools,
assemble those tools into an agent, wrap the agent in a workflow, and expose the
workflow from the product's own entry point. CPPilot keeps the framework
provider-neutral, so product code owns its workflow registry and CLI.

Use these steps:

1. Create a module in the consuming project, for example
   `dummy_assistant/workflow.py`.
2. Define narrowly scoped domain functions with complete type annotations and
   docstrings. Wrap them with `FunctionTool`; CPPilot derives their JSON schemas
   from the signatures.
3. Construct an `Agent` with product instructions, the tools, and a
   `ModelProvider`. Put request-specific values in `RunContext` rather than global
   state.
4. Implement `Workflow` to provide a stable product API and invoke the agent with
   `AgentRunner`.
5. Register or call the workflow from the consuming application's CLI, service, or
   workflow registry. Registration is application-specific and is not required by
   CPPilot itself.
6. Add the new module to the application's package or Buck target and test tool
   behavior with a fake provider before testing a live provider.

This synthetic extension shows the complete framework portion:

```python
from cppilot import (
    Agent,
    AgentRunner,
    FunctionTool,
    ModelProvider,
    RunContext,
    Workflow,
)


def lookup_dummy_records(
    context: RunContext, limit: int = 5
) -> list[dict[str, object]]:
    """Return synthetic records for the current account."""
    account_id = context.values["account_id"]
    return [
        {"account_id": account_id, "record_id": index, "score": 1 / index}
        for index in range(1, limit + 1)
    ]


class DummyAssistant(Workflow[str, str]):
    def __init__(self, provider: ModelProvider, account_id: str) -> None:
        self.provider = provider
        self.account_id = account_id

    async def run(self, prompt: str) -> str:
        agent = Agent(
            name="dummy_assistant",
            instructions=(
                "Help analyze synthetic records. "
                "Use the lookup tool instead of inventing records."
            ),
            provider=self.provider,
            tools=[FunctionTool(lookup_dummy_records)],
            context=RunContext({"account_id": self.account_id}),
        )
        result = await AgentRunner().run(agent, prompt)
        return result.output
```

Instantiate `DummyAssistant` with an adapter such as `AnthropicProvider`, or with a
custom `ModelProvider` like the one in `examples/custom_provider.py`. The consuming
application can then call `await workflow.run(prompt)` from its existing command or
service handler. Keep credentials, data access, authorization, and production
registration in that application layer.

## Security boundaries

Models and tools are untrusted inputs. Validate tool arguments, keep write roots
narrow, and use a sandbox for commands. The built-in file tools resolve paths and
reject traversal outside configured roots. `PodmanSandbox` disables networking by
default but is optional and should be configured according to the host threat model.
Local memory and session files are plain text; do not use them for secrets.

CPPilot supports Linux and macOS. The core is portable to Windows; the optional
Podman adapter depends on a compatible Podman service.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```

All examples use synthetic data. The directory is self-contained for source distributions and wheels.
