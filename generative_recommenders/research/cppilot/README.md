# CPPilot

CPPilot is a compact Python framework for provider-neutral, tool-using agents. Its
core has no runtime dependencies and does not import provider, telemetry, or sandbox
adapters.

The API is young. Minor releases may change interfaces until version 1.0.

## Install

CPPilot supports Python 3.12 and 3.14.

```bash
python -m pip install .
python -m pip install '.[anthropic]'  # or .[gemini], .[openai], .[server]
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

The custom-provider example implements `ModelProvider` without an SDK. Anthropic,
Gemini, and OpenAI-compatible adapters preserve messages, tool calls, tool results,
usage, and provider metadata. The function-tool example shows schema-derived
function tools. Local conversation and session persistence are available as
`FileMemoryStore` and `FileSessionStore` in `cppilot.local`; dependency-free durable
alternatives are `SQLiteMemoryStore` and `SQLiteSessionStore`.

```python
from pathlib import Path
from cppilot.local import FileMemoryStore, FileSessionStore

memory = FileMemoryStore(Path(".cppilot/memory"))
sessions = FileSessionStore(Path(".cppilot/sessions"))
# Pass memory to AgentRunner and reuse a conversation_id to resume context.
```

Extend CPPilot by implementing `ModelProvider`, `MemoryStore`, `SessionStore`,
`Sandbox`, `EventSink`, `AuthProvider`, `JobBackend`, `MetricsBackend`,
`SQLQueryBackend`, or `ObjectStore`. Optional modules provide FastAPI JSON/SSE
hosting, official-MCP session tools, OpenTelemetry events, dataset evaluation, and a
Docker/Podman CLI sandbox with networking disabled by default.

## Add a custom extension

A useful pattern for a product extension is to keep domain operations in tools,
assemble those tools into an agent, wrap the agent in a workflow, and expose the
workflow from the product's own entry point. CPPilot keeps the framework
provider-neutral, so product code owns its workflow registry and CLI.

Use these steps:

1. Create a module in the consuming project, for example
   a workflow module in a fictional `dummy_assistant` package.
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
6. Add the new module to the application's package and test tool
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
custom `ModelProvider` like the custom-provider example. The consuming
application can then call `await workflow.run(prompt)` from its existing command or
service handler. Keep credentials, data access, authorization, and production
registration in that application layer.

## Security boundaries

Models and tools are untrusted inputs. Validate tool arguments, keep write roots
narrow, and use a sandbox for commands. The built-in file tools resolve paths and
reject traversal outside configured roots. `ContainerSandbox` disables networking by
default but is optional and should be configured according to the host threat model.
Local memory and session files are plain text; do not use them for secrets.

CPPilot supports Linux and macOS. Filesystem writer locking uses POSIX locks; use
SQLite stores on Windows. Container execution requires a compatible Docker or
Podman service. Path grants protect against traversal and existing symlink escapes,
but are not a security boundary against a host process concurrently changing the
filesystem. Use an isolated container and trusted mounts for adversarial workloads.

## Runtime And Extensions

`AgentRunner.run()` returns a `RunResult` with output, parsed structured output,
run ID, aggregate usage, and per-agent usage. `stream()` yields live assistant text
chunks and complete tool calls/results in order, followed by aggregate `Usage`.
Consumers should concatenate text chunks. `parallel()` preserves input order and
cancels sibling runs on failure unless `return_exceptions=True`. All three raise
`MaxTurnsExceeded`, guardrail errors, and validation errors consistently. Transient
provider failures are retried only before any stream content has been published.

Instructions can be strings or sync/async callables of `RunContext`. Handoffs apply
the destination instructions and stores persist the active agent for later runs.
Output JSON Schema validation requires the `schema` extra and fails before the
first model call when that dependency or the schema is invalid. Final-result tools
also pass through output validation and guardrails. Streamed text is provisional:
output guardrails cannot retract chunks already sent to a consumer.

`FunctionTool` keeps simple signatures dependency-free. Install the `tools` extra
for Pydantic argument models and Griffe docstring descriptions. Tools may specify
a timeout, a final result, background execution, and system-prompt additions.
Background work belongs to its run and is cancelled when the run finishes or its
consumer closes the stream. Applications needing work to outlive a run should
supply a `JobBackend`.

Give `RunContext` separate `read_roots` and `write_roots`. Portable filesystem tools
use those grants; shell and code execution belong in a sandbox. Container mounts,
resource limits, image, and engine are explicit configuration, and networking is
off by default. Do not mount host credentials or sensitive directories into an
untrusted execution environment.

Registries are application-owned. `ModelRegistry` accepts named instances or
factories; `WorkflowRegistry` accepts workflow factories. Python entry points are
loaded only by explicit discovery calls, using the `cppilot.models` and
`cppilot.workflows` groups. `AgentWorkflow` provides conversation/session handling
and `WorkflowContext` owns sandbox startup and cleanup.

## CLI And Hosting

```bash
cppilot info
cppilot run "Explain a sorting algorithm" --provider openai --storage sqlite
cppilot run --provider anthropic --stream
cppilot resume SESSION_ID "Continue the explanation" --storage sqlite
```

Runs print the session ID separately from assistant output. Interactive commands
include `/help`, `/session`, `/memory`, `/clear`, `/compact`, and `/exit`. Model and
workflow selection use configured factories; a saved session rejects incompatible
configuration overrides. Rich and prompt-toolkit are optional through the `cli`
extra, and `--plain` forces the terminal fallback.

The `server` extra supplies FastAPI JSON and SSE hosting. The consuming application
must supply authentication and authorization; an API-key principal or verified
OIDC principal can be passed into `RunContext`. OIDC verification must check the
signature, issuer, audience, and expiry using the application's identity library.
Do not expose an unauthenticated server or place tokens in conversation memory.
SSE consumers must handle a terminal error event and close disconnected streams.

The `telemetry` extra supplies OpenTelemetry SDK/OTLP exporters. Run IDs correlate
run, model, retry, tool, and handoff spans and token/latency/error metrics. Export to
your own collector and dashboard; prompts and tool contents need not be exported.
The `mcp` extra supports stdio and authenticated Streamable HTTP servers with
explicit client-session lifetime management.

SQLite is the dependency-free durable backend. The `storage` extra adds async
SQLAlchemy stores and fsspec object storage; `postgres` adds asyncpg, and `s3` adds
S3-compatible filesystem support. Database query tools should use a read-only
transaction and a database identity with no write privileges. Optional
`kubernetes`, `mlflow`, and `prometheus` extras implement the generic job/metrics
extension contracts. The `eval` extra supplies classification metrics, prompt
evaluators, and bounded-concurrency dataset runners.

## Development

```bash
python -m pip install -e '.[dev]'
pytest
python -m build
```

All examples use synthetic data. The directory is self-contained for source distributions and wheels.
