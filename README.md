# pydantic-ai-harness

A prototype bridge that turns [pydantic-ai](https://ai.pydantic.dev) toolsets and prompts into a [FastMCP](https://gofastmcp.com) server.

If you've already built tools for a pydantic-ai agent, you can expose them to Claude Code, Cursor, or any other MCP client with a handful of extra lines — no rewriting, no parallel implementations.

---

## The idea

pydantic-ai's `FunctionToolset` is a clean abstraction for defining typed, dependency-injected tools:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

toolset: FunctionToolset[MyDeps] = FunctionToolset(id="my-tools")

@toolset.tool()
async def search(ctx: RunContext[MyDeps], query: str) -> list[str]:
    """Search the knowledge base."""
    return await ctx.deps.kb.search(query)

# Use it in an agent
agent = Agent(model="openai:gpt-4o", toolsets=[toolset])
```

This package lets you serve the same toolset as an MCP server:

```python
from pydantic_ai_mcp import create_mcp_server

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,       # same deps, called fresh per MCP invocation
)
await server.run_async(transport="stdio")
```

That's it. One definition, two surfaces.

---

## Installation

```bash
pip install git+https://github.com/thejens/pydantic-ai-harness.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/thejens/pydantic-ai-harness.git
```

Requires Python 3.12+ and pydantic-ai ≥ 1.107.

---

## Usage

### Tools

Pass any `AbstractToolset` — the same list you'd pass to `Agent(toolsets=[...])`:

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_mcp import create_mcp_server

@dataclass
class Deps:
    api_key: str

toolset: FunctionToolset[Deps] = FunctionToolset(id="example")

@toolset.tool()
async def current_user(ctx: RunContext[Deps]) -> str:
    """Return the name of the current user."""
    return f"Authenticated with key: {ctx.deps.api_key[:4]}..."

async def main():
    server = await create_mcp_server(
        toolsets=[toolset],
        deps=Deps(api_key="sk-..."),
        name="my-server",
    )
    await server.run_async(transport="stdio")

asyncio.run(main())
```

### Prompts

Prompt functions follow the same convention — a `RunContext` first parameter, named keyword arguments after. They become MCP [prompt templates](https://spec.modelcontextprotocol.io/specification/server/prompts/):

```python
async def expert_prompt(ctx: RunContext[Deps], domain: str) -> str:
    """System prompt that makes the model a domain expert."""
    return f"You are an expert {domain} engineer working with key {ctx.deps.api_key[:4]}..."

server = await create_mcp_server(
    toolsets=[toolset],
    deps=Deps(api_key="sk-..."),
    prompts=[expert_prompt],
)
```

The function name becomes the prompt name; the docstring becomes its description; parameters (excluding `RunContext`) become the prompt's arguments.

### Deps factory

For real applications, deps should be created fresh per call — reading secrets from the environment, opening DB connections, etc. Pass any callable instead of an instance:

```python
import os

def make_deps() -> Deps:
    return Deps(api_key=os.environ["API_KEY"])

# async factories work too
async def make_deps_async() -> Deps:
    secret = await vault.get("api_key")
    return Deps(api_key=secret)

server = await create_mcp_server(toolsets=[toolset], deps=make_deps)
```

### Session deps

Some deps are expensive to compute (auth round-trips, user-profile fetches). If your factory declares one positional argument it receives the FastMCP [`Context`](https://gofastmcp.com/servers/context), which provides session-scoped state that persists across tool calls within the same MCP session:

```python
from fastmcp.server.context import Context

async def make_deps(ctx: Context) -> Deps:
    # Fetched once per session, cached for all subsequent calls
    user_id = await ctx.get_state("user_id")
    if user_id is None:
        user_id = await auth_service.get_user()          # expensive — runs once
        await ctx.set_state("user_id", user_id)          # JSON-serializable → persists

    return Deps(user_id=user_id, request_id=new_uuid())  # cheap per-call part

server = await create_mcp_server(toolsets=[toolset], deps=make_deps)
```

`ctx.set_state` with the default `serializable=True` stores values in FastMCP's `AsyncKeyValue` store (in-memory by default, swappable for Redis via `FastMCP(session_state_store=...)`). For non-serializable objects like HTTP clients, pass `serializable=False` — those are request-scoped and rebuilt each call, but cheaply from cached serializable data.

Zero-argument factories remain fully supported and are called without a context.

### Multiple toolsets

Combine as many toolsets as you like — the same way you would with an agent:

```python
server = await create_mcp_server(
    toolsets=[search_toolset, files_toolset, metrics_toolset],
    deps=make_deps,
    prompts=[analyst_prompt, researcher_prompt],
    name="full-suite",
    instructions="Company intelligence tools.",
)
```

---

## API

```python
async def create_mcp_server(
    toolsets: Sequence[AbstractToolset[DepsT]],
    deps: DepsT | Callable[[], DepsT] | Callable[[], Awaitable[DepsT]],
    *,
    prompts: Sequence[Callable[..., Any]] | None = None,
    name: str = "pydantic-ai-mcp",
    bootstrap_deps: Any = None,
    **fastmcp_kwargs: Any,
) -> FastMCP:
```

| Parameter | Description |
|---|---|
| `toolsets` | `AbstractToolset` instances — same as `Agent(toolsets=[...])` |
| `deps` | Deps instance, sync factory, or async factory — called fresh per invocation |
| `prompts` | Prompt functions: `(ctx: RunContext[DepsT], **kwargs) -> str \| list[Message] \| PromptResult` |
| `name` | Server name shown to MCP clients |
| `bootstrap_deps` | Deps used only during startup tool discovery (safe as `None` for `FunctionToolset`) |
| `**fastmcp_kwargs` | Forwarded to `FastMCP(...)` — e.g. `instructions`, `version` |

Returns a configured [`FastMCP`](https://gofastmcp.com/servers/fastmcp) server. Call `await server.run_async(transport="stdio")` to start it.

---

## Examples

| File | What it shows |
|---|---|
| [`examples/01_simple_tools.py`](examples/01_simple_tools.py) | Minimal setup — a toolset with fixed deps |
| [`examples/02_deps_factory.py`](examples/02_deps_factory.py) | Per-call deps factory, environment config, prompts with runtime context |
| [`examples/03_reuse_across_agent_and_mcp.py`](examples/03_reuse_across_agent_and_mcp.py) | The core case — one toolset wired to both a pydantic-ai Agent and an MCP server |
| [`examples/04_session_deps.py`](examples/04_session_deps.py) | Session-scoped caching — expensive auth computed once per MCP session via `ctx.get_state` / `ctx.set_state` |

---

## How it works

`create_mcp_server` calls `toolset.get_tools()` at startup to discover tool schemas, then wraps each `ToolsetTool` in a thin FastMCP `Tool` subclass. On each MCP call:

1. Args are validated through pydantic-ai's schema validator (type coercion: `str → datetime`, `dict → BaseModel`, etc.)
2. A fresh `RunContext` is built with deps from the factory
3. `toolset.call_tool()` is called — the same path pydantic-ai's agent run loop uses

Prompt functions get the same treatment: `RunContext` is injected at render time; the remaining parameters are surfaced as typed MCP prompt arguments.

---

## Status

This is a prototype being developed in the [pydantic-ai harness repo](https://github.com/pydantic/pydantic-ai) before potential upstreaming. The shape may change as the session/deps contract is refined.

Feedback and issues welcome.
