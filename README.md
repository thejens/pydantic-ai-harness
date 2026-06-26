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

For state that should persist across MCP tool calls within a session — cached auth tokens, user preferences, accumulated results — define a Pydantic model for the persistent portion and pass it as `session_deps`:

```python
from pydantic import BaseModel
from dataclasses import dataclass

class SessionState(BaseModel):
    user_id: str | None = None   # cached after first auth call
    notes: dict[str, str] = {}   # accumulates across calls

@dataclass
class Deps:
    state: SessionState          # reference into the session store

async def make_deps(state: SessionState) -> Deps:
    if state.user_id is None:
        state.user_id = await auth_service.get_user()   # runs once, then cached
    return Deps(state=state)

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,
    session_deps=SessionState,   # load before / save after every call
)
```

Before each tool invocation, `SessionState` is deserialised from the store and passed to `make_deps`. After the call, the same instance is serialised back — so tools can read and write persistent state by mutating `ctx.deps.state` directly, with no coupling to MCP or FastMCP:

```python
@toolset.tool()
def remember(ctx: RunContext[Deps], key: str, value: str) -> str:
    ctx.deps.state.notes[key] = value   # mutate — auto-persisted
    return f"Stored {key!r}"

@toolset.tool()
def recall(ctx: RunContext[Deps], key: str) -> str:
    return ctx.deps.state.notes.get(key, "(not found)")
```

This mirrors how pydantic-ai handles mutable deps within a single agent run: the same instance is shared across all tool calls, so mutations are immediately visible to subsequent tools. `session_deps` extends that contract across MCP round-trips, making the session the unit of continuity instead of the run. The same toolset is drop-in compatible with a plain pydantic-ai Agent — just pass `SessionState()` as deps and the tools work identically.

`SessionState` fields must all have defaults (the class is instantiated with no arguments for new sessions).

### Distributed sessions with Redis

By default session state is kept in memory. For persistence across restarts and sharing across replicas, pass `session_state_store` to swap the backend:

```python
from key_value.aio.stores.redis import RedisStore

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,
    session_deps=SessionState,
    session_state_store=RedisStore(url=os.environ["REDIS_URL"]),
)
await server.run_async(transport="streamable-http", host="0.0.0.0", port=8000)
```

The load/save logic in the adapter is unchanged — only the backing store differs. `RedisStore` comes from the `py-key-value-aio[redis]` package. Any backend implementing the `AsyncKeyValue` protocol works: DynamoDB, Firestore, PostgreSQL, Valkey, and more are available in that package.

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
    deps: DepsT | Callable[[], DepsT] | Callable[[SessionDepsT], DepsT],
    *,
    session_deps: type[SessionDepsT] | None = None,
    prompts: Sequence[Callable[..., Any]] | None = None,
    name: str = "pydantic-ai-mcp",
    bootstrap_deps: Any = None,
    **fastmcp_kwargs: Any,
) -> FastMCP:
```

| Parameter | Description |
|---|---|
| `toolsets` | `AbstractToolset` instances — same as `Agent(toolsets=[...])` |
| `deps` | Deps instance, sync/async factory `() -> DepsT`, or session factory `(state: SessionDepsT) -> DepsT` |
| `session_deps` | Pydantic `BaseModel` class whose instance is loaded from the session store before each call and saved back after |
| `prompts` | Prompt functions: `(ctx: RunContext[DepsT], **kwargs) -> str \| list[Message] \| PromptResult` |
| `name` | Server name shown to MCP clients |
| `bootstrap_deps` | Deps used only during startup tool discovery (safe as `None` for `FunctionToolset`) |
| `**fastmcp_kwargs` | Forwarded to `FastMCP(...)` — e.g. `instructions`, `session_state_store` |

Returns a configured [`FastMCP`](https://gofastmcp.com/servers/fastmcp) server. Call `await server.run_async(transport="stdio")` to start it.

---

## Examples

| File | What it shows |
|---|---|
| [`examples/01_simple_tools.py`](examples/01_simple_tools.py) | Minimal setup — a toolset with fixed deps |
| [`examples/02_deps_factory.py`](examples/02_deps_factory.py) | Per-call deps factory, environment config, prompts with runtime context |
| [`examples/03_reuse_across_agent_and_mcp.py`](examples/03_reuse_across_agent_and_mcp.py) | The core case — one toolset wired to both a pydantic-ai Agent and an MCP server |
| [`examples/04_session_deps.py`](examples/04_session_deps.py) | Session-scoped state — auth cached once, notes persisted across calls; tools just mutate `ctx.deps.state` |
| [`examples/05_redis_session_store.py`](examples/05_redis_session_store.py) | Redis backing store — identical code to 04, one extra parameter for distributed persistence |

---

## How it works

`create_mcp_server` calls `toolset.get_tools()` at startup to discover tool schemas, then wraps each `ToolsetTool` in a thin FastMCP `Tool` subclass. On each MCP call:

1. If `session_deps` is set: the stored state is deserialised from the session store and passed to the `deps` factory
2. Args are validated through pydantic-ai's schema validator (type coercion: `str → datetime`, `dict → BaseModel`, etc.)
3. A fresh `RunContext` is built with the deps returned by the factory
4. `toolset.call_tool()` is called — the same path pydantic-ai's agent run loop uses
5. If `session_deps` is set: the (possibly mutated) state instance is serialised back to the store

Because the state is passed by reference into the factory and from there into `Deps`, any mutations the tool makes to `ctx.deps.state` are automatically captured in step 5 — no explicit save calls needed.

Prompt functions get the same treatment: `RunContext` is injected at render time; the remaining parameters are surfaced as typed MCP prompt arguments.

---

## Status

This is a prototype being developed in the [pydantic-ai harness repo](https://github.com/pydantic/pydantic-ai) before potential upstreaming. The shape may change as the session/deps contract is refined.

Feedback and issues welcome.
