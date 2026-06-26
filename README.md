# pydantic-ai-mcp

Serve a [pydantic-ai](https://ai.pydantic.dev) toolset as an [MCP](https://modelcontextprotocol.io) server. One definition, two surfaces.

```bash
pip install git+https://github.com/thejens/pydantic-ai-harness.git
```

---

## What it looks like

### Minimal

Define a toolset, call `create_mcp_server`, done:

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_mcp import create_mcp_server

@dataclass
class Deps:
    api_key: str

toolset: FunctionToolset[Deps] = FunctionToolset(id="demo")

@toolset.tool()
async def whoami(ctx: RunContext[Deps]) -> str:
    """Return the current user."""
    return f"Authenticated as {ctx.deps.api_key[:4]}…"

async def main():
    server = await create_mcp_server(toolsets=[toolset], deps=Deps(api_key="sk-…"))
    await server.run_async(transport="stdio")

asyncio.run(main())
```

### One toolset → agent and MCP server

The whole point: define tools once, use them everywhere:

```python
from pydantic_ai import Agent

# Agent path — unchanged
agent = Agent(model="openai:gpt-4o", toolsets=[toolset])

# MCP path — same toolset, deps called fresh per invocation
server = await create_mcp_server(toolsets=[toolset], deps=make_deps)
await server.run_async(transport="stdio")
```

### Session state

Persist data across MCP calls without touching FastMCP APIs. Make `Deps` a Pydantic model: plain fields are saved to the session store; `Field(exclude=True)` fields are ephemeral and rebuilt each call:

```python
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai_mcp import create_mcp_server

class Deps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # persisted across calls ──────────────────────────────────
    user_id: str | None = None
    notes: dict[str, str] = Field(default_factory=dict)
    # rebuilt each call (not stored) ──────────────────────────
    http: httpx.AsyncClient | None = Field(default=None, exclude=True)

async def make_deps(deps: Deps) -> Deps:
    if deps.user_id is None:
        deps.user_id = await auth()        # runs once per session, then cached
    deps.http = httpx.AsyncClient()        # fresh every call
    return deps

@toolset.tool()
def remember(ctx: RunContext[Deps], key: str, value: str) -> str:
    ctx.deps.notes[key] = value            # mutate — auto-saved after this returns
    return f"Stored {key!r}"

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,
    session_deps=Deps,                     # enables load-before / save-after
)
```

For distributed deployments, swap the in-memory store for Redis with one extra parameter:

```python
from key_value.aio.stores.redis import RedisStore

server = await create_mcp_server(
    toolsets=[toolset], deps=make_deps, session_deps=Deps,
    session_state_store=RedisStore(url=os.environ["REDIS_URL"]),
)
await server.run_async(transport="streamable-http", host="0.0.0.0", port=8000)
```

---

## Installation

```bash
pip install git+https://github.com/thejens/pydantic-ai-harness.git
# or
uv add git+https://github.com/thejens/pydantic-ai-harness.git
```

Requires Python 3.12+ and pydantic-ai ≥ 1.107.

---

## Usage

### Tools

Pass any `AbstractToolset` — the same list you'd pass to `Agent(toolsets=[...])`. The toolset's tools are discovered at startup and registered with FastMCP.

### Prompts

Prompt functions follow the same convention as tools — `RunContext` first, then named arguments. They become MCP [prompt templates](https://spec.modelcontextprotocol.io/specification/server/prompts/):

```python
async def expert_prompt(ctx: RunContext[Deps], domain: str) -> str:
    """System prompt that makes the model a domain expert."""
    return f"You are an expert {domain} engineer."

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,
    prompts=[expert_prompt],
)
```

The function name becomes the prompt name; the docstring its description; remaining parameters become MCP prompt arguments.

### Deps factory

For real applications, deps are created fresh per call. Pass any callable:

```python
def make_deps() -> Deps:
    return Deps(api_key=os.environ["API_KEY"])

async def make_deps_async() -> Deps:
    secret = await vault.get("api_key")
    return Deps(api_key=secret)

server = await create_mcp_server(toolsets=[toolset], deps=make_deps)
```

### Session deps

Make `Deps` a Pydantic model and pass it as `session_deps`. Before each tool call the serialisable fields are loaded from the store and the instance is passed to the factory. After the call the (possibly mutated) instance is saved back.

- **Plain fields** — JSON-serialisable types, persisted to the session store
- **`Field(exclude=True)` fields** — any type; omitted by `model_dump()` and restored to their default before the factory fills them in

All fields must have defaults; the model is instantiated with no arguments for new sessions.

This mirrors how pydantic-ai shares a mutable deps instance across all tool calls within a single agent run. `session_deps` extends that contract across MCP round-trips — the session is the unit of continuity instead of the run. The same toolset works unmodified with a plain pydantic-ai Agent (pass `Deps()` as deps).

### Distributed sessions with Redis

By default session state is kept in memory. Pass `session_state_store` to swap the backend:

```python
from key_value.aio.stores.redis import RedisStore

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,
    session_deps=Deps,
    session_state_store=RedisStore(url=os.environ["REDIS_URL"]),
)
```

`RedisStore` is from `py-key-value-aio[redis]`. Any backend implementing `AsyncKeyValue` works — DynamoDB, Firestore, PostgreSQL, Valkey, and more.

### Multiple toolsets

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
    deps: DepsT | Callable[[], DepsT] | Callable[[DepsT], DepsT],
    *,
    session_deps: type[DepsT] | None = None,
    prompts: Sequence[Callable[..., Any]] | None = None,
    name: str = "pydantic-ai-mcp",
    bootstrap_deps: Any = None,
    **fastmcp_kwargs: Any,
) -> FastMCP:
```

| Parameter | Description |
|---|---|
| `toolsets` | `AbstractToolset` instances — same as `Agent(toolsets=[...])` |
| `deps` | Deps instance, `() -> DepsT` factory, or `(state: DepsT) -> DepsT` session factory |
| `session_deps` | Pydantic `BaseModel` class — loaded from the session store before each call, saved back after |
| `prompts` | `(ctx: RunContext[DepsT], **kwargs) -> str \| list[Message] \| PromptResult` |
| `name` | Server name shown to MCP clients |
| `bootstrap_deps` | Deps used only during startup tool discovery (safe as `None` for `FunctionToolset`) |
| `**fastmcp_kwargs` | Forwarded to `FastMCP(...)` — e.g. `instructions`, `session_state_store` |

Returns a configured [`FastMCP`](https://gofastmcp.com/servers/fastmcp) server.

---

## Examples

| File | What it shows |
|---|---|
| [`examples/01_simple_tools.py`](examples/01_simple_tools.py) | Minimal — toolset with fixed deps |
| [`examples/02_deps_factory.py`](examples/02_deps_factory.py) | Per-call deps factory, env config, prompts |
| [`examples/03_reuse_across_agent_and_mcp.py`](examples/03_reuse_across_agent_and_mcp.py) | Core case — one toolset in both an Agent and an MCP server |
| [`examples/04_session_deps.py`](examples/04_session_deps.py) | Session state — auth cached once, notes persisted; tools just mutate `ctx.deps` |
| [`examples/05_redis_session_store.py`](examples/05_redis_session_store.py) | Redis backing — same code as 04, one extra parameter |

---

## How it works

`create_mcp_server` calls `toolset.get_tools()` at startup, then wraps each discovered tool in a thin FastMCP `Tool` subclass. On each MCP call:

1. If `session_deps` is set: deserialise stored state → construct `Deps` → pass to factory
2. Validate and coerce MCP args through pydantic-ai's schema validator
3. Build a `RunContext` and call `toolset.call_tool()` — the same path the agent run loop uses
4. If `session_deps` is set: serialise the (possibly mutated) `Deps` instance back to the store

Because state is passed by reference into the factory and into `RunContext.deps`, any mutations a tool makes to `ctx.deps` fields are automatically captured in step 4.

---

## Status

Prototype exploring the pydantic-ai ↔ MCP bridge before potential upstreaming. The API may change as the session/deps contract evolves.

Feedback and issues welcome.
