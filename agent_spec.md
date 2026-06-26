# Agent spec: pydantic-ai MCP server

You are building a Python MCP server that exposes pydantic-ai toolsets to MCP clients (Claude Code, Cursor, etc.) using the `pydantic-ai-mcp` package.

## Package

```bash
uv add git+https://github.com/thejens/pydantic-ai-harness.git
```

The only public API is `create_mcp_server`.

## Core pattern

```python
from pydantic_ai_mcp import create_mcp_server

server = await create_mcp_server(
    toolsets=[my_toolset],          # AbstractToolset instances, same as Agent(toolsets=[...])
    deps=make_deps,                 # factory called fresh before every tool invocation
    prompts=[my_prompt],            # optional prompt template functions
    name="my-server",
    **fastmcp_kwargs,               # forwarded to FastMCP(...)
)
await server.run_async(transport="stdio")   # or "streamable-http"
```

## Toolset conventions (pydantic-ai)

Tools are defined on a `FunctionToolset[DepsT]`. The first parameter of every tool function is `ctx: RunContext[DepsT]`; pydantic-ai injects it — it is NOT an MCP argument. All other parameters become MCP tool arguments.

```python
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

toolset: FunctionToolset[Deps] = FunctionToolset(id="my-tools")

@toolset.tool()
async def search(ctx: RunContext[Deps], query: str) -> list[str]:
    """Search the knowledge base."""           # docstring → MCP tool description
    return await ctx.deps.db.search(query)
```

The same toolset can be passed to both `Agent(toolsets=[...])` and `create_mcp_server(toolsets=[...])` — define it once, use it on both surfaces.

## Deps factory

Pass a factory instead of a fixed instance so deps are constructed fresh per call:

```python
async def make_deps() -> Deps:
    return Deps(
        db=await Database.connect(os.environ["DB_URL"]),
        api_key=os.environ["API_KEY"],
    )
```

## Session state

To persist data across MCP tool calls within a session, make `Deps` a Pydantic `BaseModel` and pass it as `session_deps`:

```python
from pydantic import BaseModel, ConfigDict, Field

class Deps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Plain fields → persisted to the session store (must be JSON-serialisable)
    user_id: str | None = None
    preferences: dict[str, str] = Field(default_factory=dict)

    # Field(exclude=True) → ephemeral, not persisted; rebuilt by factory each call
    db: Database | None = Field(default=None, exclude=True)
    http: httpx.AsyncClient | None = Field(default=None, exclude=True)

async def make_deps(deps: Deps) -> Deps:
    # deps.user_id / deps.preferences are already loaded from the session store
    if deps.user_id is None:
        deps.user_id = await auth_service.get_user()   # cached after first call
    deps.db = await Database.connect(os.environ["DB_URL"])   # ephemeral — rebuilt
    deps.http = httpx.AsyncClient()
    return deps
```

Tool functions just mutate `ctx.deps` — the library saves the session fields automatically after each call:

```python
@toolset.tool()
def set_preference(ctx: RunContext[Deps], key: str, value: str) -> str:
    ctx.deps.preferences[key] = value   # mutate — auto-saved
    return f"Set {key!r} = {value!r}"
```

Rules for `Deps` when using `session_deps`:
- All fields must have defaults (model is constructed with no args for new sessions)
- Non-serialisable fields (DB connections, HTTP clients, etc.) **must** use `Field(exclude=True)`
- Nested Pydantic models work; `Field(exclude=True)` on nested models is respected recursively

## Prompt templates

Prompt functions take `RunContext[DepsT]` as their first parameter. The remaining parameters become MCP prompt arguments. Return a string or list of messages.

```python
async def system_prompt(ctx: RunContext[Deps], domain: str) -> str:
    """Customisable system prompt."""
    return f"You are an expert {domain} assistant with access to {ctx.deps.user_id}'s data."
```

## Distributed / persistent sessions (Redis)

```python
from key_value.aio.stores.redis import RedisStore  # pip: py-key-value-aio[redis]

server = await create_mcp_server(
    toolsets=[toolset],
    deps=make_deps,
    session_deps=Deps,
    name="my-server",
    session_state_store=RedisStore(url=os.environ["REDIS_URL"]),
)
await server.run_async(transport="streamable-http", host="0.0.0.0", port=8000)
```

Use HTTP transport (`streamable-http`) for deployed servers; use `stdio` for local MCP clients.

## What NOT to do

- Do not import `fastmcp` in tool or factory code — the library handles the FastMCP layer
- Do not use `Field(exclude=True)` for fields that need to be shared across sessions — those should be plain fields
- Do not put non-serialisable objects in plain (non-excluded) fields — the startup check will raise a `TypeError` with a clear message
- Do not return `None` from a factory — always return the `Deps` instance
