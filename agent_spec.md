# Implementation spec: pydantic-ai-mcp

How to implement the `pydantic-ai-mcp` package. This captures the API surface,
design decisions, and non-obvious findings from the original implementation so
you don't have to rediscover them.

---

## Goal

An `MCPServer` class that:
- Subclasses `FastMCP` to inherit `run()`, `run_async()`, `http_app()`, etc.
- Accepts pydantic-ai `AbstractToolset` instances (the same list you'd pass to `Agent`)
- Registers tools via `@server.tool()` decorator (pydantic-ai RunContext convention)
- Optionally accepts pydantic-ai-style prompt functions via `@server.prompt` or `prompts=[…]`
- Optionally manages per-session state via FastMCP's session store
- Discovers and registers pydantic-ai tools during ASGI lifespan startup

---

## Module layout

```
pydantic_ai_mcp/
    __init__.py          # public: MCPServer
    _server.py           # MCPServer(FastMCP) + _validate_session_deps()
    _tool_adapter.py     # PydanticAIToolAdapter(Tool) + _load/_save session state
    _prompt_adapter.py   # PydanticAIPromptAdapter(Prompt)
    _context.py          # make_bootstrap_context(), make_call_context()
```

---

## Key external APIs

### pydantic-ai

```python
from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel    # lightest concrete Model subclass
from pydantic_ai.usage import RunUsage
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

# Constructing a RunContext (required by all toolset calls)
ctx = RunContext(deps=deps, model=TestModel(), usage=RunUsage(), max_retries=0)
ctx = RunContext(deps=deps, model=TestModel(), usage=RunUsage(), max_retries=0, tool_name="foo")

# Discovering tools at startup
async with toolset:
    tools: dict[str, ToolsetTool] = await toolset.get_tools(ctx)
# get_tools() only reads ctx.max_retries — ctx.deps is never used → safe to pass None

# Calling a tool
result = await toolset.call_tool(tool_name, validated_args, ctx, toolset_tool)

# Validating MCP args into Python types before calling (str→datetime, dict→Model, etc.)
validated = toolset_tool.args_validator.validate_python(raw_args_dict)

# Tool schema — already valid MCP inputSchema format, pass directly to FastMCP
tool_def: ToolDefinition = toolset_tool.tool_def
tool_def.name           # str
tool_def.description    # str | None
tool_def.parameters_json_schema   # {"type": "object", "properties": {...}}
```

### FastMCP

```python
from fastmcp import FastMCP
from fastmcp.tools.base import Tool, ToolResult
from fastmcp.prompts.base import Prompt, PromptArgument
from pydantic import PrivateAttr

# Tool subclass — FastMCP Tool is a Pydantic BaseModel with extra="forbid"
# Use PrivateAttr for non-schema state: stored in __pydantic_private__,
# not subject to the extra-field restriction, not part of the MCP schema.
class MyTool(Tool):
    _my_state: SomeType = PrivateAttr()

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        ...  # ToolResult = self.convert_result(any_value)

# Prompt subclass — same pattern
class MyPrompt(Prompt):
    _fn: Callable = PrivateAttr()

    async def render(self, arguments: dict[str, Any] | None = None) -> Any:
        ...  # inherited convert_result() handles str / list[Message] / PromptResult

# FastMCP constructor — MCPServer passes lifespan= and forwards **fastmcp_kwargs
FastMCP(name="...", lifespan=my_lifespan_fn, **kwargs)

# Dynamic tool/prompt registration (used in lifespan)
server.add_tool(PydanticAIToolAdapter(...))
server.add_prompt(PydanticAIPromptAdapter(...))

# Running
server.run(transport="stdio")                   # sync, uses anyio.run internally
await server.run_async(transport="stdio")
await server.run_async(transport="streamable-http", host="0.0.0.0", port=8000)

# ASGI mounting on FastAPI — inherited from FastMCP
starlette_app = server.http_app(path=None, transport="http")  # returns StarletteWithLifespan
fastapi_app.mount("/mcp", starlette_app)

# Session state — available inside Tool.run() and Prompt.render()
from fastmcp.server.dependencies import get_context
ctx = get_context()              # reads a contextvar set by FastMCP per request
await ctx.get_state(key)         # returns None if absent
await ctx.set_state(key, value)  # value must be JSON-serialisable
await ctx.delete_state(key)
ctx.session_id                   # stable per-session identifier

# Prompt rendering — NOT get_prompt()
# server.get_prompt(name, version) — second arg is VersionSpec, not arguments
await server.render_prompt(name, arguments_dict)

# Distributed session store (passed as FastMCP constructor kwarg)
from key_value.aio.stores.redis import RedisStore   # py-key-value-aio[redis]
FastMCP(..., session_state_store=RedisStore(url="redis://localhost:6379/0"))
```

---

## _server.py

`MCPServer` subclasses `FastMCP`. Tool discovery happens during lifespan startup
via `_discover_and_register()`, which calls `get_tools()` on each toolset (including
the inline `_inline_toolset` that `@server.tool()` decorates into) and calls
`self.add_tool()` / `self.add_prompt()` for each result.

```python
class MCPServer(FastMCP):
    def __init__(self, *, toolsets=(), deps=None, session_deps=None, prompts=None,
                 name="pydantic-ai-mcp", bootstrap_deps=None, **fastmcp_kwargs):

        # Set before super().__init__() — lifespan closure reads these at startup
        self._pai_toolsets = list(toolsets)
        self._pai_deps = deps
        self._pai_session_deps = session_deps
        self._pai_prompts = list(prompts or [])
        self._pai_bootstrap_deps = bootstrap_deps
        self._inline_toolset = FunctionToolset()  # receives @server.tool() calls

        user_lifespan = fastmcp_kwargs.pop("lifespan", None)

        @asynccontextmanager
        async def _pai_lifespan(server: FastMCP):
            assert isinstance(server, MCPServer)
            await server._discover_and_register()
            if user_lifespan is not None:
                async with user_lifespan(server):
                    yield
            else:
                yield

        super().__init__(name=name, lifespan=_pai_lifespan, **fastmcp_kwargs)

    def tool(self):
        """@server.tool() — pydantic-ai style (RunContext first arg)."""
        return self._inline_toolset.tool()

    def prompt(self, fn):
        """@server.prompt — pydantic-ai style (RunContext first arg)."""
        self._pai_prompts.append(fn)
        return fn

    async def _discover_and_register(self):
        all_toolsets = [*self._pai_toolsets, self._inline_toolset]
        bootstrap_ctx = make_bootstrap_context(deps=self._pai_bootstrap_deps)
        for toolset in all_toolsets:
            async with toolset:
                discovered = await toolset.get_tools(bootstrap_ctx)
            for toolset_tool in discovered.values():
                self.add_tool(PydanticAIToolAdapter.from_toolset_tool(
                    toolset, toolset_tool, self._pai_deps,
                    session_deps_cls=self._pai_session_deps,
                ))
        for fn in self._pai_prompts:
            self.add_prompt(PydanticAIPromptAdapter.from_function(
                fn, self._pai_deps, session_deps_cls=self._pai_session_deps,
            ))


def _validate_session_deps(cls):
    # Catch structural problems at construction time, not mid-request
    try:
        instance = cls()
    except Exception as exc:
        raise TypeError(f"session_deps {cls.__name__!r}: all fields must have defaults. {exc}") from exc
    try:
        cls.model_validate(instance.model_dump(mode="json"))
    except Exception as exc:
        raise TypeError(
            f"session_deps {cls.__name__!r}: serialisation round-trip failed. "
            f"Non-serialisable fields must use Field(exclude=True). {exc}"
        ) from exc
```

Key design points:
- `_inline_toolset` is created before `super().__init__()` so that the lifespan
  closure can reference it via `self` without an AttributeError.
- `user_lifespan` is composed inside `_pai_lifespan` so users can still pass a
  `lifespan=` kwarg to `MCPServer(...)` and it will run after pydantic-ai setup.
- `@server.tool()` decorates into `_inline_toolset`; discovery picks it up automatically.
- `_discover_and_register()` is also callable directly in tests (without starting the server).

---

## _context.py

Constructs `RunContext` for bootstrap and live calls.

```python
_STUB_MODEL = TestModel()   # module-level singleton — no event loop binding

def make_bootstrap_context(deps=None, *, max_retries=0) -> RunContext:
    return RunContext(deps=deps, model=_STUB_MODEL, usage=RunUsage(), max_retries=max_retries)

async def make_call_context(factory, *, session_state=None, tool_name=None, max_retries=0) -> RunContext:
    if session_state is not None:
        # session_deps path: factory receives the pre-loaded state instance
        deps = factory(session_state)
        if inspect.isawaitable(deps): deps = await deps
    else:
        deps = await _resolve_deps(factory)
    return RunContext(deps=deps, model=_STUB_MODEL, usage=RunUsage(),
                     max_retries=max_retries, tool_name=tool_name)

async def _resolve_deps(factory):
    if callable(factory):
        result = factory(get_context()) if _factory_takes_context(factory) else factory()
        return await result if inspect.isawaitable(result) else result
    return factory   # plain instance — return as-is

def _factory_takes_context(factory) -> bool:
    # A positional parameter → inject FastMCP Context; zero params → call as-is
    sig = inspect.signature(factory)
    return any(p.kind in (POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD) for p in sig.parameters.values())
```

---

## _tool_adapter.py

```python
_SESSION_STATE_KEY = "__deps__"   # key under which session state is stored

class PydanticAIToolAdapter(Tool):
    _toolset: AbstractToolset = PrivateAttr()
    _toolset_tool: ToolsetTool = PrivateAttr()
    _deps_factory: Any = PrivateAttr()
    _max_retries: int = PrivateAttr(default=0)
    _session_deps_cls: type | None = PrivateAttr(default=None)

    @classmethod
    def from_toolset_tool(cls, toolset, toolset_tool, deps_factory, max_retries=0, session_deps_cls=None):
        tool_def = toolset_tool.tool_def
        instance = cls(name=tool_def.name, description=tool_def.description,
                       parameters=tool_def.parameters_json_schema)
        instance._toolset = toolset
        instance._toolset_tool = toolset_tool
        instance._deps_factory = deps_factory
        instance._max_retries = max_retries
        instance._session_deps_cls = session_deps_cls
        return instance

    async def run(self, arguments: dict) -> ToolResult:
        session_state, fmcp_ctx = await _load_session_state(self._session_deps_cls)
        validated = self._toolset_tool.args_validator.validate_python(arguments)
        ctx = await make_call_context(self._deps_factory, session_state=session_state,
                                      tool_name=self.name, max_retries=self._max_retries)
        result = await self._toolset.call_tool(self.name, validated, ctx, self._toolset_tool)
        await _save_session_state(session_state, fmcp_ctx)
        return self.convert_result(result)
```

Session state helpers:

```python
async def _load_session_state(cls):
    if cls is None: return None, None
    fmcp_ctx = get_context()
    raw = await fmcp_ctx.get_state(_SESSION_STATE_KEY)
    state = cls.model_validate(raw) if raw is not None else cls()
    return state, fmcp_ctx

async def _save_session_state(state, fmcp_ctx):
    if state is None: return
    # mode='json' catches non-serialisable fields at dump time instead of
    # silently passing raw Python objects that blow up inside set_state()
    await fmcp_ctx.set_state(_SESSION_STATE_KEY, state.model_dump(mode="json"))
```

The load-mutate-by-reference-save cycle:
1. `state = cls()` or deserialise from store — a fresh Pydantic instance
2. `factory(state)` — factory stores the state reference inside `Deps`
3. Tool runs; if it mutates `ctx.deps.notes["x"] = "y"`, the mutation is on the same Python object
4. `state.model_dump(mode="json")` reflects the mutation — Python objects are passed by reference

---

## _prompt_adapter.py

```python
class PydanticAIPromptAdapter(Prompt):
    _fn: Callable = PrivateAttr()
    _deps_factory: Any = PrivateAttr()
    _max_retries: int = PrivateAttr(default=0)
    _session_deps_cls: type | None = PrivateAttr(default=None)

    @classmethod
    def from_function(cls, fn, deps_factory, *, max_retries=0, name=None, description=None, session_deps_cls=None):
        sig = inspect.signature(fn)
        prompt_params = list(sig.parameters.values())[1:]   # skip RunContext (first param)
        arguments = [PromptArgument(name=p.name, required=p.default is Parameter.empty)
                     for p in prompt_params]
        instance = cls(name=name or fn.__name__, description=description or getdoc(fn), arguments=arguments)
        instance._fn = fn
        instance._deps_factory = deps_factory
        instance._max_retries = max_retries
        instance._session_deps_cls = session_deps_cls
        return instance

    async def render(self, arguments=None):
        session_state, fmcp_ctx = await _load_session_state(self._session_deps_cls)
        ctx = await make_call_context(self._deps_factory, session_state=session_state,
                                      max_retries=self._max_retries)
        result = self._fn(ctx, **(arguments or {}))
        if inspect.isawaitable(result): result = await result
        await _save_session_state(session_state, fmcp_ctx)
        return result
```

---

## session_deps design

`session_deps` is a Pydantic `BaseModel` class whose instance is the unit of
per-session persistence. Users annotate fields:

- **Plain fields** — JSON-serialisable, stored in FastMCP's session store (in-memory by default, Redis etc. via `session_state_store=`)
- **`Field(exclude=True)` fields** — arbitrary types; `model_dump()` omits them, `model_validate()` restores them to their defaults; the factory fills them in each call

```python
class Deps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_id: str | None = None                                      # persisted
    notes: dict[str, str] = Field(default_factory=dict)            # persisted
    http: httpx.AsyncClient | None = Field(default=None, exclude=True)  # ephemeral
```

The factory receives the pre-loaded instance and fills in ephemeral fields:

```python
async def make_deps(deps: Deps) -> Deps:
    if deps.user_id is None:
        deps.user_id = await auth()    # cached after first call
    deps.http = httpx.AsyncClient()    # rebuilt every call
    return deps
```

Nested Pydantic models work correctly: `model_dump(mode="json")` and
`model_validate()` recurse into nested models, and `Field(exclude=True)` on
nested model fields is respected.

---

## Non-obvious findings

- **`run()` / `run_async()` inherited from FastMCP** — do not reimplement. FastMCP uses `anyio.run()` internally (not `asyncio.run()`), which ensures compatibility with both asyncio and trio.
- **Lifespan for async setup** — pydantic-ai tool discovery requires `async with toolset` + `await toolset.get_tools()`. This must run in the lifespan, not `__init__`. Use `FastMCP(lifespan=...)` and compose with any user-provided lifespan.
- **Set `_pai_*` attrs before `super().__init__()`** — the lifespan closure captures `self`, and FastMCP may access the lifespan during init. All attributes the lifespan reads must exist before `super().__init__()` returns.
- **`http_app()` inherited from FastMCP** — returns a `StarletteWithLifespan` app. When mounted on FastAPI via `app.mount("/mcp", server.http_app())`, Starlette automatically manages the sub-app's lifespan (including the pydantic-ai discovery lifespan).
- **`_discover_and_register()` callable in tests** — avoids starting the full server; call it directly then use `list_tools()`, `call_tool()`, etc.
- **`run_async()` not `run_stdio_async()`** — FastMCP's run method is `server.run_async(transport="stdio")`.
- **`render_prompt()` not `get_prompt()`** — `server.get_prompt(name, version)` takes a `VersionSpec` as second arg, not an arguments dict; use `server.render_prompt(name, args_dict)` instead.
- **`PrivateAttr` works with `extra="forbid"`** — FastMCP's `Tool` and `Prompt` base classes use `extra="forbid"`, but `PrivateAttr` fields live in `__pydantic_private__` and are not subject to that restriction.
- **Bootstrap deps can be `None`** — `FunctionToolset.get_tools()` only reads `ctx.max_retries`, never `ctx.deps`. Passing `None` for startup discovery is safe.
- **`args_validator.validate_python()` before `call_tool()`** — pydantic-ai tools expect validated Python types, not raw MCP JSON. Always validate first.
- **`model_dump(mode='json')` vs `model_dump()`** — without `mode='json'`, non-serialisable objects pass through as raw Python and only fail when `ctx.set_state()` tries to JSON-encode them. Use `mode='json'` to surface the error at dump time with a clear Pydantic message.
- **`TestModel` as stub** — the lightest concrete `Model` subclass; tools never read `ctx.model` in normal use, so this satisfies the type constraint at zero cost.
