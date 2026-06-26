from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import AbstractToolset

from ._context import make_bootstrap_context
from ._prompt_adapter import PydanticAIPromptAdapter
from ._tool_adapter import PydanticAIToolAdapter


class MCPServer(FastMCP):
    """A pydantic-ai-aware MCP server, extending FastMCP.

    Mirrors the pydantic-ai ``Agent`` interface: instantiate with toolsets and
    deps, register tools directly with ``@server.tool()``, then call ``.run()``
    to start.  All FastMCP methods (``run()``, ``run_async()``, ``http_app()``,
    ``add_middleware()``, …) are inherited and work as documented by FastMCP.

    Pydantic-ai toolsets and prompt functions are discovered during ASGI lifespan
    startup and registered with FastMCP via ``add_tool()`` / ``add_prompt()``.

    Example — decorator style::

        server = MCPServer(deps=Deps(api_key="sk-…"), name="my-server")

        @server.tool()
        async def whoami(ctx: RunContext[Deps]) -> str:
            return f"key={ctx.deps.api_key[:4]}…"

        server.run(transport="stdio")

    Example — toolset style (share toolsets with a pydantic-ai Agent)::

        server = MCPServer(toolsets=[my_toolset], deps=make_deps)
        server.run(transport="stdio")

    Example — mount on FastAPI (``http_app()`` inherited from FastMCP)::

        app = FastAPI()
        app.mount("/mcp", server.http_app())
    """

    def __init__(
        self,
        *,
        toolsets: Sequence[AbstractToolset[Any]] = (),
        deps: Any = None,
        session_deps: type[Any] | None = None,
        prompts: Sequence[Callable[..., Any]] | None = None,
        name: str = "pydantic-ai-mcp",
        bootstrap_deps: Any = None,
        **fastmcp_kwargs: Any,
    ) -> None:
        """
        Args:
            toolsets: Same list you'd pass to ``Agent(toolsets=[…])``. Each
                toolset's tools are discovered at server startup and registered
                as MCP tools. May be empty when using ``@server.tool()``.
            deps: Deps factory. Accepts:
                - A plain DepsT instance (reused for every call)
                - A sync callable ``() -> DepsT``
                - An async callable ``() -> Awaitable[DepsT]``
                - With ``session_deps`` set: a callable ``(state: SessionDepsT) -> DepsT``
            session_deps: Optional Pydantic BaseModel class persisted to the
                session store across calls. All fields must have defaults.
            prompts: Optional list of prompt functions — also registerable via
                the ``@server.prompt`` decorator.
            name: Server name shown to MCP clients.
            bootstrap_deps: Deps used only during startup tool discovery.
                Defaults to None, safe for FunctionToolset.
            **fastmcp_kwargs: Forwarded to ``FastMCP.__init__()``. If you pass
                ``lifespan=``, it will be composed with the pydantic-ai
                discovery lifespan (pydantic-ai setup runs first).
        """
        if session_deps is not None:
            _validate_session_deps(session_deps)

        # Set before super().__init__() — the lifespan closure captures self
        # and reads these fields at server startup, not at construction time.
        self._pai_toolsets = list(toolsets)
        self._pai_deps = deps
        self._pai_session_deps = session_deps
        self._pai_prompts: list[Callable[..., Any]] = list(prompts or [])
        self._pai_bootstrap_deps = bootstrap_deps
        self._inline_toolset: FunctionToolset[Any] = FunctionToolset()

        # Extract any user-provided lifespan so we can compose it with ours.
        user_lifespan = fastmcp_kwargs.pop("lifespan", None)

        @asynccontextmanager
        async def _pai_lifespan(server: FastMCP):  # type: ignore[type-arg]
            assert isinstance(server, MCPServer)
            await server._discover_and_register()
            if user_lifespan is not None:
                async with user_lifespan(server):
                    yield
            else:
                yield

        super().__init__(name=name, lifespan=_pai_lifespan, **fastmcp_kwargs)

    # ── pydantic-ai decorator API ─────────────────────────────────────────────

    def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # type: ignore[override]
        """Register a pydantic-ai style tool (RunContext[Deps] as first arg).

        Overrides FastMCP's ``tool()`` with the pydantic-ai RunContext convention.
        To share a toolset with a pydantic-ai Agent, pass it via ``toolsets=[…]``
        at construction instead.

        Example::

            @server.tool()
            async def add(ctx: RunContext[Deps], a: float, b: float) -> float:
                return (a + b) * ctx.deps.multiplier
        """
        return self._inline_toolset.tool()

    def prompt(self, fn: Callable[..., Any]) -> Callable[..., Any]:  # type: ignore[override]
        """Register a pydantic-ai style prompt (RunContext[Deps] as first arg).

        Overrides FastMCP's ``prompt()`` with the pydantic-ai RunContext convention.

        Example::

            @server.prompt
            async def expert_prompt(ctx: RunContext[Deps], domain: str) -> str:
                return f"You are an expert {domain} engineer."
        """
        self._pai_prompts.append(fn)
        return fn

    # ── internal ──────────────────────────────────────────────────────────────

    async def _discover_and_register(self) -> None:
        """Discover pydantic-ai tools and register them with FastMCP.

        Called automatically during lifespan startup. Can also be awaited
        directly in tests to populate the server without starting it.
        """
        all_toolsets = [*self._pai_toolsets, self._inline_toolset]
        bootstrap_ctx = make_bootstrap_context(deps=self._pai_bootstrap_deps)

        for toolset in all_toolsets:
            async with toolset:
                discovered = await toolset.get_tools(bootstrap_ctx)
            for toolset_tool in discovered.values():
                self.add_tool(
                    PydanticAIToolAdapter.from_toolset_tool(
                        toolset,
                        toolset_tool,
                        self._pai_deps,
                        session_deps_cls=self._pai_session_deps,
                    )
                )

        for fn in self._pai_prompts:
            self.add_prompt(
                PydanticAIPromptAdapter.from_function(
                    fn,
                    self._pai_deps,
                    session_deps_cls=self._pai_session_deps,
                )
            )


def _validate_session_deps(cls: type[Any]) -> None:
    """Eagerly surface structural problems with a session_deps class.

    Checks:
    1. The class can be instantiated with no arguments (required for new sessions).
    2. The default instance can be round-tripped through model_dump(mode='json') +
       model_validate(), which is the exact path taken on every tool call.

    A PydanticSerializationError here means a non-excluded field holds a
    value that isn't JSON-serializable — mark it with Field(exclude=True).
    """
    try:
        instance = cls()
    except Exception as exc:
        raise TypeError(
            f"session_deps class {cls.__name__!r} could not be instantiated with no "
            f"arguments. All fields must have defaults. Original error: {exc}"
        ) from exc

    try:
        dumped = instance.model_dump(mode="json")
        cls.model_validate(dumped)
    except Exception as exc:
        raise TypeError(
            f"session_deps class {cls.__name__!r} failed serialization round-trip. "
            f"Non-serializable fields must be marked with Field(exclude=True). "
            f"Original error: {exc}"
        ) from exc
