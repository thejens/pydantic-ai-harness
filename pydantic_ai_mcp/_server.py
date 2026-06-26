from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fastmcp import FastMCP
from pydantic_ai.toolsets.abstract import AbstractToolset

from ._context import make_bootstrap_context
from ._prompt_adapter import PydanticAIPromptAdapter
from ._tool_adapter import PydanticAIToolAdapter


async def create_mcp_server(
    toolsets: Sequence[AbstractToolset[Any]],
    deps: Any,
    *,
    session_deps: type[Any] | None = None,
    prompts: Sequence[Callable[..., Any]] | None = None,
    name: str = "pydantic-ai-mcp",
    bootstrap_deps: Any = None,
    **fastmcp_kwargs: Any,
) -> FastMCP:
    """Create a FastMCP server from pydantic-ai toolsets and prompt functions.

    Args:
        toolsets: Same list you'd pass to Agent(toolsets=[...]). Each toolset's
            tools are discovered at startup and registered as MCP tools.
        deps: Deps factory. Accepts:
            - A plain DepsT instance (reused for every call)
            - A sync callable () -> DepsT
            - An async callable () -> Awaitable[DepsT]
            - With session_deps set: a callable (state: SessionDepsT) -> DepsT
              that receives the pre-loaded session state before each call.
        session_deps: Optional Pydantic BaseModel class whose instance is
            persisted to FastMCP's session store across calls. Before each
            tool or prompt invocation the stored state is deserialised and
            passed as the first argument to ``deps``. After the call the
            (possibly mutated) instance is serialised back — so tools can
            read and write session state simply by mutating ctx.deps fields,
            with no direct coupling to FastMCP's APIs.

            All fields must have defaults (the class is constructed with no
            arguments when no prior state exists for a session).

            The backing store is FastMCP's in-memory store by default. Pass
            ``session_state_store=RedisStore(url=...)`` in fastmcp_kwargs to
            use Redis (or any other AsyncKeyValue backend) for persistence
            across restarts and distributed deployments.
        prompts: Optional list of prompt functions with signature
            (ctx: RunContext[DepsT], **kwargs) -> str | list[Message] | PromptResult.
            The first RunContext parameter is stripped; remaining params become
            MCP PromptArguments.
        name: Server name shown to MCP clients.
        bootstrap_deps: Deps used only during startup get_tools() discovery.
            Defaults to None, which is safe for FunctionToolset (it never reads ctx.deps).
        **fastmcp_kwargs: Forwarded verbatim to FastMCP(name=name, ...).

    Returns:
        A configured FastMCP server, ready to run.
    """
    bootstrap_ctx = make_bootstrap_context(deps=bootstrap_deps)

    tool_adapters: list[PydanticAIToolAdapter] = []
    for toolset in toolsets:
        async with toolset:
            discovered = await toolset.get_tools(bootstrap_ctx)
        for toolset_tool in discovered.values():
            tool_adapters.append(
                PydanticAIToolAdapter.from_toolset_tool(
                    toolset,
                    toolset_tool,
                    deps,
                    session_deps_cls=session_deps,
                )
            )

    server = FastMCP(name=name, tools=tool_adapters, **fastmcp_kwargs)

    for fn in prompts or []:
        server.add_prompt(
            PydanticAIPromptAdapter.from_function(
                fn,
                deps,
                session_deps_cls=session_deps,
            )
        )

    return server
