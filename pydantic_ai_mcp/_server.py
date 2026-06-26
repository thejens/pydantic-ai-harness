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
    prompts: Sequence[Callable[..., Any]] | None = None,
    name: str = "pydantic-ai-mcp",
    bootstrap_deps: Any = None,
    **fastmcp_kwargs: Any,
) -> FastMCP:
    """Create a FastMCP server from pydantic-ai toolsets and prompt functions.

    Args:
        toolsets: Same list you'd pass to Agent(toolsets=[...]). Each toolset's
            tools are discovered at startup and registered as MCP tools.
        deps: Deps instance or factory used per call. Accepts:
            - A plain DepsT instance (reused for every call)
            - A sync callable () -> DepsT
            - An async callable () -> Awaitable[DepsT]
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
                PydanticAIToolAdapter.from_toolset_tool(toolset, toolset_tool, deps)
            )

    server = FastMCP(name=name, tools=tool_adapters, **fastmcp_kwargs)

    for fn in prompts or []:
        server.add_prompt(PydanticAIPromptAdapter.from_function(fn, deps))

    return server
