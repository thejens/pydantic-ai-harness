"""
Reuse across Agent and MCP — the core motivation for this package.

The same toolset is registered with both a pydantic-ai Agent (for programmatic use)
and an MCP server (so Claude Code and other MCP clients can call the tools directly).
No duplication. One definition, two surfaces.

Run as MCP server:
    uv run python examples/03_reuse_across_agent_and_mcp.py --mcp

Run as agent (prints result and exits):
    uv run python examples/03_reuse_across_agent_and_mcp.py --agent
"""
import asyncio
import sys
from dataclasses import dataclass

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


# ── shared deps ──────────────────────────────────────────────────────────────

@dataclass
class Deps:
    http: httpx.AsyncClient


def make_deps() -> Deps:
    return Deps(http=httpx.AsyncClient(timeout=10))


# ── shared toolset ───────────────────────────────────────────────────────────

web_toolset: FunctionToolset[Deps] = FunctionToolset(id="web")


@web_toolset.tool()
async def fetch_url(ctx: RunContext[Deps], url: str) -> str:
    """Fetch the text content of a URL."""
    response = await ctx.deps.http.get(url)
    response.raise_for_status()
    return response.text[:2000]   # truncate for readability


@web_toolset.tool()
async def check_status(ctx: RunContext[Deps], url: str) -> int:
    """Return the HTTP status code for a URL."""
    response = await ctx.deps.http.head(url)
    return response.status_code


# ── prompt ───────────────────────────────────────────────────────────────────

async def web_researcher_prompt(ctx: RunContext[Deps], topic: str) -> str:
    """Research assistant prompt for a given topic."""
    return (
        f"You are a meticulous web researcher. "
        f"Your goal is to gather accurate, up-to-date information about: {topic}. "
        "Use fetch_url to read sources and check_status to verify links are live."
    )


# ── agent surface ─────────────────────────────────────────────────────────────

async def run_as_agent() -> None:
    agent: Agent[Deps, str] = Agent(
        model="openai:gpt-4o-mini",
        toolsets=[web_toolset],
        output_type=str,
    )
    deps = make_deps()
    async with deps.http:
        result = await agent.run(
            "What is the HTTP status of https://example.com?",
            deps=deps,
        )
    print(result.output)


# ── MCP surface ───────────────────────────────────────────────────────────────

async def run_as_mcp() -> None:
    server = await create_mcp_server(
        toolsets=[web_toolset],
        deps=make_deps,
        prompts=[web_researcher_prompt],
        name="web-tools",
        instructions="HTTP tools for fetching and inspecting web pages.",
    )
    await server.run_async(transport="stdio")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--mcp"
    if mode == "--agent":
        asyncio.run(run_as_agent())
    else:
        asyncio.run(run_as_mcp())
