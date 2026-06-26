"""
Simple tools example — the minimal case.

Shows how to expose a FunctionToolset as an MCP server with a single deps object.
Run with:
    uv run python examples/01_simple_tools.py
Then connect any MCP client (e.g. Claude Code) to the stdio process.
"""
import asyncio
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


@dataclass
class Deps:
    user_name: str
    multiplier: float = 1.0


# Build a toolset exactly as you would for Agent(toolsets=[math_toolset])
math_toolset: FunctionToolset[Deps] = FunctionToolset(id="math")


@math_toolset.tool()
def add(ctx: RunContext[Deps], a: float, b: float) -> float:
    """Add two numbers."""
    return (a + b) * ctx.deps.multiplier


@math_toolset.tool()
def greet(ctx: RunContext[Deps]) -> str:
    """Return a greeting for the current user."""
    return f"Hello, {ctx.deps.user_name}!"


async def main() -> None:
    deps = Deps(user_name="Alice", multiplier=2.0)

    server = await create_mcp_server(
        toolsets=[math_toolset],
        deps=deps,
        name="simple-math",
    )
    await server.run_async(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
