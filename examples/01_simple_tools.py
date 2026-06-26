"""
Simple tools example — the minimal case.

Shows how to expose a FunctionToolset as an MCP server with a single deps object.
Run with:
    uv run python examples/01_simple_tools.py
Then connect any MCP client (e.g. Claude Code) to the stdio process.
"""
from dataclasses import dataclass

from pydantic_ai import RunContext

from pydantic_ai_mcp import MCPServer


@dataclass
class Deps:
    user_name: str
    multiplier: float = 1.0


server = MCPServer(deps=Deps(user_name="Alice", multiplier=2.0), name="simple-math")


@server.tool()
def add(ctx: RunContext[Deps], a: float, b: float) -> float:
    """Add two numbers."""
    return (a + b) * ctx.deps.multiplier


@server.tool()
def greet(ctx: RunContext[Deps]) -> str:
    """Return a greeting for the current user."""
    return f"Hello, {ctx.deps.user_name}!"


if __name__ == "__main__":
    server.run(transport="stdio")
