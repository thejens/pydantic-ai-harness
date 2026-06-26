"""
Deps factory example — per-call dependency injection.

Shows the real-world pattern: deps are created fresh for every MCP call,
reading from environment variables or a config file. This is how you'd
connect your toolset to a live database, API client, or secret manager.

Run with:
    DB_URL=sqlite:///example.db uv run python examples/02_deps_factory.py
"""
import os
from dataclasses import dataclass, field
from datetime import datetime

from pydantic_ai import RunContext

from pydantic_ai_mcp import MCPServer


@dataclass
class AppDeps:
    db_url: str
    request_id: str = field(default_factory=lambda: datetime.now().isoformat())


def make_deps() -> AppDeps:
    """Called once per tool/prompt invocation — read config from environment."""
    return AppDeps(
        db_url=os.environ.get("DB_URL", "sqlite:///default.db"),
    )


# ── server ────────────────────────────────────────────────────────────────────

server = MCPServer(
    deps=make_deps,               # factory — called per invocation
    name="database-assistant",
    instructions="Tools and prompts for working with the connected database.",
)


@server.tool()
async def get_connection_info(ctx: RunContext[AppDeps]) -> dict[str, str]:
    """Return information about the active database connection."""
    return {
        "url": ctx.deps.db_url,
        "request_id": ctx.deps.request_id,
    }


@server.tool()
async def query_rows(ctx: RunContext[AppDeps], table: str, limit: int = 10) -> str:
    """Return a summary of rows in the given table (simulated)."""
    return f"Would query SELECT * FROM {table} LIMIT {limit} on {ctx.deps.db_url}"


@server.prompt
async def sql_expert_prompt(ctx: RunContext[AppDeps], dialect: str = "SQLite") -> str:
    """System prompt that makes the model an expert on this database."""
    return (
        f"You are an expert {dialect} database engineer. "
        f"The active database is: {ctx.deps.db_url}. "
        "Help the user write efficient, correct SQL queries."
    )


if __name__ == "__main__":
    server.run(transport="stdio")
