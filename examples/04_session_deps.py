"""
Session deps example — per-session dependency caching.

Some deps are expensive to build (auth round-trips, DB handshakes, user-profile
fetches). This example shows how to compute them once per MCP session and reuse
them across tool calls, using FastMCP's built-in session state.

The key: if your deps factory accepts one argument, it receives the FastMCP
Context. From there you can read/write session-scoped state with ctx.get_state()
and ctx.set_state().

Run with:
    uv run python examples/04_session_deps.py
"""
import asyncio
import time
from dataclasses import dataclass

from fastmcp.server.context import Context
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


# ── deps ──────────────────────────────────────────────────────────────────────

@dataclass
class Deps:
    user_id: str
    session_id: str
    call_count: int   # per-call state (not cached)


# ── simulated expensive auth ──────────────────────────────────────────────────

async def _fetch_user_id() -> str:
    """Pretend this hits an auth service. Takes 200 ms."""
    await asyncio.sleep(0.2)
    return f"user-{int(time.time())}"


# ── context-aware deps factory ────────────────────────────────────────────────

_call_counts: dict[str, int] = {}   # not worth serializing — resets on restart

async def make_deps(ctx: Context) -> Deps:
    """Called per tool invocation.

    user_id is fetched once per MCP session and stored in session state.
    Subsequent calls in the same session retrieve it instantly.
    """
    # Session-scoped: persists across tool calls within this MCP session
    user_id = await ctx.get_state("user_id")
    if user_id is None:
        print("  [auth] fetching user_id for new session…", flush=True)
        user_id = await _fetch_user_id()
        await ctx.set_state("user_id", user_id)   # serializable → survives reconnect
        print(f"  [auth] cached user_id={user_id}", flush=True)
    else:
        print(f"  [auth] reused cached user_id={user_id}", flush=True)

    session_id = ctx.session_id
    _call_counts[session_id] = _call_counts.get(session_id, 0) + 1

    return Deps(
        user_id=user_id,
        session_id=session_id,
        call_count=_call_counts[session_id],
    )


# ── toolset ───────────────────────────────────────────────────────────────────

toolset: FunctionToolset[Deps] = FunctionToolset(id="session-demo")


@toolset.tool()
def whoami(ctx: RunContext[Deps]) -> dict[str, str | int]:
    """Return identity and call statistics for the current session."""
    return {
        "user_id": ctx.deps.user_id,
        "session_id": ctx.deps.session_id,
        "calls_this_session": ctx.deps.call_count,
    }


@toolset.tool()
def greet(ctx: RunContext[Deps], name: str) -> str:
    """Greet someone, personalised to the authenticated user."""
    return f"Hi {name}! You are authenticated as {ctx.deps.user_id}."


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    server = await create_mcp_server(
        toolsets=[toolset],
        deps=make_deps,           # one positional arg → receives FastMCP Context
        name="session-demo",
        instructions=(
            "Demo server showing per-session dep caching. "
            "Call whoami() twice — the second call reuses the cached user_id."
        ),
    )
    await server.run_async(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
