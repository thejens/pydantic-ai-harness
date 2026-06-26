"""
Redis session store — distributed, persistent sessions.

By default FastMCP keeps session state in memory: it's fast, but sessions
vanish on restart and can't be shared across processes. Swapping to a Redis
backend takes one line and makes sessions:

  - Persistent  — survive server restarts
  - Distributed — shared across multiple server replicas
  - Configurable TTL — expired automatically by Redis

This example runs an HTTP server (streamable-http transport) because that's
the transport used in production deployments. Each HTTP request carries an
"mcp-session-id" header that FastMCP uses as the stable session key — the
same session_id is sent back by Redis regardless of which replica handles
the request.

Prerequisites:
    docker run -p 6379:6379 redis:latest

Run:
    REDIS_URL=redis://localhost:6379/0 uv run python examples/05_redis_session_store.py

Connect any MCP client to http://localhost:8000/mcp (streamable-http).
"""
import asyncio
import os
import time
from dataclasses import dataclass

from fastmcp.server.context import Context
from fastmcp.server.dependencies import get_context
from key_value.aio.stores.redis import RedisStore
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


# ── deps ──────────────────────────────────────────────────────────────────────

@dataclass
class Deps:
    user_id: str      # resolved once per session (from Redis cache)
    session_id: str


# ── simulated auth service ────────────────────────────────────────────────────

async def _authenticate() -> str:
    """Simulate an auth round-trip that takes ~200 ms."""
    await asyncio.sleep(0.2)
    return f"user-{int(time.time())}"


# ── context-aware factory — reads/writes to Redis via FastMCP ─────────────────

async def make_deps(ctx: Context) -> Deps:
    """Resolve deps for each tool call.

    user_id is fetched from the auth service once per MCP session, then cached
    in Redis. Every replica that receives a subsequent request from the same
    session skips the auth call and reads from Redis directly.
    """
    user_id: str | None = await ctx.get_state("user_id")

    if user_id is None:
        print(f"  [session {ctx.session_id[:8]}] auth: first call — fetching user_id")
        user_id = await _authenticate()
        await ctx.set_state("user_id", user_id)
        print(f"  [session {ctx.session_id[:8]}] auth: cached user_id={user_id} in Redis")
    else:
        print(f"  [session {ctx.session_id[:8]}] auth: reused user_id={user_id} from Redis")

    return Deps(user_id=user_id, session_id=ctx.session_id)


# ── toolset ───────────────────────────────────────────────────────────────────

toolset: FunctionToolset[Deps] = FunctionToolset(id="demo")


@toolset.tool()
def whoami(ctx: RunContext[Deps]) -> dict[str, str]:
    """Return current session identity."""
    return {
        "user_id": ctx.deps.user_id,
        "session_id": ctx.deps.session_id,
    }


@toolset.tool()
async def store_note(ctx: RunContext[Deps], key: str, value: str) -> str:
    """Persist a note for this session (survives server restart)."""
    fmcp_ctx = get_context()   # FastMCP context available anywhere during a request
    await fmcp_ctx.set_state(f"note:{key}", value)
    return f"Stored note[{key!r}] for session {ctx.deps.session_id[:8]}"


@toolset.tool()
async def read_note(ctx: RunContext[Deps], key: str) -> str | None:
    """Read a previously stored note."""
    fmcp_ctx = Context.get()
    return await fmcp_ctx.get_state(f"note:{key}")


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # RedisStore implements AsyncKeyValue — drop it straight into FastMCP.
    # All session state (user_id, notes, anything set via ctx.set_state) is
    # stored here, keyed by session_id, shared across all replicas.
    session_store = RedisStore(url=redis_url)

    server = await create_mcp_server(
        toolsets=[toolset],
        deps=make_deps,
        name="redis-session-demo",
        instructions="Demonstrates Redis-backed session state shared across replicas.",
        # FastMCP forwards this to _state_store — one line to go distributed.
        session_state_store=session_store,
    )

    print(f"Starting MCP server with Redis session store at {redis_url}")
    print("Connect via: http://localhost:8000/mcp (streamable-http)")
    await server.run_async(transport="streamable-http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
