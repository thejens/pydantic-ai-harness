"""
Redis session store — distributed, persistent sessions.

Drop ``session_state_store=RedisStore(url=...)`` into the same pattern from
example 04: the session state is now stored in Redis instead of memory, making
it persistent across server restarts and shared across replicas.

Tools don't change at all — they still just mutate ctx.deps.state fields.

Prerequisites:
    docker compose -f examples/docker-compose.yml up -d

Run:
    REDIS_URL=redis://localhost:6379/0 uv run python examples/05_redis_session_store.py

Connect any MCP client to http://localhost:8000/mcp (streamable-http transport).
To see persistence in action:
  1. Call remember('color', 'blue')
  2. Restart the server
  3. Call recall('color') — still returns 'blue'
"""
import asyncio
import os
import time
from dataclasses import dataclass

from key_value.aio.stores.redis import RedisStore
from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


# ── session state ─────────────────────────────────────────────────────────────

class SessionState(BaseModel):
    user_id: str | None = None
    notes: dict[str, str] = {}


# ── deps ──────────────────────────────────────────────────────────────────────

@dataclass
class Deps:
    state: SessionState


# ── factory ───────────────────────────────────────────────────────────────────

async def make_deps(state: SessionState) -> Deps:
    if state.user_id is None:
        print("  [auth] first call — authenticating…")
        await asyncio.sleep(0.1)
        state.user_id = f"user-{int(time.time())}"
        print(f"  [auth] user_id={state.user_id} cached in Redis")
    else:
        print(f"  [auth] reused user_id={state.user_id} from Redis")
    return Deps(state=state)


# ── toolset ───────────────────────────────────────────────────────────────────

toolset: FunctionToolset[Deps] = FunctionToolset(id="redis-demo")


@toolset.tool()
def whoami(ctx: RunContext[Deps]) -> dict[str, str | None]:
    """Return the authenticated user for this session."""
    return {"user_id": ctx.deps.state.user_id}


@toolset.tool()
def remember(ctx: RunContext[Deps], key: str, value: str) -> str:
    """Store a note that survives server restarts."""
    ctx.deps.state.notes[key] = value
    return f"Stored {key!r} = {value!r}"


@toolset.tool()
def recall(ctx: RunContext[Deps], key: str) -> str:
    """Retrieve a previously stored note."""
    value = ctx.deps.state.notes.get(key)
    return value if value is not None else f"(nothing stored for {key!r})"


@toolset.tool()
def forget(ctx: RunContext[Deps], key: str) -> str:
    """Delete a stored note."""
    ctx.deps.state.notes.pop(key, None)
    return f"Deleted {key!r}"


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    server = await create_mcp_server(
        toolsets=[toolset],
        deps=make_deps,
        session_deps=SessionState,
        name="redis-session-demo",
        instructions="Redis-backed sessions. State persists across server restarts.",
        # One parameter swaps the backing store from in-memory to Redis.
        # The session_deps load/save logic in the adapter is unchanged.
        session_state_store=RedisStore(url=redis_url),
    )

    print(f"MCP server running  —  Redis at {redis_url}")
    print("Connect via: http://localhost:8000/mcp")
    await server.run_async(transport="streamable-http", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
