"""
Redis session store — identical pattern to example 04, distributed backing.

Swapping from in-memory to Redis requires exactly one extra parameter:
``session_state_store=RedisStore(url=...)``. Tool code is unchanged.

What Redis adds:
  - Persistence: session state survives server restarts
  - Distribution: multiple replicas share the same session state
  - TTL: Redis expires stale sessions automatically

The serializable/ephemeral split from example 04 still applies:
  - Serializable fields (no ``exclude``) → stored in Redis, shared across replicas
  - Ephemeral fields (``Field(exclude=True)``) → rebuilt by the factory each call

Prerequisites:
    docker compose -f examples/docker-compose.yml up -d

Run:
    REDIS_URL=redis://localhost:6379/0 uv run python examples/05_redis_session_store.py

Connect any MCP client to http://localhost:8000/mcp (streamable-http transport).
Demo:
  1. Call remember('color', 'blue')
  2. Restart the server (Ctrl-C, re-run)
  3. Call recall('color') — still returns 'blue'
"""
import asyncio
import os
import time
from typing import Any

from key_value.aio.stores.redis import RedisStore
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

from pydantic_ai_mcp import MCPServer


# ── deps ──────────────────────────────────────────────────────────────────────

class Deps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable — stored in Redis, survives restarts, shared across replicas
    user_id: str | None = None
    notes: dict[str, str] = Field(default_factory=dict)

    # Ephemeral — rebuilt each call from the restored session data, never stored
    http_headers: dict[str, str] | None = Field(default=None, exclude=True)


# ── factory ───────────────────────────────────────────────────────────────────

async def make_deps(deps: Deps) -> Deps:
    if deps.user_id is None:
        print("  [auth] first call — authenticating…")
        await asyncio.sleep(0.1)
        deps.user_id = f"user-{int(time.time())}"
        print(f"  [auth] user_id={deps.user_id} cached in Redis")
    else:
        print(f"  [auth] reused user_id={deps.user_id} from Redis")

    # Build ephemeral resource from the now-available session data
    deps.http_headers = {"X-User-Id": deps.user_id}
    return deps


# ── server ────────────────────────────────────────────────────────────────────

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

server = MCPServer(
    deps=make_deps,
    session_deps=Deps,
    name="redis-session-demo",
    instructions="Redis-backed session demo. State persists across server restarts.",
    session_state_store=RedisStore(url=redis_url),   # only difference from example 04
)


@server.tool()
def whoami(ctx: RunContext[Deps]) -> dict[str, Any]:
    """Return the current user and the per-call ephemeral resource."""
    return {
        "user_id": ctx.deps.user_id,
        "http_headers": ctx.deps.http_headers,
    }


@server.tool()
def remember(ctx: RunContext[Deps], key: str, value: str) -> str:
    """Store a note that survives server restarts."""
    ctx.deps.notes[key] = value
    return f"Stored {key!r} = {value!r}"


@server.tool()
def recall(ctx: RunContext[Deps], key: str) -> str:
    """Retrieve a previously stored note."""
    value = ctx.deps.notes.get(key)
    return value if value is not None else f"(nothing stored for {key!r})"


@server.tool()
def forget(ctx: RunContext[Deps], key: str) -> str:
    """Delete a stored note."""
    ctx.deps.notes.pop(key, None)
    return f"Deleted {key!r}"


if __name__ == "__main__":
    print(f"MCP server running  —  Redis at {redis_url}")
    print("Connect via: http://localhost:8000/mcp")
    server.run(transport="streamable-http", host="0.0.0.0", port=8000)
