"""
Session deps — one Deps model, serializable and ephemeral fields mixed.

A Pydantic model can hold both kinds of fields:

  - Serializable fields (str, int, dict, …) — persisted to the session store
    across every MCP call, shared across replicas.

  - Ephemeral fields — excluded from serialization with ``Field(exclude=True)``.
    Restored to their defaults before each call; the factory fills them back in.

The library loads the serializable fields from the store, constructs a ``Deps``
instance, calls the factory with it (so the factory can fill in ephemeral
resources using the already-restored session data), then saves the serializable
fields back after the tool returns.

Tool functions just mutate ``ctx.deps`` directly — no awareness of which fields
are persistent vs. ephemeral is needed.

This mirrors how pydantic-ai shares a mutable deps instance across all tool calls
within a single agent run. ``session_deps`` extends that contract across MCP
round-trips, with the session as the unit of continuity.

Run:
    uv run python examples/04_session_deps.py
Connect any MCP client to the stdio process.
"""
import asyncio
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext

from pydantic_ai_mcp import MCPServer


# ── deps — one model, two kinds of fields ─────────────────────────────────────

class Deps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── serializable: persisted to the session store ──
    user_id: str | None = None
    notes: dict[str, str] = Field(default_factory=dict)

    # ── ephemeral: rebuilt each call, not persisted ───
    # Use Field(exclude=True) so model_dump() omits these and model_validate()
    # restores them to their default (None here). The factory fills them in.
    expensive_resource: Any | None = Field(default=None, exclude=True)


# ── factory — receives the pre-loaded Deps, fills in ephemeral fields ─────────

async def make_deps(deps: Deps) -> Deps:
    """Called before every tool invocation.

    ``deps.user_id`` and ``deps.notes`` are already loaded from the session
    store. Authenticate once, then fill in non-serializable resources every call.
    """
    if deps.user_id is None:
        # Simulates an auth round-trip. Runs once per session.
        print("  [auth] first call — authenticating…", flush=True)
        await asyncio.sleep(0.1)
        deps.user_id = f"user-{int(time.time())}"    # mutation → will be persisted
        print(f"  [auth] user_id={deps.user_id}", flush=True)
    else:
        print(f"  [auth] reused user_id={deps.user_id}", flush=True)

    # Ephemeral resource built fresh each call using the now-available user_id.
    deps.expensive_resource = f"<resource for {deps.user_id}>"
    return deps


# ── server ────────────────────────────────────────────────────────────────────

server = MCPServer(
    deps=make_deps,
    session_deps=Deps,    # Deps IS the session model — same class, one definition
    name="session-demo",
    instructions=(
        "Session state demo. Try: remember('x', '1') → recall('x') → "
        "restart server → recall('x')  (state survives until process exits)."
    ),
)


@server.tool()
def whoami(ctx: RunContext[Deps]) -> dict[str, Any]:
    """Return the current session identity and ephemeral resource handle."""
    return {
        "user_id": ctx.deps.user_id,
        "resource": ctx.deps.expensive_resource,
    }


@server.tool()
def remember(ctx: RunContext[Deps], key: str, value: str) -> str:
    """Store a note that persists for the lifetime of this MCP session."""
    ctx.deps.notes[key] = value    # mutate — auto-persisted after this returns
    return f"Stored {key!r} = {value!r}"


@server.tool()
def recall(ctx: RunContext[Deps], key: str) -> str:
    """Retrieve a note stored earlier in this session."""
    value = ctx.deps.notes.get(key)
    return value if value is not None else f"(nothing stored for {key!r})"


@server.tool()
def forget(ctx: RunContext[Deps], key: str) -> str:
    """Delete a note from the session."""
    ctx.deps.notes.pop(key, None)  # mutate — auto-persisted
    return f"Deleted {key!r}"


if __name__ == "__main__":
    server.run(transport="stdio")
