"""
Session deps — transparent state persistence across MCP calls.

The ``session_deps`` parameter tells pydantic-ai-mcp to load a Pydantic model
from the session store before each call and save it back after. Tools mutate
``ctx.deps`` normally — no awareness of FastMCP or sessions required.

This mirrors how pydantic-ai handles mutable deps within a single agent run:
the same deps instance is shared across all tool calls, so any field the first
tool sets is visible to the next. ``session_deps`` extends that contract across
MCP round-trips, making the session the unit of continuity instead of the run.

The same toolset is drop-in compatible with a regular pydantic-ai Agent — just
pass a ``SessionState()`` as deps and the tools work identically.

Run with:
    uv run python examples/04_session_deps.py
Then connect any MCP client to the stdio process.
"""
import asyncio
import time
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


# ── session state (the persistent, serializable part) ─────────────────────────

class SessionState(BaseModel):
    """Persisted across every tool call in this MCP session.

    All fields must have defaults — the model is constructed with no arguments
    when a session is brand new.
    """
    user_id: str | None = None     # cached after first auth round-trip
    notes: dict[str, str] = {}


# ── deps (may include non-serializable resources) ─────────────────────────────

@dataclass
class Deps:
    state: SessionState            # reference into the session store


# ── deps factory — receives the pre-loaded session state ──────────────────────

async def make_deps(state: SessionState) -> Deps:
    """Called before every tool invocation.

    ``state`` is already populated from the session store. Mutate it here
    to cache expensive-to-compute values — the mutations are saved after
    the tool returns.
    """
    if state.user_id is None:
        # Simulates an auth round-trip. Runs once per session, never again.
        print("  [auth] first call — authenticating…", flush=True)
        await asyncio.sleep(0.1)
        state.user_id = f"user-{int(time.time())}"
        print(f"  [auth] user_id={state.user_id} (will be cached)", flush=True)
    else:
        print(f"  [auth] reused cached user_id={state.user_id}", flush=True)
    return Deps(state=state)


# ── toolset ───────────────────────────────────────────────────────────────────

toolset: FunctionToolset[Deps] = FunctionToolset(id="session-demo")


@toolset.tool()
def whoami(ctx: RunContext[Deps]) -> dict[str, str | None]:
    """Return the authenticated user for this session."""
    return {"user_id": ctx.deps.state.user_id}


@toolset.tool()
def remember(ctx: RunContext[Deps], key: str, value: str) -> str:
    """Store a note for this session."""
    ctx.deps.state.notes[key] = value    # just mutate — auto-saved
    return f"Stored {key!r} = {value!r}"


@toolset.tool()
def recall(ctx: RunContext[Deps], key: str) -> str:
    """Retrieve a note stored earlier in this session."""
    value = ctx.deps.state.notes.get(key)
    return value if value is not None else f"(nothing stored for {key!r})"


@toolset.tool()
def forget(ctx: RunContext[Deps], key: str) -> str:
    """Delete a note from the session."""
    ctx.deps.state.notes.pop(key, None)  # just mutate — auto-saved
    return f"Deleted {key!r}"


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    server = await create_mcp_server(
        toolsets=[toolset],
        deps=make_deps,
        session_deps=SessionState,   # enables load-before / save-after
        name="session-demo",
        instructions=(
            "Session state demo. Try: remember('x', '1') → recall('x') → forget('x') → recall('x')."
        ),
    )
    await server.run_async(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
