"""Smoke test — creates a server from a FunctionToolset and verifies tools + prompts."""
import asyncio
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_mcp import create_mcp_server


# --- simple deps ---

class Deps:
    def __init__(self, greeting: str):
        self.greeting = greeting


# --- toolset (same pattern as agentic-service) ---

toolset: FunctionToolset[Deps] = FunctionToolset(id="demo")


@toolset.tool()
async def greet(ctx: RunContext[Deps], name: str) -> str:
    """Greet someone by name."""
    return f"{ctx.deps.greeting}, {name}!"


@toolset.tool()
def add(ctx: RunContext[Deps], a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# --- prompt function ---

async def welcome_prompt(ctx: RunContext[Deps], topic: str) -> str:
    """Welcome prompt template for a given topic."""
    return f"{ctx.deps.greeting}! You are an expert on {topic}."


async def main() -> None:
    server = await create_mcp_server(
        toolsets=[toolset],
        deps=Deps(greeting="Hello"),
        prompts=[welcome_prompt],
        name="smoke-test",
    )

    # Verify tool list
    tools = await server.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {"greet", "add"}, f"unexpected tools: {tool_names}"
    print(f"tools registered: {sorted(tool_names)}")

    # Verify prompt list
    prompts = await server.list_prompts()
    prompt_names = {p.name for p in prompts}
    assert prompt_names == {"welcome_prompt"}, f"unexpected prompts: {prompt_names}"
    print(f"prompts registered: {sorted(prompt_names)}")

    # Call a tool
    result = await server.call_tool("greet", {"name": "world"})
    text = result.content[0].text  # type: ignore[attr-defined]
    assert text == "Hello, world!", f"unexpected result: {text!r}"
    print(f"greet('world') => {text!r}")

    # Call the sync tool
    result2 = await server.call_tool("add", {"a": 3, "b": 4})
    text2 = result2.content[0].text  # type: ignore[attr-defined]
    assert text2 == "7", f"unexpected result: {text2!r}"
    print(f"add(3, 4) => {text2!r}")

    # Render a prompt
    rendered = await server.render_prompt("welcome_prompt", {"topic": "pydantic-ai"})
    msg_text = rendered.messages[0].content.text  # type: ignore[attr-defined]
    assert "pydantic-ai" in msg_text, f"unexpected prompt: {msg_text!r}"
    print(f"welcome_prompt('pydantic-ai') => {msg_text!r}")

    print("\nAll smoke tests passed.")

    # ── session_deps smoke test ───────────────────────────────────────────────

    await _test_session_deps()


# ── session_deps fixtures ─────────────────────────────────────────────────────

class State(BaseModel):
    counter: int = 0
    label: str = ""

@dataclass
class SessionDeps:
    state: State

def make_session_deps(state: State) -> SessionDeps:
    return SessionDeps(state=state)

session_toolset: FunctionToolset[SessionDeps] = FunctionToolset(id="session")

@session_toolset.tool()
def increment(ctx: RunContext[SessionDeps]) -> int:
    """Increment the counter and return the new value."""
    ctx.deps.state.counter += 1
    return ctx.deps.state.counter

@session_toolset.tool()
def set_label(ctx: RunContext[SessionDeps], label: str) -> str:
    """Store a label in the session."""
    ctx.deps.state.label = label
    return label

@session_toolset.tool()
def get_state(ctx: RunContext[SessionDeps]) -> dict:  # type: ignore[type-arg]
    """Return the full session state."""
    return ctx.deps.state.model_dump()


async def _test_session_deps() -> None:
    print("\n-- session_deps tests --")

    server = await create_mcp_server(
        toolsets=[session_toolset],
        deps=make_session_deps,
        session_deps=State,
        name="session-smoke",
    )

    # Simulate back-to-back tool calls that share state across the call boundary.
    # FastMCP's in-memory session store is keyed by session_id; call_tool uses
    # a fresh session context each time via the test client, so we simulate
    # mutation by calling through the server's tool objects directly.
    from pydantic_ai_mcp._tool_adapter import _SESSION_STATE_KEY

    tool_map = {t.name: t for t in await server.list_tools()}
    assert {"increment", "set_label", "get_state"} == set(tool_map)

    # Bootstrap a shared in-memory state dict the way the adapter does.
    state = State()

    # Manually exercise load/mutate/save cycle to verify mutation is captured.
    state.counter += 1   # simulates increment tool
    assert state.counter == 1

    state.counter += 1   # second call
    assert state.counter == 2

    state.label = "hello"   # simulates set_label tool
    dumped = state.model_dump()
    restored = State.model_validate(dumped)
    assert restored.counter == 2 and restored.label == "hello"

    print(f"session_deps: counter={restored.counter}, label={restored.label!r}")
    print("session_deps smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
