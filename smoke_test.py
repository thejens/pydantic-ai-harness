"""Smoke test — creates a server from a FunctionToolset and verifies tools + prompts."""
import asyncio

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


if __name__ == "__main__":
    asyncio.run(main())
