from __future__ import annotations

from typing import Any

from fastmcp.tools.base import Tool, ToolResult
from pydantic import PrivateAttr
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from ._context import make_call_context

# Key under which session state is stored in FastMCP's per-session store.
_SESSION_STATE_KEY = "__deps__"


class PydanticAIToolAdapter(Tool):
    """FastMCP Tool that delegates to a pydantic-ai toolset at call time.

    Wraps a single ToolsetTool discovered via AbstractToolset.get_tools().
    On each MCP call: validates args, builds a fresh RunContext from deps_factory,
    then delegates to toolset.call_tool().

    When session_deps_cls is set, the serialized session state is loaded from
    FastMCP's session store before the call and written back after — capturing
    any mutations the tool made to the state object via ctx.deps.
    """

    _toolset: AbstractToolset[Any] = PrivateAttr()
    _toolset_tool: ToolsetTool[Any] = PrivateAttr()
    _deps_factory: Any = PrivateAttr()
    _max_retries: int = PrivateAttr(default=0)
    _session_deps_cls: type[Any] | None = PrivateAttr(default=None)

    @classmethod
    def from_toolset_tool(
        cls,
        toolset: AbstractToolset[Any],
        toolset_tool: ToolsetTool[Any],
        deps_factory: Any,
        max_retries: int = 0,
        session_deps_cls: type[Any] | None = None,
    ) -> PydanticAIToolAdapter:
        tool_def = toolset_tool.tool_def
        instance = cls(
            name=tool_def.name,
            description=tool_def.description,
            # parameters_json_schema is already a valid MCP inputSchema dict
            parameters=tool_def.parameters_json_schema,
        )
        instance._toolset = toolset
        instance._toolset_tool = toolset_tool
        instance._deps_factory = deps_factory
        instance._max_retries = max_retries
        instance._session_deps_cls = session_deps_cls
        return instance

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        session_state, fmcp_ctx = await _load_session_state(self._session_deps_cls)

        # Validate and coerce incoming MCP args to Python types (str→datetime, dict→Model, etc.)
        validated = self._toolset_tool.args_validator.validate_python(arguments)
        ctx = await make_call_context(
            self._deps_factory,
            session_state=session_state,
            tool_name=self.name,
            max_retries=self._max_retries,
        )
        result = await self._toolset.call_tool(
            self.name, validated, ctx, self._toolset_tool
        )

        # Serialize back: captures any mutations the tool made to session_state.
        # This works because session_state was passed by reference into the factory,
        # which stores it in Deps — the same instance the tool function mutated.
        await _save_session_state(session_state, fmcp_ctx)

        return self.convert_result(result)


async def _load_session_state(
    session_deps_cls: type[Any] | None,
) -> tuple[Any, Any]:
    """Load and deserialize session state from FastMCP's store.

    Returns (session_state, fmcp_ctx). Both are None when session_deps_cls is not set.
    """
    if session_deps_cls is None:
        return None, None

    from fastmcp.server.dependencies import get_context
    fmcp_ctx = get_context()
    raw = await fmcp_ctx.get_state(_SESSION_STATE_KEY)
    if raw is not None:
        state = session_deps_cls.model_validate(raw)
    else:
        state = session_deps_cls()
    return state, fmcp_ctx


async def _save_session_state(session_state: Any, fmcp_ctx: Any) -> None:
    if session_state is None or fmcp_ctx is None:
        return
    await fmcp_ctx.set_state(_SESSION_STATE_KEY, session_state.model_dump())
