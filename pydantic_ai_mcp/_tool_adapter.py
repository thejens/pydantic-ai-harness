from __future__ import annotations

from typing import Any

from fastmcp.tools.base import Tool, ToolResult
from pydantic import PrivateAttr
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

from ._context import make_call_context


class PydanticAIToolAdapter(Tool):
    """FastMCP Tool that delegates to a pydantic-ai toolset at call time.

    Wraps a single ToolsetTool discovered via AbstractToolset.get_tools().
    On each MCP call: validates args, builds a fresh RunContext from deps_factory,
    then delegates to toolset.call_tool().
    """

    _toolset: AbstractToolset[Any] = PrivateAttr()
    _toolset_tool: ToolsetTool[Any] = PrivateAttr()
    _deps_factory: Any = PrivateAttr()
    _max_retries: int = PrivateAttr(default=0)

    @classmethod
    def from_toolset_tool(
        cls,
        toolset: AbstractToolset[Any],
        toolset_tool: ToolsetTool[Any],
        deps_factory: Any,
        max_retries: int = 0,
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
        return instance

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        # Validate and coerce incoming MCP args to Python types (str→datetime, dict→Model, etc.)
        validated = self._toolset_tool.args_validator.validate_python(arguments)
        ctx = await make_call_context(
            self._deps_factory,
            tool_name=self.name,
            max_retries=self._max_retries,
        )
        result = await self._toolset.call_tool(
            self.name, validated, ctx, self._toolset_tool
        )
        return self.convert_result(result)
