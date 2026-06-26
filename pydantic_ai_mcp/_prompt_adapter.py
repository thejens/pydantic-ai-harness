from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from fastmcp.prompts.base import Prompt, PromptArgument
from pydantic import PrivateAttr

from ._context import make_call_context
from ._tool_adapter import _load_session_state, _save_session_state


class PydanticAIPromptAdapter(Prompt):
    """FastMCP Prompt backed by a pydantic-ai-style function.

    The wrapped function must have signature:
        async def my_prompt(ctx: RunContext[DepsT], arg1: str, arg2: int = 0) -> str: ...

    The first parameter (RunContext) is detected by position and stripped from the
    MCP argument list. Remaining parameters become MCP PromptArguments.

    The function can return str, list[Message | str], or PromptResult — the inherited
    convert_result() handles all three.
    """

    _fn: Callable[..., Any] = PrivateAttr()
    _deps_factory: Any = PrivateAttr()
    _max_retries: int = PrivateAttr(default=0)
    _session_deps_cls: type[Any] | None = PrivateAttr(default=None)

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        deps_factory: Any,
        *,
        max_retries: int = 0,
        name: str | None = None,
        description: str | None = None,
        session_deps_cls: type[Any] | None = None,
    ) -> PydanticAIPromptAdapter:
        sig = inspect.signature(fn)
        all_params = list(sig.parameters.values())

        # First param is RunContext — skip it; everything else becomes an MCP argument.
        prompt_params = all_params[1:]

        arguments = [
            PromptArgument(
                name=p.name,
                required=p.default is inspect.Parameter.empty,
            )
            for p in prompt_params
        ]

        instance = cls(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn),
            arguments=arguments,
        )
        instance._fn = fn
        instance._deps_factory = deps_factory
        instance._max_retries = max_retries
        instance._session_deps_cls = session_deps_cls
        return instance

    async def render(self, arguments: dict[str, Any] | None = None) -> Any:
        session_state, fmcp_ctx = await _load_session_state(self._session_deps_cls)

        ctx = await make_call_context(
            self._deps_factory,
            session_state=session_state,
            max_retries=self._max_retries,
        )
        result = self._fn(ctx, **(arguments or {}))
        if inspect.isawaitable(result):
            result = await result

        await _save_session_state(session_state, fmcp_ctx)

        # convert_result() is inherited from Prompt and handles str / list[Message] / PromptResult
        return result
