from __future__ import annotations

import inspect
from typing import Any

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

# Shared stub — pydantic-ai tools never invoke ctx.model in normal use;
# it only exists to satisfy RunContext's type contract.
_STUB_MODEL = TestModel()


def make_bootstrap_context(deps: Any = None, *, max_retries: int = 0) -> RunContext[Any]:
    """Minimal RunContext for get_tools() discovery at server startup.

    FunctionToolset.get_tools() only reads ctx.max_retries, never ctx.deps,
    so deps=None is safe here.
    """
    return RunContext(
        deps=deps,
        model=_STUB_MODEL,
        usage=RunUsage(),
        max_retries=max_retries,
    )


async def make_call_context(
    factory: Any,
    *,
    tool_name: str | None = None,
    max_retries: int = 0,
) -> RunContext[Any]:
    """RunContext for a live tool or prompt call with fresh deps per invocation."""
    deps = await _resolve_deps(factory)
    return RunContext(
        deps=deps,
        model=_STUB_MODEL,
        usage=RunUsage(),
        max_retries=max_retries,
        tool_name=tool_name,
    )


async def _resolve_deps(factory: Any) -> Any:
    if callable(factory):
        if _factory_takes_context(factory):
            from fastmcp.server.dependencies import get_context
            result = factory(get_context())
        else:
            result = factory()
        if inspect.isawaitable(result):
            return await result
        return result
    return factory


def _factory_takes_context(factory: Any) -> bool:
    """Return True if the factory declares at least one positional parameter.

    A positional parameter signals that the factory wants the FastMCP Context
    injected. Zero-parameter factories are called as-is (backward compatible).
    """
    try:
        sig = inspect.signature(factory)
        return any(
            p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for p in sig.parameters.values()
        )
    except (ValueError, TypeError):
        return False
