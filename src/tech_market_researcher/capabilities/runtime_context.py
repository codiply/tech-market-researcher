"""Tool that exposes the per-run runtime context."""

from langchain.tools import ToolRuntime, tool

from tech_market_researcher.core.common import AgentRuntimeContext


@tool
def runtime_context_debugger(runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    """Return the current runtime context as a JSON string."""
    return runtime.context.model_dump_json()
