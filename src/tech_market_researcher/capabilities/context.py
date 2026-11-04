"""Debugging tools that expose the per-run agent context."""

from langchain.tools import ToolRuntime, tool

from tech_market_researcher.core.common import AgentContext


@tool
def context_debugger(runtime: ToolRuntime[AgentContext]) -> str:
    """Return the current run context as a JSON string."""
    return runtime.context.model_dump_json()
