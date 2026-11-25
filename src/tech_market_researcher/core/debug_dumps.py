"""Debug dumps written under the .debug folder on agent creation."""

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from tech_market_researcher.core.configuration import DebugConfig
from tech_market_researcher.core.skills_dump import dump_agent_skills
from tech_market_researcher.core.tool_definitions_dump import dump_agent_tool_definitions


def write_debug_dumps(debug_config: DebugConfig, agent: CompiledStateGraph[Any, Any]) -> None:
    """Write the debug dumps enabled in the config: the agent's tool definitions and available skills."""
    if debug_config.tool_definitions.enabled:
        dump_agent_tool_definitions(agent)
    if debug_config.skills.enabled:
        dump_agent_skills(agent)
