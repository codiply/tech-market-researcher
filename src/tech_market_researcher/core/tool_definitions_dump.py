"""Dump the agent's bound tool definitions to the debug folder for inspection."""

import json
import logging
from pathlib import Path
from typing import Any, cast

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from tech_market_researcher.core.constants import (
    BASE_DEBUG_DIR_PATH,
    DEBUG_AGENT_FOLDER,
    DEBUG_TOOL_DEFINITIONS_FOLDER,
    TOOL_DEFINITIONS_FILE_NAME,
)

logger = logging.getLogger(__name__)

# Graph node holding the ToolNode that merges the built-in, middleware and user tools.
TOOLS_NODE_NAME = "tools"


def _tool_definition(tool: BaseTool) -> dict[str, Any]:
    """Serialize one tool in the provider tool-call format (name, description, parameters).

    Uses the tool's tool-call schema (what the model is actually offered), not its raw input
    schema: injected arguments such as ToolRuntime are excluded from the former but would
    crash JSON-schema serialization of the latter.
    """
    # The declared type is a union (pydantic v2/v1 model or dict); the tools in this codebase
    # always use a v2 model.
    schema = cast(type[BaseModel], tool.tool_call_schema)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema.model_json_schema(),
        },
    }


def dump_agent_tool_definitions(agent: CompiledStateGraph[Any, Any]) -> Path:
    """Write every tool bound to the agent to the debug folder, and return the file path.

    The tool set comes from the graph's `tools` node, so it matches exactly what the model
    is offered: the built-in deepagents tools, the middleware tools, and the configured
    capabilities tools.
    """
    tool_node = agent.nodes[TOOLS_NODE_NAME].bound
    if not hasattr(tool_node, "tools_by_name"):
        msg = f"agent node '{TOOLS_NODE_NAME}' is not a tool node; cannot dump the tool definitions"
        raise RuntimeError(msg)
    tools = sorted(tool_node.tools_by_name.values(), key=lambda tool: tool.name)
    out_path = BASE_DEBUG_DIR_PATH / DEBUG_AGENT_FOLDER / DEBUG_TOOL_DEFINITIONS_FOLDER / TOOL_DEFINITIONS_FILE_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([_tool_definition(tool) for tool in tools], indent=2) + "\n")
    logger.info("Dumped %d tool definition(s) to %s", len(tools), out_path)
    return out_path
