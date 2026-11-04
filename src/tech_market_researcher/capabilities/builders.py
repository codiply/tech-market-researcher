"""Builders that turn the capabilities config into agent tools and middlewares."""

from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from tech_market_researcher.capabilities.search import get_search_tool
from tech_market_researcher.core.configuration import CapabilitiesConfig


def build_middleware_list(config: CapabilitiesConfig) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build the middleware list enabled in the capabilities config."""
    middlewares: list[AgentMiddleware[Any, Any, Any]] = []
    if config.todo_list.enabled:
        middlewares.append(TodoListMiddleware())
    return middlewares


def build_tool_list(config: CapabilitiesConfig) -> list[BaseTool]:
    """Build the tools list from the capabilities config."""
    tools = []
    if config.search.enabled:
        tools.append(get_search_tool(config.search))
    return tools
