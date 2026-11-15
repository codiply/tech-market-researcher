"""Builders that turn the capabilities config into agent tools and middlewares."""

from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from tech_market_researcher.capabilities.clock import current_datetime
from tech_market_researcher.capabilities.indexing import build_indexing_tools
from tech_market_researcher.capabilities.runtime_context import runtime_context_debugger
from tech_market_researcher.capabilities.search import get_search_tool
from tech_market_researcher.capabilities.webpage import fetch_page
from tech_market_researcher.core.configuration import CapabilitiesConfig
from tech_market_researcher.integrations.chroma import get_client


def build_middleware_list(config: CapabilitiesConfig) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build the middleware list enabled in the capabilities config."""
    middlewares: list[AgentMiddleware[Any, Any, Any]] = []
    if config.todo_list.enabled:
        middlewares.append(TodoListMiddleware())
    return middlewares


WEB_PAGE_INDEX_TOOL_NAME_PREFIX = "web_page"


async def build_tool_list(config: CapabilitiesConfig) -> list[BaseTool]:
    """Build the tools list from the capabilities config."""
    tools = []
    if config.search.enabled:
        tools.append(get_search_tool(config.search))
    if config.fetch_page.enabled:
        tools.append(fetch_page)
    if config.runtime_context_debugger.enabled:
        tools.append(runtime_context_debugger)
    if config.current_datetime.enabled:
        tools.append(current_datetime)
    if config.web_page_indexer.enabled:
        indexer = config.web_page_indexer
        assert indexer.collection_name is not None  # Validated by WebPageIndexerConfig
        assert indexer.embeddings_model_name is not None  # Validated by WebPageIndexerConfig
        client = await get_client()
        tools.extend(
            await build_indexing_tools(
                client,
                indexer.collection_name,
                indexer.embeddings_model_name,
                WEB_PAGE_INDEX_TOOL_NAME_PREFIX,
            )
        )
    return tools
