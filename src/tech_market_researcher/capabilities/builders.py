"""Builders that turn the capabilities config into agent tools and middlewares."""

from typing import Any

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import BaseTool

from tech_market_researcher.capabilities.clock import current_datetime
from tech_market_researcher.capabilities.indexing import IndexToolsParams, MetadataParameter, build_index_tools
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
WEB_PAGE_INDEX_COLLECTION_DESCRIPTION = (
    "A vector index of web pages the agent has fetched during its research. "
    "Each document is the text content of a fetched page. "
    "Use it to recall the content of pages you have already fetched instead of fetching them again."
)
WEB_PAGE_INDEX_METADATA_PARAMETERS = [
    MetadataParameter(
        name="source_url",
        description=(
            "The URL of the web page this document was fetched from. Always include it when "
            "storing a page so the source of the content can be identified and the page can be "
            "re-fetched later if needed. Must be the full URL, including the scheme "
            "(e.g. https://example.com/article). Do not use it for anything other than the "
            "page's URL."
        ),
    ),
]


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
            await build_index_tools(
                client,
                IndexToolsParams(
                    collection_name=indexer.collection_name,
                    embedding_model_name=indexer.embeddings_model_name,
                    tool_name_prefix=WEB_PAGE_INDEX_TOOL_NAME_PREFIX,
                    collection_description=WEB_PAGE_INDEX_COLLECTION_DESCRIPTION,
                    metadata_parameters=WEB_PAGE_INDEX_METADATA_PARAMETERS,
                ),
            )
        )
    return tools
