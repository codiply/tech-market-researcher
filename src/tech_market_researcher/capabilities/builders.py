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
from tech_market_researcher.core.configuration import ApplicationConfig, CapabilitiesConfig
from tech_market_researcher.core.webpage import fetch_as_markdown
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
WEB_PAGE_INDEX_DOCUMENT_ID_DESCRIPTION = (
    "The URL of the web page this record stores. Always use the page's full URL, including the "
    "scheme (e.g. https://example.com/article), as the ID, so re-fetching the same page replaces "
    "its previous record instead of duplicating it."
)


async def build_tool_list(config: ApplicationConfig) -> list[BaseTool]:
    """Build the tools list from the capabilities config."""
    capabilities = config.supervisor.capabilities
    tools = []
    if capabilities.search.enabled:
        tools.append(get_search_tool(capabilities.search))
    if capabilities.fetch_page.enabled:
        tools.append(fetch_page)
    if capabilities.runtime_context_debugger.enabled:
        tools.append(runtime_context_debugger)
    if capabilities.current_datetime.enabled:
        tools.append(current_datetime)
    if capabilities.web_page_indexer.enabled:
        indexer = capabilities.web_page_indexer
        assert indexer.collection_name is not None  # Validated by WebPageIndexerConfig
        assert indexer.embeddings_model is not None  # Validated by WebPageIndexerConfig
        client = await get_client()
        tools.extend(
            await build_index_tools(
                client,
                IndexToolsParams(
                    tool_name_prefix=WEB_PAGE_INDEX_TOOL_NAME_PREFIX,
                    collection_name=indexer.collection_name,
                    collection_description=WEB_PAGE_INDEX_COLLECTION_DESCRIPTION,
                    is_document_id_required=True,
                    document_id_description=WEB_PAGE_INDEX_DOCUMENT_ID_DESCRIPTION,
                    metadata_parameters=WEB_PAGE_INDEX_METADATA_PARAMETERS,
                    embeddings_model=config.get_model_connection_details(indexer.embeddings_model),
                    document_fetcher=fetch_as_markdown,
                    chunk_size_tokens=indexer.chunk_size_tokens,
                    chunk_overlap_tokens=indexer.chunk_overlap_tokens,
                ),
            )
        )
    return tools
