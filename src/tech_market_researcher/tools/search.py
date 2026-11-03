"""Web search tools backed by Tavily and a local SearXNG instance."""

import os
from typing import Annotated, Literal

import httpx
from langchain.tools import InjectedToolArg, tool
from langchain_core.tools import BaseTool
from tavily import TavilyClient

from tech_market_researcher.core.configuration import SearchProvider, SearchToolConfig
from tech_market_researcher.core.constants import SEARXNG_URL
from tech_market_researcher.core.webpage import fetch_as_markdown

SEARCH_MAX_RESULT_COUNT_DEFAULT = 5
SEARCH_MAX_RESULT_COUNT_MAX = 25


def _get_tavily_client() -> TavilyClient:
    """Lazily build the Tavily client so import does not require the API key."""
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _format_results(query: str, results: list[dict]) -> str:
    """Fetch result pages as markdown and format them for the model."""
    result_texts = []
    for result in results:
        url = result["url"]
        title = result["title"]
        content = fetch_as_markdown(url)
        result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n" + "\n".join(result_texts)


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: int = SEARCH_MAX_RESULT_COUNT_DEFAULT,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (max 25, default: 5)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    effective_max_results = min(max_results, SEARCH_MAX_RESULT_COUNT_MAX)
    search_results = _get_tavily_client().search(
        query,
        max_results=effective_max_results,
        topic=topic,
    )
    return _format_results(query, search_results.get("results", []))


@tool(parse_docstring=True)
def searxng_search(
    query: str,
    max_results: int = SEARCH_MAX_RESULT_COUNT_DEFAULT,
) -> str:
    """Search the web for information on a given query.

    Uses the local SearXNG metasearch instance to discover relevant URLs, then
    fetches and returns the full webpage content as markdown.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (max 25, default: 5)

    Returns:
        Formatted search results with full webpage content
    """
    response = httpx.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json"},
        timeout=60.0,
    )
    response.raise_for_status()
    effective_max_results = min(max_results, SEARCH_MAX_RESULT_COUNT_MAX)
    results = response.json().get("results", [])[:effective_max_results]
    return _format_results(query, results)


def get_search_tool(search: SearchToolConfig) -> BaseTool:
    """Return the search tool for the configured provider (validated by the config model)."""
    if search.provider is SearchProvider.SEARXNG:
        return searxng_search
    return tavily_search
