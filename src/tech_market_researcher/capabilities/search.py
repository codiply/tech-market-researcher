"""Web search tools backed by Tavily and a local SearXNG instance.

These tools only return search results (title, URL, snippet). Fetching full
page content is a separate capability (see capabilities/webpage.py)."""

import os
from typing import Annotated, Literal

import httpx
import mlflow
from langchain.tools import InjectedToolArg, tool
from langchain_core.tools import BaseTool
from tavily import TavilyClient

from tech_market_researcher.core.configuration import SearchProvider, SearchToolConfig
from tech_market_researcher.core.constants import SEARXNG_URL

SEARCH_MAX_RESULT_COUNT_DEFAULT = 5
SEARCH_MAX_RESULT_COUNT_MAX = 25


def _get_tavily_client() -> TavilyClient:
    """Lazily build the Tavily client so import does not require the API key."""
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _to_retriever_chunks(results: list[dict]) -> list[dict]:
    """Shape raw search results into the document chunks a RETRIEVER span reports."""
    return [
        {
            "content": f"{result['title']}\n{result.get('content') or ''}".strip(),
            "metadata": {"doc_uri": result["url"], "title": result["title"]},
        }
        for result in results
    ]


def _format_results(query: str, results: list[dict]) -> str:
    """Format search results (title, URL, snippet) for the model."""
    result_texts = [
        f"{index}. **{result['title']}**\n   URL: {result['url']}\n   {result.get('content') or ''}".rstrip()
        for index, result in enumerate(results, start=1)
    ]
    return f"Found {len(results)} result(s) for '{query}':\n\n" + "\n\n".join(result_texts)


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: int = SEARCH_MAX_RESULT_COUNT_DEFAULT,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs and returns the result titles, URLs
    and snippets.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (max 25, default: 5)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Numbered search results with title, URL and snippet
    """
    effective_max_results = min(max_results, SEARCH_MAX_RESULT_COUNT_MAX)
    with mlflow.start_span(name="tavily_search", span_type="RETRIEVER") as span:
        span.set_inputs({"query": query, "max_results": effective_max_results, "topic": topic})
        search_results = _get_tavily_client().search(
            query,
            max_results=effective_max_results,
            topic=topic,
        )
        results = search_results.get("results", [])
        span.set_outputs(_to_retriever_chunks(results))
    return _format_results(query, results)


@tool(parse_docstring=True)
def searxng_search(
    query: str,
    max_results: int = SEARCH_MAX_RESULT_COUNT_DEFAULT,
) -> str:
    """Search the web for information on a given query.

    Uses the local SearXNG metasearch instance to discover relevant URLs and
    returns the result titles, URLs and snippets.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (max 25, default: 5)

    Returns:
        Numbered search results with title, URL and snippet
    """
    effective_max_results = min(max_results, SEARCH_MAX_RESULT_COUNT_MAX)
    with mlflow.start_span(name="searxng_search", span_type="RETRIEVER") as span:
        span.set_inputs({"query": query, "max_results": effective_max_results})
        response = httpx.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=60.0,
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:effective_max_results]
        span.set_outputs(_to_retriever_chunks(results))
    return _format_results(query, results)


def get_search_tool(search: SearchToolConfig) -> BaseTool:
    """Return the search tool for the configured provider (validated by the config model)."""
    if search.provider is SearchProvider.SEARXNG:
        return searxng_search
    return tavily_search
