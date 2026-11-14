"""Web search tool backed by Tavily or a local SearXNG instance, depending on the config.

The tool and its RETRIEVER span are always named ``web_search`` regardless of the
configured provider (recorded as a span attribute). These tools only return search
results (title, URL, snippet). Fetching full page content is a separate capability
(see capabilities/webpage.py)."""

import os
from collections.abc import Callable
from typing import Annotated, Literal

import httpx
import mlflow
from langchain.tools import InjectedToolArg
from langchain_core.tools import BaseTool, StructuredTool
from tavily import TavilyClient

from tech_market_researcher.core.configuration import SearchProvider, SearchToolConfig
from tech_market_researcher.core.environment import SEARXNG_URL

SEARCH_TOOL_NAME = "web_search"
SEARCH_MAX_RESULT_COUNT_DEFAULT = 5
SEARCH_MAX_RESULT_COUNT_MAX = 25


def _get_tavily_client() -> TavilyClient:
    """Lazily build the Tavily client so import does not require the API key."""
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def _fetch_tavily(query: str, max_results: int, topic: str) -> list[dict]:
    """Fetch raw search results from the Tavily API."""
    search_results = _get_tavily_client().search(
        query,
        max_results=max_results,
        topic=topic,
    )
    return search_results.get("results", [])


def _fetch_searxng(query: str, max_results: int, _topic: str) -> list[dict]:
    """Fetch raw search results from the local SearXNG metasearch instance.

    _topic is not supported by SearXNG; it is accepted only for a uniform provider API."""
    response = httpx.get(
        f"{SEARXNG_URL}/search",
        params={"q": query, "format": "json"},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json().get("results", [])[:max_results]


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


def get_search_tool(search: SearchToolConfig) -> BaseTool:
    """Return the web_search tool backed by the configured provider (validated by the config model)."""
    fetch: Callable[[str, int, str], list[dict]]
    if search.provider is SearchProvider.SEARXNG:
        fetch = _fetch_searxng
    elif search.provider is SearchProvider.TAVILY:
        fetch = _fetch_tavily
    else:
        raise ValueError(f"Unknown search provider: {search.provider}")

    def web_search(
        query: str,
        max_results: int = SEARCH_MAX_RESULT_COUNT_DEFAULT,
        topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    ) -> str:
        """Search the web for information on a given query.

        Returns the result titles, URLs and snippets.

        Args:
            query: Search query to execute
            max_results: Maximum number of results to return (max 25, default: 5)

        Returns:
            Numbered search results with title, URL and snippet
        """
        effective_max_results = min(max_results, SEARCH_MAX_RESULT_COUNT_MAX)
        with mlflow.start_span(name=SEARCH_TOOL_NAME, span_type="RETRIEVER") as span:
            span.set_inputs({"query": query, "max_results": effective_max_results, "topic": topic})
            span.set_attributes({"provider": search.provider.value})
            results = fetch(query, effective_max_results, topic)
            span.set_outputs(_to_retriever_chunks(results))
        return _format_results(query, results)

    return StructuredTool.from_function(func=web_search, name=SEARCH_TOOL_NAME, description=web_search.__doc__)
