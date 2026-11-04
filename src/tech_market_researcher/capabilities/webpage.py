"""Tool for fetching a webpage and returning its content in a given format."""

from enum import Enum

from langchain.tools import tool

from tech_market_researcher.core.webpage import fetch_as_markdown


class FetchPageOutputFormat(str, Enum):
    """Output formats for the fetch_page tool."""

    MARKDOWN = "markdown"


@tool(parse_docstring=True)
def fetch_page(
    url: str,
    output_format: FetchPageOutputFormat = FetchPageOutputFormat.MARKDOWN,
) -> str:
    """Fetch a webpage and return its content in the requested output format.

    Use this to read the full content of a page.

    Args:
        url: URL of the page to fetch
        output_format: Format to return the page content in ('markdown' only for now)

    Returns:
        Page content in the requested output format, or an error message if the fetch fails
    """
    if output_format is FetchPageOutputFormat.MARKDOWN:
        return fetch_as_markdown(url)
    raise ValueError(f"Unsupported output format: {output_format}")
