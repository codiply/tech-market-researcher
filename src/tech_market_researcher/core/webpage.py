"""Webpage fetching helpers."""

import logging

import httpx
from markdownify import markdownify

logger = logging.getLogger(__name__)


def fetch_as_markdown(url: str, timeout: float = 10.0) -> str:
    """Fetch a webpage and convert its HTML to markdown."""
    logger.info("Fetching webpage (timeout=%.1fs)", timeout)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        logger.info("Webpage fetched successfully")
        return markdownify(response.text)
    except httpx.HTTPError as e:
        logger.warning("Webpage fetch failed: %s", e)
        return f"Error fetching {url}: {e!s}"
