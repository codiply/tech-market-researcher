"""Unit tests for the webpage helpers."""

import pytest

from tech_market_researcher.core.webpage import domain_from_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        pytest.param("https://example.com", "example.com", id="bare domain"),
        pytest.param("https://example.com/article", "example.com", id="domain with path"),
        pytest.param("https://example.com/a/b?q=1#frag", "example.com", id="path, query and fragment ignored"),
        pytest.param("http://EXAMPLE.COM/article", "example.com", id="host is lowercased"),
        pytest.param("https://example.com:8080/article", "example.com", id="port is dropped"),
        pytest.param("https://sub.domain.example.co.uk/path", "sub.domain.example.co.uk", id="subdomain is returned"),
        pytest.param("https://user:pass@example.com/", "example.com", id="credentials are dropped"),
        pytest.param("example.com/article", None, id="no scheme, so no host"),
        pytest.param("", None, id="empty string"),
    ],
)
def test_domain_from_url(url: str, expected: str | None) -> None:
    """Extracts the hostname from a URL, or None when the URL carries no host."""
    assert domain_from_url(url) == expected
