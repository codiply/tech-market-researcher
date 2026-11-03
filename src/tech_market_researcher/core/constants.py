"""Application-wide constants."""

import os

DEFAULT_CONFIG_NAME = "base"

# Local SearXNG instance (see docker/docker-compose.yml); overridable via env.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8384")
