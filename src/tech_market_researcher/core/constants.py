"""Application-wide constants."""

import os

DEFAULT_CONFIG_NAME = "base"

DEFAULT_USER_ID = "codiply"
DEFAULT_SESSION_ID = "session-0001"
DEFAULT_TIME_ZONE = "Europe/Madrid"

# Local SearXNG instance (see docker/docker-compose.yml); overridable via env.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8384")
