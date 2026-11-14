"""Settings read from environment variables."""

import os

# Local SearXNG instance (see docker/docker-compose.yml); overridable via env.
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8384")

# The Tailscale host name of this machine, if any; web UIs use it as their public URL so
# devices on the Tailscale mesh can reach them.
TAILSCALE_HOST_NAME = os.environ.get("TAILSCALE_HOST_NAME")
