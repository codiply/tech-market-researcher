#!/usr/bin/env bash
# Stack helper — wraps `docker compose` for this repo.
# Usage: ./stack.sh <up|down|ps|logs|restart|pull> [extra args...]
set -euo pipefail

COMPOSE_FILE="$(dirname "$(readlink -f "$0")")/docker/docker-compose.yml"
ROOT="$(dirname "$(readlink -f "$0")")"

# Use the .env from the repo root, if present
ENV_ARGS=()
if [[ -f "$ROOT/.env" ]]; then
  ENV_ARGS=(--env-file "$ROOT/.env")
fi

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  echo "Usage: $0 <up|down|ps|logs|restart|pull> [extra args...]" >&2
  echo "Examples:" >&2
  echo "  $0 up -d        # start the stack in the background" >&2
  echo "  $0 down         # stop and remove containers" >&2
  echo "  $0 ps           # show container status" >&2
  echo "  $0 logs -f      # tail logs" >&2
  exit 1
fi
shift

exec docker compose "${ENV_ARGS[@]}" -f "$COMPOSE_FILE" "$cmd" "$@"
