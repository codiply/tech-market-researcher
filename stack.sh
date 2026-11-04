#!/usr/bin/env bash
# Stack helper — wraps `docker compose` for this repo.
# Usage: ./stack.sh <up|down|ps|logs|restart|pull|build> [extra args...]
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
  echo "Usage: $0 <up|down|ps|logs|restart|pull|build> [extra args...]" >&2
  echo "Examples:" >&2
  echo "  $0 up -d        # start the stack, then provision the MLflow gateway" >&2
  echo "  $0 build        # build (or rebuild) the local images" >&2
  echo "  $0 down         # stop and remove containers" >&2
  echo "  $0 ps           # show container status" >&2
  echo "  $0 logs -f      # tail logs" >&2
  exit 1
fi
shift

# Ensure storage dirs exist with the ownership the containers expect.
# If a dir is missing, dockerd (running as root) would recreate it as
# root:root 0755, and the container user would fail with "file access
# denied" / "data directory is owned by root".
#   minio      -> uid 100 (minio user in the alpine/minio mirror image;
#                 note: the official minio/minio image uses uid 1001)
#   mlflow_db  -> uid 999  (postgres user)
#   pgadmin    -> uid 5050 (pgadmin user in the dpage/pgadmin4 image; gid 0)
#   searxng_*  -> uid 977  (searxng user in the searxng/searxng image)
#   searxng_valkey runs as root, so only the mkdir is needed (no chown)
ensure_storage_dirs() {
  local -A owners=( [minio]=100 [mlflow_db]=999 [pgadmin]=5050 [searxng_config]=977 [searxng_cache]=977 [searxng_valkey]=0 )
  local -A groups=( [minio]=101 [mlflow_db]=999 [pgadmin]=0 [searxng_config]=977 [searxng_cache]=977 [searxng_valkey]=0 )
  local dir path uid
  for dir in minio mlflow_db pgadmin searxng_config searxng_cache searxng_valkey; do
    path="$ROOT/storage/$dir"
    uid="${owners[$dir]}"
    mkdir -p "$path"
    [[ "$uid" == "0" ]] && continue
    if [[ "$(stat -c '%u' "$path")" != "$uid" ]]; then
      echo "storage/$dir is owned by uid $(stat -c '%u' "$path"); chowning to $uid:${groups[$dir]}..."
      if [[ "$(id -u)" == "0" ]]; then
        chown "$uid:${groups[$dir]}" "$path"
      else
        sudo chown "$uid:${groups[$dir]}" "$path"
      fi
    fi
  done
}

# Wait for the MLflow server to answer HTTP 200 on /health.
# (A raw TCP probe is not enough: the port accepts connections before the
# server is ready to serve them, causing "Connection reset by peer".)
wait_for_mlflow() {
  echo "Waiting for MLflow server on localhost:5000..."
  local tries
  for tries in $(seq 1 90); do
    if uv --directory "$ROOT" run python -c '
import sys, urllib.request
try:
    sys.exit(0 if urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=2).status == 200 else 1)
except Exception:
    sys.exit(1)
' 2>/dev/null; then
      echo "MLflow server is up."
      return 0
    fi
    sleep 2
  done
  echo "error: MLflow server did not come up within 180s" >&2
  return 1
}

# Bootstrap the MLflow stack (idempotent).
# MLFLOW_TRACKING_USERNAME attributes dataset records to the automation instead of the local user.
# (read_mlflow_creds() only looks at MLFLOW_TRACKING_USERNAME, not MLFLOW_USER.)
bootstrap() {
  wait_for_mlflow || return 1
  MLFLOW_TRACKING_USERNAME=bootstrap uv --directory "$ROOT" run bootstrap
}

case "$cmd" in
  up|restart)
    ensure_storage_dirs
    docker compose "${ENV_ARGS[@]}" -f "$COMPOSE_FILE" "$cmd" "$@"
    bootstrap
    exit $?
    ;;
  build)
    docker compose "${ENV_ARGS[@]}" -f "$COMPOSE_FILE" build "$@"
    ;;
esac

exec docker compose "${ENV_ARGS[@]}" -f "$COMPOSE_FILE" "$cmd" "$@"
