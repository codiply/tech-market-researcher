#!/bin/sh
# pgAdmin entrypoint wrapper:
#   1. renders /tmp/servers.json from the compose environment (MLFLOW_DB_*,
#      PHOENIX_DB_*, AGENT_DB_*), which the image's entrypoint imports at
#      startup, adding the MLflow, Phoenix, and agent databases to the
#      server list;
#   2. starts the image's real entrypoint in the background;
#   3. once the web server answers (the entrypoint imports the server list
#      before it starts the server, so by then the DB is in its final state),
#      saves the database passwords via save_password.py (encrypted with the
#      pgAdmin login password, which is the crypt key pgAdmin uses in server
#      mode) so the UI never prompts for it;
#   4. stays in the foreground, following the app's lifecycle.
set -eu

cat > /tmp/servers.json <<JSON
{
  "Servers": {
    "1": {
      "Name": "${PGADMIN_SERVER_NAME:-MLflow (Postgres)}",
      "Group": "Servers",
      "Host": "mlflow_db",
      "Port": 5433,
      "MaintenanceDB": "${MLFLOW_DB_NAME:-mlflow}",
      "Username": "${MLFLOW_DB_USER:-mlflow}",
      "Password": "${MLFLOW_DB_PASSWORD:?set MLFLOW_DB_PASSWORD in the environment}"
    },
    "2": {
      "Name": "${PHOENIX_SERVER_NAME:-Phoenix (Postgres)}",
      "Group": "Servers",
      "Host": "phoenix_db",
      "Port": 5432,
      "MaintenanceDB": "${PHOENIX_DB_NAME:-phoenix}",
      "Username": "${PHOENIX_DB_USER:?set PHOENIX_DB_USER in the environment}",
      "Password": "${PHOENIX_DB_PASSWORD:?set PHOENIX_DB_PASSWORD in the environment}"
    },
    "3": {
      "Name": "${AGENT_SERVER_NAME:-Agent (Postgres)}",
      "Group": "Servers",
      "Host": "agent_db",
      "Port": 5432,
      "MaintenanceDB": "${AGENT_DB_NAME:-agent}",
      "Username": "${AGENT_DB_USER:?set AGENT_DB_USER in the environment}",
      "Password": "${AGENT_DB_PASSWORD:?set AGENT_DB_PASSWORD in the environment}"
    }
  }
}
JSON

/entrypoint.sh &
APP_PID=$!

# Forward termination signals to the app.
trap 'kill -TERM "$APP_PID" 2>/dev/null || true; wait "$APP_PID"' TERM INT

# Wait until the app answers locally. The entrypoint imports the server list
# before it starts the web server, so once this succeeds the config DB is in
# its final state (with PGADMIN_REPLACE_SERVERS_ON_STARTUP the re-import
# would still be in flight if we only waited for a server name to appear).
for _ in $(seq 1 90); do
    if /venv/bin/python3 - <<'PY' 2>/dev/null
import os, sys, urllib.request
port = os.environ.get("PGADMIN_LISTEN_PORT", "80")
url = f"http://127.0.0.1:{port}/"
sys.exit(0 if urllib.request.urlopen(url, timeout=2).status == 200 else 1)
PY
    then
        break
    fi
    sleep 1
done

# Persist the database password so the UI connects without prompting.
# Retry a few times in case the running app holds the SQLite write lock.
for _ in $(seq 1 3); do
    if /venv/bin/python3 /pgadmin-init/save_password.py; then
        break
    fi
    sleep 2
done

wait "$APP_PID"
