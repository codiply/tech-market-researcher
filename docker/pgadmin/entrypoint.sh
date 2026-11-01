#!/bin/sh
# pgAdmin entrypoint wrapper:
#   1. renders /tmp/servers.json from the compose environment (MLFLOW_DB_*),
#      which the image's entrypoint imports at startup, adding the MLflow
#      database to the server list;
#   2. starts the image's real entrypoint in the background;
#   3. once the imported server exists in the config DB, saves the database
#      password via save_password.py (encrypted with the pgAdmin login
#      password, which is the crypt key pgAdmin uses in server mode) so the
#      UI never prompts for it;
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
    }
  }
}
JSON

/entrypoint.sh &
APP_PID=$!

# Forward termination signals to the app.
trap 'kill -TERM "$APP_PID" 2>/dev/null || true; wait "$APP_PID"' TERM INT

# Wait for the real entrypoint to import the server into the config DB.
for _ in $(seq 1 90); do
    if /venv/bin/python3 - <<'PY' 2>/dev/null
import os, sqlite3, sys
con = sqlite3.connect("file:/var/lib/pgadmin/pgadmin4.db?mode=ro", uri=True)
name = os.environ.get("PGADMIN_SERVER_NAME", "MLflow (Postgres)")
sys.exit(0 if con.execute(
    "SELECT 1 FROM server WHERE name = ?", (name,)).fetchone() else 1)
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
