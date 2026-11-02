# mypy: ignore-errors
# pylint: skip-file
"""Save the MLflow database password on the pre-configured pgAdmin server.

pgAdmin, in server mode, uses the user's login password as the key to
encrypt saved server passwords (see the store_crypt_key signal handler). So
we encrypt the database password with PGADMIN_DEFAULT_PASSWORD and set
save_password, exactly as the UI's "save password" would. The UI then
decrypts it on login and connects without prompting.

Runs in pgAdmin's app context, after the image entrypoint has imported the
server from /tmp/servers.json. Keeps .env as the source of truth: on every
container start the stored password is (re)encrypted from MLFLOW_DB_PASSWORD
if it differs.
"""

import os
import sys

os.chdir("/pgadmin4")
sys.path.insert(0, "/pgadmin4")

# These modules only exist inside the pgAdmin container image; the script runs
# there, after the chdir/sys.path setup above, so the imports cannot resolve
# in the dev environment.
from pgadmin import create_app
from pgadmin.model import Server, db
from pgadmin.utils.crypto import decrypt, encrypt

import config

SERVER_NAME = os.environ.get("PGADMIN_SERVER_NAME", "MLflow (Postgres)")
db_password = os.environ.get("MLFLOW_DB_PASSWORD", "")
login_password = os.environ.get("PGADMIN_DEFAULT_PASSWORD", "")

app = create_app(config.APP_NAME + "-setup")
app.PGADMIN_RUNTIME = False
with app.test_request_context():
    server = Server.query.filter_by(name=SERVER_NAME).first()
    if server is None:
        print(f"pgadmin-init: server '{SERVER_NAME}' not found; " "not saving password")
        sys.exit(0)
    if not login_password or not db_password:
        print("pgadmin-init: missing login/db password; not saving")
        sys.exit(0)

    stored = None
    if server.password:
        try:
            stored = decrypt(server.password, login_password).decode()
        except Exception:  # pylint: disable=broad-exception-caught
            # Any failure (stale key, non-encrypted value) means "not saved".
            stored = None
    if stored == db_password and server.save_password:
        print(
            f"pgadmin-init: database password for '{SERVER_NAME}' " "is already saved"
        )
    else:
        server.password = encrypt(db_password, login_password)
        server.save_password = 1
        db.session.commit()
        print(f"pgadmin-init: saved database password for '{SERVER_NAME}'")
