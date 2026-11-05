# mypy: ignore-errors
# pylint: skip-file
"""Save database passwords on the pre-configured pgAdmin servers.

pgAdmin, in server mode, uses the user's login password as the key to
encrypt saved server passwords (see the store_crypt_key signal handler). So
we encrypt each server's database password with PGADMIN_DEFAULT_PASSWORD and
set save_password, exactly as the UI's "save password" would. The UI then
decrypts it on login and connects without prompting.

Reads the pre-configured server list from /tmp/servers.json (rendered by
entrypoint.sh from the compose environment) and keeps it as the source of
truth: on every container start each stored password is (re)encrypted from
the file if it differs.
"""

import json
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

SERVERS_JSON = os.environ.get("PGADMIN_SERVER_JSON_FILE", "/tmp/servers.json")


def main() -> None:
    with open(SERVERS_JSON, encoding="utf-8") as fh:
        specs = json.load(fh).get("Servers", {}).values()

    login_password = os.environ.get("PGADMIN_DEFAULT_PASSWORD", "")

    app = create_app(config.APP_NAME + "-setup")
    app.PGADMIN_RUNTIME = False
    with app.test_request_context():
        for spec in specs:
            name = spec.get("Name")
            db_password = spec.get("Password", "")
            if not name:
                continue
            server = Server.query.filter_by(name=name).first()
            if server is None:
                print(f"pgadmin-init: server '{name}' not found; not saving password")
                continue
            if not login_password or not db_password:
                print(f"pgadmin-init: missing login/db password for '{name}'; not saving")
                continue

            stored = None
            if server.password:
                try:
                    stored = decrypt(server.password, login_password).decode()
                except Exception:  # pylint: disable=broad-exception-caught
                    # Any failure (stale key, non-encrypted value) means "not saved".
                    stored = None
            if stored == db_password and server.save_password:
                print(f"pgadmin-init: database password for '{name}' is already saved")
            else:
                server.password = encrypt(db_password, login_password)
                server.save_password = 1
                db.session.commit()
                print(f"pgadmin-init: saved database password for '{name}'")


if __name__ == "__main__":
    main()
