"""Application-wide constants."""

import re
from importlib.metadata import version
from pathlib import Path

PACKAGE_VERSION = version("tech-market-researcher")
PROJECT_NAME = "tech-market-researcher"

BASE_DEBUG_DIR_PATH = Path(".debug")
# Folders under BASE_DEBUG_DIR_PATH for rendered configs: the latest debug dump and the timestamped archives.
DEBUG_CONFIG_FOLDER = "config"
DEBUG_CONFIG_LATEST_FOLDER = "latest"
DEBUG_CONFIG_ARCHIVE_FOLDER = "archive"
# Maximum number of timestamped config archives to keep per loader; older ones are pruned.
DEBUG_CONFIG_ARCHIVE_MAX_COUNT = 1000

# On-disk directory the agent's filesystem tools are rooted at (virtual mode).
AGENT_BACKEND_ROOT_DIR = Path(".storage/agent/backend")
# Defaults for the agent's file logging (see LoggingConfig).
AGENT_DEFAULT_LOGS_DIR = Path(".storage/agent/logs")
AGENT_DEFAULT_LOG_LEVEL = "INFO"
AGENT_LOG_FILE_NAME = "agent.log"
# Maximum size of a single file the agent can read, in megabytes.
AGENT_BACKEND_MAX_FILE_SIZE_MB = 10

# Env var enabling the dump of rendered configs under BASE_DEBUG_DIR_PATH (True/true/1).
DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR = "DEBUG_DUMP_RENDERED_CONFIGS"
# Env var enabling the archiving of rendered config dumps into the archive folder (True/true/1).
ARCHIVE_RENDERED_CONFIGS_ENV_VAR = "ARCHIVE_RENDERED_CONFIGS"

DEFAULT_CONFIG_NAME = "dev"

DEFAULT_USER_ID = "codiply"
DEFAULT_SESSION_ID = "session-0001"
DEFAULT_TIME_ZONE = "Europe/Madrid"

# Postgres identifiers are capped at 63 bytes, and schema names are interpolated
# into DDL, so they must be plain identifiers within that limit.
POSTGRES_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
