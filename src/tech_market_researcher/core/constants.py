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
# Folder under BASE_DEBUG_DIR_PATH for the agent's debug dumps, and the file the tool definitions are written to.
DEBUG_AGENT_FOLDER = "agent"
DEBUG_TOOL_DEFINITIONS_FOLDER = "tool-definitions"
TOOL_DEFINITIONS_FILE_NAME = "tool-definitions.json"
DEBUG_SKILLS_FOLDER = "skills"
SKILLS_FILE_NAME = "skills.md"

# On-disk directory the agent's filesystem tools are rooted at (virtual mode).
AGENT_BACKEND_ROOT_DIR = Path(".storage/agent/backend")
# Defaults for the agent's file logging (see LoggingConfig).
AGENT_DEFAULT_LOGS_DIR = Path(".storage/agent/logs")
AGENT_DEFAULT_LOG_LEVEL = "INFO"
AGENT_DEFAULT_LOG_ROTATION_WHEN = "midnight"
AGENT_DEFAULT_LOG_ROTATION_BACKUP_COUNT = 14
AGENT_DEFAULT_LOG_ROTATION_MAX_SIZE_MB = 10.0
AGENT_LOG_FILE_NAME = "agent.log"
# Maximum size of a single file the agent can read, in megabytes.
AGENT_BACKEND_MAX_FILE_SIZE_MB = 10
# On-disk directory holding skill bundles, laid out as <bundle>/<version>/<skill>.
SKILL_BUNDLES_DIR = Path("assets") / "skill-bundles"
# Directory under the agent's backend root where enabled skill bundles are materialized.
SKILLS_BACKEND_DIR = "skills"

# Env var enabling the dump of rendered configs under BASE_DEBUG_DIR_PATH (True/true/1).
DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR = "DEBUG_DUMP_RENDERED_CONFIGS"
# Env var enabling the archiving of rendered config dumps into the archive folder (True/true/1).
ARCHIVE_RENDERED_CONFIGS_ENV_VAR = "ARCHIVE_RENDERED_CONFIGS"

DEFAULT_CONFIG_NAME = "dev"
# Checkpoint keys for tagging conversations per user.
AGENT_CHECKPOINT_THREAD_ID_KEY = "thread_id"
AGENT_CHECKPOINT_USER_ID_METADATA_KEY = "tech_market_researcher.user_id"

DEFAULT_USER_ID = "codiply"
DEFAULT_TIME_ZONE = "Europe/Madrid"

# Postgres identifiers are capped at 63 bytes, and schema names are interpolated
# into DDL, so they must be plain identifiers within that limit.
POSTGRES_SCHEMA_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
