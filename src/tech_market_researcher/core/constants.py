"""Application-wide constants."""

from importlib.metadata import version
from pathlib import Path

PACKAGE_VERSION = version("tech-market-researcher")
PROJECT_NAME = "tech-market-researcher"

BASE_DEBUG_DIR_PATH = Path(".debug")

# Env var enabling the dump of rendered configs under BASE_DEBUG_DIR_PATH (True/true/1).
DEBUG_DUMP_RENDERED_CONFIGS_ENV_VAR = "DEBUG_DUMP_RENDERED_CONFIGS"

DEFAULT_CONFIG_NAME = "dev"

DEFAULT_USER_ID = "codiply"
DEFAULT_SESSION_ID = "session-0001"
DEFAULT_TIME_ZONE = "Europe/Madrid"
