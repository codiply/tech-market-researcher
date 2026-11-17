"""File-based logging setup for the agent."""

import logging
from logging.handlers import RotatingFileHandler

from tech_market_researcher.core.configuration import LoggingConfig
from tech_market_researcher.core.constants import AGENT_LOG_FILE_NAME

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def setup_agent_logging(config: LoggingConfig) -> None:
    """Attach a rotating file handler, writing the agent's logs to <logging.directory>/agent.log.

    No stream handler is added, so the logs never interfere with the console output of the
    application frontends. The configured level applies to the tech_market_researcher package;
    third-party loggers are capped at INFO, so their debug output (which can contain request
    URLs and other details) is not captured. Safe to call multiple times; only the first call takes effect.
    """
    log_file = (config.directory / AGENT_LOG_FILE_NAME).resolve()
    root = logging.getLogger()
    already_configured = any(
        isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(log_file) for handler in root.handlers
    )
    if already_configured:
        return

    config.directory.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_file, maxBytes=_MAX_FILE_SIZE_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("tech_market_researcher").setLevel(logging.getLevelNamesMapping()[config.level.upper()])
