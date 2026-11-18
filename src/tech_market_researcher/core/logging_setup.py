"""File-based logging setup for the agent."""

import logging
import os
import re
import time
from logging.handlers import TimedRotatingFileHandler

from tech_market_researcher.core.configuration import LoggingConfig
from tech_market_researcher.core.constants import AGENT_LOG_FILE_NAME

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class SizeAndDateRotatingFileHandler(TimedRotatingFileHandler):
    """Rotates the log file at a fixed time of day and also caps each file's size.

    A size-triggered rollover within the same interval appends an incrementing
    counter to the date suffix (e.g. agent.log.2026-01-21.1); such files are
    included in the backup-count cleanup.
    """

    def __init__(self, filename: str, when: str, backup_count: int, max_bytes: int, encoding: str | None) -> None:
        super().__init__(filename, when=when, backupCount=backup_count, encoding=encoding)
        self._max_bytes = max_bytes

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if super().shouldRollover(record):
            return True
        if self._max_bytes <= 0 or self.stream is None:
            return False
        message = self.format(record).encode(self.encoding or "utf-8", "replace")
        return self.stream.tell() + len(message) >= self._max_bytes

    def doRollover(self) -> None:
        t = self.rolloverAt - self.interval
        time_tuple = time.gmtime(t) if self.utc else time.localtime(t)
        base_name = self.baseFilename + "." + time.strftime(self.suffix, time_tuple)
        destination = base_name
        counter = 1
        while os.path.exists(destination):
            destination = f"{base_name}.{counter}"
            counter += 1
        if self.stream:
            self.stream.close()
            self.stream = None
        self.rotate(self.baseFilename, destination)
        if self.backupCount > 0:
            for file in self.getFilesToDelete():
                os.remove(file)
        if not self.delay:
            self.stream = self._open()
        self.rolloverAt = self.computeRollover(int(time.time()))

    def getFilesToDelete(self) -> list[str]:
        # The parent only matches plain date suffixes; here we also accept the
        # counter suffix added by size-triggered rollovers within one interval.
        counter_suffix = re.compile(r"\.\d+$")
        prefix = os.path.basename(self.baseFilename) + "."
        result = []
        for name in os.listdir(os.path.dirname(self.baseFilename)):
            if not name.startswith(prefix):
                continue
            counter_match = counter_suffix.fullmatch(name[len(prefix) :])
            if counter_match:
                counter = int(counter_match[0][1:])
            elif self.extMatch.fullmatch(name[len(prefix) :]):
                counter = 0
            else:
                continue
            # Date first, then counter, so .2 sorts before .10.
            result.append((name[len(prefix) :], counter, os.path.join(os.path.dirname(self.baseFilename), name)))
        if len(result) < self.backupCount:
            return []
        result.sort()
        return [path for _, _, path in result[: len(result) - self.backupCount]]


def setup_agent_logging(config: LoggingConfig) -> None:
    """Attach a file handler to the root logger, writing the agent's logs to <logging.directory>/agent.log.

    The log file rolls over on the schedule given by <logging.rotation_when> and when it
    exceeds <logging.rotation_max_size_mb> MB; the newest <logging.rotation_backup_count>
    rotated files are kept, older ones deleted.

    No stream handler is added, so the logs never interfere with the console output of the
    application frontends. The configured level applies to the tech_market_researcher package;
    third-party loggers are capped at INFO, so their debug output (which can contain request
    URLs and other details) is not captured. Safe to call multiple times; only the first call takes effect.
    """
    log_file = (config.directory / AGENT_LOG_FILE_NAME).resolve()
    root = logging.getLogger()
    already_configured = any(
        isinstance(handler, SizeAndDateRotatingFileHandler) and handler.baseFilename == str(log_file)
        for handler in root.handlers
    )
    if already_configured:
        return

    config.directory.mkdir(parents=True, exist_ok=True)
    handler = SizeAndDateRotatingFileHandler(
        str(log_file),
        when=config.rotation_when,
        backup_count=config.rotation_backup_count,
        max_bytes=int(config.rotation_max_size_mb * 1024 * 1024),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("tech_market_researcher").setLevel(logging.getLevelNamesMapping()[config.level.upper()])
