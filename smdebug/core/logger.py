# Standard Library
import logging
import os
import socket
import sys
from collections import defaultdict

# First Party
from smdebug.core.config_constants import LOG_DUPLICATION_THRESHOLD

_logger_initialized = False


class MaxLevelFilter(logging.Filter):
    """Filters (lets through) all messages with level < LEVEL"""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        # "<" instead of "<=": since logger.setLevel is inclusive, this should be exclusive
        return record.levelno < self.level


class DuplicateLogFilter:
    """Filters duplicate messages to prevent spamming users"""

    def __init__(self):
        self.msgs = defaultdict(int)
        self.repeat_threshold = LOG_DUPLICATION_THRESHOLD

    def filter(self, record):
        self.msgs[record.msg] += 1
        return self.msgs[record.msg] <= self.repeat_threshold


def _get_log_level():
    default = "info"
    log_level = os.environ.get("SMDEBUG_LOG_LEVEL", default=default)
    log_level = log_level.lower()
    allowed_levels = ["info", "debug", "warning", "error", "critical", "off"]
    if log_level not in allowed_levels:
        log_level = default

    level = None
    if log_level is None or log_level == "off":
        level = None
    else:
        if log_level == "critical":
            level = logging.CRITICAL
        elif log_level == "error":
            level = logging.ERROR
        elif log_level == "warning":
            level = logging.WARNING
        elif log_level == "info":
            level = logging.INFO
        elif log_level == "debug":
            level = logging.DEBUG
    return level


def get_logger(name="smdebug"):
    global _logger_initialized
    if not _logger_initialized:
        worker_pid = f"{socket.gethostname()}:{os.getpid()}"
        log_context = os.environ.get("SMDEBUG_LOG_CONTEXT", default=worker_pid)
        level = _get_log_level()
        logger = logging.getLogger(name)

        logger.handlers = []
        log_formatter = logging.Formatter(
            fmt="[%(asctime)s.%(msecs)03d "
            + log_context
            + " %(levelname)s %(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_formatter)

        if os.environ.get("SMDEBUG_LOG_ALL_TO_STDOUT", default="TRUE").lower() == "false":
            stderr_handler = logging.StreamHandler(sys.stderr)
            min_level = logging.DEBUG
            # lets through all levels less than ERROR
            stdout_handler.addFilter(MaxLevelFilter(logging.ERROR))
            stdout_handler.setLevel(min_level)

            stderr_handler.setLevel(max(min_level, logging.ERROR))
            stderr_handler.setFormatter(log_formatter)
            logger.addHandler(stderr_handler)

        logger.addHandler(stdout_handler)
        logger.addFilter(DuplicateLogFilter())

        # SMDEBUG_LOG_PATH is the full path to log file
        # by default, log is only written to stdout&stderr
        # if this is set, it is written to file
        path = os.environ.get("SMDEBUG_LOG_PATH", default=None)
        if path is not None:
            fh = logging.FileHandler(path)
            fh.setFormatter(log_formatter)
            logger.addHandler(fh)

        if level:
            logger.setLevel(level)
        else:
            logger.disabled = True
        logger.propagate = False
        _logger_initialized = True
    return logging.getLogger(name)
