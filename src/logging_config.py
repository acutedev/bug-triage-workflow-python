"""Logging configuration for the Bug Triage Workflow.

This module centralizes JSON logger setup so workflow executors can log
consistently without duplicating formatting code or modifying the global
root logger.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

APP_LOGGER_NAME = "bug_triage_workflow"

# Separate logger for full provider-error diagnostics. propagate=False keeps it
# off the console; it writes only to a file handler configured at startup.
DIAGNOSTIC_LOGGER_NAME = f"{APP_LOGGER_NAME}.provider_diagnostic"

RESERVED_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in RESERVED_LOG_RECORD_KEYS and not key.startswith("_")
        }
        if extra_fields:
            log_payload["extra"] = extra_fields

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log_payload, default=str, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the application logger for JSON console output.

    This function intentionally configures only the application logger, not the
    global root logger. It is idempotent: repeated calls update existing handler
    levels and formatters without attaching duplicate handlers.
    """
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = JsonLogFormatter()

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def _secure_opener(path: str, flags: int) -> int:
    """Open a file with mode 0600, hardening pre-existing files to the same mode."""
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
    except Exception:
        os.close(fd)
        raise
    return fd


class _SecureRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that creates files with mode 0600."""

    def _open(self):
        return open(
            self.baseFilename,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            opener=_secure_opener,
        )


def configure_diagnostic_logging(
    path: str | Path | None = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure a file-only logger for full provider-error diagnostic output.

    This logger does not propagate, so tracebacks and raw provider exception
    details are written exclusively to the diagnostic file and never reach the
    console handler.

    When path is None the log is written to a process-specific path in the OS
    temp directory (contains the current PID so concurrent processes don't
    share a file). Set BUG_TRIAGE_DIAGNOSTIC_LOG to override.

    If the path cannot be opened the logger falls back to a NullHandler so
    the CLI can still start. It is idempotent: repeated calls update existing
    handler levels and formatters without attaching duplicates.
    """
    if path is None:
        resolved = Path(tempfile.gettempdir()) / f"bug_triage_diagnostic_{os.getpid()}.log"
    else:
        resolved = Path(path)

    logger = logging.getLogger(DIAGNOSTIC_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = JsonLogFormatter()

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
        return logger

    try:
        file_handler = _SecureRotatingFileHandler(
            resolved,
            maxBytes=1_000_000,
            backupCount=2,
            encoding="utf-8",
        )
    except Exception:
        null_handler = logging.NullHandler()
        logger.addHandler(null_handler)
        return logger

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the application logger or a child logger under it."""
    if name is None:
        return logging.getLogger(APP_LOGGER_NAME)

    if not name.strip():
        raise ValueError("logger name must not be empty")

    clean_name = name.strip()
    if clean_name == APP_LOGGER_NAME or clean_name.startswith(f"{APP_LOGGER_NAME}."):
        return logging.getLogger(clean_name)

    return logging.getLogger(f"{APP_LOGGER_NAME}.{clean_name}")
