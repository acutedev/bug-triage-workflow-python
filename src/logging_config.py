"""Logging configuration for the Bug Triage Workflow.

This module centralizes JSON logger setup so workflow executors can log
consistently without duplicating formatting code or modifying the global
root logger.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

APP_LOGGER_NAME = "bug_triage_workflow"

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
