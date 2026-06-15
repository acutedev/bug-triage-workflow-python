"""Tests for logging configuration."""

import io
import json
import logging
import sys

import pytest

from src.logging_config import (
    APP_LOGGER_NAME,
    JsonLogFormatter,
    configure_logging,
    get_logger,
)


@pytest.fixture
def isolated_application_logger():
    logger = logging.getLogger(APP_LOGGER_NAME)
    original_level = logger.level
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate

    for handler in original_handlers:
        logger.removeHandler(handler)

    yield logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(original_level)
    logger.propagate = original_propagate
    for handler in original_handlers:
        logger.addHandler(handler)


def test_configure_logging_configures_only_application_logger(
    isolated_application_logger, monkeypatch
):
    root_logger = logging.getLogger()
    original_root_level = root_logger.level
    original_root_handlers = list(root_logger.handlers)

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    configured_logger = configure_logging(logging.DEBUG)

    assert configured_logger is isolated_application_logger
    assert configured_logger.name == APP_LOGGER_NAME
    assert configured_logger.level == logging.DEBUG
    assert configured_logger.propagate is False
    assert len(configured_logger.handlers) == 1

    handler = configured_logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.level == logging.DEBUG
    assert isinstance(handler.formatter, JsonLogFormatter)

    assert root_logger.level == original_root_level
    assert list(root_logger.handlers) == original_root_handlers

    configured_logger.debug(
        "Workflow started",
        extra={"executor": "workflow", "status": "received"},
    )

    payload = json.loads(stream.getvalue())
    assert payload["level"] == "DEBUG"
    assert payload["logger"] == APP_LOGGER_NAME
    assert payload["message"] == "Workflow started"
    assert payload["extra"] == {
        "executor": "workflow",
        "status": "received",
    }


def test_configure_logging_is_idempotent_when_handlers_already_exist(
    isolated_application_logger,
):
    existing_handler = logging.NullHandler()
    isolated_application_logger.addHandler(existing_handler)

    configured_logger = configure_logging(logging.ERROR)

    assert configured_logger is isolated_application_logger
    assert isolated_application_logger.level == logging.ERROR
    assert isolated_application_logger.handlers == [existing_handler]
    assert existing_handler.level == logging.ERROR
    assert isinstance(existing_handler.formatter, JsonLogFormatter)


def test_get_logger_returns_application_logger_when_name_is_none():
    logger = get_logger()
    assert logger is logging.getLogger(APP_LOGGER_NAME)


def test_get_logger_returns_child_logger_under_application_namespace():
    logger = get_logger("workflow")
    assert logger is logging.getLogger(f"{APP_LOGGER_NAME}.workflow")


def test_get_logger_does_not_duplicate_application_prefix():
    logger = get_logger(f"{APP_LOGGER_NAME}.workflow")
    assert logger is logging.getLogger(f"{APP_LOGGER_NAME}.workflow")


@pytest.mark.parametrize("name", ["", "   "])
def test_get_logger_rejects_empty_names(name):
    with pytest.raises(ValueError, match="logger name must not be empty"):
        get_logger(name)


def test_json_log_formatter_outputs_expected_fields():
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name=f"{APP_LOGGER_NAME}.workflow",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Workflow started",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "INFO"
    assert payload["logger"] == f"{APP_LOGGER_NAME}.workflow"
    assert payload["message"] == "Workflow started"
    assert "timestamp" in payload


def test_json_log_formatter_includes_extra_fields():
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name=f"{APP_LOGGER_NAME}.workflow",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Human review required",
        args=(),
        exc_info=None,
    )
    record.executor = "policy_router"
    record.route = "request_human_approval"

    formatted = formatter.format(record)
    payload = json.loads(formatted)

    assert payload["level"] == "WARNING"
    assert payload["extra"] == {
        "executor": "policy_router",
        "route": "request_human_approval",
    }
