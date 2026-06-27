"""Tests for logging configuration."""

import io
import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest

from src.logging_config import (
    APP_LOGGER_NAME,
    DIAGNOSTIC_LOGGER_NAME,
    JsonLogFormatter,
    configure_diagnostic_logging,
    configure_logging,
    get_logger,
)


@pytest.fixture
def isolated_application_logger():
    """Isolate the app and diagnostic loggers for tests in this module."""
    saved: dict[str, dict] = {}
    for name in (APP_LOGGER_NAME, DIAGNOSTIC_LOGGER_NAME):
        lg = logging.getLogger(name)
        saved[name] = {
            "level": lg.level,
            "handlers": list(lg.handlers),
            "propagate": lg.propagate,
        }
        for h in saved[name]["handlers"]:
            lg.removeHandler(h)

    yield logging.getLogger(APP_LOGGER_NAME)

    for name in (APP_LOGGER_NAME, DIAGNOSTIC_LOGGER_NAME):
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()
        lg.setLevel(saved[name]["level"])
        lg.propagate = saved[name]["propagate"]
        for h in saved[name]["handlers"]:
            lg.addHandler(h)


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


def test_configure_logging_console_handler_emits_exc_info(
    isolated_application_logger, monkeypatch
):
    """The console handler must NOT strip exc_info — unrelated tracebacks must appear."""
    stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stream)

    configured_logger = configure_logging()

    try:
        raise RuntimeError("unexpected-error-XYZZY")
    except RuntimeError:
        configured_logger.exception("Something went wrong")

    output = stream.getvalue()
    payload = json.loads(output)
    assert "exception" in payload
    assert "unexpected-error-XYZZY" in payload["exception"]
    assert "Traceback" in payload["exception"]


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


# ---------------------------------------------------------------------------
# configure_diagnostic_logging tests
# ---------------------------------------------------------------------------


def test_configure_diagnostic_logging_uses_file_handler(
    isolated_application_logger, tmp_path
):
    diag_path = tmp_path / "diag.log"
    diag_logger = configure_diagnostic_logging(path=diag_path)

    assert diag_logger.name == DIAGNOSTIC_LOGGER_NAME
    assert diag_logger.propagate is False
    assert len(diag_logger.handlers) == 1
    assert isinstance(diag_logger.handlers[0], logging.FileHandler)


def test_configure_diagnostic_logging_retains_full_exc_info(
    isolated_application_logger, tmp_path
):
    """The diagnostic file must include exc_info and traceback text."""
    diag_path = tmp_path / "diag.log"
    diag_logger = configure_diagnostic_logging(path=diag_path)

    try:
        raise RuntimeError("provider-secret-XYZ")
    except RuntimeError:
        diag_logger.exception("Provider error calling classifier service")

    content = diag_path.read_text(encoding="utf-8")
    payload = json.loads(content)
    assert "exception" in payload
    assert "provider-secret-XYZ" in payload["exception"]
    assert "Traceback" in payload["exception"]


def test_configure_diagnostic_logging_does_not_propagate(
    isolated_application_logger, monkeypatch, tmp_path
):
    """Diagnostic logger must not write to the console handler."""
    console_stream = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console_stream)
    configure_logging()  # sets up console handler on the app logger

    diag_path = tmp_path / "diag.log"
    diag_logger = configure_diagnostic_logging(path=diag_path)

    try:
        raise RuntimeError("should-not-reach-console")
    except RuntimeError:
        diag_logger.exception("Provider error")

    assert "should-not-reach-console" not in console_stream.getvalue()


def test_configure_diagnostic_logging_default_path_uses_tempdir(
    isolated_application_logger,
):
    """When no path is given, the handler writes to the OS temp directory."""
    import os

    diag_logger = configure_diagnostic_logging()

    assert len(diag_logger.handlers) == 1
    handler = diag_logger.handlers[0]
    assert isinstance(handler, logging.FileHandler)
    # Default filename must be process-specific (contain PID) and in tempdir.
    base = Path(handler.baseFilename)
    assert base.parent == Path(tempfile.gettempdir())
    assert str(os.getpid()) in base.name


def test_configure_diagnostic_logging_is_idempotent(
    isolated_application_logger, tmp_path
):
    existing_handler = logging.FileHandler(tmp_path / "diag.log")
    diag_logger = logging.getLogger(DIAGNOSTIC_LOGGER_NAME)
    diag_logger.addHandler(existing_handler)

    configure_diagnostic_logging(path=tmp_path / "other.log", level=logging.WARNING)

    assert diag_logger.handlers == [existing_handler]
    assert diag_logger.level == logging.WARNING
    assert existing_handler.level == logging.WARNING


# ---------------------------------------------------------------------------
# File permissions (0600)
# ---------------------------------------------------------------------------


def test_diagnostic_file_default_mode_is_0600(isolated_application_logger, tmp_path):
    """Default diagnostic file must be created with mode 0600."""
    import os

    diag_path = tmp_path / "diag_default.log"
    configure_diagnostic_logging(path=diag_path)

    # Emit one record so the file is created.
    logging.getLogger(DIAGNOSTIC_LOGGER_NAME).debug("ping")

    stat = os.stat(diag_path)
    assert oct(stat.st_mode & 0o777) == oct(0o600)


def test_diagnostic_file_override_path_mode_is_0600(
    isolated_application_logger, tmp_path
):
    """An explicit override path must also be created with mode 0600."""
    import os

    diag_path = tmp_path / "custom_diag.log"
    configure_diagnostic_logging(path=diag_path)
    logging.getLogger(DIAGNOSTIC_LOGGER_NAME).debug("ping")

    stat = os.stat(diag_path)
    assert oct(stat.st_mode & 0o777) == oct(0o600)


# ---------------------------------------------------------------------------
# Startup resilience
# ---------------------------------------------------------------------------


def test_invalid_path_falls_back_to_null_handler(isolated_application_logger):
    """An unwritable/invalid path must return a logger with a NullHandler."""
    diag_logger = configure_diagnostic_logging(path="/nonexistent/no/such/dir/diag.log")

    assert diag_logger.name == DIAGNOSTIC_LOGGER_NAME
    assert diag_logger.propagate is False
    assert len(diag_logger.handlers) == 1
    assert isinstance(diag_logger.handlers[0], logging.NullHandler)


def test_parent_is_file_falls_back_to_null_handler(
    isolated_application_logger, tmp_path, monkeypatch
):
    """Parent path that is a regular file (not a directory) must install NullHandler."""
    import io

    # Create a regular file where the parent directory would be.
    fake_parent = tmp_path / "not_a_dir"
    fake_parent.write_text("i am a file")
    diag_path = fake_parent / "diag.log"

    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    diag_logger = configure_diagnostic_logging(path=diag_path)

    assert len(diag_logger.handlers) == 1
    assert isinstance(diag_logger.handlers[0], logging.NullHandler)
    assert not diag_path.exists()
    assert "--- Logging error ---" not in stderr_buf.getvalue()

    # A later call with a valid path must replace the NullHandler.
    valid_path = tmp_path / "diag.log"
    configure_diagnostic_logging(path=valid_path)
    assert len(diag_logger.handlers) == 1
    assert isinstance(diag_logger.handlers[0], logging.FileHandler)


def test_fallback_logger_does_not_emit_to_console(
    isolated_application_logger, monkeypatch
):
    """Fallback NullHandler logger must not write anything to stdout/stderr."""
    import io

    console = io.StringIO()
    monkeypatch.setattr(sys, "stdout", console)
    monkeypatch.setattr(sys, "stderr", console)

    diag_logger = configure_diagnostic_logging(path="/nonexistent/no/such/dir/diag.log")
    try:
        raise RuntimeError("should-not-appear-anywhere")
    except RuntimeError:
        diag_logger.exception("Provider error")

    assert "should-not-appear-anywhere" not in console.getvalue()


# ---------------------------------------------------------------------------
# Bounded, non-shared diagnostics
# ---------------------------------------------------------------------------


def test_default_filename_is_process_specific(isolated_application_logger):
    """Default diagnostic filename must embed the current PID."""
    import os

    diag_logger = configure_diagnostic_logging()
    handler = diag_logger.handlers[0]
    assert str(os.getpid()) in Path(handler.baseFilename).name


def test_diagnostic_handler_is_rotating_with_bounded_size(
    isolated_application_logger, tmp_path
):
    """Diagnostic handler must be a RotatingFileHandler with size and backup limits."""
    from logging.handlers import RotatingFileHandler

    diag_path = tmp_path / "diag.log"
    diag_logger = configure_diagnostic_logging(path=diag_path)

    handler = diag_logger.handlers[0]
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes > 0
    assert handler.backupCount > 0


def test_preexisting_0644_diagnostic_file_hardened_to_0600(
    isolated_application_logger, tmp_path
):
    """A pre-existing diagnostic file with mode 0644 must be hardened to 0600."""
    import os
    import stat

    diag_path = tmp_path / "diag.log"
    diag_path.write_text("old content\n")
    os.chmod(diag_path, 0o644)
    assert stat.S_IMODE(os.stat(diag_path).st_mode) == 0o644

    diag_logger = configure_diagnostic_logging(path=diag_path)
    diag_logger.warning("probe record")

    mode = stat.S_IMODE(os.stat(diag_path).st_mode)
    assert mode == 0o600, f"Expected 0600, got {oct(mode)}"
    assert diag_path.read_text().__contains__("probe record")


# ---------------------------------------------------------------------------
# Regression: sticky NullHandler fallback (defect 1)
# ---------------------------------------------------------------------------


def test_null_handler_replaced_when_valid_path_provided_later(
    isolated_application_logger, tmp_path
):
    """NullHandler installed for an invalid path must be replaced on a later valid call."""
    configure_diagnostic_logging(path="/nonexistent/no/such/dir/diag.log")
    diag_logger = logging.getLogger(DIAGNOSTIC_LOGGER_NAME)
    assert isinstance(diag_logger.handlers[0], logging.NullHandler)

    diag_path = tmp_path / "diag.log"
    configure_diagnostic_logging(path=diag_path)

    assert len(diag_logger.handlers) == 1
    assert not isinstance(diag_logger.handlers[0], logging.NullHandler)
    assert isinstance(diag_logger.handlers[0], logging.FileHandler)
    assert diag_logger.propagate is False


def test_valid_handler_remains_idempotent_on_repeated_configure(
    isolated_application_logger, tmp_path
):
    """A real file handler must not be duplicated by a second valid configure call."""
    diag_path = tmp_path / "diag.log"
    configure_diagnostic_logging(path=diag_path)
    configure_diagnostic_logging(path=diag_path)

    diag_logger = logging.getLogger(DIAGNOSTIC_LOGGER_NAME)
    assert len(diag_logger.handlers) == 1
    assert isinstance(diag_logger.handlers[0], logging.FileHandler)


# ---------------------------------------------------------------------------
# Regression: premature diagnostic-file creation (defect 2)
# ---------------------------------------------------------------------------


def test_handler_does_not_create_file_before_first_emit(
    isolated_application_logger, tmp_path
):
    """Installing the handler must not create the diagnostic file."""
    diag_path = tmp_path / "diag.log"
    configure_diagnostic_logging(path=diag_path)

    assert not diag_path.exists(), "Diagnostic file must not be created before first emit"


def test_first_emit_creates_file_with_mode_0600(
    isolated_application_logger, tmp_path
):
    """First log record must create the file; it must have mode 0600."""
    import os

    diag_path = tmp_path / "diag.log"
    configure_diagnostic_logging(path=diag_path)

    assert not diag_path.exists()
    logging.getLogger(DIAGNOSTIC_LOGGER_NAME).debug("first record")
    assert diag_path.exists()
    assert oct(os.stat(diag_path).st_mode & 0o777) == oct(0o600)


def test_invalid_delayed_path_nonfatal_and_no_stderr_output(
    isolated_application_logger, tmp_path, monkeypatch
):
    """A delayed open failure at emit time must not write to stderr."""
    import io

    diag_dir = tmp_path / "subdir"
    diag_dir.mkdir()
    diag_path = diag_dir / "diag.log"
    configure_diagnostic_logging(path=diag_path)

    # Remove the directory so the delayed open fails at emit time.
    diag_dir.rmdir()

    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    logging.getLogger(DIAGNOSTIC_LOGGER_NAME).debug("probe")

    stderr_out = stderr_buf.getvalue()
    assert "--- Logging error ---" not in stderr_out
    assert "Traceback" not in stderr_out
