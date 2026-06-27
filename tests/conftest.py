"""Shared test helpers for the bug triage test suite."""

from __future__ import annotations

import io
import logging

import pytest

from src.logging_config import APP_LOGGER_NAME, DIAGNOSTIC_LOGGER_NAME


class _InteractiveTty(io.IOBase):
    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        raise EOFError


@pytest.fixture
def isolated_app_logger():
    """Remove and restore app and diagnostic logger handlers around a test.

    Prevents tests that configure real logging from leaking handlers (and their
    underlying streams) into subsequent tests.
    """
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
