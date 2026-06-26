"""Shared test helpers for the bug triage test suite."""

from __future__ import annotations

import io


class _InteractiveTty(io.IOBase):
    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        raise EOFError
