"""Tests for CLI argument parsing and input resolution."""

from __future__ import annotations

import asyncio
import io
import sys

import pytest

from src import main as main_module
from src.config import AppConfig
from tests.conftest import _InteractiveTty


class CapturingLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, tuple[object, object, object]]] = []

    def exception(self, message: str) -> None:
        self.exceptions.append((message, sys.exc_info()))


def make_config() -> AppConfig:
    return AppConfig(
        llm_provider="openai",
        llm_api_key="test-key",
        llm_model="gpt-test",
        human_approval_enabled=True,
    )


def configure_cli_test(monkeypatch, *, piped_stdin: bool = False):
    logger = CapturingLogger()
    config = make_config()
    monkeypatch.setattr(main_module, "configure_logging", lambda **_: logger)
    monkeypatch.setattr(main_module, "configure_diagnostic_logging", lambda **_: logger)
    monkeypatch.setattr(main_module, "load_config", lambda: config)
    if not piped_stdin:
        monkeypatch.setattr(sys, "stdin", _InteractiveTty())
    return config, logger


def run_cli(argv: list[str]) -> int:
    return asyncio.run(main_module.main(argv))


# ---------------------------------------------------------------------------
# --demo flag
# ---------------------------------------------------------------------------


def test_demo_flag_calls_run_demo(monkeypatch):
    config, _ = configure_cli_test(monkeypatch)
    captured: dict = {}

    async def fake_run_demo(config_arg=None):
        captured["config"] = config_arg

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli(["--demo"])

    assert exit_code == 0
    assert captured["config"] is config


# ---------------------------------------------------------------------------
# --text flag
# ---------------------------------------------------------------------------


def test_text_flag_calls_run_with_report(monkeypatch):
    config, _ = configure_cli_test(monkeypatch)
    captured: dict = {}

    async def fake_run_with_report(report_text, config_arg):
        captured["text"] = report_text
        captured["config"] = config_arg

    monkeypatch.setattr(main_module, "run_with_report", fake_run_with_report)

    exit_code = run_cli(["--text", "The checkout button crashes in Chrome."])

    assert exit_code == 0
    assert captured["text"] == "The checkout button crashes in Chrome."
    assert captured["config"] is config


def test_text_flag_strips_surrounding_whitespace(monkeypatch):
    configure_cli_test(monkeypatch)
    captured: dict = {}

    async def fake_run_with_report(report_text, config_arg):
        captured["text"] = report_text

    monkeypatch.setattr(main_module, "run_with_report", fake_run_with_report)

    report = "  Login fails with 500 on prod.\nSteps: open /login, click submit.\n"
    exit_code = run_cli(["--text", report])

    assert exit_code == 0
    assert captured["text"] == report.strip()


# ---------------------------------------------------------------------------
# --file flag
# ---------------------------------------------------------------------------


def test_file_flag_reads_file_and_calls_run_with_report(monkeypatch, tmp_path):
    config, _ = configure_cli_test(monkeypatch)
    captured: dict = {}

    report = "login fails on Firefox"
    report_file = tmp_path / "report.txt"
    report_file.write_text(report)

    async def fake_run_with_report(report_text, config_arg):
        captured["text"] = report_text
        captured["config"] = config_arg

    monkeypatch.setattr(main_module, "run_with_report", fake_run_with_report)

    exit_code = run_cli(["--file", str(report_file)])

    assert exit_code == 0
    assert captured["text"] == report
    assert captured["config"] is config


def test_file_flag_strips_surrounding_whitespace(monkeypatch, tmp_path):
    configure_cli_test(monkeypatch)
    captured: dict = {}

    report_file = tmp_path / "report.txt"
    report_file.write_text("\n\n  crash on startup  \n\n")

    async def fake_run_with_report(report_text, config_arg):
        captured["text"] = report_text

    monkeypatch.setattr(main_module, "run_with_report", fake_run_with_report)

    run_cli(["--file", str(report_file)])

    assert captured["text"] == "crash on startup"


# ---------------------------------------------------------------------------
# stdin piped
# ---------------------------------------------------------------------------


def test_stdin_piped_calls_run_with_report(monkeypatch):
    config, _ = configure_cli_test(monkeypatch, piped_stdin=True)
    captured: dict = {}

    async def fake_run_with_report(report_text, config_arg):
        captured["text"] = report_text
        captured["config"] = config_arg

    monkeypatch.setattr(main_module, "run_with_report", fake_run_with_report)
    monkeypatch.setattr(sys, "stdin", io.StringIO("crash on startup"))

    exit_code = run_cli([])

    assert exit_code == 0
    assert captured["text"] == "crash on startup"
    assert captured["config"] is config


def test_stdin_piped_strips_surrounding_whitespace(monkeypatch):
    configure_cli_test(monkeypatch, piped_stdin=True)
    captured: dict = {}

    async def fake_run_with_report(report_text, config_arg):
        captured["text"] = report_text

    monkeypatch.setattr(main_module, "run_with_report", fake_run_with_report)
    monkeypatch.setattr(sys, "stdin", io.StringIO("  button broken  \n"))

    run_cli([])

    assert captured["text"] == "button broken"


# ---------------------------------------------------------------------------
# No input with interactive stdin → usage error
# ---------------------------------------------------------------------------


def test_no_input_interactive_stdin_returns_usage_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _InteractiveTty())

    exit_code = run_cli([])

    assert exit_code == 2
    assert capsys.readouterr().err.strip()


def test_no_input_usage_error_message_mentions_options(monkeypatch, capsys):
    configure_cli_test(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _InteractiveTty())

    run_cli([])

    stderr = capsys.readouterr().err
    assert any(opt in stderr for opt in ("--demo", "--text", "--file"))


# ---------------------------------------------------------------------------
# Conflicting input sources
# ---------------------------------------------------------------------------


def test_demo_and_text_together_returns_usage_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    exit_code = run_cli(["--demo", "--text", "something"])

    assert exit_code == 2


def test_demo_and_file_together_returns_usage_error(monkeypatch, capsys, tmp_path):
    configure_cli_test(monkeypatch)
    f = tmp_path / "r.txt"
    f.write_text("something")

    exit_code = run_cli(["--demo", "--file", str(f)])

    assert exit_code == 2


def test_text_and_file_together_returns_usage_error(monkeypatch, capsys, tmp_path):
    configure_cli_test(monkeypatch)
    f = tmp_path / "r.txt"
    f.write_text("something")

    exit_code = run_cli(["--text", "something", "--file", str(f)])

    assert exit_code == 2


# ---------------------------------------------------------------------------
# Empty / whitespace-only input
# ---------------------------------------------------------------------------


def test_empty_text_flag_returns_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    exit_code = run_cli(["--text", ""])

    assert exit_code == 2
    assert capsys.readouterr().err.strip()


def test_whitespace_only_text_flag_returns_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    exit_code = run_cli(["--text", "   \n\t  "])

    assert exit_code == 2
    assert capsys.readouterr().err.strip()


def test_empty_file_returns_error(monkeypatch, capsys, tmp_path):
    configure_cli_test(monkeypatch)
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ")

    exit_code = run_cli(["--file", str(f)])

    assert exit_code == 2
    assert capsys.readouterr().err.strip()


def test_piped_empty_stdin_returns_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("  \n  "))

    exit_code = run_cli([])

    assert exit_code == 2
    assert capsys.readouterr().err.strip()


# ---------------------------------------------------------------------------
# File not found / unreadable
# ---------------------------------------------------------------------------


def test_file_not_found_returns_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    exit_code = run_cli(["--file", "/nonexistent/path/to/report.txt"])

    assert exit_code == 2
    assert capsys.readouterr().err.strip()


def test_file_not_found_error_mentions_path(monkeypatch, capsys):
    configure_cli_test(monkeypatch)
    path = "/nonexistent/path/to/report.txt"

    run_cli(["--file", path])

    assert path in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Piped stdin combined with an explicit input flag → mutual exclusivity error
# ---------------------------------------------------------------------------


def test_piped_stdin_with_demo_flag_returns_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin report"))

    exit_code = run_cli(["--demo"])

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert stderr.strip()


def test_piped_stdin_with_demo_flag_error_message_is_useful(monkeypatch, capsys):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin report"))

    run_cli(["--demo"])

    stderr = capsys.readouterr().err
    assert "--demo" in stderr or "stdin" in stderr.lower()


def test_piped_stdin_with_text_flag_returns_error(monkeypatch, capsys):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin report"))

    exit_code = run_cli(["--text", "flag report"])

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert stderr.strip()


def test_piped_stdin_with_text_flag_error_message_is_useful(monkeypatch, capsys):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin report"))

    run_cli(["--text", "flag report"])

    stderr = capsys.readouterr().err
    assert "--text" in stderr or "stdin" in stderr.lower()


def test_piped_stdin_with_file_flag_returns_error(monkeypatch, capsys, tmp_path):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin report"))
    f = tmp_path / "report.txt"
    f.write_text("file report")

    exit_code = run_cli(["--file", str(f)])

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert stderr.strip()


def test_piped_stdin_with_file_flag_error_message_is_useful(monkeypatch, capsys, tmp_path):
    configure_cli_test(monkeypatch, piped_stdin=True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("stdin report"))
    f = tmp_path / "report.txt"
    f.write_text("file report")

    run_cli(["--file", str(f)])

    stderr = capsys.readouterr().err
    assert "--file" in stderr or "stdin" in stderr.lower()
