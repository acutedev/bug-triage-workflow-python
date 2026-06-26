"""Tests for command-line entry point behavior."""

import asyncio
import sys

from openai import OpenAIError

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


def configure_cli_test(monkeypatch):
    logger = CapturingLogger()
    config = make_config()
    monkeypatch.setattr(main_module, "configure_logging", lambda: logger)
    monkeypatch.setattr(main_module, "load_config", lambda: config)
    monkeypatch.setattr(sys, "stdin", _InteractiveTty())
    return config, logger


def run_cli(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = ["--demo"]
    return asyncio.run(main_module.main(argv))


def test_main_successful_run_returns_zero(monkeypatch, capsys):
    config, logger = configure_cli_test(monkeypatch)
    captured: dict[str, AppConfig | None] = {}

    async def fake_run_demo(config_arg=None):
        captured["config"] = config_arg

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    assert exit_code == 0
    assert captured["config"] is config
    assert logger.exceptions == []
    assert capsys.readouterr().err == ""


def test_main_configuration_error_returns_two(monkeypatch, capsys):
    _, logger = configure_cli_test(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda: (_ for _ in ()).throw(ValueError("LLM_API_KEY is required")),
    )

    async def fake_run_demo(config_arg=None):
        raise AssertionError("run_demo should not run when configuration fails")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    assert exit_code == 2
    assert "Configuration error: LLM_API_KEY is required" in capsys.readouterr().err
    assert logger.exceptions == []


def test_main_unexpected_configuration_error_returns_one_and_logs_exception(
    monkeypatch,
    capsys,
):
    _, logger = configure_cli_test(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda: (_ for _ in ()).throw(
            RuntimeError("configuration loader failed")
        ),
    )

    async def fake_run_demo(config_arg=None):
        raise AssertionError(
            "run_demo should not run when configuration fails"
        )

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    exit_code = run_cli()
    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert (
        "Unexpected error: bug triage workflow failed. See logs for details."
        in stderr
    )
    assert "configuration loader failed" not in stderr
    assert len(logger.exceptions) == 1
    message, exc_info = logger.exceptions[0]
    assert message == "Bug triage CLI failed unexpectedly"
    assert exc_info[0] is RuntimeError
    assert str(exc_info[1]) == "configuration loader failed"
    assert exc_info[2] is not None


def test_main_keyboard_interrupt_returns_130(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    assert exit_code == 130
    assert "Operation cancelled by user." in capsys.readouterr().err


def test_main_eof_error_returns_one(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise EOFError

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    assert exit_code == 1
    assert "Input closed; bug triage workflow cancelled." in capsys.readouterr().err


def test_main_provider_error_returns_one(monkeypatch, capsys):
    configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise OpenAIError("provider unavailable")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    assert exit_code == 1
    assert "Provider error: provider unavailable" in capsys.readouterr().err


def test_main_generic_runtime_error_returns_one_and_logs_exception(
    monkeypatch,
    capsys,
):
    _, logger = configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "Unexpected error: bug triage workflow failed. See logs for details." in stderr
    assert "unexpected failure" not in stderr

    assert len(logger.exceptions) == 1
    message, exc_info = logger.exceptions[0]
    assert message == "Bug triage CLI failed unexpectedly"
    assert exc_info[0] is RuntimeError
    assert str(exc_info[1]) == "unexpected failure"
    assert exc_info[2] is not None
