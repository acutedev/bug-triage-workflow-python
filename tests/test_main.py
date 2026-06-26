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
        self.errors: list[str] = []

    def exception(self, message: str) -> None:
        self.exceptions.append((message, sys.exc_info()))

    def error(self, message: str) -> None:
        self.errors.append(message)


def make_config() -> AppConfig:
    return AppConfig(
        llm_provider="openai",
        llm_api_key="test-key",
        llm_model="gpt-test",
        human_approval_enabled=True,
    )


def configure_cli_test(monkeypatch):
    logger = CapturingLogger()
    diag_logger = CapturingLogger()
    # Attach so tests that need diagnostic details can reach it via logger.diagnostic
    logger.diagnostic = diag_logger
    config = make_config()
    monkeypatch.setattr(main_module, "configure_logging", lambda **_: logger)
    monkeypatch.setattr(
        main_module, "configure_diagnostic_logging", lambda **_: diag_logger
    )
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


_NORMALIZED_PROVIDER_MSG = (
    "Provider error: the classifier service could not complete the request."
    " See logs for details."
)


def test_main_provider_error_returns_one(monkeypatch, capsys):
    _, logger = configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise OpenAIError("provider unavailable")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert _NORMALIZED_PROVIDER_MSG in stderr
    assert "provider unavailable" not in stderr


def test_provider_error_logs_full_exception_details(monkeypatch, capsys):
    """Full provider exception detail must go to the diagnostic logger, not stderr."""
    _, logger = configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise OpenAIError("secret-detail: connection refused to https://api.openai.com/v1")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    run_cli()

    # Diagnostic logger must capture the full exception with exc_info.
    diag = logger.diagnostic
    assert len(diag.exceptions) == 1
    message, exc_info = diag.exceptions[0]
    assert "provider" in message.lower() or "classifier" in message.lower()
    assert exc_info[0] is OpenAIError
    assert "secret-detail" in str(exc_info[1])
    assert exc_info[2] is not None

    # Main console logger must NOT receive exc_info for the provider error.
    assert len(logger.exceptions) == 0


def test_provider_error_normalized_message_shown(monkeypatch, capsys):
    """The exact normalized provider message must appear on stderr."""
    configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise OpenAIError("any internal detail")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    run_cli()

    assert _NORMALIZED_PROVIDER_MSG in capsys.readouterr().err


def test_provider_error_raw_text_absent_from_output(monkeypatch, capsys):
    """Raw provider exception text must not appear on stdout or stderr."""
    configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise OpenAIError("raw-internal-error-text-12345")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    run_cli()

    out = capsys.readouterr()
    assert "raw-internal-error-text-12345" not in out.out
    assert "raw-internal-error-text-12345" not in out.err


def test_provider_error_no_sensitive_values_in_output(monkeypatch, capsys):
    """API keys or secret-shaped values embedded in the error must not reach output."""
    configure_cli_test(monkeypatch)

    async def fake_run_demo(config_arg=None):
        raise OpenAIError("Unauthorized: sk-proj-SECRETKEY1234567890ABCDEF")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    run_cli()

    out = capsys.readouterr()
    combined = out.out + out.err
    assert "sk-proj-SECRETKEY1234567890ABCDEF" not in combined
    assert "Unauthorized" not in combined


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


def test_help_exits_zero_without_api_key(monkeypatch, capsys):
    """--help must succeed before load_config is ever reached."""
    logger = CapturingLogger()
    monkeypatch.setattr(main_module, "configure_logging", lambda **_: logger)
    monkeypatch.setattr(
        main_module, "configure_diagnostic_logging", lambda **_: CapturingLogger()
    )
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("load_config must not be called for --help")
        ),
    )
    exit_code = asyncio.run(main_module.main(["--help"]))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "--demo" in out
    assert "--text" in out
    assert "--file" in out


def test_help_short_flag_exits_zero_without_api_key(monkeypatch, capsys):
    """-h must succeed before load_config is ever reached."""
    logger = CapturingLogger()
    monkeypatch.setattr(main_module, "configure_logging", lambda **_: logger)
    monkeypatch.setattr(
        main_module, "configure_diagnostic_logging", lambda **_: CapturingLogger()
    )
    monkeypatch.setattr(
        main_module,
        "load_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("load_config must not be called for -h")
        ),
    )
    exit_code = asyncio.run(main_module.main(["-h"]))
    assert exit_code == 0
    assert "--demo" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Real-logging terminal-safety regression
# ---------------------------------------------------------------------------

_SECRET_VALUE = "sk-proj-REALSECRET9876543210ABCDEF"
_ENDPOINT_VALUE = "https://api.openai.com/v1/chat/completions"
_RAW_MARKER = "RAW_PROVIDER_EXCEPTION_MARKER_XYZZY"


def _setup_real_logging_cli_test(monkeypatch, *, diag_path, configure_diag=True):
    """Shared setup for real-logging integration tests.

    Redirects sys.stdout to a StringIO so the StreamHandler captures it instead
    of the real stdout (which would be closed by isolated_app_logger teardown).
    Returns the console StringIO buffer.

    When configure_diag is False, the diagnostic_logging monkeypatch is skipped
    so the caller can install its own.
    """
    import io as _io
    from src.logging_config import (
        configure_diagnostic_logging as real_diag,
        configure_logging as real_configure_logging,
    )

    console_buf = _io.StringIO()
    monkeypatch.setattr(sys, "stdout", console_buf)

    monkeypatch.setattr(main_module, "configure_logging", lambda **_: real_configure_logging())
    if configure_diag:
        monkeypatch.setattr(
            main_module,
            "configure_diagnostic_logging",
            lambda **_: real_diag(path=diag_path),
        )
    monkeypatch.setattr(main_module, "load_config", lambda: make_config())
    monkeypatch.setattr(sys, "stdin", _InteractiveTty())
    return console_buf


def test_real_logging_provider_error_terminal_is_sanitized(
    isolated_app_logger, monkeypatch, capsys, tmp_path
):
    """Terminal output must not expose sensitive provider error details.

    Proves: normalized message appears; secret value, endpoint, raw marker, and
    Traceback are absent from both stdout and stderr; exit code is 1.
    """
    diag_path = tmp_path / "diag.log"
    console_buf = _setup_real_logging_cli_test(monkeypatch, diag_path=diag_path)

    raw_error_text = (
        f"Unauthorized: {_SECRET_VALUE} endpoint={_ENDPOINT_VALUE} {_RAW_MARKER}"
    )

    async def fake_run_demo(config_arg=None):
        raise OpenAIError(raw_error_text)

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()
    stderr = capsys.readouterr().err
    combined_terminal = console_buf.getvalue() + stderr

    assert exit_code == 1
    assert _NORMALIZED_PROVIDER_MSG in stderr
    assert _SECRET_VALUE not in combined_terminal
    assert _ENDPOINT_VALUE not in combined_terminal
    assert _RAW_MARKER not in combined_terminal
    assert "Traceback" not in combined_terminal
    assert "Unauthorized" not in combined_terminal


def test_real_logging_provider_error_diagnostic_file_retains_full_details(
    isolated_app_logger, monkeypatch, capsys, tmp_path
):
    """Full provider exception and traceback must be written to the diagnostic file.

    Proves: diagnostic file contains raw marker, secret-shaped value, endpoint,
    and Traceback text even when terminal is clean.
    """
    diag_path = tmp_path / "diag.log"
    _setup_real_logging_cli_test(monkeypatch, diag_path=diag_path)

    raw_error_text = (
        f"Unauthorized: {_SECRET_VALUE} endpoint={_ENDPOINT_VALUE} {_RAW_MARKER}"
    )

    async def fake_run_demo(config_arg=None):
        raise OpenAIError(raw_error_text)

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    run_cli()

    diag_content = diag_path.read_text(encoding="utf-8")
    assert _RAW_MARKER in diag_content
    assert _SECRET_VALUE in diag_content
    assert _ENDPOINT_VALUE in diag_content
    assert "Traceback" in diag_content


def test_real_logging_env_var_overrides_diagnostic_path(
    isolated_app_logger, monkeypatch, capsys, tmp_path
):
    """BUG_TRIAGE_DIAGNOSTIC_LOG env var must redirect the diagnostic file.

    Uses the real configure_diagnostic_logging (no mock) so the env-var path
    flows through main()'s os.environ.get() call.
    """
    from src.logging_config import configure_diagnostic_logging as real_diag

    override_path = tmp_path / "custom_diag.log"
    console_buf = _setup_real_logging_cli_test(
        monkeypatch, diag_path=None, configure_diag=False
    )

    # Use the real configure_diagnostic_logging; set the env var so main() passes
    # it through as path=str(override_path).
    monkeypatch.setattr(main_module, "configure_diagnostic_logging", real_diag)
    monkeypatch.setenv("BUG_TRIAGE_DIAGNOSTIC_LOG", str(override_path))

    async def fake_run_demo(config_arg=None):
        raise OpenAIError(f"{_RAW_MARKER} secret={_SECRET_VALUE}")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)
    run_cli()

    assert override_path.exists(), "Diagnostic log was not written to the override path"
    content = override_path.read_text(encoding="utf-8")
    assert _RAW_MARKER in content


def test_real_logging_unexpected_error_retains_console_traceback(
    isolated_app_logger, monkeypatch, capsys, tmp_path
):
    """An unrelated unexpected exception must still emit exc_info to the console logger.

    This proves that the provider-specific sanitization does NOT globally suppress
    tracebacks for other errors.
    """
    diag_path = tmp_path / "diag.log"
    console_buf = _setup_real_logging_cli_test(monkeypatch, diag_path=diag_path)

    async def fake_run_demo(config_arg=None):
        raise RuntimeError("unexpected-runtime-failure-XYZZY")

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = run_cli()
    console_out = console_buf.getvalue()

    assert exit_code == 1
    # The console JSON log must include the exception/traceback for unrelated errors.
    assert "unexpected-runtime-failure-XYZZY" in console_out
    assert "Traceback" in console_out


# ---------------------------------------------------------------------------
# Startup resilience — invalid diagnostic path
# ---------------------------------------------------------------------------


def test_invalid_diagnostic_path_does_not_prevent_startup(
    isolated_app_logger, monkeypatch, capsys
):
    """A bad BUG_TRIAGE_DIAGNOSTIC_LOG path must not crash the CLI."""
    from src.logging_config import (
        configure_diagnostic_logging as real_diag,
        configure_logging as real_configure_logging,
    )

    import io as _io

    console_buf = _io.StringIO()
    monkeypatch.setattr(sys, "stdout", console_buf)
    monkeypatch.setattr(main_module, "configure_logging", lambda **_: real_configure_logging())
    monkeypatch.setattr(main_module, "configure_diagnostic_logging", real_diag)
    monkeypatch.setenv(
        "BUG_TRIAGE_DIAGNOSTIC_LOG", "/nonexistent/no/such/dir/diag.log"
    )
    monkeypatch.setattr(main_module, "load_config", lambda: make_config())
    monkeypatch.setattr(sys, "stdin", _InteractiveTty())

    async def fake_run_demo(config_arg=None):
        pass

    monkeypatch.setattr(main_module, "run_demo", fake_run_demo)

    exit_code = asyncio.run(main_module.main(["--demo"]))
    assert exit_code == 0
    assert "Traceback" not in capsys.readouterr().err


def test_help_exits_zero_with_invalid_diagnostic_path(
    isolated_app_logger, monkeypatch, capsys
):
    """--help must exit 0 even when BUG_TRIAGE_DIAGNOSTIC_LOG is invalid."""
    from src.logging_config import (
        configure_diagnostic_logging as real_diag,
        configure_logging as real_configure_logging,
    )

    import io as _io

    monkeypatch.setattr(sys, "stdout", _io.StringIO())
    monkeypatch.setattr(main_module, "configure_logging", lambda **_: real_configure_logging())
    monkeypatch.setattr(main_module, "configure_diagnostic_logging", real_diag)
    monkeypatch.setenv(
        "BUG_TRIAGE_DIAGNOSTIC_LOG", "/nonexistent/no/such/dir/diag.log"
    )

    exit_code = asyncio.run(main_module.main(["--help"]))
    assert exit_code == 0
    assert "Traceback" not in capsys.readouterr().err
