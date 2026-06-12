"""Tests for application configuration loading."""

import pytest

from src import config as config_module
from src.config import AppConfig, load_config


@pytest.fixture(autouse=True)
def disable_dotenv_loading(monkeypatch):
    monkeypatch.setattr(config_module, "_load_dotenv_if_available", lambda: None)


def test_load_config_reads_valid_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("HUMAN_APPROVAL_ENABLED", "false")

    config = load_config()

    assert config == AppConfig(
        llm_provider="openai",
        llm_api_key="test-key",
        llm_model="gpt-test",
        human_approval_enabled=False,
    )


def test_load_config_uses_defaults_when_optional_values_are_missing(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("HUMAN_APPROVAL_ENABLED", raising=False)

    config = load_config()

    assert config.llm_provider == "openai"
    assert config.llm_model == "gpt-4.1-mini"
    assert config.human_approval_enabled is True


def test_load_config_can_skip_api_key_requirement(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    config = load_config(require_api_key=False)

    assert config.llm_api_key == ""


def test_load_config_rejects_missing_required_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        load_config()


def test_load_config_rejects_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unsupported")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with pytest.raises(ValueError, match="LLM_PROVIDER must be one of: openai"):
        load_config()


def test_load_config_rejects_blank_model(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "   ")

    with pytest.raises(ValueError, match="LLM_MODEL cannot be blank"):
        load_config()


def test_load_config_rejects_invalid_human_approval_boolean(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("HUMAN_APPROVAL_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="HUMAN_APPROVAL_ENABLED must be a boolean value"):
        load_config()
