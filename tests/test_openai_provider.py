"""Tests for the OpenAI classifier provider adapter."""

import pytest

from src.config import AppConfig
from agent_framework import Agent

from src.models import TriageClassification
import src.openai_provider as openai_provider
from src.openai_provider import build_classifier_agent


def make_config(
    *,
    provider: str = "openai",
    api_key: str = "test-key",
    model: str = "gpt-4.1-mini",
) -> AppConfig:
    return AppConfig(
        llm_provider=provider,
        llm_api_key=api_key,
        llm_model=model,
        human_approval_enabled=True,
    )


# === Native MAF Agent build_classifier_agent tests ===

class FakeChatClient:
    """Minimal injected chat client used to construct a native MAF Agent."""


def test_build_classifier_agent_returns_native_maf_agent():
    agent = build_classifier_agent(
        make_config(),
        client=FakeChatClient(),
    )

    assert isinstance(agent, Agent)
    assert agent.name == "classifier_agent"
    assert agent.default_options["response_format"] is TriageClassification


def test_build_classifier_agent_rejects_non_openai_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER=openai"):
        build_classifier_agent(
            make_config(provider="anthropic"),
            client=FakeChatClient(),
        )


def test_build_classifier_agent_requires_api_key_without_injected_client():
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        build_classifier_agent(make_config(api_key=""))


def test_build_classifier_agent_allows_injected_client_without_api_key():
    agent = build_classifier_agent(
        make_config(api_key=""),
        client=FakeChatClient(),
    )

    assert isinstance(agent, Agent)
    assert agent.name == "classifier_agent"


def test_build_classifier_agent_configures_bounded_default_openai_client(
    monkeypatch,
):
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["async_client_kwargs"] = kwargs

    class FakeOpenAIChatClient:
        def __init__(self, **kwargs):
            captured["chat_client_kwargs"] = kwargs

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(openai_provider, "OpenAIChatClient", FakeOpenAIChatClient)

    agent = build_classifier_agent(make_config(api_key="secret-test-key"))

    assert isinstance(agent, Agent)
    assert agent.name == "classifier_agent"
    assert captured["async_client_kwargs"] == {
        "api_key": "secret-test-key",
        "timeout": openai_provider.OPENAI_REQUEST_TIMEOUT_SECONDS,
        "max_retries": openai_provider.OPENAI_MAX_RETRIES,
    }
    chat_client_kwargs = captured["chat_client_kwargs"]
    assert isinstance(chat_client_kwargs, dict)
    assert chat_client_kwargs["model"] == "gpt-4.1-mini"
    assert isinstance(chat_client_kwargs["async_client"], FakeAsyncOpenAI)


def test_build_classifier_agent_logs_do_not_include_api_key(caplog):
    with caplog.at_level("INFO", logger="bug_triage_workflow.openai_provider"):
        build_classifier_agent(
            make_config(api_key="super-secret-provider-key"),
            client=FakeChatClient(),
        )

    assert "Native classifier agent created" in caplog.text
    assert "super-secret-provider-key" not in caplog.text
