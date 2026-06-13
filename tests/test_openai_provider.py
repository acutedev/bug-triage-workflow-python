"""Tests for the OpenAI classifier provider adapter."""

from types import SimpleNamespace

import pytest

from src.config import AppConfig
from agent_framework import Agent

from src.openai_provider import OpenAIClassifierClient, build_classifier_agent


class FakeResponses:
    def __init__(self, output_text: str | None = '{"category":"ui_bug"}') -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FailingResponses:
    def create(self, **kwargs):
        raise RuntimeError("OpenAI unavailable")


class FakeOpenAIClient:
    def __init__(self, output_text: str | None = '{"category":"ui_bug"}') -> None:
        self.responses = FakeResponses(output_text)


class FailingOpenAIClient:
    def __init__(self) -> None:
        self.responses = FailingResponses()


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


def test_openai_classifier_client_calls_responses_api_with_prompt_and_model():
    fake_client = FakeOpenAIClient(output_text='{"category":"ui_bug"}')
    adapter = OpenAIClassifierClient(make_config(), client=fake_client)

    result = adapter("Classify this bug")

    assert result == '{"category":"ui_bug"}'
    assert fake_client.responses.calls == [
        {
            "model": "gpt-4.1-mini",
            "input": "Classify this bug",
            "temperature": 0,
        }
    ]


def test_openai_classifier_client_rejects_non_openai_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER=openai"):
        OpenAIClassifierClient(
            make_config(provider="anthropic"),
            client=FakeOpenAIClient(),
        )


def test_openai_classifier_client_rejects_missing_api_key():
    with pytest.raises(ValueError, match="LLM_API_KEY is required"):
        OpenAIClassifierClient(make_config(api_key=""), client=FakeOpenAIClient())


@pytest.mark.parametrize("output_text", [None, "", "   "])
def test_openai_classifier_client_rejects_missing_output_text(output_text):
    adapter = OpenAIClassifierClient(
        make_config(),
        client=FakeOpenAIClient(output_text=output_text),
    )

    with pytest.raises(ValueError, match="non-empty output_text"):
        adapter("Classify this bug")


def test_openai_classifier_client_logs_api_failure(caplog):
    adapter = OpenAIClassifierClient(make_config(), client=FailingOpenAIClient())

    with caplog.at_level("ERROR", logger="bug_triage_workflow.openai_provider"):
        with pytest.raises(RuntimeError, match="OpenAI unavailable"):
            adapter("Classify this bug")

    assert "OpenAI classification request failed" in caplog.text
    assert any(
        getattr(record, "executor", None) == "OpenAIClassifierClient"
        and getattr(record, "model", None) == "gpt-4.1-mini"
        for record in caplog.records
    )
