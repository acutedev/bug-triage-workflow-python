"""Tests for deterministic demo scenario runner helpers."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any

import pytest
from agent_framework import (
    Agent,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)

from scripts import run_demo_scenario as demo
from src.models import (
    BugCategory,
    HumanReviewAction,
    RouteName,
    Sentiment,
    TriageClassification,
    Urgency,
    WorkflowResult,
    WorkflowStatus,
)


class CapturingLogger:
    def __init__(self) -> None:
        self.exceptions: list[tuple[str, tuple[object, object, object]]] = []

    def exception(self, message: str) -> None:
        self.exceptions.append((message, sys.exc_info()))


def make_classification(
    *,
    recommended_route: RouteName = RouteName.CREATE_STANDARD_TICKET,
) -> TriageClassification:
    return TriageClassification(
        category=BugCategory.UI_BUG,
        urgency=Urgency.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
        missing_info=[],
        recommended_route=recommended_route,
        reasoning="Deterministic demo test classification.",
        confidence=0.9,
    )


def make_completed_result(
    route: RouteName,
    *,
    human_review_required: bool = False,
    human_review_action: HumanReviewAction | None = None,
    approval_granted: bool | None = None,
) -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=route,
        classification=make_classification(recommended_route=route),
        human_review_required=human_review_required,
        human_review_action=human_review_action,
        approval_granted=approval_granted,
        final_action="Demo final action.",
        error=None,
    )


def make_report_rejected_result() -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowStatus.REPORT_REJECTED,
        selected_route=RouteName.LOG_REJECTION,
        classification=make_classification(
            recommended_route=RouteName.REQUEST_HUMAN_APPROVAL
        ),
        human_review_required=True,
        human_review_action=HumanReviewAction.REJECT_REPORT,
        approval_granted=False,
        final_action=None,
        error=None,
    )


def make_classifier_failure_result(
    *,
    error: str = "Bug classification failed: invalid classifier response",
) -> WorkflowResult:
    return WorkflowResult(
        status=WorkflowStatus.FAILED,
        selected_route=None,
        classification=None,
        human_review_required=False,
        human_review_action=None,
        approval_granted=None,
        final_action=None,
        error=error,
    )


def matching_result_for_scenario(scenario_name: str) -> WorkflowResult:
    if scenario_name == "standard-ticket":
        return make_completed_result(RouteName.CREATE_STANDARD_TICKET)
    if scenario_name == "request-more-info":
        return make_completed_result(RouteName.REQUEST_MORE_INFO)
    if scenario_name == "escalation-approved":
        return make_completed_result(
            RouteName.CREATE_ESCALATION_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.APPROVE_ESCALATION,
            approval_granted=True,
        )
    if scenario_name == "standard-ticket-selected":
        return make_completed_result(
            RouteName.CREATE_STANDARD_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approval_granted=None,
        )
    if scenario_name == "report-rejected":
        return make_report_rejected_result()
    if scenario_name == "direct-escalation":
        return make_completed_result(RouteName.CREATE_ESCALATION_TICKET)
    if scenario_name == "classifier-failure":
        return make_classifier_failure_result()
    if scenario_name == "adversarial-security":
        classification = TriageClassification(
            category=BugCategory.SECURITY,
            urgency=Urgency.CRITICAL,
            sentiment=Sentiment.NEUTRAL,
            missing_info=[],
            recommended_route=RouteName.CREATE_ESCALATION_TICKET,
            reasoning="Credential exposure — genuine security incident.",
            confidence=0.95,
        )
        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            classification=classification,
            human_review_required=True,
            human_review_action=HumanReviewAction.APPROVE_ESCALATION,
            approval_granted=True,
            final_action="Escalation ticket created.",
            error=None,
        )
    if scenario_name == "adversarial-benign-quote":
        classification = TriageClassification(
            category=BugCategory.UI_BUG,
            urgency=Urgency.LOW,
            sentiment=Sentiment.NEUTRAL,
            missing_info=[],
            recommended_route=RouteName.CREATE_STANDARD_TICKET,
            reasoning="CSS layout issue — quoted text is not a security incident.",
            confidence=0.9,
        )
        return WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            classification=classification,
            human_review_required=False,
            human_review_action=None,
            approval_granted=None,
            final_action="Standard ticket created.",
            error=None,
        )

    if scenario_name == "low-confidence-review":
        return make_completed_result(
            RouteName.CREATE_STANDARD_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approval_granted=None,
        )

    raise AssertionError(f"Unsupported scenario in test: {scenario_name}")


def test_scenario_registry_contains_exactly_supported_names():
    assert list(demo.SCENARIOS) == [
        "standard-ticket",
        "request-more-info",
        "escalation-approved",
        "standard-ticket-selected",
        "report-rejected",
        "direct-escalation",
        "classifier-failure",
        "adversarial-security",
        "adversarial-benign-quote",
        "low-confidence-review",
    ]


def test_invalid_scenario_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(demo, "configure_logging", CapturingLogger)

    exit_code = demo.main(["not-a-scenario"])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "Supported scenarios:" in captured.err
    assert "not-a-scenario" not in captured.out


def test_print_event_does_not_probe_request_id_for_non_request_events(capsys):
    class StartedEvent:
        type = "started"
        executor_id = "preprocess_executor"

        @property
        def request_id(self):
            raise RuntimeError("request_id should not be read")

    demo.print_event(StartedEvent())

    assert (
        capsys.readouterr().out.strip()
        == "event=started | executor=preprocess_executor"
    )


def test_print_event_uses_source_executor_for_request_info(capsys):
    class RequestInfoEvent:
        type = "request_info"
        executor_id = None
        source_executor_id = "request_human_review_executor"

        @property
        def request_id(self):
            raise RuntimeError("request_id should not be read")

    demo.print_event(RequestInfoEvent())

    assert capsys.readouterr().out.strip() == (
        "event=request_info | executor=request_human_review_executor | "
        "request_info=paused"
    )


@pytest.mark.parametrize("scenario_name", list(demo.SCENARIOS))
def test_scenario_validation_accepts_matching_result(scenario_name):
    demo.validate_scenario_result(
        scenario_name,
        matching_result_for_scenario(scenario_name),
    )


@pytest.mark.parametrize("scenario_name", list(demo.SCENARIOS))
def test_scenario_validation_rejects_incorrect_result(scenario_name):
    with pytest.raises(demo.ScenarioValidationError):
        demo.validate_scenario_result(
            scenario_name,
            make_classifier_failure_result(error="Bug preprocessing failed: invalid"),
        )


def test_successful_scenario_prints_validation_passed(monkeypatch, capsys):
    async def fake_run_scenario(spec):
        return matching_result_for_scenario(spec.name)

    monkeypatch.setattr(demo, "run_scenario", fake_run_scenario)

    exit_code = asyncio.run(demo.run_selected_scenario("standard-ticket"))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "DEMO SCENARIO: standard-ticket" in captured.out
    assert "DEMO VALIDATION PASSED" in captured.out


def test_result_validation_failure_returns_nonzero(monkeypatch, capsys):
    async def fake_run_scenario(spec):
        return make_classifier_failure_result(
            error="Bug preprocessing failed: wrong scenario"
        )

    monkeypatch.setattr(demo, "configure_logging", CapturingLogger)
    monkeypatch.setattr(demo, "run_scenario", fake_run_scenario)

    exit_code = demo.main(["standard-ticket"])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "Configuration or validation error:" in captured.err
    assert "DEMO VALIDATION PASSED" not in captured.out


def test_unexpected_runner_failure_returns_one_and_logs_exception(
    monkeypatch,
    capsys,
):
    logger = CapturingLogger()
    runner_error = RuntimeError("internal runner exploded")

    async def fake_run_selected_scenario(scenario_name):
        raise runner_error

    monkeypatch.setattr(demo, "configure_logging", lambda: logger)
    monkeypatch.setattr(
        demo,
        "run_selected_scenario",
        fake_run_selected_scenario,
    )

    exit_code = demo.main(["standard-ticket"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert (
        "Unexpected error: demo scenario failed. See logs for details."
        in captured.err
    )
    assert "internal runner exploded" not in captured.err
    assert len(logger.exceptions) == 1
    message, exc_info = logger.exceptions[0]
    assert message == "Demo scenario failed unexpectedly"
    assert exc_info[0] is RuntimeError
    assert exc_info[1] is runner_error
    assert exc_info[2] is not None


def test_no_secrets_appear_in_printed_output(monkeypatch, capsys):
    secret = "sk-" + "demo-secret-should-not-print"
    monkeypatch.setenv("LLM_API_KEY", secret)

    async def fake_run_scenario(spec):
        return matching_result_for_scenario(spec.name)

    monkeypatch.setattr(demo, "configure_logging", CapturingLogger)
    monkeypatch.setattr(demo, "run_scenario", fake_run_scenario)

    exit_code = demo.main(["standard-ticket"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert secret not in captured.out
    assert secret not in captured.err


# --- Optional classifier expectation tests ---


def _spec_with(**kwargs) -> demo.ScenarioSpec:
    """Build a minimal ScenarioSpec with overrides for testing optional fields."""
    return demo.ScenarioSpec(
        name="standard-ticket",
        report="irrelevant",
        expected_status=WorkflowStatus.COMPLETED,
        expected_route=RouteName.CREATE_STANDARD_TICKET,
        expected_human_review_required=False,
        expected_human_review_action=None,
        expected_approval_granted=None,
        **kwargs,
    )


def _result_with_category_urgency(
    category: BugCategory, urgency: Urgency
) -> WorkflowResult:
    classification = TriageClassification(
        category=category,
        urgency=urgency,
        sentiment=Sentiment.NEUTRAL,
        missing_info=[],
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
        reasoning="test",
        confidence=0.9,
    )
    return WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_STANDARD_TICKET,
        classification=classification,
        human_review_required=False,
        human_review_action=None,
        approval_granted=None,
        final_action="done",
        error=None,
    )


def test_expected_category_passes_when_correct():
    spec = _spec_with(expected_category=BugCategory.UI_BUG)
    result = _result_with_category_urgency(BugCategory.UI_BUG, Urgency.MEDIUM)
    demo.validate_result(spec, result)  # should not raise


def test_expected_category_fails_when_incorrect():
    spec = _spec_with(expected_category=BugCategory.SECURITY)
    result = _result_with_category_urgency(BugCategory.UI_BUG, Urgency.MEDIUM)
    with pytest.raises(demo.ScenarioValidationError, match="category"):
        demo.validate_result(spec, result)


def test_allowed_urgencies_passes_when_correct():
    spec = _spec_with(allowed_urgencies=frozenset({Urgency.LOW, Urgency.MEDIUM}))
    result = _result_with_category_urgency(BugCategory.UI_BUG, Urgency.MEDIUM)
    demo.validate_result(spec, result)  # should not raise


def test_allowed_urgencies_fails_when_incorrect():
    spec = _spec_with(allowed_urgencies=frozenset({Urgency.CRITICAL}))
    result = _result_with_category_urgency(BugCategory.UI_BUG, Urgency.MEDIUM)
    with pytest.raises(demo.ScenarioValidationError, match="urgency"):
        demo.validate_result(spec, result)


def test_scenarios_without_optional_fields_validate_unchanged():
    spec = _spec_with()  # no expected_category or allowed_urgencies
    result = _result_with_category_urgency(BugCategory.SECURITY, Urgency.CRITICAL)
    demo.validate_result(spec, result)  # should not raise — fields not checked


# --- Adversarial scenario registration tests ---


def test_scenario_registry_contains_adversarial_scenarios():
    assert "adversarial-security" in demo.SCENARIOS
    assert "adversarial-benign-quote" in demo.SCENARIOS


def test_adversarial_security_scenario_configuration():
    spec = demo.SCENARIOS["adversarial-security"]
    assert spec.expected_category == BugCategory.SECURITY
    assert spec.allowed_urgencies is not None
    assert Urgency.HIGH in spec.allowed_urgencies
    assert Urgency.CRITICAL in spec.allowed_urgencies
    assert spec.expected_human_review_required is True
    assert spec.expected_approval_granted is True


def test_adversarial_benign_quote_scenario_configuration():
    spec = demo.SCENARIOS["adversarial-benign-quote"]
    assert spec.expected_category == BugCategory.UI_BUG
    assert spec.allowed_urgencies == frozenset({Urgency.LOW, Urgency.MEDIUM})
    assert spec.expected_route == RouteName.CREATE_STANDARD_TICKET
    assert spec.expected_human_review_required is False
    assert spec.expected_approval_granted is None


# --- Fake-agent end-to-end execution of adversarial contracts ---


class _DeterministicClassifierClient:
    """Returns a fixed valid classification JSON without any network calls."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._text = json.dumps(response)

    def get_response(
        self,
        messages: Any,
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        text = self._text

        if stream:
            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                yield ChatResponseUpdate(
                    contents=[Content.from_text(text)],
                    role="assistant",
                    finish_reason="stop",
                )

            def _finalize(updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
                return ChatResponse.from_updates(
                    updates,
                    output_format_type=(options or {}).get("response_format"),
                )

            return ResponseStream(_stream(), finalizer=_finalize)

        async def _get() -> ChatResponse:
            return ChatResponse(
                messages=Message(role="assistant", contents=[text])
            )

        return _get()


def _make_deterministic_agent(response: dict[str, Any]) -> Agent:
    return Agent(
        client=_DeterministicClassifierClient(response),
        name="classifier_agent",
        instructions="Return bug classification JSON.",
        default_options={"response_format": TriageClassification},
    )


def test_fake_agent_executes_adversarial_security_contract():
    """Requirement 6: fake agent runs adversarial-security without live OpenAI calls."""
    spec = demo.SCENARIOS["adversarial-security"]
    agent = _make_deterministic_agent({
        "category": "security",
        "urgency": "critical",
        "sentiment": "neutral",
        "missing_info": [],
        "recommended_route": "create_escalation_ticket",
        "reasoning": "Credential exposure — genuine security incident.",
        "confidence": 0.95,
    })
    result = asyncio.run(
        demo.run_scenario_workflow(spec, agent, event_printer=lambda _: None)
    )
    demo.validate_result(spec, result)


def test_fake_agent_executes_adversarial_benign_quote_contract():
    """Requirement 6: fake agent runs adversarial-benign-quote without live OpenAI calls."""
    spec = demo.SCENARIOS["adversarial-benign-quote"]
    agent = _make_deterministic_agent({
        "category": "ui_bug",
        "urgency": "low",
        "sentiment": "neutral",
        "missing_info": [],
        "recommended_route": "create_standard_ticket",
        "reasoning": "CSS layout issue — quoted text is not a security incident.",
        "confidence": 0.9,
    })
    result = asyncio.run(
        demo.run_scenario_workflow(spec, agent, event_printer=lambda _: None)
    )
    demo.validate_result(spec, result)


# --- Low-confidence-review scenario tests ---


def test_scenario_registry_contains_low_confidence_review():
    assert "low-confidence-review" in demo.SCENARIOS


def test_low_confidence_review_scenario_configuration():
    spec = demo.SCENARIOS["low-confidence-review"]
    assert spec.expected_route == RouteName.CREATE_STANDARD_TICKET
    assert spec.expected_human_review_required is True
    assert spec.expected_human_review_action == HumanReviewAction.CREATE_STANDARD_TICKET
    assert spec.expected_approval_granted is None
    assert spec.review_decision is not None
    assert spec.review_decision.action == HumanReviewAction.CREATE_STANDARD_TICKET
    assert spec.use_fake_classifier is True


def test_low_confidence_review_fake_agent_emits_confidence_below_threshold():
    spec = demo.SCENARIOS["low-confidence-review"]
    agent = _make_deterministic_agent({
        "category": "ui_bug",
        "urgency": "low",
        "sentiment": "neutral",
        "missing_info": [],
        "recommended_route": "create_standard_ticket",
        "reasoning": "Cosmetic layout regression, low classifier confidence.",
        "confidence": 0.60,
    })
    result = asyncio.run(
        demo.run_scenario_workflow(spec, agent, event_printer=lambda _: None)
    )
    assert result.classification is not None
    assert result.classification.confidence < 0.70


def test_low_confidence_review_fake_agent_completes_via_human_review():
    spec = demo.SCENARIOS["low-confidence-review"]
    agent = _make_deterministic_agent({
        "category": "ui_bug",
        "urgency": "low",
        "sentiment": "neutral",
        "missing_info": [],
        "recommended_route": "create_standard_ticket",
        "reasoning": "Cosmetic layout regression, low classifier confidence.",
        "confidence": 0.60,
    })
    result = asyncio.run(
        demo.run_scenario_workflow(spec, agent, event_printer=lambda _: None)
    )
    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert result.human_review_required is True
    assert result.human_review_action == HumanReviewAction.CREATE_STANDARD_TICKET
    assert result.approval_granted is None
    demo.validate_result(spec, result)
