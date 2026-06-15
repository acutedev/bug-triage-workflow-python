"""Tests for deterministic demo scenario runner helpers."""

from __future__ import annotations

import asyncio
import sys

import pytest

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
    secret = "sk-demo-secret-should-not-print"
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
