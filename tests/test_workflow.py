"""Integration tests for Microsoft Agent Framework bug triage orchestration."""

import asyncio

import src.workflow as workflow_module
from src.models import (
    BugCategory,
    HumanApprovalDecision,
    RouteName,
    Sentiment,
    Urgency,
    WorkflowResult,
    WorkflowStatus,
)
from src.workflow import (
    build_bug_triage_workflow,
    run_bug_triage_workflow,
    stream_bug_triage_workflow,
)


def make_llm_response(
    *,
    category: BugCategory = BugCategory.UI_BUG,
    urgency: Urgency = Urgency.MEDIUM,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    missing_info: list[str] | None = None,
    recommended_route: RouteName = RouteName.CREATE_STANDARD_TICKET,
) -> dict[str, object]:
    return {
        "category": category.value,
        "urgency": urgency.value,
        "sentiment": sentiment.value,
        "missing_info": missing_info or [],
        "recommended_route": recommended_route.value,
        "reasoning": "Validated classification for workflow integration testing.",
        "confidence": 0.9,
    }


def security_bug_report() -> str:
    return (
        "In production on Chrome using Windows, when I open the account page, "
        "I can view another user's private data instead of my own data. "
        "It should only display my account."
    )


def test_build_bug_triage_workflow_contains_expected_executors():
    workflow = build_bug_triage_workflow(
        lambda prompt: make_llm_response()
    )

    executor_ids = {
        workflow_executor.id
        for workflow_executor in workflow.get_executors_list()
    }
    assert executor_ids == {
        "preprocess_executor",
        "classifier_executor",
        "router_executor",
        "request_more_info_executor",
        "create_standard_ticket_executor",
        "request_human_approval_executor",
        "unexpected_route_executor",
        "create_escalation_ticket_executor",
        "log_rejection_executor",
    }


def test_workflow_runs_classifier_in_worker_thread(monkeypatch):
    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(
        run_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            lambda prompt: make_llm_response(),
        )
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert len(to_thread_calls) == 1
    called_function, called_args, called_kwargs = to_thread_calls[0]
    assert getattr(called_function, "__name__", None) == "classify_bug_report"
    assert called_args[0].normalized_text.startswith("In production on Chrome")
    assert callable(called_args[1])
    assert called_kwargs == {}


def test_workflow_creates_standard_ticket_for_complete_safe_report():
    bug_report = (
        "In production on Chrome using macOS, when I click save, "
        "the page shows an error instead of saving. "
        "It should save successfully."
    )

    result = asyncio.run(
        run_bug_triage_workflow(
            bug_report,
            lambda prompt: make_llm_response(
                category=BugCategory.UI_BUG,
                urgency=Urgency.MEDIUM,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.CREATE_STANDARD_TICKET,
            ),
        )
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert result.final_action == "Create a standard bug ticket."
    assert result.human_approval_required is False


def test_workflow_requests_more_information_when_report_is_incomplete():
    result = asyncio.run(
        run_bug_triage_workflow(
            "The login page is broken.",
            lambda prompt: make_llm_response(
                category=BugCategory.AUTHENTICATION,
                urgency=Urgency.MEDIUM,
                sentiment=Sentiment.CONFUSED,
                missing_info=["browser", "device_or_os"],
                recommended_route=RouteName.REQUEST_MORE_INFO,
            ),
        )
    )

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.REQUEST_MORE_INFO
    assert result.final_action == (
        "Request additional information from the bug reporter."
    )


def test_workflow_pauses_for_human_approval_and_resumes_with_approval():
    async def run_scenario():
        workflow = build_bug_triage_workflow(
            lambda prompt: make_llm_response(
                category=BugCategory.SECURITY,
                urgency=Urgency.CRITICAL,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            )
        )

        initial_stream = workflow.run(
            security_bug_report(),
            stream=True,
            include_status_events=True,
        )
        initial_events = [event async for event in initial_stream]

        request_event = next(
            event for event in initial_events if event.type == "request_info"
        )

        resumed_stream = workflow.run(
            stream=True,
            responses={
                request_event.request_id: HumanApprovalDecision(
                    required=True,
                    approval_granted=True,
                    approver="integration-test",
                    notes="Approved for escalation.",
                )
            },
            include_status_events=True,
        )
        resumed_events = [event async for event in resumed_stream]
        return initial_events, resumed_events

    initial_events, resumed_events = asyncio.run(run_scenario())

    waiting_result = next(
        event.data
        for event in initial_events
        if isinstance(getattr(event, "data", None), WorkflowResult)
    )
    final_result = next(
        event.data
        for event in resumed_events
        if isinstance(getattr(event, "data", None), WorkflowResult)
    )

    assert waiting_result.status == WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL
    assert waiting_result.selected_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert waiting_result.human_approval_required is True
    assert waiting_result.approval_granted is None
    assert waiting_result.final_action is None

    assert final_result.status == WorkflowStatus.COMPLETED
    assert final_result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert final_result.human_approval_required is True
    assert final_result.approval_granted is True
    assert final_result.final_action == (
        "Create an escalation ticket for human-reviewed handling."
    )


def test_workflow_pauses_for_human_approval_and_resumes_with_rejection():
    async def run_scenario():
        workflow = build_bug_triage_workflow(
            lambda prompt: make_llm_response(
                category=BugCategory.SECURITY,
                urgency=Urgency.CRITICAL,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            )
        )

        initial_stream = workflow.run(
            security_bug_report(),
            stream=True,
            include_status_events=True,
        )
        initial_events = [event async for event in initial_stream]

        request_event = next(
            event for event in initial_events if event.type == "request_info"
        )

        resumed_stream = workflow.run(
            stream=True,
            responses={
                request_event.request_id: HumanApprovalDecision(
                    required=True,
                    approval_granted=False,
                    approver="integration-test",
                    notes="Rejected after review.",
                )
            },
            include_status_events=True,
        )
        resumed_events = [event async for event in resumed_stream]
        return initial_events, resumed_events

    initial_events, resumed_events = asyncio.run(run_scenario())

    waiting_result = next(
        event.data
        for event in initial_events
        if isinstance(getattr(event, "data", None), WorkflowResult)
    )
    final_result = next(
        event.data
        for event in resumed_events
        if isinstance(getattr(event, "data", None), WorkflowResult)
    )

    assert waiting_result.status == WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL
    assert waiting_result.selected_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert waiting_result.human_approval_required is True
    assert waiting_result.approval_granted is None
    assert waiting_result.final_action is None
    assert final_result.status == WorkflowStatus.REJECTED
    assert final_result.selected_route == RouteName.LOG_REJECTION
    assert final_result.human_approval_required is True
    assert final_result.approval_granted is False
    assert final_result.final_action is None


def test_workflow_returns_failed_result_for_invalid_classifier_output():
    result = asyncio.run(
        run_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            lambda prompt: {"invalid": "classifier response"},
        )
    )

    assert result.status == WorkflowStatus.FAILED
    assert result.error
    assert result.final_action is None

def test_workflow_stream_emits_final_workflow_result():
    async def collect_events():
        stream = stream_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            lambda prompt: make_llm_response(),
        )

        return [event async for event in stream]

    events = asyncio.run(collect_events())

    assert events
    assert any(
        isinstance(getattr(event, "data", None), WorkflowResult)
        for event in events
    )
