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


def print_workflow_result(label: str, result: WorkflowResult) -> None:
    print(f"\n=== {label} ===")
    for event in result.event_log:
        print(
            f"{event.status.value} | "
            f"{event.executor} | "
            f"{event.message} | "
            f"{event.data}"
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


def test_workflow_uses_direct_escalation_graph_when_approval_is_disabled():
    workflow = build_bug_triage_workflow(
        lambda prompt: make_llm_response(),
        human_approval_enabled=False,
    )

    executor_ids = {
        workflow_executor.id
        for workflow_executor in workflow.get_executors_list()
    }

    assert "create_direct_escalation_ticket_executor" in executor_ids
    assert "request_human_approval_executor" not in executor_ids
    assert "create_escalation_ticket_executor" not in executor_ids
    assert "log_rejection_executor" not in executor_ids


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

    print_workflow_result("worker-thread workflow", result)

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

    print_workflow_result("standard ticket workflow", result)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert result.final_action == "Create a standard bug ticket."
    assert result.human_approval_required is False
    assert [event.status for event in result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]


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

    print_workflow_result("request more info workflow", result)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.REQUEST_MORE_INFO
    assert result.final_action == (
        "Request additional information from the bug reporter."
    )
    assert [event.status for event in result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]


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

    print_workflow_result("waiting for human approval", waiting_result)
    print_workflow_result("approved human review", final_result)

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
    assert [event.status for event in final_result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
        WorkflowStatus.APPROVED,
        WorkflowStatus.COMPLETED,
    ]


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

    print_workflow_result("waiting for human approval", waiting_result)
    print_workflow_result("rejected human review", final_result)

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
    assert [event.status for event in final_result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
        WorkflowStatus.REJECTED,
    ]


def test_risky_report_escalates_directly_when_human_approval_is_disabled():
    result = asyncio.run(
        run_bug_triage_workflow(
            security_bug_report(),
            lambda prompt: make_llm_response(
                category=BugCategory.SECURITY,
                urgency=Urgency.CRITICAL,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            ),
            human_approval_enabled=False,
        )
    )

    print_workflow_result("direct escalation workflow", result)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert result.human_approval_required is False
    assert result.approval_granted is None
    assert result.final_action == (
        "Create an escalation ticket directly because human approval "
        "is disabled by configuration."
    )
    assert [event.status for event in result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert (
        result.event_log[-1].executor
        == "create_direct_escalation_ticket_executor"
    )
    assert result.event_log[-1].data["human_approval_enabled"] is False
    assert (
        result.event_log[-1].data["policy_route"]
        == RouteName.REQUEST_HUMAN_APPROVAL.value
    )
    assert (
        result.event_log[-1].data["effective_route"]
        == RouteName.CREATE_ESCALATION_TICKET.value
    )


def test_disabled_human_approval_stream_does_not_request_information():
    async def collect_events():
        stream = stream_bug_triage_workflow(
            security_bug_report(),
            lambda prompt: make_llm_response(
                category=BugCategory.SECURITY,
                urgency=Urgency.CRITICAL,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            ),
            human_approval_enabled=False,
        )

        return [event async for event in stream]

    events = asyncio.run(collect_events())

    assert events
    assert all(event.type != "request_info" for event in events)

    final_result = next(
        event.data
        for event in events
        if isinstance(getattr(event, "data", None), WorkflowResult)
    )

    assert final_result.status == WorkflowStatus.COMPLETED
    assert final_result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert final_result.human_approval_required is False
    assert final_result.approval_granted is None


def test_workflow_returns_failed_result_for_preprocessing_validation_error(
    monkeypatch,
):
    def raise_preprocessing_error(raw_text: str):
        raise ValueError("Invalid bug report input.")

    monkeypatch.setattr(
        workflow_module,
        "preprocess_bug_report",
        raise_preprocessing_error,
    )

    result = asyncio.run(
        run_bug_triage_workflow(
            "Invalid bug report.",
            lambda prompt: make_llm_response(),
        )
    )

    print_workflow_result("failed preprocessing workflow", result)

    assert result.status == WorkflowStatus.FAILED
    assert result.error == (
        "Bug preprocessing failed: Invalid bug report input."
    )
    assert result.final_action is None
    assert [event.status for event in result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.FAILED,
    ]
    assert result.event_log[-1].executor == "preprocess_executor"
    assert result.event_log[-1].data["error_type"] == "ValueError"


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

    print_workflow_result("failed classifier workflow", result)

    assert result.status == WorkflowStatus.FAILED
    assert result.error
    assert result.final_action is None
    assert [event.status for event in result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.FAILED,
    ]
    assert result.event_log[-1].executor == "classifier_executor"
    assert result.event_log[-1].data["error_type"] == "ValidationError"


def test_workflow_returns_failed_result_for_unexpected_llm_client_error():
    def raise_client_error(prompt: str):
        raise RuntimeError("LLM provider unavailable.")

    result = asyncio.run(
        run_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            raise_client_error,
        )
    )

    print_workflow_result("failed LLM client workflow", result)

    assert result.status == WorkflowStatus.FAILED
    assert result.error == (
        "Bug classification failed: LLM provider unavailable."
    )
    assert result.final_action is None
    assert [event.status for event in result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.FAILED,
    ]
    assert result.event_log[-1].executor == "classifier_executor"
    assert result.event_log[-1].data["error_type"] == "RuntimeError"


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

    print("\n=== raw MAF stream events ===")
    for event in events:
        source_executor_id = (
            event.source_executor_id
            if event.type == "request_info"
            else None
        )
        print(
            f"{event.type} | "
            f"{source_executor_id} | "
            f"{getattr(event, 'data', None)}"
        )

    assert events
    assert any(
        isinstance(getattr(event, "data", None), WorkflowResult)
        for event in events
    )
