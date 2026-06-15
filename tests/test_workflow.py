"""Integration tests for Microsoft Agent Framework bug triage orchestration."""

import asyncio
import json
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

import src.workflow as workflow_module
from src.models import (
    BugCategory,
    HumanReviewAction,
    HumanReviewDecision,
    RouteName,
    Sentiment,
    TriageClassification,
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


class FakeClassifierChatClient:
    """Deterministic chat client used by a real native MAF Agent in tests."""

    def __init__(
        self,
        response: dict[str, object] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.response = make_llm_response() if response is None else response
        self.error = error
        self.call_count = 0
        self.received_messages: list[object] = []

    def get_response(
        self,
        messages: str | Message | list[str] | list[Message],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        del kwargs
        self.call_count += 1
        self.received_messages.append(messages)
        response_text = json.dumps(self.response)

        if stream:
            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                if self.error is not None:
                    raise self.error
                yield ChatResponseUpdate(
                    contents=[Content.from_text(response_text)],
                    role="assistant",
                    finish_reason="stop",
                )

            def _finalize(
                updates: Sequence[ChatResponseUpdate],
            ) -> ChatResponse:
                return ChatResponse.from_updates(
                    updates,
                    output_format_type=(options or {}).get("response_format"),
                )

            return ResponseStream(_stream(), finalizer=_finalize)

        async def _get() -> ChatResponse:
            if self.error is not None:
                raise self.error
            return ChatResponse(
                messages=Message(
                    role="assistant",
                    contents=[response_text],
                )
            )

        return _get()


def make_classifier_agent(
    response: dict[str, object] | None = None,
    *,
    error: Exception | None = None,
) -> tuple[Agent, FakeClassifierChatClient]:
    client = FakeClassifierChatClient(response, error=error)
    agent = Agent(
        client=client,
        name="classifier_agent",
        instructions="Return only the requested bug classification JSON.",
        default_options={"response_format": TriageClassification},
    )
    return agent, client


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


def workflow_results_from_events(events: list[object]) -> list[WorkflowResult]:
    return [
        event.data
        for event in events
        if isinstance(getattr(event, "data", None), WorkflowResult)
    ]


def workflow_statuses(result: WorkflowResult) -> list[WorkflowStatus]:
    return [event.status for event in result.event_log]


def event_executor_ids(events: list[object]) -> set[str]:
    return {
        executor_id
        for event in events
        if (executor_id := getattr(event, "executor_id", None)) is not None
    }


async def run_human_review_scenario(
    action: HumanReviewAction,
) -> tuple[list[object], list[object]]:
    workflow = build_bug_triage_workflow(
        make_classifier_agent(
            make_llm_response(
                category=BugCategory.SECURITY,
                urgency=Urgency.CRITICAL,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            )
        )[0]
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
            request_event.request_id: HumanReviewDecision(
                required=True,
                action=action,
                approver="integration-test",
                notes=f"{action.value} selected.",
            )
        },
        include_status_events=True,
    )
    resumed_events = [event async for event in resumed_stream]
    return initial_events, resumed_events


def test_build_bug_triage_workflow_contains_expected_executors():
    classifier_agent, _ = make_classifier_agent()
    workflow = build_bug_triage_workflow(classifier_agent)

    executor_ids = {
        workflow_executor.id
        for workflow_executor in workflow.get_executors_list()
    }
    assert executor_ids == {
        "preprocess_executor",
        "classifier_request_executor",
        "classifier_agent",
        "classifier_response_executor",
        "router_executor",
        "request_more_info_executor",
        "create_standard_ticket_executor",
        "request_human_review_executor",
        "unexpected_route_executor",
        "create_escalation_ticket_executor",
        "log_rejection_executor",
        "unexpected_human_review_action_executor",
    }


def test_workflow_uses_direct_escalation_graph_when_approval_is_disabled():
    classifier_agent, _ = make_classifier_agent()
    workflow = build_bug_triage_workflow(
        classifier_agent,
        human_approval_enabled=False,
    )

    executor_ids = {
        workflow_executor.id
        for workflow_executor in workflow.get_executors_list()
    }

    assert "create_direct_escalation_ticket_executor" in executor_ids
    assert "request_human_review_executor" not in executor_ids
    assert "create_escalation_ticket_executor" not in executor_ids
    assert "log_rejection_executor" not in executor_ids


def test_workflow_runs_native_classifier_agent():
    classifier_agent, client = make_classifier_agent()

    result = asyncio.run(
        run_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            classifier_agent,
        )
    )

    print_workflow_result("native-agent workflow", result)

    assert result.status == WorkflowStatus.COMPLETED
    assert client.call_count == 1
    assert client.received_messages
    received_messages = client.received_messages[0]
    assert isinstance(received_messages, list)
    assert any(
        isinstance(message, Message)
        and "In production on Chrome" in message.text
        for message in received_messages
    )
    assert result.event_log[2].executor == "classifier_agent"


def test_workflow_creates_standard_ticket_for_complete_safe_report():
    bug_report = (
        "In production on Chrome using macOS, when I click save, "
        "the page shows an error instead of saving. "
        "It should save successfully."
    )

    result = asyncio.run(
        run_bug_triage_workflow(
            bug_report,
            make_classifier_agent(
                make_llm_response(
                    category=BugCategory.UI_BUG,
                    urgency=Urgency.MEDIUM,
                    sentiment=Sentiment.NEUTRAL,
                    recommended_route=RouteName.CREATE_STANDARD_TICKET,
                )
            )[0],
        )
    )

    print_workflow_result("standard ticket workflow", result)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert result.final_action == "Create a standard bug ticket."
    assert result.human_review_required is False
    assert result.human_review_action is None
    assert result.approval_granted is None
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
            make_classifier_agent(
                make_llm_response(
                    category=BugCategory.AUTHENTICATION,
                    urgency=Urgency.MEDIUM,
                    sentiment=Sentiment.CONFUSED,
                    missing_info=["browser", "device_or_os"],
                    recommended_route=RouteName.REQUEST_MORE_INFO,
                )
            )[0],
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


def test_human_review_approved_creates_escalation_ticket():
    initial_events, resumed_events = asyncio.run(
        run_human_review_scenario(HumanReviewAction.APPROVE_ESCALATION)
    )

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

    print_workflow_result("awaiting human review", waiting_result)
    print_workflow_result("escalation approved human review", final_result)

    assert waiting_result.status == WorkflowStatus.AWAITING_HUMAN_REVIEW
    assert waiting_result.selected_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert waiting_result.human_review_required is True
    assert waiting_result.human_review_action is None
    assert waiting_result.approval_granted is None
    assert waiting_result.final_action is None

    assert final_result.status == WorkflowStatus.COMPLETED
    assert final_result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert final_result.human_review_required is True
    assert final_result.human_review_action == HumanReviewAction.APPROVE_ESCALATION
    assert final_result.approval_granted is True
    assert final_result.final_action == (
        "Create an escalation ticket for human-reviewed handling."
    )
    resumed_executor_ids = event_executor_ids(resumed_events)
    assert "create_escalation_ticket_executor" in resumed_executor_ids
    assert "create_standard_ticket_executor" not in resumed_executor_ids
    assert "log_rejection_executor" not in resumed_executor_ids
    assert [event.status for event in final_result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.AWAITING_HUMAN_REVIEW,
        WorkflowStatus.ESCALATION_APPROVED,
        WorkflowStatus.COMPLETED,
    ]
    for status in [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.AWAITING_HUMAN_REVIEW,
    ]:
        assert workflow_statuses(final_result).count(status) == 1


def test_human_review_standard_option_creates_standard_ticket():
    initial_events, resumed_events = asyncio.run(
        run_human_review_scenario(HumanReviewAction.CREATE_STANDARD_TICKET)
    )

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

    print_workflow_result("awaiting human review", waiting_result)
    print_workflow_result("standard-ticket human review", final_result)

    assert waiting_result.status == WorkflowStatus.AWAITING_HUMAN_REVIEW
    assert waiting_result.selected_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert waiting_result.human_review_required is True
    assert waiting_result.human_review_action is None
    assert waiting_result.approval_granted is None
    assert waiting_result.final_action is None

    assert final_result.status == WorkflowStatus.COMPLETED
    assert final_result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert final_result.human_review_required is True
    assert final_result.human_review_action == HumanReviewAction.CREATE_STANDARD_TICKET
    assert final_result.approval_granted is None
    assert final_result.final_action == "Create a standard bug ticket."
    assert final_result.classification == waiting_result.classification
    resumed_executor_ids = event_executor_ids(resumed_events)
    assert "create_standard_ticket_executor" in resumed_executor_ids
    assert "create_escalation_ticket_executor" not in resumed_executor_ids
    assert "log_rejection_executor" not in resumed_executor_ids
    assert [event.status for event in final_result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.AWAITING_HUMAN_REVIEW,
        WorkflowStatus.STANDARD_TICKET_SELECTED,
        WorkflowStatus.COMPLETED,
    ]
    assert final_result.event_log[-2].executor == "request_human_review_executor"
    assert final_result.event_log[-2].message == (
        "Human reviewer selected standard-ticket handling instead of escalation."
    )
    assert final_result.event_log[-1].executor == "create_standard_ticket_executor"


def test_human_review_rejected_logs_rejection():
    initial_events, resumed_events = asyncio.run(
        run_human_review_scenario(HumanReviewAction.REJECT_REPORT)
    )

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

    print_workflow_result("awaiting human review", waiting_result)
    print_workflow_result("report rejected human review", final_result)

    assert waiting_result.status == WorkflowStatus.AWAITING_HUMAN_REVIEW
    assert waiting_result.selected_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert waiting_result.human_review_required is True
    assert waiting_result.human_review_action is None
    assert waiting_result.approval_granted is None
    assert waiting_result.final_action is None

    assert final_result.status == WorkflowStatus.REPORT_REJECTED
    assert final_result.selected_route == RouteName.LOG_REJECTION
    assert final_result.human_review_required is True
    assert final_result.human_review_action == HumanReviewAction.REJECT_REPORT
    assert final_result.approval_granted is False
    assert final_result.final_action is None
    resumed_executor_ids = event_executor_ids(resumed_events)
    assert "log_rejection_executor" in resumed_executor_ids
    assert "create_standard_ticket_executor" not in resumed_executor_ids
    assert "create_escalation_ticket_executor" not in resumed_executor_ids
    assert [event.status for event in final_result.event_log] == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.AWAITING_HUMAN_REVIEW,
        WorkflowStatus.REPORT_REJECTED,
    ]
    assert final_result.event_log[-1].message == "Human reviewer rejected the report."


def test_risky_report_escalates_directly_when_human_review_is_disabled():
    result = asyncio.run(
        run_bug_triage_workflow(
            security_bug_report(),
            make_classifier_agent(
                make_llm_response(
                    category=BugCategory.SECURITY,
                    urgency=Urgency.CRITICAL,
                    sentiment=Sentiment.NEUTRAL,
                    recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
                )
            )[0],
            human_approval_enabled=False,
        )
    )

    print_workflow_result("direct escalation workflow", result)

    assert result.status == WorkflowStatus.COMPLETED
    assert result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert result.human_review_required is False
    assert result.human_review_action is None
    assert result.approval_granted is None
    assert result.final_action == (
        "Create an escalation ticket directly because human review "
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
    assert result.event_log[-1].data["human_review_enabled"] is False
    assert (
        result.event_log[-1].data["policy_route"]
        == RouteName.REQUEST_HUMAN_APPROVAL.value
    )
    assert (
        result.event_log[-1].data["effective_route"]
        == RouteName.CREATE_ESCALATION_TICKET.value
    )


def test_disabled_human_review_stream_does_not_request_information():
    async def collect_events():
        stream = stream_bug_triage_workflow(
            security_bug_report(),
            make_classifier_agent(
                make_llm_response(
                    category=BugCategory.SECURITY,
                    urgency=Urgency.CRITICAL,
                    sentiment=Sentiment.NEUTRAL,
                    recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
                )
            )[0],
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
    assert final_result.human_review_required is False
    assert final_result.human_review_action is None
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
            make_classifier_agent()[0],
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
            make_classifier_agent({"invalid": "classifier response"})[0],
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
    assert result.event_log[-1].executor == "classifier_response_executor"
    assert result.event_log[-1].data["error_type"] == "ValidationError"


def test_workflow_returns_failed_result_for_unexpected_llm_client_error():
    result = asyncio.run(
        run_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            make_classifier_agent(
                error=RuntimeError("LLM provider unavailable.")
            )[0],
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
    assert result.event_log[-1].executor == "classifier_agent"
    assert result.event_log[-1].data["error_type"] == "RuntimeError"


def test_workflow_stream_emits_failed_result_for_unexpected_llm_client_error():
    async def collect_events():
        stream = stream_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            make_classifier_agent(
                error=RuntimeError("LLM provider unavailable.")
            )[0],
        )

        return [event async for event in stream]

    events = asyncio.run(collect_events())

    workflow_results = workflow_results_from_events(events)

    assert len(workflow_results) == 1
    final_result = workflow_results[0]
    assert final_result.status == WorkflowStatus.FAILED
    assert final_result.error == (
        "Bug classification failed: LLM provider unavailable."
    )
    assert final_result.final_action is None
    assert workflow_statuses(final_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.FAILED,
    ]
    assert final_result.event_log[-1].executor == "classifier_agent"
    assert final_result.event_log[-1].data["error_type"] == "RuntimeError"


def test_workflow_does_not_mislabel_router_exception_as_classifier_failure(
    monkeypatch,
):
    def raise_router_error(classification, preprocessed_report):
        raise RuntimeError("router policy failed")

    monkeypatch.setattr(workflow_module, "route_triage", raise_router_error)

    with pytest.raises(RuntimeError, match="router policy failed"):
        asyncio.run(
            run_bug_triage_workflow(
                (
                    "In production on Chrome using macOS, when I click save, "
                    "the page shows an error instead of saving. "
                    "It should save successfully."
                ),
                make_classifier_agent()[0],
            )
        )


def test_workflow_stream_does_not_mislabel_router_exception_as_classifier_failure(
    monkeypatch,
):
    def raise_router_error(classification, preprocessed_report):
        raise RuntimeError("router policy failed")

    monkeypatch.setattr(workflow_module, "route_triage", raise_router_error)

    async def collect_events():
        stream = stream_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            make_classifier_agent()[0],
        )
        return [event async for event in stream]

    with pytest.raises(RuntimeError, match="router policy failed"):
        asyncio.run(collect_events())


def test_workflow_stream_emits_final_workflow_result():
    async def collect_events():
        stream = stream_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            make_classifier_agent()[0],
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
    workflow_results = workflow_results_from_events(events)
    assert len(workflow_results) == 1
    final_result = workflow_results[0]
    assert final_result.status == WorkflowStatus.COMPLETED
    assert final_result.event_log[-1].status == WorkflowStatus.COMPLETED


def test_separate_run_calls_do_not_share_trace_events():
    first_result = asyncio.run(
        run_bug_triage_workflow(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            make_classifier_agent()[0],
        )
    )
    second_result = asyncio.run(
        run_bug_triage_workflow(
            "The login page is broken.",
            make_classifier_agent(
                make_llm_response(
                    category=BugCategory.AUTHENTICATION,
                    urgency=Urgency.MEDIUM,
                    sentiment=Sentiment.CONFUSED,
                    missing_info=["browser", "device_or_os"],
                    recommended_route=RouteName.REQUEST_MORE_INFO,
                )
            )[0],
        )
    )

    assert first_result.event_log is not second_result.event_log
    assert workflow_statuses(first_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert workflow_statuses(second_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert first_result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert second_result.selected_route == RouteName.REQUEST_MORE_INFO


def test_separate_streaming_calls_do_not_share_trace_events():
    async def collect_final_result(raw_text: str, classifier_agent: Agent):
        events = [
            event
            async for event in stream_bug_triage_workflow(
                raw_text,
                classifier_agent,
            )
        ]
        workflow_results = workflow_results_from_events(events)
        assert len(workflow_results) == 1
        return workflow_results[0]

    first_result = asyncio.run(
        collect_final_result(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            ),
            make_classifier_agent()[0],
        )
    )
    second_result = asyncio.run(
        collect_final_result(
            "The login page is broken.",
            make_classifier_agent(
                make_llm_response(
                    category=BugCategory.AUTHENTICATION,
                    urgency=Urgency.MEDIUM,
                    sentiment=Sentiment.CONFUSED,
                    missing_info=["browser", "device_or_os"],
                    recommended_route=RouteName.REQUEST_MORE_INFO,
                )
            )[0],
        )
    )

    assert workflow_statuses(first_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert workflow_statuses(second_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert first_result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert second_result.selected_route == RouteName.REQUEST_MORE_INFO


def test_concurrent_independent_runs_do_not_contaminate_trace_events():
    async def run_scenario():
        return await asyncio.gather(
            run_bug_triage_workflow(
                (
                    "In production on Chrome using macOS, when I click save, "
                    "the page shows an error instead of saving. "
                    "It should save successfully."
                ),
                make_classifier_agent()[0],
            ),
            run_bug_triage_workflow(
                "The login page is broken.",
                make_classifier_agent(
                    make_llm_response(
                        category=BugCategory.AUTHENTICATION,
                        urgency=Urgency.MEDIUM,
                        sentiment=Sentiment.CONFUSED,
                        missing_info=["browser", "device_or_os"],
                        recommended_route=RouteName.REQUEST_MORE_INFO,
                    )
                )[0],
            ),
        )

    first_result, second_result = asyncio.run(run_scenario())

    assert first_result.event_log is not second_result.event_log
    assert workflow_statuses(first_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert workflow_statuses(second_result) == [
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.COMPLETED,
    ]
    assert first_result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert second_result.selected_route == RouteName.REQUEST_MORE_INFO


def test_built_workflow_object_cannot_be_reused_for_unrelated_reports():
    async def run_scenario():
        workflow = build_bug_triage_workflow(make_classifier_agent()[0])

        await workflow.run(
            (
                "In production on Chrome using macOS, when I click save, "
                "the page shows an error instead of saving. "
                "It should save successfully."
            )
        )

        with pytest.raises(RuntimeError, match="single-use"):
            await workflow.run(
                (
                    "In production on Chrome using macOS, when I click delete, "
                    "the page shows an error instead of deleting. "
                    "It should delete successfully."
                )
            )

    asyncio.run(run_scenario())
