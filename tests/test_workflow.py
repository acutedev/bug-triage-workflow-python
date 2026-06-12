"""Integration tests for Microsoft Agent Framework bug triage orchestration."""

import asyncio

from src.models import (
    BugCategory,
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
    }


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


def test_workflow_requests_human_approval_for_security_report():
    bug_report = (
        "In production on Chrome using Windows, when I open the account page, "
        "I can view another user's private data instead of my own data. "
        "It should only display my account."
    )

    result = asyncio.run(
        run_bug_triage_workflow(
            bug_report,
            lambda prompt: make_llm_response(
                category=BugCategory.SECURITY,
                urgency=Urgency.CRITICAL,
                sentiment=Sentiment.NEUTRAL,
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            ),
        )
    )

    assert result.status == WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL
    assert result.selected_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert result.human_approval_required is True
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
