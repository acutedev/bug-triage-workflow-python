"""Tests for workflow models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.models import (
    BugCategory,
    BugReportInput,
    HumanReviewAction,
    HumanReviewDecision,
    PreprocessedBugReport,
    RouteDecision,
    RouteName,
    Sentiment,
    TriageClassification,
    Urgency,
    WorkflowEvent,
    WorkflowResult,
    WorkflowStatus,
)


# BugReportInput tests


def test_valid_bug_report_input():
    report = BugReportInput(raw_text="The login page crashes when I submit the form.")
    assert report.raw_text == "The login page crashes when I submit the form."


def test_empty_bug_report_input_rejected():
    with pytest.raises(ValidationError):
        BugReportInput(raw_text="   ")


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        BugReportInput(
            raw_text="The login page crashes when I submit the form.",
            unexpected_field="should not be accepted",
        )


# PreprocessedBugReport tests


def test_valid_preprocessed_bug_report():
    preprocessed = PreprocessedBugReport(
        raw_text="The login page crashes.",
        normalized_text="The login page crashes.",
        extracted_fields={"module": "login"},
        missing_info=["browser", "device"],
        has_obvious_missing_info=True,
    )
    assert preprocessed.normalized_text == "The login page crashes."
    assert preprocessed.extracted_fields == {"module": "login"}
    assert preprocessed.missing_info == ["browser", "device"]
    assert preprocessed.has_obvious_missing_info is True


def test_blank_normalized_text_rejected():
    with pytest.raises(ValidationError):
        PreprocessedBugReport(
            raw_text="The login page crashes.",
            normalized_text="   ",
        )


def test_blank_preprocessed_missing_info_item_rejected():
    with pytest.raises(ValidationError):
        PreprocessedBugReport(
            raw_text="The login page crashes.",
            normalized_text="The login page crashes.",
            missing_info=["browser", "   "],
        )


def test_preprocessed_missing_info_requires_flag_true():
    with pytest.raises(ValidationError):
        PreprocessedBugReport(
            raw_text="The login page crashes.",
            normalized_text="The login page crashes.",
            missing_info=["browser"],
            has_obvious_missing_info=False,
        )


def test_preprocessed_empty_missing_info_requires_flag_false():
    with pytest.raises(ValidationError):
        PreprocessedBugReport(
            raw_text="The login page crashes.",
            normalized_text="The login page crashes.",
            missing_info=[],
            has_obvious_missing_info=True,
        )


# TriageClassification tests


def test_valid_triage_classification():
    classification = TriageClassification(
        category=BugCategory.AUTHENTICATION,
        urgency=Urgency.HIGH,
        sentiment=Sentiment.FRUSTRATED,
        missing_info=["reproduction steps"],
        recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
        reasoning="The report indicates a critical login failure.",
        confidence=0.85,
    )
    assert classification.category == BugCategory.AUTHENTICATION
    assert classification.confidence == 0.85


def test_invalid_confidence_below_zero_rejected():
    with pytest.raises(ValidationError):
        TriageClassification(
            category=BugCategory.UI_BUG,
            urgency=Urgency.MEDIUM,
            sentiment=Sentiment.NEUTRAL,
            missing_info=[],
            recommended_route=RouteName.CREATE_STANDARD_TICKET,
            reasoning="Some UI issue.",
            confidence=-0.1,
        )


def test_invalid_confidence_above_one_rejected():
    with pytest.raises(ValidationError):
        TriageClassification(
            category=BugCategory.UI_BUG,
            urgency=Urgency.MEDIUM,
            sentiment=Sentiment.NEUTRAL,
            missing_info=[],
            recommended_route=RouteName.CREATE_STANDARD_TICKET,
            reasoning="Some UI issue.",
            confidence=1.1,
        )


def test_confidence_boundaries_are_accepted():
    low_confidence = TriageClassification(
        category=BugCategory.UI_BUG,
        urgency=Urgency.LOW,
        sentiment=Sentiment.NEUTRAL,
        missing_info=[],
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
        reasoning="Low confidence classification is still valid.",
        confidence=0.0,
    )

    high_confidence = TriageClassification(
        category=BugCategory.UI_BUG,
        urgency=Urgency.LOW,
        sentiment=Sentiment.NEUTRAL,
        missing_info=[],
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
        reasoning="Full confidence classification is valid.",
        confidence=1.0,
    )

    assert low_confidence.confidence == 0.0
    assert high_confidence.confidence == 1.0


def test_empty_reasoning_rejected():
    with pytest.raises(ValidationError):
        TriageClassification(
            category=BugCategory.SECURITY,
            urgency=Urgency.HIGH,
            sentiment=Sentiment.ANGRY,
            missing_info=["exploit details"],
            recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            reasoning="",
            confidence=0.9,
        )


def test_blank_classification_missing_info_item_rejected():
    with pytest.raises(ValidationError):
        TriageClassification(
            category=BugCategory.AUTHENTICATION,
            urgency=Urgency.HIGH,
            sentiment=Sentiment.FRUSTRATED,
            missing_info=["browser", ""],
            recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            reasoning="The report indicates a critical login failure.",
            confidence=0.85,
        )


def test_extra_fields_rejected_in_triage_classification():
    with pytest.raises(ValidationError):
        TriageClassification(
            category=BugCategory.AUTHENTICATION,
            urgency=Urgency.HIGH,
            sentiment=Sentiment.FRUSTRATED,
            missing_info=[],
            recommended_route=RouteName.REQUEST_HUMAN_APPROVAL,
            reasoning="The login issue blocks users.",
            confidence=0.9,
            unexpected_llm_field="hallucinated value",
        )


# RouteDecision tests


def test_valid_route_decision():
    decision = RouteDecision(
        selected_route=RouteName.REQUEST_MORE_INFO,
        reason="Important reproduction details are missing.",
    )
    assert decision.selected_route == RouteName.REQUEST_MORE_INFO
    assert decision.reason == "Important reproduction details are missing."


def test_blank_route_decision_reason_rejected():
    with pytest.raises(ValidationError):
        RouteDecision(
            selected_route=RouteName.REQUEST_MORE_INFO,
            reason="   ",
        )


# WorkflowResult tests


def test_workflow_result_without_selected_route_when_failed():
    result = WorkflowResult(
        status=WorkflowStatus.FAILED,
        selected_route=None,
        classification=None,
        human_review_required=False,
        approval_granted=None,
        final_action=None,
        error="validation failure",
    )
    assert result.status == WorkflowStatus.FAILED
    assert result.selected_route is None
    assert result.final_action is None
    assert result.error == "validation failure"


def test_failed_workflow_result_without_error_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(status=WorkflowStatus.FAILED)


def test_failed_workflow_result_with_blank_error_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(status=WorkflowStatus.FAILED, error="   ")


def test_failed_workflow_result_with_final_action_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.FAILED,
            error="validation failure",
            final_action="Create a ticket anyway.",
        )


def test_failed_workflow_result_with_approval_granted_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.FAILED,
            error="validation failure",
            human_review_required=True,
            approval_granted=False,
        )


def test_completed_workflow_result_requires_final_action():
    with pytest.raises(ValidationError):
        WorkflowResult(status=WorkflowStatus.COMPLETED)


def test_completed_workflow_result_with_final_action_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_STANDARD_TICKET,
        final_action="Created standard bug ticket.",
    )
    assert result.status == WorkflowStatus.COMPLETED
    assert result.final_action == "Created standard bug ticket."


def test_completed_workflow_result_with_error_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            final_action="Created standard bug ticket.",
            error="unexpected error",
        )


def test_workflow_result_updated_at_cannot_be_before_created_at():
    created_at = datetime.now(UTC)

    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            final_action="Created standard bug ticket.",
            created_at=created_at,
            updated_at=created_at - timedelta(seconds=1),
        )


# WorkflowEvent tests


def test_valid_workflow_event():
    event = WorkflowEvent(
        status=WorkflowStatus.PREPROCESSED,
        executor="preprocess_bug_report",
        message="Bug report preprocessed.",
        data={"missing_info_count": 2},
    )
    assert event.status == WorkflowStatus.PREPROCESSED
    assert event.executor == "preprocess_bug_report"
    assert event.message == "Bug report preprocessed."
    assert event.data == {"missing_info_count": 2}
    assert event.created_at is not None


def test_blank_workflow_event_message_rejected():
    with pytest.raises(ValidationError):
        WorkflowEvent(status=WorkflowStatus.PREPROCESSED, message="   ")


def test_blank_workflow_event_executor_rejected():
    with pytest.raises(ValidationError):
        WorkflowEvent(
            status=WorkflowStatus.PREPROCESSED,
            executor="   ",
            message="Bug report preprocessed.",
        )


def test_workflow_result_event_log_uses_workflow_events():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_STANDARD_TICKET,
        final_action="Created standard bug ticket.",
        event_log=[
            WorkflowEvent(
                status=WorkflowStatus.COMPLETED,
                executor="create_standard_ticket_executor",
                message="Created standard bug ticket.",
            )
        ],
    )
    assert len(result.event_log) == 1
    assert result.event_log[0].executor == "create_standard_ticket_executor"


def test_workflow_result_final_event_status_must_match_result_status():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            final_action="Created standard bug ticket.",
            event_log=[
                WorkflowEvent(
                    status=WorkflowStatus.ROUTED,
                    executor="router_executor",
                    message="Bug report routed.",
                )
            ],
        )


def test_workflow_result_empty_event_log_is_still_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_STANDARD_TICKET,
        final_action="Created standard bug ticket.",
        event_log=[],
    )

    assert result.event_log == []


# HumanReviewDecision tests


@pytest.mark.parametrize(
    "action",
    [
        HumanReviewAction.APPROVE_ESCALATION,
        HumanReviewAction.CREATE_STANDARD_TICKET,
        HumanReviewAction.REJECT_REPORT,
    ],
)
def test_human_review_decision_accepts_all_actions(action):
    decision = HumanReviewDecision(
        required=True,
        action=action,
        approver="alice",
        notes="Reviewed by support.",
    )
    assert decision.required is True
    assert decision.action == action
    assert decision.approver == "alice"
    assert decision.notes == "Reviewed by support."


def test_human_review_decision_rejects_invalid_action():
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            required=True,
            action="not_a_review_action",
            approver="alice",
        )


def test_human_review_decision_rejects_required_false():
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            required=False,
            action=HumanReviewAction.APPROVE_ESCALATION,
            approver="alice",
        )


def test_human_review_decision_rejects_blank_approver():
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            required=True,
            action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approver="   ",
        )


def test_human_review_decision_rejects_blank_optional_notes():
    with pytest.raises(ValidationError):
        HumanReviewDecision(
            required=True,
            action=HumanReviewAction.REJECT_REPORT,
            approver="bob",
            notes="   ",
        )


# WorkflowResult human review state tests


def test_awaiting_human_review_requires_human_review_flag():
    with pytest.raises(ValidationError):
        WorkflowResult(status=WorkflowStatus.AWAITING_HUMAN_REVIEW)


def test_awaiting_human_review_with_flag_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.AWAITING_HUMAN_REVIEW,
        selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
        human_review_required=True,
    )
    assert result.status == WorkflowStatus.AWAITING_HUMAN_REVIEW
    assert result.human_review_required is True
    assert result.human_review_action is None


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("approval_granted", False),
        ("human_review_action", HumanReviewAction.REJECT_REPORT),
        ("final_action", "Create an escalation ticket."),
        ("error", "unexpected error"),
    ],
)
def test_awaiting_human_review_rejects_terminal_fields(
    field_name,
    field_value,
):
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.AWAITING_HUMAN_REVIEW,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_review_required=True,
            **{field_name: field_value},
        )


def test_approval_granted_requires_human_review_flag():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            approval_granted=True,
        )


def test_human_review_action_requires_human_review_flag():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
        )


@pytest.mark.parametrize(
    ("action", "approval_granted"),
    [
        (HumanReviewAction.APPROVE_ESCALATION, None),
        (HumanReviewAction.CREATE_STANDARD_TICKET, True),
        (HumanReviewAction.REJECT_REPORT, None),
    ],
)
def test_human_review_action_must_match_approval_summary(
    action,
    approval_granted,
):
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_review_required=True,
            human_review_action=action,
            approval_granted=approval_granted,
        )


def test_escalation_approved_status_requires_escalation_review_action():
    result = WorkflowResult(
        status=WorkflowStatus.ESCALATION_APPROVED,
        selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
        human_review_required=True,
        human_review_action=HumanReviewAction.APPROVE_ESCALATION,
        approval_granted=True,
    )
    assert result.status == WorkflowStatus.ESCALATION_APPROVED
    assert result.human_review_action == HumanReviewAction.APPROVE_ESCALATION
    assert result.approval_granted is True


def test_escalation_approved_status_without_escalation_action_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ESCALATION_APPROVED,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_review_required=True,
            approval_granted=True,
        )


def test_standard_ticket_selected_status_requires_standard_ticket_action():
    result = WorkflowResult(
        status=WorkflowStatus.STANDARD_TICKET_SELECTED,
        selected_route=RouteName.CREATE_STANDARD_TICKET,
        human_review_required=True,
        human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
        approval_granted=None,
    )
    assert result.status == WorkflowStatus.STANDARD_TICKET_SELECTED
    assert result.human_review_action == HumanReviewAction.CREATE_STANDARD_TICKET
    assert result.approval_granted is None


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("final_action", "Create a standard bug ticket."),
        ("error", "unexpected error"),
    ],
)
def test_standard_ticket_selected_status_rejects_terminal_fields(field_name, field_value):
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.STANDARD_TICKET_SELECTED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            **{field_name: field_value},
        )


def test_report_rejected_status_requires_reject_report_action():
    result = WorkflowResult(
        status=WorkflowStatus.REPORT_REJECTED,
        selected_route=RouteName.LOG_REJECTION,
        human_review_required=True,
        human_review_action=HumanReviewAction.REJECT_REPORT,
        approval_granted=False,
    )
    assert result.status == WorkflowStatus.REPORT_REJECTED
    assert result.human_review_action == HumanReviewAction.REJECT_REPORT
    assert result.approval_granted is False


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("final_action", "Create an escalation ticket."),
        ("error", "unexpected error"),
    ],
)
def test_report_rejected_status_rejects_terminal_fields(field_name, field_value):
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.REPORT_REJECTED,
            selected_route=RouteName.LOG_REJECTION,
            human_review_required=True,
            human_review_action=HumanReviewAction.REJECT_REPORT,
            approval_granted=False,
            **{field_name: field_value},
        )


def test_report_rejected_status_with_granted_approval_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.REPORT_REJECTED,
            selected_route=RouteName.LOG_REJECTION,
            human_review_required=True,
            human_review_action=HumanReviewAction.REJECT_REPORT,
            approval_granted=True,
        )


def test_escalation_ticket_route_requires_escalation_review_action():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.REJECT_REPORT,
            approval_granted=False,
            final_action="Created urgent escalation ticket.",
        )


def test_escalation_ticket_route_with_escalation_action_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_ESCALATION_TICKET,
        human_review_required=True,
        human_review_action=HumanReviewAction.APPROVE_ESCALATION,
        approval_granted=True,
        final_action="Created urgent escalation ticket.",
    )
    assert result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert result.human_review_action == HumanReviewAction.APPROVE_ESCALATION
    assert result.approval_granted is True


def test_standard_ticket_route_after_review_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_STANDARD_TICKET,
        human_review_required=True,
        human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
        approval_granted=None,
        final_action="Create a standard bug ticket.",
    )
    assert result.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert result.human_review_action == HumanReviewAction.CREATE_STANDARD_TICKET
    assert result.approval_granted is None


def test_standard_ticket_route_after_review_rejects_escalation_action():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.APPROVE_ESCALATION,
            approval_granted=True,
            final_action="Create a standard bug ticket.",
        )


def test_log_rejection_route_requires_reject_report_action():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.REPORT_REJECTED,
            selected_route=RouteName.LOG_REJECTION,
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approval_granted=None,
        )


def test_direct_escalation_without_required_approval_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_ESCALATION_TICKET,
        human_review_required=False,
        human_review_action=None,
        approval_granted=None,
        final_action="Created escalation ticket directly.",
    )

    assert result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert result.human_review_required is False
    assert result.human_review_action is None
    assert result.approval_granted is None
    assert result.final_action == "Created escalation ticket directly."
