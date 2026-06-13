"""Tests for workflow models."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from src.models import (
    BugCategory,
    BugReportInput,
    HumanApprovalDecision,
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
        human_approval_required=False,
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
                status=WorkflowStatus.PREPROCESSED,
                executor="preprocess_bug_report",
                message="Bug report preprocessed.",
            )
        ],
    )
    assert len(result.event_log) == 1
    assert result.event_log[0].executor == "preprocess_bug_report"


# HumanApprovalDecision tests


def test_human_approval_decision_can_represent_approval_granted():
    approved = HumanApprovalDecision(required=True, approval_granted=True, approver="alice")
    assert approved.required is True
    assert approved.approval_granted is True
    assert approved.approver == "alice"


def test_human_approval_decision_can_represent_rejection():
    rejected = HumanApprovalDecision(
        required=True,
        approval_granted=False,
        approver="bob",
        notes="Not approved",
    )
    assert rejected.required is True
    assert rejected.approval_granted is False
    assert rejected.approver == "bob"
    assert rejected.notes == "Not approved"


def test_human_approval_not_required_without_decision_rejected():
    with pytest.raises(ValidationError):
        HumanApprovalDecision(required=False)


def test_human_approval_not_required_with_decision_rejected():
    with pytest.raises(ValidationError):
        HumanApprovalDecision(required=False, approval_granted=True)


def test_human_approval_required_without_decision_rejected():
    with pytest.raises(ValidationError):
        HumanApprovalDecision(required=True)


def test_human_approval_granted_without_approver_rejected():
    with pytest.raises(ValidationError):
        HumanApprovalDecision(required=True, approval_granted=True)


def test_human_approval_rejected_without_approver_rejected():
    with pytest.raises(ValidationError):
        HumanApprovalDecision(required=True, approval_granted=False)


def test_blank_human_approval_text_fields_rejected():
    with pytest.raises(ValidationError):
        HumanApprovalDecision(required=True, approval_granted=True, approver="   ")

    with pytest.raises(ValidationError):
        HumanApprovalDecision(
            required=True,
            approval_granted=False,
            approver="bob",
            notes="   ",
        )


# WorkflowResult human approval state tests


def test_waiting_for_human_approval_requires_human_approval_flag():
    with pytest.raises(ValidationError):
        WorkflowResult(status=WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL)


def test_waiting_for_human_approval_with_flag_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
        selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
        human_approval_required=True,
    )
    assert result.status == WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL
    assert result.human_approval_required is True


def test_approval_granted_requires_human_approval_flag():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            approval_granted=True,
        )


def test_approved_status_requires_granted_approval():
    result = WorkflowResult(
        status=WorkflowStatus.APPROVED,
        selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
        human_approval_required=True,
        approval_granted=True,
    )
    assert result.status == WorkflowStatus.APPROVED
    assert result.approval_granted is True


def test_approved_status_without_granted_approval_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.APPROVED,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_approval_required=True,
            approval_granted=None,
        )


def test_rejected_status_requires_rejected_approval():
    result = WorkflowResult(
        status=WorkflowStatus.REJECTED,
        selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
        human_approval_required=True,
        approval_granted=False,
    )
    assert result.status == WorkflowStatus.REJECTED
    assert result.approval_granted is False


def test_rejected_status_with_granted_approval_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.REJECTED,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_approval_required=True,
            approval_granted=True,
        )


def test_escalation_ticket_route_requires_granted_approval():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            human_approval_required=True,
            approval_granted=False,
        )


def test_escalation_ticket_route_with_granted_approval_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_ESCALATION_TICKET,
        human_approval_required=True,
        approval_granted=True,
        final_action="Created urgent escalation ticket.",
    )
    assert result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert result.approval_granted is True


def test_direct_escalation_without_required_approval_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.CREATE_ESCALATION_TICKET,
        human_approval_required=False,
        approval_granted=None,
        final_action="Created escalation ticket directly.",
    )

    assert result.selected_route == RouteName.CREATE_ESCALATION_TICKET
    assert result.human_approval_required is False
    assert result.approval_granted is None
    assert result.final_action == "Created escalation ticket directly."
