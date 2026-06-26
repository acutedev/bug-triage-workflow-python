"""Characterization tests for WorkflowResult invariants.

These tests document existing validation behavior that was not directly covered
by test_models.py. Each rejection test targets a specific validator and uses
match= to confirm the intended error message fired.

Note: some PREPROCESSED field errors (human_review_action, approval_granted)
cannot be isolated because _validate_empty_review_summary_for_status checks
human_review_required first, and the global _validate_human_review_summary
rejects inputs where human_review_action or approval_granted are set without
human_review_required=True. Those cases assert the earliest authoritative
PREPROCESSED invariant (human_review_required) with a comment explaining the
ordering constraint.
"""

import pytest
from pydantic import ValidationError

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
        reasoning="Characterization test classification.",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# PREPROCESSED — review-summary fields are forbidden (parallel to RECEIVED)
#
# Validator order inside _validate_empty_review_summary_for_status:
#   1. human_review_required
#   2. human_review_action
#   3. approval_granted
#
# To pass _validate_human_review_summary (global, runs first), inputs that
# include human_review_action or approval_granted must also set
# human_review_required=True. That means the PREPROCESSED check for those
# fields is always preempted by the human_review_required check. The cases
# below therefore assert the earliest authoritative PREPROCESSED invariant.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"human_review_required": True},
            "human_review_required must be false when status is preprocessed",
        ),
        (
            # human_review_action cannot be isolated: human_review_required must
            # be True to pass the global summary check, but then the PREPROCESSED
            # validator rejects human_review_required first.
            {
                "human_review_required": True,
                "human_review_action": HumanReviewAction.CREATE_STANDARD_TICKET,
            },
            "human_review_required must be false when status is preprocessed",
        ),
        (
            # approval_granted cannot be isolated: same ordering constraint as
            # human_review_action above. APPROVE_ESCALATION requires
            # approval_granted=True to satisfy the global summary check.
            {
                "human_review_required": True,
                "human_review_action": HumanReviewAction.APPROVE_ESCALATION,
                "approval_granted": True,
            },
            "human_review_required must be false when status is preprocessed",
        ),
    ],
)
def test_preprocessed_workflow_result_rejects_review_summary_fields(
    overrides: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValidationError, match=expected_message):
        WorkflowResult(
            status=WorkflowStatus.PREPROCESSED,
            **overrides,
        )


# ---------------------------------------------------------------------------
# CLASSIFIED — error and final_action are forbidden at this intermediate stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("error", "something went wrong"),
        ("final_action", "Create a ticket."),
    ],
)
def test_classified_workflow_result_rejects_terminal_fields(field_name, field_value):
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.CLASSIFIED,
            classification=make_classification(),
            **{field_name: field_value},
        )


# ---------------------------------------------------------------------------
# ROUTED — error field is forbidden (final_action already tested)
# ---------------------------------------------------------------------------


def test_routed_workflow_result_with_error_rejected():
    with pytest.raises(ValidationError):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            classification=make_classification(),
            error="something went wrong",
        )


# ---------------------------------------------------------------------------
# AWAITING_HUMAN_REVIEW — selected_route must be exactly REQUEST_HUMAN_APPROVAL
# ---------------------------------------------------------------------------


def test_awaiting_human_review_rejects_wrong_selected_route():
    with pytest.raises(
        ValidationError,
        match="selected_route must be request_human_approval while awaiting human review",
    ):
        WorkflowResult(
            status=WorkflowStatus.AWAITING_HUMAN_REVIEW,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            classification=make_classification(
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL
            ),
            human_review_required=True,
        )


# ---------------------------------------------------------------------------
# ESCALATION_APPROVED — classification required and route must be exact
# ---------------------------------------------------------------------------


def test_escalation_approved_status_requires_classification():
    with pytest.raises(
        ValidationError,
        match="classification is required when status is escalation_approved",
    ):
        WorkflowResult(
            status=WorkflowStatus.ESCALATION_APPROVED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.APPROVE_ESCALATION,
            approval_granted=True,
        )


def test_escalation_approved_status_rejects_wrong_selected_route():
    with pytest.raises(
        ValidationError,
        match="selected_route must be create_escalation_ticket",
    ):
        WorkflowResult(
            status=WorkflowStatus.ESCALATION_APPROVED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            classification=make_classification(
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL
            ),
            human_review_required=True,
            human_review_action=HumanReviewAction.APPROVE_ESCALATION,
            approval_granted=True,
        )


# ---------------------------------------------------------------------------
# STANDARD_TICKET_SELECTED — classification required, route must be exact,
# human_review_action must be present
# ---------------------------------------------------------------------------


def test_standard_ticket_selected_status_requires_classification():
    with pytest.raises(
        ValidationError,
        match="classification is required when status is standard_ticket_selected",
    ):
        WorkflowResult(
            status=WorkflowStatus.STANDARD_TICKET_SELECTED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approval_granted=None,
        )


def test_standard_ticket_selected_status_rejects_wrong_selected_route():
    with pytest.raises(
        ValidationError,
        match="selected_route must be create_standard_ticket",
    ):
        WorkflowResult(
            status=WorkflowStatus.STANDARD_TICKET_SELECTED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            classification=make_classification(
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL
            ),
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approval_granted=None,
        )


def test_standard_ticket_selected_status_requires_human_review_action():
    with pytest.raises(
        ValidationError,
        match="create_standard_ticket route requires create_standard_ticket human review action",
    ):
        WorkflowResult(
            status=WorkflowStatus.STANDARD_TICKET_SELECTED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            classification=make_classification(
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL
            ),
            human_review_required=True,
        )


# ---------------------------------------------------------------------------
# FAILED — human_review_action must be None
# ---------------------------------------------------------------------------


def test_failed_workflow_result_may_preserve_human_review_required():
    result = WorkflowResult(
        status=WorkflowStatus.FAILED,
        human_review_required=True,
        error="Workflow failed during human review.",
    )
    assert result.human_review_required is True


def test_failed_workflow_result_with_human_review_action_rejected():
    with pytest.raises(
        ValidationError,
        match="human_review_action must be None when status is failed",
    ):
        WorkflowResult(
            status=WorkflowStatus.FAILED,
            error="workflow failed",
            human_review_required=True,
            human_review_action=HumanReviewAction.CREATE_STANDARD_TICKET,
            approval_granted=None,
        )


# ---------------------------------------------------------------------------
# AWAITING_HUMAN_REVIEW — final_action error uses status-specific phrasing
# ---------------------------------------------------------------------------


def test_awaiting_human_review_final_action_preserves_specific_error():
    with pytest.raises(
        ValidationError,
        match="final_action must be None while awaiting human review",
    ):
        WorkflowResult(
            status=WorkflowStatus.AWAITING_HUMAN_REVIEW,
            classification=make_classification(
                recommended_route=RouteName.REQUEST_HUMAN_APPROVAL
            ),
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_review_required=True,
            final_action="Done.",
        )


# ---------------------------------------------------------------------------
# COMPLETED + REQUEST_MORE_INFO — valid happy path
# ---------------------------------------------------------------------------


def test_completed_request_more_info_without_review_is_valid():
    result = WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=RouteName.REQUEST_MORE_INFO,
        classification=make_classification(
            recommended_route=RouteName.REQUEST_MORE_INFO
        ),
        human_review_required=False,
        final_action="Requested additional information from the reporter.",
    )
    assert result.status is WorkflowStatus.COMPLETED
    assert result.selected_route is RouteName.REQUEST_MORE_INFO
    assert result.human_review_required is False
    assert result.human_review_action is None
    assert result.approval_granted is None


# ---------------------------------------------------------------------------
# Check-ordering characterization tests
# These pin the order in which field checks fire so that the lookup-table
# refactor cannot silently reorder validation without a test catching it.
# ---------------------------------------------------------------------------


def test_classified_final_action_checked_before_classification():
    with pytest.raises(
        ValidationError,
        match="final_action must be None when status is classified",
    ):
        WorkflowResult(
            status=WorkflowStatus.CLASSIFIED,
            final_action="Create a ticket.",
        )


def test_routed_error_checked_before_classification():
    with pytest.raises(
        ValidationError,
        match="error must be None when status is routed",
    ):
        WorkflowResult(
            status=WorkflowStatus.ROUTED,
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            error="something went wrong",
        )


def test_completed_route_checked_before_error():
    with pytest.raises(
        ValidationError,
        match="selected_route is required when status is completed",
    ):
        WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            classification=make_classification(),
            final_action="Done.",
            error="also present",
        )


def test_awaiting_human_review_classification_checked_before_error():
    with pytest.raises(
        ValidationError,
        match="classification is required when status is awaiting_human_review",
    ):
        WorkflowResult(
            status=WorkflowStatus.AWAITING_HUMAN_REVIEW,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            human_review_required=True,
            error="also present",
        )
