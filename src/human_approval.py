"""Human review executor support for bug triage workflows."""

from __future__ import annotations

from typing import Any

from agent_framework import Executor, WorkflowContext, handler, response_handler

from src.models import (
    HumanReviewAction,
    HumanReviewDecision,
    RouteName,
    WorkflowResult,
    WorkflowStatus,
)
from src.workflow_messages import (
    HumanApprovalOutcome,
    HumanApprovalRequest,
    RoutedBugReport,
)
from src.workflow_trace import WorkflowTrace


class HumanReviewExecutor(Executor):
    """Pause the workflow for a typed human review decision."""

    def __init__(self, trace: WorkflowTrace) -> None:
        super().__init__(id="request_human_review_executor")
        self._trace = trace

    @handler
    async def request_approval(
        self,
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[HumanApprovalOutcome, WorkflowResult],
    ) -> None:
        self._trace.append(
            WorkflowStatus.AWAITING_HUMAN_REVIEW,
            "Waiting for human review.",
            "request_human_review_executor",
        )
        waiting_result = WorkflowResult(
            status=WorkflowStatus.AWAITING_HUMAN_REVIEW,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            classification=routed_report.classification,
            human_review_required=True,
            approval_granted=None,
            final_action=None,
            error=None,
            event_log=self._trace.snapshot(),
        )
        await ctx.yield_output(waiting_result)

        await ctx.request_info(
            request_data=HumanApprovalRequest(
                routed_report=routed_report,
                prompt=(
                    "Choose escalation, standard-ticket handling, or rejection "
                    "for this bug report. "
                    f"Category: {routed_report.classification.category.value}; "
                    f"urgency: {routed_report.classification.urgency.value}; "
                    f"sentiment: {routed_report.classification.sentiment.value}; "
                    f"reasoning: {routed_report.classification.reasoning}; "
                    f"route reason: "
                    f"{routed_report.route_decision.reason or 'not provided'}."
                ),
            ),
            response_type=HumanReviewDecision,
        )

    @response_handler(
        request=HumanApprovalRequest,
        response=HumanReviewDecision,
        output=HumanApprovalOutcome,
    )
    async def handle_decision(
        self,
        original_request,
        decision,
        ctx,
    ) -> None:
        await ctx.send_message(
            HumanApprovalOutcome(
                routed_report=original_request.routed_report,
                decision=decision,
            )
        )


def review_action_matches(expected_action: HumanReviewAction):
    """Create a switch-case predicate for a human review action."""

    def condition(message: Any) -> bool:
        return (
            isinstance(message, HumanApprovalOutcome)
            and message.decision.action == expected_action
        )

    return condition
