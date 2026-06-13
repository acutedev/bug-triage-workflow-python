"""Human approval executor support for bug triage workflows."""

from __future__ import annotations

from typing import Any

from agent_framework import Executor, WorkflowContext, handler, response_handler

from src.models import (
    HumanApprovalDecision,
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


class HumanApprovalExecutor(Executor):
    """Pause the workflow for a typed human approval decision."""

    def __init__(self, trace: WorkflowTrace) -> None:
        super().__init__(id="request_human_approval_executor")
        self._trace = trace

    @handler
    async def request_approval(
        self,
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[HumanApprovalOutcome, WorkflowResult],
    ) -> None:
        self._trace.append(
            WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
            "Waiting for human approval.",
            "request_human_approval_executor",
        )
        waiting_result = WorkflowResult(
            status=WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            classification=routed_report.classification,
            human_approval_required=True,
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
                    "Approve or reject escalation of this bug report. "
                    f"Category: {routed_report.classification.category.value}; "
                    f"urgency: {routed_report.classification.urgency.value}; "
                    f"sentiment: {routed_report.classification.sentiment.value}; "
                    f"reasoning: {routed_report.classification.reasoning}; "
                    f"route reason: "
                    f"{routed_report.route_decision.reason or 'not provided'}."
                ),
            ),
            response_type=HumanApprovalDecision,
        )

    @response_handler(
        request=HumanApprovalRequest,
        response=HumanApprovalDecision,
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


def approval_matches(expected_approval: bool):
    """Create a switch-case predicate for an approval outcome."""

    def condition(message: Any) -> bool:
        return (
            isinstance(message, HumanApprovalOutcome)
            and message.decision.approval_granted is expected_approval
        )

    return condition
