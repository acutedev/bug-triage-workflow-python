"""Reusable WorkflowResult builders for bug triage orchestration."""

from __future__ import annotations

from src.models import WorkflowResult, WorkflowStatus
from src.workflow_messages import RoutedBugReport
from src.workflow_trace import WorkflowTrace


def build_completed_result(
    routed_report: RoutedBugReport,
    trace: WorkflowTrace,
    *,
    final_action: str,
    executor: str,
    human_approval_required: bool = False,
) -> WorkflowResult:
    """Build a validated completed workflow result."""

    trace.append(
        WorkflowStatus.COMPLETED,
        final_action,
        executor,
    )

    return WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=routed_report.route_decision.selected_route,
        classification=routed_report.classification,
        human_approval_required=human_approval_required,
        approval_granted=None,
        final_action=final_action,
        error=None,
        event_log=trace.snapshot(),
    )


def build_failed_result(
    error: Exception,
    *,
    stage: str,
    executor: str,
    trace: WorkflowTrace,
) -> WorkflowResult:
    """Build a validated terminal result for an expected workflow failure."""

    error_message = f"Bug {stage} failed: {error}"
    trace.append(
        WorkflowStatus.FAILED,
        error_message,
        executor,
        error_type=type(error).__name__,
    )

    return WorkflowResult(
        status=WorkflowStatus.FAILED,
        selected_route=None,
        classification=None,
        human_approval_required=False,
        approval_granted=None,
        final_action=None,
        error=error_message,
        event_log=trace.snapshot(),
    )
