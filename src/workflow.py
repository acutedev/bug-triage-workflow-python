"""Microsoft Agent Framework orchestration for bug triage.

The workflow layer connects the already-tested business components:

raw report -> preprocess -> classify -> route -> terminal action

Business logic remains in preprocess.py, classifier.py, and router.py. This
module is responsible only for Microsoft Agent Framework orchestration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agent_framework import (
    Case,
    Default,
    Executor,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    handler,
    response_handler,
)
from typing_extensions import Never

from src.classifier import LLMClassifierClient, classify_bug_report
from src.models import (
    HumanApprovalDecision,
    PreprocessedBugReport,
    RouteDecision,
    RouteName,
    TriageClassification,
    WorkflowEvent,
    WorkflowResult,
    WorkflowStatus,
)
from src.preprocess import preprocess_bug_report
from src.router import route_triage


@dataclass(frozen=True)
class ClassifiedBugReport:
    """Message passed from the classifier executor to the router."""

    preprocessed_report: PreprocessedBugReport
    classification: TriageClassification


@dataclass(frozen=True)
class RoutedBugReport:
    """Message passed from the router to a terminal branch executor."""

    preprocessed_report: PreprocessedBugReport
    classification: TriageClassification
    route_decision: RouteDecision


@dataclass(frozen=True)
class HumanApprovalRequest:
    """Information presented to the human approver."""

    routed_report: RoutedBugReport
    prompt: str


@dataclass(frozen=True)
class HumanApprovalOutcome:
    """Human decision passed to the approved or rejected branch."""

    routed_report: RoutedBugReport
    decision: HumanApprovalDecision


class HumanApprovalExecutor(Executor):
    """Pause the workflow for a typed human approval decision."""

    def __init__(self) -> None:
        super().__init__(id="request_human_approval_executor")

    @handler
    async def request_approval(
        self,
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[HumanApprovalOutcome, WorkflowResult],
    ) -> None:
        waiting_result = WorkflowResult(
            status=WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            classification=routed_report.classification,
            human_approval_required=True,
            approval_granted=None,
            final_action=None,
            error=None,
            event_log=[
                *_base_event_log(routed_report),
                _workflow_event(
                    WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
                    "Waiting for human approval.",
                    "request_human_approval_executor",
                ),
            ],
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


def _route_matches(expected_route: RouteName):
    """Create a switch-case predicate for a specific route."""

    def condition(message: Any) -> bool:
        return (
            isinstance(message, RoutedBugReport)
            and message.route_decision.selected_route == expected_route
        )

    return condition


def _approval_matches(expected_approval: bool):
    """Create a switch-case predicate for an approval outcome."""

    def condition(message: Any) -> bool:
        return (
            isinstance(message, HumanApprovalOutcome)
            and message.decision.approval_granted is expected_approval
        )

    return condition


def _workflow_event(
    status: WorkflowStatus,
    message: str,
    executor: str,
    **data: Any,
) -> WorkflowEvent:
    """Create one structured workflow trace entry."""

    return WorkflowEvent(
        status=status,
        message=message,
        executor=executor,
        data=data,
    )


def _base_event_log(routed_report: RoutedBugReport) -> list[WorkflowEvent]:
    """Build the shared trace for a successfully routed bug report."""

    return [
        _workflow_event(
            WorkflowStatus.RECEIVED,
            "Bug report received.",
            "preprocess_executor",
        ),
        _workflow_event(
            WorkflowStatus.PREPROCESSED,
            "Bug report preprocessed.",
            "preprocess_executor",
            missing_info=routed_report.preprocessed_report.missing_info,
        ),
        _workflow_event(
            WorkflowStatus.CLASSIFIED,
            "Bug report classified.",
            "classifier_executor",
            category=routed_report.classification.category.value,
            urgency=routed_report.classification.urgency.value,
            sentiment=routed_report.classification.sentiment.value,
        ),
        _workflow_event(
            WorkflowStatus.ROUTED,
            "Bug report routed.",
            "router_executor",
            selected_route=routed_report.route_decision.selected_route.value,
        ),
    ]


def _completed_result(
    routed_report: RoutedBugReport,
    *,
    final_action: str,
    human_approval_required: bool = False,
) -> WorkflowResult:
    """Build a validated completed workflow result."""

    return WorkflowResult(
        status=WorkflowStatus.COMPLETED,
        selected_route=routed_report.route_decision.selected_route,
        classification=routed_report.classification,
        human_approval_required=human_approval_required,
        approval_granted=None,
        final_action=final_action,
        error=None,
        event_log=[
            *_base_event_log(routed_report),
            _workflow_event(
                WorkflowStatus.COMPLETED,
                final_action,
                routed_report.route_decision.selected_route.value,
            ),
        ],
    )


def _failed_result(
    preprocessed_report: PreprocessedBugReport,
    error: Exception,
) -> WorkflowResult:
    """Build a validated terminal result for a classifier failure."""

    error_message = f"Bug classification failed: {error}"
    return WorkflowResult(
        status=WorkflowStatus.FAILED,
        selected_route=None,
        classification=None,
        human_approval_required=False,
        approval_granted=None,
        final_action=None,
        error=error_message,
        event_log=[
            _workflow_event(
                WorkflowStatus.RECEIVED,
                "Bug report received.",
                "preprocess_executor",
            ),
            _workflow_event(
                WorkflowStatus.PREPROCESSED,
                "Bug report preprocessed.",
                "preprocess_executor",
                missing_info=preprocessed_report.missing_info,
            ),
            _workflow_event(
                WorkflowStatus.FAILED,
                error_message,
                "classifier_executor",
            ),
        ],
    )


def build_bug_triage_workflow(llm_client: LLMClassifierClient) -> Workflow:
    """Build a fresh Microsoft Agent Framework bug triage workflow.

    A fresh workflow is created for each run so executor state cannot leak
    between workflow executions.
    """

    @executor(id="preprocess_executor")
    async def preprocess_executor(
        raw_text: str,
        ctx: WorkflowContext[PreprocessedBugReport],
    ) -> None:
        preprocessed_report = preprocess_bug_report(raw_text)
        await ctx.send_message(preprocessed_report)

    @executor(id="classifier_executor")
    async def classifier_executor(
        preprocessed_report: PreprocessedBugReport,
        ctx: WorkflowContext[ClassifiedBugReport, WorkflowResult],
    ) -> None:
        try:
            classification = await asyncio.to_thread(
                classify_bug_report,
                preprocessed_report,
                llm_client,
            )
        except ValueError as error:
            await ctx.yield_output(
                _failed_result(preprocessed_report, error)
            )
            return

        await ctx.send_message(
            ClassifiedBugReport(
                preprocessed_report=preprocessed_report,
                classification=classification,
            )
        )

    @executor(id="router_executor")
    async def router_executor(
        classified_report: ClassifiedBugReport,
        ctx: WorkflowContext[RoutedBugReport],
    ) -> None:
        route_decision = route_triage(
            classified_report.classification,
            classified_report.preprocessed_report,
        )

        await ctx.send_message(
            RoutedBugReport(
                preprocessed_report=classified_report.preprocessed_report,
                classification=classified_report.classification,
                route_decision=route_decision,
            )
        )

    @executor(id="request_more_info_executor")
    async def request_more_info_executor(
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        result = _completed_result(
            routed_report,
            final_action="Request additional information from the bug reporter.",
        )
        await ctx.yield_output(result)

    @executor(id="create_standard_ticket_executor")
    async def create_standard_ticket_executor(
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        result = _completed_result(
            routed_report,
            final_action="Create a standard bug ticket.",
        )
        await ctx.yield_output(result)

    human_approval_executor = HumanApprovalExecutor()

    @executor(id="create_escalation_ticket_executor")
    async def create_escalation_ticket_executor(
        outcome: HumanApprovalOutcome,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            classification=outcome.routed_report.classification,
            human_approval_required=True,
            approval_granted=True,
            final_action=(
                "Create an escalation ticket for human-reviewed handling."
            ),
            error=None,
            event_log=[
                *_base_event_log(outcome.routed_report),
                _workflow_event(
                    WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
                    "Waiting for human approval.",
                    "request_human_approval_executor",
                ),
                _workflow_event(
                    WorkflowStatus.APPROVED,
                    "Human reviewer approved escalation.",
                    "request_human_approval_executor",
                    approver=outcome.decision.approver,
                ),
                _workflow_event(
                    WorkflowStatus.COMPLETED,
                    "Escalation ticket created.",
                    "create_escalation_ticket_executor",
                ),
            ],
        )
        await ctx.yield_output(result)

    @executor(id="log_rejection_executor")
    async def log_rejection_executor(
        outcome: HumanApprovalOutcome,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        result = WorkflowResult(
            status=WorkflowStatus.REJECTED,
            selected_route=RouteName.LOG_REJECTION,
            classification=outcome.routed_report.classification,
            human_approval_required=True,
            approval_granted=False,
            final_action=None,
            error=None,
            event_log=[
                *_base_event_log(outcome.routed_report),
                _workflow_event(
                    WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
                    "Waiting for human approval.",
                    "request_human_approval_executor",
                ),
                _workflow_event(
                    WorkflowStatus.REJECTED,
                    "Human reviewer rejected escalation.",
                    "log_rejection_executor",
                    approver=outcome.decision.approver,
                ),
            ],
        )
        await ctx.yield_output(result)

    @executor(id="unexpected_route_executor")
    async def unexpected_route_executor(
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            selected_route=routed_report.route_decision.selected_route,
            classification=routed_report.classification,
            human_approval_required=False,
            approval_granted=None,
            final_action=None,
            error=(
                "No Microsoft Agent Framework branch was configured for route "
                f"{routed_report.route_decision.selected_route.value}."
            ),
            event_log=[
                *_base_event_log(routed_report),
                _workflow_event(
                    WorkflowStatus.FAILED,
                    "No workflow branch was configured for the selected route.",
                    "unexpected_route_executor",
                    selected_route=(
                        routed_report.route_decision.selected_route.value
                    ),
                ),
            ],
        )
        await ctx.yield_output(result)

    return (
        WorkflowBuilder(
            start_executor=preprocess_executor,
            name="bug_triage_workflow",
            description=(
                "Preprocesses, classifies, and deterministically routes bug reports."
            ),
            output_from=[
                classifier_executor,
                request_more_info_executor,
                create_standard_ticket_executor,
                human_approval_executor,
                create_escalation_ticket_executor,
                log_rejection_executor,
                unexpected_route_executor,
            ],
        )
        .add_edge(preprocess_executor, classifier_executor)
        .add_edge(classifier_executor, router_executor)
        .add_switch_case_edge_group(
            router_executor,
            [
                Case(
                    condition=_route_matches(RouteName.REQUEST_MORE_INFO),
                    target=request_more_info_executor,
                ),
                Case(
                    condition=_route_matches(RouteName.CREATE_STANDARD_TICKET),
                    target=create_standard_ticket_executor,
                ),
                Case(
                    condition=_route_matches(RouteName.REQUEST_HUMAN_APPROVAL),
                    target=human_approval_executor,
                ),
                Default(target=unexpected_route_executor),
            ],
        )
        .add_switch_case_edge_group(
            human_approval_executor,
            [
                Case(
                    condition=_approval_matches(True),
                    target=create_escalation_ticket_executor,
                ),
                Default(target=log_rejection_executor),
            ],
        )
        .build()
    )


async def run_bug_triage_workflow(
    raw_text: str,
    llm_client: LLMClassifierClient,
) -> WorkflowResult:
    """Run the workflow and return its single validated output."""

    workflow = build_bug_triage_workflow(llm_client)
    run_result = await workflow.run(raw_text)
    outputs = run_result.get_outputs()

    if len(outputs) != 1:
        raise RuntimeError(
            f"Bug triage workflow expected one output but received {len(outputs)}"
        )

    result = outputs[0]
    if not isinstance(result, WorkflowResult):
        raise TypeError("Bug triage workflow output was not a WorkflowResult")

    return result


def stream_bug_triage_workflow(
    raw_text: str,
    llm_client: LLMClassifierClient,
):
    """Return a Microsoft Agent Framework event stream for the workflow."""

    workflow = build_bug_triage_workflow(llm_client)
    return workflow.run(
        raw_text,
        stream=True,
        include_status_events=True,
    )
