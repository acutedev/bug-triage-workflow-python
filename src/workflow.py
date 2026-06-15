"""Microsoft Agent Framework orchestration for bug triage.

The workflow layer connects the already-tested business components:

raw report -> preprocess -> native classifier agent -> route -> terminal action

Business logic remains in preprocess.py, classifier.py, and router.py. This
module is responsible only for Microsoft Agent Framework orchestration.
"""

from __future__ import annotations

from typing import Any

from agent_framework import (
    Agent,
    AgentExecutorRequest,
    AgentExecutorResponse,
    Case,
    Default,
    Message,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    executor,
)
from typing_extensions import Never

from src.classifier import build_classification_prompt, parse_classification_response
from src.human_approval import HumanReviewExecutor, review_action_matches
from src.models import (
    HumanReviewAction,
    PreprocessedBugReport,
    RouteDecision,
    RouteName,
    WorkflowResult,
    WorkflowStatus,
)
from src.preprocess import preprocess_bug_report
from src.router import route_triage
from src.workflow_messages import (
    ClassifiedBugReport,
    HumanReviewOutcome,
    RoutedBugReport,
)
from src.workflow_results import build_completed_result, build_failed_result
from src.workflow_trace import WorkflowTrace


PREPROCESSED_REPORT_STATE_KEY = "preprocessed_report"


def _route_matches(expected_route: RouteName):
    """Create a switch-case predicate for a specific route."""

    def condition(message: Any) -> bool:
        return (
            isinstance(message, RoutedBugReport)
            and message.route_decision.selected_route == expected_route
        )

    return condition


def build_bug_triage_workflow(
    classifier_agent: Agent,
    *,
    human_approval_enabled: bool = True,
    trace: WorkflowTrace | None = None,
) -> Workflow:
    """Build a fresh Microsoft Agent Framework bug triage workflow.

    Built workflow objects are single-use for independent bug reports. Use the
    public run/stream helpers, or build a fresh workflow for each new report, so
    executor state cannot leak between executions. Human approval resume uses
    the same built workflow and remains supported.
    """
    workflow_trace = trace if trace is not None else WorkflowTrace()
    workflow_started = False

    @executor(id="preprocess_executor")
    async def preprocess_executor(
        raw_text: str,
        ctx: WorkflowContext[PreprocessedBugReport, WorkflowResult],
    ) -> None:
        nonlocal workflow_started
        if workflow_started:
            raise RuntimeError(
                "Bug triage workflows are single-use; build a fresh workflow "
                "for each independent report."
            )
        workflow_started = True

        workflow_trace.append(
            WorkflowStatus.RECEIVED,
            "Bug report received.",
            "preprocess_executor",
        )
        try:
            preprocessed_report = preprocess_bug_report(raw_text)
        except ValueError as error:
            await ctx.yield_output(
                build_failed_result(
                    error,
                    stage="preprocessing",
                    executor="preprocess_executor",
                    trace=workflow_trace,
                )
            )
            return

        workflow_trace.append(
            WorkflowStatus.PREPROCESSED,
            "Bug report preprocessed.",
            "preprocess_executor",
            missing_info=preprocessed_report.missing_info,
        )
        await ctx.send_message(preprocessed_report)

    @executor(id="classifier_request_executor")
    async def classifier_request_executor(
        preprocessed_report: PreprocessedBugReport,
        ctx: WorkflowContext[AgentExecutorRequest],
    ) -> None:
        """Store typed report state and submit a request to the native MAF agent."""
        ctx.set_state(PREPROCESSED_REPORT_STATE_KEY, preprocessed_report)
        prompt = build_classification_prompt(preprocessed_report)

        workflow_trace.enter_classifier_provider_boundary()
        try:
            await ctx.send_message(
                AgentExecutorRequest(
                    messages=[Message("user", contents=[prompt])],
                    should_respond=True,
                )
            )
        except Exception:
            workflow_trace.exit_classifier_provider_boundary()
            raise

    @executor(id="classifier_response_executor")
    async def classifier_response_executor(
        response: AgentExecutorResponse,
        ctx: WorkflowContext[ClassifiedBugReport, WorkflowResult],
    ) -> None:
        """Validate native agent output and restore the typed report from state."""
        workflow_trace.exit_classifier_provider_boundary()
        preprocessed_report: PreprocessedBugReport = ctx.get_state(
            PREPROCESSED_REPORT_STATE_KEY
        )

        try:
            classification = parse_classification_response(
                response.agent_response.text
            )
        except Exception as error:
            await ctx.yield_output(
                build_failed_result(
                    error,
                    stage="classification",
                    executor="classifier_response_executor",
                    trace=workflow_trace,
                )
            )
            return

        workflow_trace.append(
            WorkflowStatus.CLASSIFIED,
            "Bug report classified.",
            "classifier_agent",
            category=classification.category.value,
            urgency=classification.urgency.value,
            sentiment=classification.sentiment.value,
        )

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
        workflow_trace.append(
            WorkflowStatus.ROUTED,
            "Bug report routed.",
            "router_executor",
            selected_route=route_decision.selected_route.value,
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
        result = build_completed_result(
            routed_report,
            workflow_trace,
            final_action="Request additional information from the bug reporter.",
            executor="request_more_info_executor",
        )
        await ctx.yield_output(result)

    @executor(id="create_standard_ticket_executor")
    async def create_standard_ticket_executor(
        routed_or_reviewed_report: RoutedBugReport | HumanReviewOutcome,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        human_review_action = None
        human_review_required = False

        if isinstance(routed_or_reviewed_report, HumanReviewOutcome):
            outcome = routed_or_reviewed_report
            human_review_action = HumanReviewAction.CREATE_STANDARD_TICKET
            human_review_required = True
            workflow_trace.append(
                WorkflowStatus.STANDARD_TICKET_SELECTED,
                (
                    "Human reviewer selected standard-ticket handling "
                    "instead of escalation."
                ),
                "request_human_review_executor",
                approver=outcome.decision.approver,
                human_review_action=outcome.decision.action.value,
            )
            routed_report = RoutedBugReport(
                preprocessed_report=outcome.routed_report.preprocessed_report,
                classification=outcome.routed_report.classification,
                route_decision=RouteDecision(
                    selected_route=RouteName.CREATE_STANDARD_TICKET,
                    reason=(
                        "Human reviewer selected standard-ticket handling "
                        "instead of escalation."
                    ),
                ),
            )
        else:
            routed_report = routed_or_reviewed_report

        result = build_completed_result(
            routed_report,
            workflow_trace,
            final_action="Create a standard bug ticket.",
            executor="create_standard_ticket_executor",
            human_review_required=human_review_required,
            human_review_action=human_review_action,
        )
        await ctx.yield_output(result)

    human_review_executor = HumanReviewExecutor(workflow_trace)

    @executor(id="create_direct_escalation_ticket_executor")
    async def create_direct_escalation_ticket_executor(
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        final_action = (
            "Create an escalation ticket directly because human review "
            "is disabled by configuration."
        )
        workflow_trace.append(
            WorkflowStatus.COMPLETED,
            final_action,
            "create_direct_escalation_ticket_executor",
            human_review_enabled=False,
            policy_route=routed_report.route_decision.selected_route.value,
            effective_route=RouteName.CREATE_ESCALATION_TICKET.value,
        )
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            classification=routed_report.classification,
            human_review_required=False,
            human_review_action=None,
            approval_granted=None,
            final_action=final_action,
            error=None,
            event_log=workflow_trace.snapshot(),
        )
        await ctx.yield_output(result)

    @executor(id="create_escalation_ticket_executor")
    async def create_escalation_ticket_executor(
        outcome: HumanReviewOutcome,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        workflow_trace.append(
            WorkflowStatus.ESCALATION_APPROVED,
            "Human reviewer approved escalation.",
            "request_human_review_executor",
            approver=outcome.decision.approver,
        )
        workflow_trace.append(
            WorkflowStatus.COMPLETED,
            "Escalation ticket created.",
            "create_escalation_ticket_executor",
        )
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            selected_route=RouteName.CREATE_ESCALATION_TICKET,
            classification=outcome.routed_report.classification,
            human_review_required=True,
            human_review_action=HumanReviewAction.APPROVE_ESCALATION,
            approval_granted=True,
            final_action=(
                "Create an escalation ticket for human-reviewed handling."
            ),
            error=None,
            event_log=workflow_trace.snapshot(),
        )
        await ctx.yield_output(result)

    @executor(id="log_rejection_executor")
    async def log_rejection_executor(
        outcome: HumanReviewOutcome,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        workflow_trace.append(
            WorkflowStatus.REPORT_REJECTED,
            "Human reviewer rejected the report.",
            "log_rejection_executor",
            approver=outcome.decision.approver,
            human_review_action=outcome.decision.action.value,
        )
        result = WorkflowResult(
            status=WorkflowStatus.REPORT_REJECTED,
            selected_route=RouteName.LOG_REJECTION,
            classification=outcome.routed_report.classification,
            human_review_required=True,
            human_review_action=HumanReviewAction.REJECT_REPORT,
            approval_granted=False,
            final_action=None,
            error=None,
            event_log=workflow_trace.snapshot(),
        )
        await ctx.yield_output(result)

    @executor(id="unexpected_human_review_action_executor")
    async def unexpected_human_review_action_executor(
        outcome: HumanReviewOutcome,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        error_message = (
            "No Microsoft Agent Framework branch was configured for human "
            f"review action {outcome.decision.action.value}."
        )
        workflow_trace.append(
            WorkflowStatus.FAILED,
            "No workflow branch was configured for the selected human review action.",
            "unexpected_human_review_action_executor",
            human_review_action=outcome.decision.action.value,
        )
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            selected_route=outcome.routed_report.route_decision.selected_route,
            classification=outcome.routed_report.classification,
            human_review_required=True,
            human_review_action=None,
            approval_granted=None,
            final_action=None,
            error=error_message,
            event_log=workflow_trace.snapshot(),
        )
        await ctx.yield_output(result)

    @executor(id="unexpected_route_executor")
    async def unexpected_route_executor(
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        error_message = (
            "No Microsoft Agent Framework branch was configured for route "
            f"{routed_report.route_decision.selected_route.value}."
        )
        workflow_trace.append(
            WorkflowStatus.FAILED,
            "No workflow branch was configured for the selected route.",
            "unexpected_route_executor",
            selected_route=routed_report.route_decision.selected_route.value,
        )
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            selected_route=routed_report.route_decision.selected_route,
            classification=routed_report.classification,
            human_review_required=False,
            human_review_action=None,
            approval_granted=None,
            final_action=None,
            error=error_message,
            event_log=workflow_trace.snapshot(),
        )
        await ctx.yield_output(result)

    human_review_target = (
        human_review_executor
        if human_approval_enabled
        else create_direct_escalation_ticket_executor
    )

    output_executors = [
        preprocess_executor,
        classifier_response_executor,
        request_more_info_executor,
        create_standard_ticket_executor,
        unexpected_route_executor,
    ]

    if human_approval_enabled:
        output_executors.extend(
            [
                human_review_executor,
                create_escalation_ticket_executor,
                log_rejection_executor,
                unexpected_human_review_action_executor,
            ]
        )
    else:
        output_executors.append(create_direct_escalation_ticket_executor)

    workflow_builder = (
        WorkflowBuilder(
            start_executor=preprocess_executor,
            name="bug_triage_workflow",
            description=(
                "Preprocesses, classifies, and deterministically routes bug reports."
            ),
            output_from=output_executors,
        )
        .add_edge(preprocess_executor, classifier_request_executor)
        .add_edge(classifier_request_executor, classifier_agent)
        .add_edge(classifier_agent, classifier_response_executor)
        .add_edge(classifier_response_executor, router_executor)
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
                    target=human_review_target,
                ),
                Default(target=unexpected_route_executor),
            ],
        )
    )

    if human_approval_enabled:
        workflow_builder = workflow_builder.add_switch_case_edge_group(
            human_review_executor,
            [
                Case(
                    condition=review_action_matches(
                        HumanReviewAction.APPROVE_ESCALATION
                    ),
                    target=create_escalation_ticket_executor,
                ),
                Case(
                    condition=review_action_matches(
                        HumanReviewAction.CREATE_STANDARD_TICKET
                    ),
                    target=create_standard_ticket_executor,
                ),
                Case(
                    condition=review_action_matches(HumanReviewAction.REJECT_REPORT),
                    target=log_rejection_executor,
                ),
                Default(target=unexpected_human_review_action_executor),
            ],
        )

    return workflow_builder.build()


async def run_bug_triage_workflow(
    raw_text: str,
    classifier_agent: Agent,
    *,
    human_approval_enabled: bool = True,
) -> WorkflowResult:
    """Run the workflow and return its single validated output."""

    workflow_trace = WorkflowTrace()
    workflow = build_bug_triage_workflow(
        classifier_agent,
        human_approval_enabled=human_approval_enabled,
        trace=workflow_trace,
    )
    try:
        run_result = await workflow.run(raw_text)
    except Exception as error:
        if workflow_trace.is_classifier_provider_boundary_active():
            workflow_trace.exit_classifier_provider_boundary()
            return build_failed_result(
                error,
                stage="classification",
                executor="classifier_agent",
                trace=workflow_trace,
            )
        raise

    outputs = run_result.get_outputs()

    if len(outputs) != 1:
        raise RuntimeError(
            f"Bug triage workflow expected one output but received {len(outputs)}"
        )

    result = outputs[0]
    if not isinstance(result, WorkflowResult):
        raise TypeError("Bug triage workflow output was not a WorkflowResult")

    return result



async def stream_bug_triage_workflow(
    raw_text: str,
    classifier_agent: Agent,
    *,
    human_approval_enabled: bool = True,
):
    """Stream workflow events and convert native-agent failures to typed output."""

    workflow_trace = WorkflowTrace()
    workflow = build_bug_triage_workflow(
        classifier_agent,
        human_approval_enabled=human_approval_enabled,
        trace=workflow_trace,
    )

    try:
        async for event in workflow.run(
            raw_text,
            stream=True,
            include_status_events=True,
        ):
            yield event
    except Exception as error:
        if not workflow_trace.is_classifier_provider_boundary_active():
            raise

        workflow_trace.exit_classifier_provider_boundary()
        yield WorkflowEvent(
            "output",
            executor_id="classifier_agent",
            data=build_failed_result(
                error,
                stage="classification",
                executor="classifier_agent",
                trace=workflow_trace,
            ),
        )
