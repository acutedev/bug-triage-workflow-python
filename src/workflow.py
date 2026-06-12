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
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
)
from typing_extensions import Never

from src.classifier import LLMClassifierClient, classify_bug_report
from src.models import (
    PreprocessedBugReport,
    RouteDecision,
    RouteName,
    TriageClassification,
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


def _route_matches(expected_route: RouteName):
    """Create a switch-case predicate for a specific route."""

    def condition(message: Any) -> bool:
        return (
            isinstance(message, RoutedBugReport)
            and message.route_decision.selected_route == expected_route
        )

    return condition


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
        event_log=[],
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
        ctx: WorkflowContext[ClassifiedBugReport],
    ) -> None:
        classification = await asyncio.to_thread(
            classify_bug_report,
            preprocessed_report,
            llm_client,
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

    @executor(id="request_human_approval_executor")
    async def request_human_approval_executor(
        routed_report: RoutedBugReport,
        ctx: WorkflowContext[Never, WorkflowResult],
    ) -> None:
        result = WorkflowResult(
            status=WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL,
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            classification=routed_report.classification,
            human_approval_required=True,
            approval_granted=None,
            final_action=None,
            error=None,
            event_log=[],
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
            event_log=[],
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
                request_more_info_executor,
                create_standard_ticket_executor,
                request_human_approval_executor,
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
                    target=request_human_approval_executor,
                ),
                Default(target=unexpected_route_executor),
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
