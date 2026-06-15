"""Internal transport messages passed between workflow executors."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import (
    HumanReviewDecision,
    PreprocessedBugReport,
    RouteDecision,
    TriageClassification,
)


@dataclass(frozen=True)
class ClassifiedBugReport:
    """Message passed from the native classifier agent adapter to the router."""

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
    """Human review decision passed to the selected terminal branch."""

    routed_report: RoutedBugReport
    decision: HumanReviewDecision
