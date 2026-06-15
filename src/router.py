"""Router module applying deterministic workflow routing policy.

The router is intentionally rule-based. The LLM classifier may recommend a
route, but Python policy makes the final decision so risky actions remain
controlled and testable.
"""

from __future__ import annotations

from src.logging_config import get_logger
from src.models import (
    BugCategory,
    PreprocessedBugReport,
    RouteDecision,
    RouteName,
    Sentiment,
    TriageClassification,
    Urgency,
)

logger = get_logger("router")

_RISKY_CATEGORIES = {BugCategory.SECURITY, BugCategory.DATA_LOSS}
_HIGH_RISK_URGENCIES = {Urgency.HIGH, Urgency.CRITICAL}
_HIGH_EMOTION_SENTIMENTS = {Sentiment.FRUSTRATED, Sentiment.ANGRY}


def _has_missing_information(
    preprocessed_report: PreprocessedBugReport | None,
    classification: TriageClassification,
) -> bool:
    """Return whether either preprocessing or classification found missing info."""
    if preprocessed_report and preprocessed_report.has_obvious_missing_info:
        return True
    return bool(classification.missing_info)


def _requires_human_review(classification: TriageClassification) -> bool:
    """Return whether classification requires human review before action."""
    if classification.category in _RISKY_CATEGORIES:
        return True

    if classification.urgency == Urgency.CRITICAL:
        return True

    if (
        classification.urgency in _HIGH_RISK_URGENCIES
        and classification.sentiment in _HIGH_EMOTION_SENTIMENTS
    ):
        return True

    return False


def route_triage(
    classification: TriageClassification,
    preprocessed_report: PreprocessedBugReport | None = None,
) -> RouteDecision:
    """Select the next route for a classified bug report.

    Routing priority:
    1. Risky/security/data-loss/critical cases require human review.
    2. Missing information requests clarification.
    3. Everything else becomes a standard ticket.

    Args:
        classification: Validated LLM classification.
        preprocessed_report: Optional preprocessing result used to consider
            deterministic missing-info signals.

    Returns:
        A validated RouteDecision.
    """
    if _requires_human_review(classification):
        decision = RouteDecision(
            selected_route=RouteName.REQUEST_HUMAN_APPROVAL,
            reason="Risky or high-urgency report requires human review before action.",
        )
    elif _has_missing_information(preprocessed_report, classification):
        decision = RouteDecision(
            selected_route=RouteName.REQUEST_MORE_INFO,
            reason="Important bug report details are missing.",
        )
    else:
        decision = RouteDecision(
            selected_route=RouteName.CREATE_STANDARD_TICKET,
            reason="Report has enough information for a standard bug ticket.",
        )

    logger.info(
        "Triage route selected",
        extra={
            "executor": "route_triage",
            "selected_route": decision.selected_route.value,
            "recommended_route": classification.recommended_route.value,
            "category": classification.category.value,
            "urgency": classification.urgency.value,
            "sentiment": classification.sentiment.value,
        },
    )

    if decision.selected_route != classification.recommended_route:
        logger.warning(
            "LLM route recommendation overridden by policy router",
            extra={
                "executor": "route_triage",
                "selected_route": decision.selected_route.value,
                "recommended_route": classification.recommended_route.value,
            },
        )

    return decision
