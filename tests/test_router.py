"""Tests for deterministic triage routing."""

import logging

from src.models import (
    BugCategory,
    PreprocessedBugReport,
    RouteName,
    Sentiment,
    TriageClassification,
    Urgency,
)
from src.router import route_triage


def make_classification(
    *,
    category: BugCategory = BugCategory.UI_BUG,
    urgency: Urgency = Urgency.MEDIUM,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    missing_info: list[str] | None = None,
    recommended_route: RouteName = RouteName.CREATE_STANDARD_TICKET,
) -> TriageClassification:
    return TriageClassification(
        category=category,
        urgency=urgency,
        sentiment=sentiment,
        missing_info=missing_info or [],
        recommended_route=recommended_route,
        reasoning="Validated classification for router testing.",
        confidence=0.9,
    )


def make_preprocessed_report(*, has_missing_info: bool = False) -> PreprocessedBugReport:
    return PreprocessedBugReport(
        raw_text="The page crashes when I click save.",
        normalized_text="The page crashes when I click save.",
        extracted_fields={"module": "ui_bug"},
        missing_info=["browser"] if has_missing_info else [],
        has_obvious_missing_info=has_missing_info,
    )


# Human approval route tests


def test_security_category_requires_human_approval():
    classification = make_classification(
        category=BugCategory.SECURITY,
        urgency=Urgency.MEDIUM,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_HUMAN_APPROVAL


def test_data_loss_category_requires_human_approval():
    classification = make_classification(
        category=BugCategory.DATA_LOSS,
        urgency=Urgency.MEDIUM,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_HUMAN_APPROVAL


def test_critical_urgency_requires_human_approval():
    classification = make_classification(
        category=BugCategory.PERFORMANCE,
        urgency=Urgency.CRITICAL,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_HUMAN_APPROVAL



def test_high_urgency_angry_sentiment_requires_human_approval():
    classification = make_classification(
        category=BugCategory.AUTHENTICATION,
        urgency=Urgency.HIGH,
        sentiment=Sentiment.ANGRY,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_HUMAN_APPROVAL


def test_high_urgency_frustrated_sentiment_requires_human_approval():
    classification = make_classification(
        category=BugCategory.AUTHENTICATION,
        urgency=Urgency.HIGH,
        sentiment=Sentiment.FRUSTRATED,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_HUMAN_APPROVAL


# Missing information route tests


def test_classification_missing_info_routes_to_request_more_info():
    classification = make_classification(
        missing_info=["browser"],
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_MORE_INFO


def test_preprocessed_missing_info_routes_to_request_more_info():
    classification = make_classification(recommended_route=RouteName.CREATE_STANDARD_TICKET)
    preprocessed_report = make_preprocessed_report(has_missing_info=True)

    decision = route_triage(classification, preprocessed_report)

    assert decision.selected_route == RouteName.REQUEST_MORE_INFO



# Standard ticket route tests


def test_high_urgency_neutral_sentiment_does_not_require_human_approval():
    classification = make_classification(
        category=BugCategory.AUTHENTICATION,
        urgency=Urgency.HIGH,
        sentiment=Sentiment.NEUTRAL,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.CREATE_STANDARD_TICKET


def test_complete_non_risky_report_routes_to_standard_ticket():
    classification = make_classification(
        category=BugCategory.UI_BUG,
        urgency=Urgency.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )
    preprocessed_report = make_preprocessed_report(has_missing_info=False)

    decision = route_triage(classification, preprocessed_report)

    assert decision.selected_route == RouteName.CREATE_STANDARD_TICKET


# Policy priority tests


def test_risky_report_takes_priority_over_missing_info():
    classification = make_classification(
        category=BugCategory.SECURITY,
        urgency=Urgency.HIGH,
        missing_info=["browser"],
        recommended_route=RouteName.REQUEST_MORE_INFO,
    )

    decision = route_triage(classification)

    assert decision.selected_route == RouteName.REQUEST_HUMAN_APPROVAL



# Logging tests


def test_safe_matching_llm_recommendation_is_not_overridden(caplog):
    classification = make_classification(
        category=BugCategory.UI_BUG,
        urgency=Urgency.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    with caplog.at_level(logging.WARNING, logger="bug_triage_workflow.router"):
        decision = route_triage(classification)

    assert decision.selected_route == RouteName.CREATE_STANDARD_TICKET
    assert "LLM route recommendation overridden by policy router" not in caplog.text


def test_route_triage_logs_selected_route(caplog):
    classification = make_classification(recommended_route=RouteName.CREATE_STANDARD_TICKET)

    with caplog.at_level(logging.INFO, logger="bug_triage_workflow.router"):
        route_triage(classification)

    assert "Triage route selected" in caplog.text
    assert any(
        record.executor == "route_triage"
        and record.selected_route == RouteName.CREATE_STANDARD_TICKET.value
        and record.recommended_route == RouteName.CREATE_STANDARD_TICKET.value
        for record in caplog.records
    )


def test_route_triage_logs_warning_when_llm_recommendation_is_overridden(caplog):
    classification = make_classification(
        category=BugCategory.SECURITY,
        recommended_route=RouteName.CREATE_STANDARD_TICKET,
    )

    with caplog.at_level(logging.WARNING, logger="bug_triage_workflow.router"):
        route_triage(classification)

    assert "LLM route recommendation overridden by policy router" in caplog.text
    assert any(
        record.executor == "route_triage"
        and record.selected_route == RouteName.REQUEST_HUMAN_APPROVAL.value
        and record.recommended_route == RouteName.CREATE_STANDARD_TICKET.value
        for record in caplog.records
    )
