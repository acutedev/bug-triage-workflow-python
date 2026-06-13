"""Tests for LLM-backed bug report classification."""

import json
import logging

import pytest
from pydantic import ValidationError

from src.classifier import (
    CLASSIFIER_AGENT_INSTRUCTIONS,
    build_classification_prompt,
    parse_classification_response,
)
from src.models import (
    BugCategory,
    PreprocessedBugReport,
    RouteName,
    Sentiment,
    Urgency,
)


def make_preprocessed_report() -> PreprocessedBugReport:
    return PreprocessedBugReport(
        raw_text="In production on Chrome, the login page crashes.",
        normalized_text="In production on Chrome, the login page crashes.",
        extracted_fields={
            "browser": "Chrome",
            "environment": "production",
            "module": "authentication",
        },
        missing_info=[],
        has_obvious_missing_info=False,
    )


def valid_llm_response() -> dict[str, object]:
    return {
        "category": "authentication",
        "urgency": "high",
        "sentiment": "frustrated",
        "missing_info": [],
        "recommended_route": "request_human_approval",
        "reasoning": "Login crashes in production and blocks users.",
        "confidence": 0.91,
    }


def test_classifier_agent_instructions_hold_business_rules():
    assert "TriageClassification" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "request_more_info" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "request_human_approval" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "create_standard_ticket" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "Do not invent missing information" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "confidence score" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "chain-of-thought" in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "category, urgency, sentiment" not in CLASSIFIER_AGENT_INSTRUCTIONS
    assert "BUG_REPORT_JSON" not in CLASSIFIER_AGENT_INSTRUCTIONS


def test_build_classification_prompt_includes_only_report_context():
    preprocessed_report = make_preprocessed_report()

    prompt = build_classification_prompt(preprocessed_report)
    marker = "BUG_REPORT_JSON:\n"
    payload = json.loads(prompt.split(marker, maxsplit=1)[1])

    assert prompt.startswith("Classify the following preprocessed bug report.")
    assert marker in prompt
    assert payload["raw_text"] == preprocessed_report.raw_text
    assert payload["normalized_text"] == preprocessed_report.normalized_text
    assert payload["extracted_fields"] == preprocessed_report.extracted_fields
    assert payload["preprocessor_missing_info"] == []
    assert "schema" not in payload
    assert "routing_guidance" not in payload


def test_parse_classification_response_accepts_mapping_response():
    classification = parse_classification_response(valid_llm_response())

    assert classification.category == BugCategory.AUTHENTICATION
    assert classification.urgency == Urgency.HIGH
    assert classification.sentiment == Sentiment.FRUSTRATED
    assert classification.recommended_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert classification.confidence == 0.91


def test_parse_classification_response_accepts_json_string_response():
    classification = parse_classification_response(json.dumps(valid_llm_response()))

    assert classification.category == BugCategory.AUTHENTICATION
    assert classification.reasoning == "Login crashes in production and blocks users."


def test_parse_classification_response_rejects_invalid_json_string_response():
    with pytest.raises(ValueError, match="LLM response was not valid JSON"):
        parse_classification_response("{not valid json")


def test_parse_classification_response_rejects_non_object_json_response():
    with pytest.raises(ValueError, match="LLM response JSON must be an object"):
        parse_classification_response('["not", "an", "object"]')


def test_parse_classification_response_rejects_unsupported_response_type():
    with pytest.raises(TypeError, match="LLM response must be a JSON string or mapping"):
        parse_classification_response(123)  # type: ignore[arg-type]


def test_parse_classification_response_rejects_invalid_classification_contract():
    response = valid_llm_response()
    response["confidence"] = 1.5

    with pytest.raises(ValidationError):
        parse_classification_response(response)


def test_parse_classification_response_rejects_extra_llm_fields():
    response = valid_llm_response()
    response["unexpected_field"] = "should be rejected"

    with pytest.raises(ValidationError):
        parse_classification_response(response)


def test_parse_classification_response_logs_completion(caplog):
    with caplog.at_level(logging.INFO, logger="bug_triage_workflow.classifier"):
        parse_classification_response(valid_llm_response())

    assert "Bug report classification completed" in caplog.text
    assert any(
        getattr(record, "executor", None) == "classifier_agent"
        and getattr(record, "category", None) == BugCategory.AUTHENTICATION.value
        and getattr(record, "urgency", None) == Urgency.HIGH.value
        and getattr(record, "sentiment", None) == Sentiment.FRUSTRATED.value
        and getattr(record, "recommended_route", None) == RouteName.REQUEST_HUMAN_APPROVAL.value
        and getattr(record, "confidence", None) == 0.91
        for record in caplog.records
    )


def test_parse_classification_response_logs_validation_failure(caplog):
    response = valid_llm_response()
    response["confidence"] = -0.1

    with caplog.at_level(logging.ERROR, logger="bug_triage_workflow.classifier"):
        with pytest.raises(ValidationError):
            parse_classification_response(response)

    assert "Bug report classification failed validation" in caplog.text
    assert any(
        getattr(record, "executor", None) == "classifier_agent"
        for record in caplog.records
    )
    assert all(record.exc_info is None for record in caplog.records)
