"""Tests for LLM-backed bug report classification."""

import json
import logging

import pytest
from pydantic import ValidationError

from src.classifier import build_classification_prompt, classify_bug_report
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


def test_build_classification_prompt_includes_report_context_and_schema():
    preprocessed_report = make_preprocessed_report()

    prompt = build_classification_prompt(preprocessed_report)
    marker = "BUG_REPORT_JSON:\n"
    payload = json.loads(prompt.split(marker, maxsplit=1)[1])

    assert "Return only valid JSON" in prompt
    assert marker in prompt
    assert payload["raw_text"] == preprocessed_report.raw_text
    assert payload["normalized_text"] == preprocessed_report.normalized_text
    assert payload["extracted_fields"] == preprocessed_report.extracted_fields
    assert payload["preprocessor_missing_info"] == []
    assert "authentication" in payload["schema"]["category"]
    assert "request_human_approval" in payload["schema"]["recommended_route"]
    assert "routing_guidance" in payload


def test_classify_bug_report_accepts_mapping_response():
    preprocessed_report = make_preprocessed_report()
    prompts: list[str] = []

    def llm_client(prompt: str) -> dict[str, object]:
        prompts.append(prompt)
        return valid_llm_response()

    classification = classify_bug_report(preprocessed_report, llm_client)

    assert prompts
    assert classification.category == BugCategory.AUTHENTICATION
    assert classification.urgency == Urgency.HIGH
    assert classification.sentiment == Sentiment.FRUSTRATED
    assert classification.recommended_route == RouteName.REQUEST_HUMAN_APPROVAL
    assert classification.confidence == 0.91


def test_classify_bug_report_accepts_json_string_response():
    preprocessed_report = make_preprocessed_report()

    classification = classify_bug_report(
        preprocessed_report,
        lambda prompt: json.dumps(valid_llm_response()),
    )

    assert classification.category == BugCategory.AUTHENTICATION
    assert classification.reasoning == "Login crashes in production and blocks users."


def test_classify_bug_report_rejects_invalid_json_string_response():
    preprocessed_report = make_preprocessed_report()

    with pytest.raises(ValueError, match="LLM response was not valid JSON"):
        classify_bug_report(preprocessed_report, lambda prompt: "{not valid json")


def test_classify_bug_report_rejects_non_object_json_response():
    preprocessed_report = make_preprocessed_report()

    with pytest.raises(ValueError, match="LLM response JSON must be an object"):
        classify_bug_report(preprocessed_report, lambda prompt: '["not", "an", "object"]')


def test_classify_bug_report_rejects_unsupported_response_type():
    preprocessed_report = make_preprocessed_report()

    with pytest.raises(TypeError, match="LLM response must be a JSON string or mapping"):
        classify_bug_report(preprocessed_report, lambda prompt: 123)  # type: ignore[return-value]


def test_classify_bug_report_rejects_invalid_classification_contract():
    preprocessed_report = make_preprocessed_report()
    response = valid_llm_response()
    response["confidence"] = 1.5

    with pytest.raises(ValidationError):
        classify_bug_report(preprocessed_report, lambda prompt: response)


def test_classify_bug_report_rejects_extra_llm_fields():
    preprocessed_report = make_preprocessed_report()
    response = valid_llm_response()
    response["unexpected_field"] = "should be rejected"

    with pytest.raises(ValidationError):
        classify_bug_report(preprocessed_report, lambda prompt: response)


def test_classify_bug_report_logs_start_and_completion(caplog):
    preprocessed_report = make_preprocessed_report()

    with caplog.at_level(logging.INFO, logger="bug_triage_workflow.classifier"):
        classify_bug_report(preprocessed_report, lambda prompt: valid_llm_response())

    assert "Bug report classification started" in caplog.text
    assert "Bug report classification completed" in caplog.text
    assert any(
        record.executor == "classify_bug_report"
        and record.extracted_field_names == ["browser", "environment", "module"]
        and record.preprocessor_missing_info_count == 0
        for record in caplog.records
    )
    assert any(
        getattr(record, "executor", None) == "classify_bug_report"
        and getattr(record, "category", None) == BugCategory.AUTHENTICATION.value
        and getattr(record, "urgency", None) == Urgency.HIGH.value
        and getattr(record, "sentiment", None) == Sentiment.FRUSTRATED.value
        and getattr(record, "recommended_route", None) == RouteName.REQUEST_HUMAN_APPROVAL.value
        and getattr(record, "confidence", None) == 0.91
        for record in caplog.records
    )


def test_classify_bug_report_logs_validation_failure(caplog):
    preprocessed_report = make_preprocessed_report()
    response = valid_llm_response()
    response["confidence"] = -0.1

    with caplog.at_level(logging.ERROR, logger="bug_triage_workflow.classifier"):
        with pytest.raises(ValidationError):
            classify_bug_report(preprocessed_report, lambda prompt: response)

    assert "Bug report classification failed validation" in caplog.text
    assert any(
        getattr(record, "executor", None) == "classify_bug_report"
        for record in caplog.records
    )
