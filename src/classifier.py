"""Prompt and validation helpers for bug report triage classification.

The native Microsoft Agent Framework classifier node uses this module to build
its prompt and validate the agent response against the strict
TriageClassification contract. The callable classifier entry point remains for
focused unit tests and backward-compatible provider-independent use.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from src.logging_config import get_logger
from src.models import (
    BugCategory,
    PreprocessedBugReport,
    RouteName,
    Sentiment,
    TriageClassification,
    Urgency,
)

logger = get_logger("classifier")

LLMClassifierClient = Callable[[str], str | Mapping[str, Any]]

_CLASSIFICATION_SYSTEM_INSTRUCTIONS = """You are a bug triage classifier.
Return only valid JSON matching the requested schema.
Do not include markdown, code fences, explanations, or extra fields.
"""

_CLASSIFICATION_SCHEMA_DESCRIPTION = {
    "category": [category.value for category in BugCategory],
    "urgency": [urgency.value for urgency in Urgency],
    "sentiment": [sentiment.value for sentiment in Sentiment],
    "missing_info": "array of strings describing important missing details",
    "recommended_route": [route.value for route in RouteName],
    "reasoning": "brief explanation for the classification",
    "confidence": "number between 0.0 and 1.0",
}

CLASSIFIER_AGENT_INSTRUCTIONS = _CLASSIFICATION_SYSTEM_INSTRUCTIONS


def build_classification_prompt(preprocessed_report: PreprocessedBugReport) -> str:
    """Build the prompt sent to the LLM classifier."""
    payload = {
        "raw_text": preprocessed_report.raw_text,
        "normalized_text": preprocessed_report.normalized_text,
        "extracted_fields": preprocessed_report.extracted_fields,
        "preprocessor_missing_info": preprocessed_report.missing_info,
        "schema": _CLASSIFICATION_SCHEMA_DESCRIPTION,
        "routing_guidance": {
            "request_more_info": "Use when required report details are missing.",
            "request_human_approval": "Use for security, data loss, critical urgency, or risky/high-emotion cases.",
            "create_standard_ticket": "Use for complete non-risky bug reports.",
        },
    }

    return (
        f"{_CLASSIFICATION_SYSTEM_INSTRUCTIONS}\n"
        "Classify the following preprocessed bug report.\n"
        "BUG_REPORT_JSON:\n"
        f"{json.dumps(payload, sort_keys=True)}"
    )


def _coerce_llm_response_to_mapping(response: str | Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert an LLM response into a mapping for Pydantic validation."""
    if isinstance(response, Mapping):
        return response

    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc

        if not isinstance(parsed, Mapping):
            raise ValueError("LLM response JSON must be an object")

        return parsed

    raise TypeError("LLM response must be a JSON string or mapping")


def parse_classification_response(
    response: str | Mapping[str, Any],
    *,
    executor: str = "classifier_agent",
) -> TriageClassification:
    """Parse and validate a classifier agent response."""
    response_mapping = _coerce_llm_response_to_mapping(response)

    try:
        classification = TriageClassification.model_validate(response_mapping)
    except ValidationError:
        logger.exception(
            "Bug report classification failed validation",
            extra={"executor": executor},
        )
        raise

    logger.info(
        "Bug report classification completed",
        extra={
            "executor": executor,
            "category": classification.category.value,
            "urgency": classification.urgency.value,
            "sentiment": classification.sentiment.value,
            "recommended_route": classification.recommended_route.value,
            "confidence": classification.confidence,
        },
    )

    return classification


def classify_bug_report(
    preprocessed_report: PreprocessedBugReport,
    llm_client: LLMClassifierClient,
) -> TriageClassification:
    """Classify a preprocessed bug report using an injected LLM client.

    Args:
        preprocessed_report: Validated preprocessed report.
        llm_client: Callable that accepts a prompt and returns either a JSON
            string or a mapping compatible with TriageClassification.

    Returns:
        A validated TriageClassification.

    Raises:
        ValueError: If the LLM returns malformed JSON.
        ValidationError: If the JSON does not match the strict classification contract.
        TypeError: If the LLM client returns an unsupported response type.
    """
    prompt = build_classification_prompt(preprocessed_report)

    logger.info(
        "Bug report classification started",
        extra={
            "executor": "classify_bug_report",
            "extracted_field_names": sorted(preprocessed_report.extracted_fields.keys()),
            "preprocessor_missing_info_count": len(preprocessed_report.missing_info),
        },
    )

    response = llm_client(prompt)
    return parse_classification_response(
        response,
        executor="classify_bug_report",
    )
