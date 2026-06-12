"""Preprocessing executor for incoming bug reports.

This module implements the plain-function workflow executor. It prepares raw
bug report text for the classifier by normalizing whitespace, extracting simple
rule-based fields, and identifying obvious missing information.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from src.logging_config import get_logger
from src.models import BugReportInput, PreprocessedBugReport

logger = get_logger("preprocess")

_BROWSER_KEYWORDS: Mapping[str, str] = {
    "chrome": "Chrome",
    "safari": "Safari",
    "firefox": "Firefox",
    "edge": "Edge",
    "microsoft edge": "Edge",
    "brave": "Brave",
    "opera": "Opera",
    "vivaldi": "Vivaldi",
    "internet explorer": "Internet Explorer",
    "ie": "Internet Explorer",
}

_ENVIRONMENT_KEYWORDS: Mapping[str, str] = {
    "production": "production",
    "prod": "production",
    "live": "production",
    "staging": "staging",
    "stage": "staging",
    "test": "test",
    "testing": "test",
    "qa": "qa",
    "uat": "uat",
    "sandbox": "sandbox",
    "dev": "development",
    "development": "development",
    "local": "local",
}

_DEVICE_OR_OS_KEYWORDS: Mapping[str, str] = {
    "mac": "macOS",
    "macos": "macOS",
    "mac os": "macOS",
    "osx": "macOS",
    "macbook": "macOS",
    "iphone": "iPhone",
    "ios": "iOS",
    "ipad": "iPad",
    "ipados": "iPadOS",
    "android": "Android",
    "windows": "Windows",
    "win": "Windows",
    "pc": "Windows",
    "linux": "Linux",
    "ubuntu": "Linux",
    "chromebook": "ChromeOS",
    "chromeos": "ChromeOS",
    "chrome os": "ChromeOS",
    "mobile": "mobile",
    "tablet": "tablet",
    "desktop": "desktop",
    "laptop": "laptop",
}

_MODULE_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "authentication": ("login", "log in", "signin", "sign in", "password", "auth"),
    "billing": ("billing", "invoice", "payment", "charge", "refund", "subscription"),
    "performance": ("slow", "timeout", "latency", "lag", "loading", "performance"),
    "data_loss": ("data loss", "missing data", "deleted", "disappeared", "corrupt"),
    "security": ("security", "permission", "unauthorized", "vulnerability", "exploit"),
    "ui_bug": ("button", "screen", "page", "layout", "modal", "dropdown", "form"),
}

_REPRODUCTION_KEYWORDS = (
    "steps",
    "reproduce",
    "replicate",
    "when i",
    "after i",
    "before i",
    "while i",
    "if i",
    "click",
    "tap",
    "select",
    "choose",
    "enter",
    "type",
    "submit",
    "save",
    "upload",
    "download",
    "open",
    "refresh",
    "navigate",
    "go to",
)

_EXPECTED_BEHAVIOR_KEYWORDS = (
    "expect",
    "expected",
    "should",
    "supposed to",
    "intended",
    "want it to",
    "needs to",
)

_ACTUAL_BEHAVIOR_KEYWORDS = (
    "actual",
    "instead",
    "error",
    "crash",
    "crashes",
    "fails",
    "failed",
    "failure",
    "does not",
    "doesn't",
    "cannot",
    "can't",
    "unable",
    "broken",
    "blank",
    "stuck",
    "hangs",
    "freezes",
    "shows",
    "displays",
    "404",
    "500",
)


def normalize_text(raw_text: str) -> str:
    """Normalize whitespace in raw bug report text."""
    return " ".join(raw_text.split())


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Build a case-insensitive whole-keyword regex pattern.

    Multi-word keywords allow flexible whitespace between words. The boundary
    checks prevent false positives such as matching "ios" inside "curious" or
    "edge" inside "knowledge".
    """
    escaped_words = [re.escape(part) for part in keyword.split()]
    flexible_keyword = r"\s+".join(escaped_words)
    return re.compile(rf"(?<!\w){flexible_keyword}(?!\w)", re.IGNORECASE)


def _contains_keyword(text: str, keyword: str) -> bool:
    return bool(_keyword_pattern(keyword).search(text))


def _contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    return any(_contains_keyword(text, keyword) for keyword in keywords)


def _extract_first_keyword_match(text: str, keyword_map: Mapping[str, str]) -> str | None:
    for keyword, value in keyword_map.items():
        if _contains_keyword(text, keyword):
            return value
    return None


def _detect_module(text: str) -> str | None:
    for module, keywords in _MODULE_KEYWORDS.items():
        if _contains_any_keyword(text, keywords):
            return module
    return None


def extract_fields(normalized_text: str) -> dict[str, str]:
    """Extract obvious structured fields from normalized bug report text."""
    extracted_fields: dict[str, str] = {}

    module = _detect_module(normalized_text)
    if module:
        extracted_fields["module"] = module

    browser = _extract_first_keyword_match(normalized_text, _BROWSER_KEYWORDS)
    if browser:
        extracted_fields["browser"] = browser

    environment = _extract_first_keyword_match(normalized_text, _ENVIRONMENT_KEYWORDS)
    if environment:
        extracted_fields["environment"] = environment

    device_or_os = _extract_first_keyword_match(normalized_text, _DEVICE_OR_OS_KEYWORDS)
    if device_or_os:
        extracted_fields["device_or_os"] = device_or_os

    return extracted_fields


def detect_missing_info(normalized_text: str, extracted_fields: Mapping[str, str]) -> list[str]:
    """Return important details that appear to be missing from the report."""
    missing_info: list[str] = []

    if not _contains_any_keyword(normalized_text, _REPRODUCTION_KEYWORDS):
        missing_info.append("steps_to_reproduce")

    if "environment" not in extracted_fields:
        missing_info.append("environment")

    if "browser" not in extracted_fields:
        missing_info.append("browser")

    if "device_or_os" not in extracted_fields:
        missing_info.append("device_or_os")

    if not _contains_any_keyword(normalized_text, _EXPECTED_BEHAVIOR_KEYWORDS):
        missing_info.append("expected_behavior")

    if not _contains_any_keyword(normalized_text, _ACTUAL_BEHAVIOR_KEYWORDS):
        missing_info.append("actual_behavior")

    return missing_info


def _coerce_bug_report_input(report: BugReportInput | str) -> BugReportInput:
    if isinstance(report, BugReportInput):
        return report

    if isinstance(report, str):
        return BugReportInput(raw_text=report)

    raise TypeError("report must be a BugReportInput or raw string")


def preprocess_bug_report(report: BugReportInput | str) -> PreprocessedBugReport:
    """Preprocess an incoming bug report.

    Args:
        report: Either a validated BugReportInput or a raw bug report string.

    Returns:
        A validated PreprocessedBugReport ready for classification.
    """
    bug_report = _coerce_bug_report_input(report)
    normalized_text = normalize_text(bug_report.raw_text)
    extracted_fields = extract_fields(normalized_text)
    missing_info = detect_missing_info(normalized_text, extracted_fields)

    logger.info(
        "Bug report preprocessed",
        extra={
            "executor": "preprocess_bug_report",
            "missing_info_count": len(missing_info),
            "extracted_field_names": sorted(extracted_fields.keys()),
        },
    )

    return PreprocessedBugReport(
        raw_text=bug_report.raw_text,
        normalized_text=normalized_text,
        extracted_fields=extracted_fields,
        missing_info=missing_info,
        has_obvious_missing_info=bool(missing_info),
    )
