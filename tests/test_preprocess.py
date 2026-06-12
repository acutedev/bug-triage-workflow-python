"""Tests for bug report preprocessing."""

import logging

import pytest
from pydantic import ValidationError

from src.models import BugReportInput
from src.preprocess import (
    detect_missing_info,
    extract_fields,
    normalize_text,
    preprocess_bug_report,
)


# normalize_text tests


def test_normalize_text_collapses_whitespace():
    raw_text = "Login   page\n\ncrashes\twhen I submit."

    normalized = normalize_text(raw_text)

    assert normalized == "Login page crashes when I submit."


# extract_fields tests


def test_extract_fields_detects_authentication_module():
    fields = extract_fields("The login page crashes after I submit my password.")

    assert fields["module"] == "authentication"


def test_extract_fields_detects_billing_module():
    fields = extract_fields("The invoice and payment screen fails.")

    assert fields["module"] == "billing"


def test_extract_fields_detects_data_loss_module():
    fields = extract_fields("Customer records disappeared and there may be data loss.")

    assert fields["module"] == "data_loss"


def test_extract_fields_detects_ui_bug_module():
    fields = extract_fields("The dropdown layout is broken on the settings page.")

    assert fields["module"] == "ui_bug"


def test_extract_fields_detects_browser_environment_and_device():
    fields = extract_fields("In production on Chrome using macOS, the form fails.")

    assert fields["browser"] == "Chrome"
    assert fields["environment"] == "production"
    assert fields["device_or_os"] == "macOS"


def test_extract_fields_detects_additional_browsers():
    assert extract_fields("The app fails in Firefox.")["browser"] == "Firefox"
    assert extract_fields("The app fails in Brave.")["browser"] == "Brave"
    assert extract_fields("The app fails in Microsoft Edge.")["browser"] == "Edge"
    assert extract_fields("The app fails in Opera.")["browser"] == "Opera"


def test_extract_fields_detects_additional_environments():
    assert extract_fields("This breaks in staging.")["environment"] == "staging"
    assert extract_fields("This breaks in the test environment.")["environment"] == "test"
    assert extract_fields("This breaks in QA.")["environment"] == "qa"
    assert extract_fields("This breaks in UAT.")["environment"] == "uat"
    assert extract_fields("This breaks in sandbox.")["environment"] == "sandbox"


def test_extract_fields_detects_additional_devices_and_operating_systems():
    assert extract_fields("This fails on iPad.")["device_or_os"] == "iPad"
    assert extract_fields("This fails on iPadOS.")["device_or_os"] == "iPadOS"
    assert extract_fields("This fails on Chromebook.")["device_or_os"] == "ChromeOS"
    assert extract_fields("This fails on Ubuntu.")["device_or_os"] == "Linux"
    assert extract_fields("This fails on mobile.")["device_or_os"] == "mobile"
    assert extract_fields("This fails on laptop.")["device_or_os"] == "laptop"


def test_extract_fields_is_case_insensitive():
    fields = extract_fields("In PRODUCTION on CHROME using MACOS, the login form fails.")

    assert fields["browser"] == "Chrome"
    assert fields["environment"] == "production"
    assert fields["device_or_os"] == "macOS"
    assert fields["module"] == "authentication"


def test_extract_fields_does_not_match_keywords_inside_larger_words():
    fields = extract_fields("The curious knowledge base article is unclear.")

    assert "device_or_os" not in fields
    assert "browser" not in fields


def test_extract_fields_returns_empty_dict_when_nothing_obvious_found():
    fields = extract_fields("Something is wrong and I need help.")

    assert fields == {}


# detect_missing_info tests


def test_detect_missing_info_returns_expected_missing_items():
    missing_info = detect_missing_info(
        normalized_text="Something is broken.",
        extracted_fields={},
    )

    assert missing_info == [
        "steps_to_reproduce",
        "environment",
        "browser",
        "device_or_os",
        "expected_behavior",
    ]


def test_detect_missing_info_includes_actual_behavior_when_no_failure_signal_present():
    missing_info = detect_missing_info(
        normalized_text="Something needs attention.",
        extracted_fields={},
    )

    assert "actual_behavior" in missing_info


def test_detect_missing_info_omits_items_that_are_present():
    text = (
        "In production on Chrome using macOS, when I click submit, "
        "the page crashes instead of saving. It should save successfully."
    )
    fields = extract_fields(text)

    missing_info = detect_missing_info(text, fields)

    assert missing_info == []


def test_detect_missing_info_accepts_broader_reproduction_keywords():
    text = (
        "In staging on Firefox using Windows, after I tap save, "
        "the screen freezes instead of saving. I expect it to save."
    )
    fields = extract_fields(text)

    missing_info = detect_missing_info(text, fields)

    assert missing_info == []


def test_detect_missing_info_accepts_common_actual_behavior_keywords():
    text = (
        "In QA on Edge using Android, when I upload a file, "
        "the page shows a 500 error. It should show a success message."
    )
    fields = extract_fields(text)

    missing_info = detect_missing_info(text, fields)

    assert missing_info == []


# preprocess_bug_report tests


def test_preprocess_bug_report_accepts_raw_string():
    result = preprocess_bug_report(
        "In production on Chrome using macOS, when I click submit, "
        "the login page crashes instead of saving. It should save successfully."
    )

    assert result.raw_text.startswith("In production")
    assert result.normalized_text.startswith("In production")
    assert result.extracted_fields["module"] == "authentication"
    assert result.extracted_fields["browser"] == "Chrome"
    assert result.extracted_fields["environment"] == "production"
    assert result.extracted_fields["device_or_os"] == "macOS"
    assert result.missing_info == []
    assert result.has_obvious_missing_info is False


def test_preprocess_bug_report_accepts_bug_report_input_model():
    report = BugReportInput(raw_text="The billing invoice page fails.")

    result = preprocess_bug_report(report)

    assert result.raw_text == "The billing invoice page fails."
    assert result.normalized_text == "The billing invoice page fails."
    assert result.extracted_fields["module"] == "billing"
    assert result.has_obvious_missing_info is True


def test_preprocess_bug_report_rejects_blank_raw_string():
    with pytest.raises(ValidationError):
        preprocess_bug_report("   ")


def test_preprocess_bug_report_rejects_unsupported_input_type():
    with pytest.raises(TypeError, match="report must be a BugReportInput or raw string"):
        preprocess_bug_report(123)  # type: ignore[arg-type]


def test_preprocess_bug_report_logs_execution(caplog):
    with caplog.at_level(logging.INFO, logger="bug_triage_workflow.preprocess"):
        preprocess_bug_report(
            "In production on Chrome using macOS, when I click submit, "
            "the page crashes instead of saving. It should save successfully."
        )

    assert "Bug report preprocessed" in caplog.text
    assert any(
        record.executor == "preprocess_bug_report"
        and record.missing_info_count == 0
        and record.extracted_field_names == ["browser", "device_or_os", "environment", "module"]
        for record in caplog.records
    )
