"""Command-line demo for the bug triage workflow."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

from openai import OpenAIError

from src.config import AppConfig, load_config
from src.logging_config import configure_logging
from src.models import HumanReviewAction, HumanReviewDecision, WorkflowResult
from src.openai_provider import build_classifier_agent
from src.workflow import build_bug_triage_workflow

SAMPLE_BUG_REPORT = """
Users can reset another user's password by changing the account ID in the
password-reset request.

Environment: production
Browser: Chrome 137
Operating system: macOS
Module: authentication

Steps to reproduce:
1. Sign in as a standard user.
2. Open the password-reset page.
3. Change the account ID in the request.
4. Submit the request.

Expected result:
Only the signed-in user's password can be reset.

Actual result:
The password for another account is reset.

This appears to be a critical security vulnerability affecting production.
""".strip()


def print_section(title: str) -> None:
    """Print a visible console section heading."""
    border = "=" * 78
    print(f"\n{border}")
    print(title)
    print(border)


def print_json(value: Any) -> None:
    """Print a model or regular Python value as formatted JSON."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def print_workflow_result(result: WorkflowResult) -> None:
    """Print the final validated workflow result."""
    print_section("FINAL WORKFLOW RESULT")
    print_json(result)


def print_flow_summary(result: WorkflowResult) -> None:
    """Print the business flow captured in the final workflow trace."""
    print_section("WORKFLOW FLOW")
    for event in result.event_log:
        executor = event.executor or "workflow"
        print(f"{event.status.value} | {executor} | {event.message}")


def print_stream_event(event: Any) -> WorkflowResult | None:
    """Print one Microsoft Agent Framework stream event.

    Return the final WorkflowResult when the event carries one.
    """
    event_type = getattr(event, "type", type(event).__name__)
    executor_id = getattr(event, "executor_id", None)

    request_id: str | None = None
    if event_type == "request_info":
        executor_id = getattr(event, "source_executor_id", executor_id)
        request_id = event.request_id

    data = getattr(event, "data", None)

    print(f"\n[{event_type}]")
    if executor_id:
        print(f"executor: {executor_id}")
    if request_id:
        print(f"request_id: {request_id}")

    if isinstance(data, WorkflowResult):
        print("data: final WorkflowResult")
        return data

    if data is not None:
        print("data:")
        print_json(data)

    return None


def read_human_review_decision() -> HumanReviewDecision:
    """Prompt the operator for an explicit human review action."""
    action_by_selection = {
        "1": HumanReviewAction.APPROVE_ESCALATION,
        "2": HumanReviewAction.CREATE_STANDARD_TICKET,
        "3": HumanReviewAction.REJECT_REPORT,
    }

    while True:
        print("\nChoose an action:")
        print("1. Approve escalation")
        print("2. Create a standard ticket instead")
        print("3. Reject the report")
        selection = input("Selection [1/2/3]: ").strip()

        action = action_by_selection.get(selection)
        if action is not None:
            break

        print("Please enter 1, 2, or 3.")

    approver = input("Approver name: ").strip()
    while not approver:
        print("Approver name is required.")
        approver = input("Approver name: ").strip()

    notes = input("Optional notes: ").strip() or None

    return HumanReviewDecision(
        required=True,
        action=action,
        approver=approver,
        notes=notes,
    )


async def run_demo(config: AppConfig | None = None) -> WorkflowResult:
    """Run the sample bug report through the real workflow graph."""
    configure_logging()
    if config is None:
        config = load_config()

    classifier_agent = build_classifier_agent(config)
    workflow = build_bug_triage_workflow(
        classifier_agent,
        human_approval_enabled=config.human_approval_enabled,
    )

    print_section("SAMPLE BUG REPORT")
    print(SAMPLE_BUG_REPORT)

    print_section("STREAMING WORKFLOW EVENTS")
    final_result: WorkflowResult | None = None
    request_event: Any | None = None

    async for event in workflow.run(
        SAMPLE_BUG_REPORT,
        stream=True,
        include_status_events=True,
    ):
        result = print_stream_event(event)
        if result is not None:
            final_result = result

        if getattr(event, "type", None) == "request_info":
            request_event = event

    if request_event is not None:
        request_data = getattr(request_event, "data", None)
        prompt = getattr(request_data, "prompt", None)

        print_section("HUMAN REVIEW REQUIRED")
        if prompt:
            print(prompt)

        decision = read_human_review_decision()

        print_section("RESUMING WORKFLOW")
        async for event in workflow.run(
            stream=True,
            responses={request_event.request_id: decision},
            include_status_events=True,
        ):
            result = print_stream_event(event)
            if result is not None:
                final_result = result

    if final_result is None:
        raise RuntimeError("Bug triage demo completed without a WorkflowResult.")

    print_flow_summary(final_result)
    print_workflow_result(final_result)
    return final_result


async def main() -> int:
    """Command-line entry point."""
    logger = configure_logging()

    try:
        try:
            config = load_config()
        except ValueError as error:
            print(f"Configuration error: {error}", file=sys.stderr)
            return 2
        await run_demo(config)
    except KeyboardInterrupt:
        print("Operation cancelled by user.", file=sys.stderr)
        return 130
    except EOFError:
        print("Input closed; bug triage demo cancelled.", file=sys.stderr)
        return 1
    except OpenAIError as error:
        print(f"Provider error: {error}", file=sys.stderr)
        return 1
    except Exception:
        logger.exception("Bug triage CLI failed unexpectedly")
        print(
            "Unexpected error: bug triage demo failed. See logs for details.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
