"""Pydantic models for the Bug Triage Workflow.

This module defines strict Pydantic models and enums used across the
triage workflow. Keep models small and focused; business logic belongs
in executor modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base model that rejects unexpected fields at all workflow boundaries."""

    model_config = ConfigDict(extra="forbid")


class WorkflowStatus(str, Enum):
    RECEIVED = "received"
    PREPROCESSED = "preprocessed"
    CLASSIFIED = "classified"
    ROUTED = "routed"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    ESCALATION_APPROVED = "escalation_approved"
    STANDARD_TICKET_SELECTED = "standard_ticket_selected"
    REPORT_REJECTED = "report_rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class BugCategory(str, Enum):
    AUTHENTICATION = "authentication"
    UI_BUG = "ui_bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA_LOSS = "data_loss"
    INTEGRATION = "integration"
    UNKNOWN = "unknown"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Sentiment(str, Enum):
    NEUTRAL = "neutral"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


class RouteName(str, Enum):
    REQUEST_MORE_INFO = "request_more_info"
    CREATE_STANDARD_TICKET = "create_standard_ticket"
    REQUEST_HUMAN_APPROVAL = "request_human_approval"
    CREATE_ESCALATION_TICKET = "create_escalation_ticket"
    LOG_REJECTION = "log_rejection"


class HumanReviewAction(str, Enum):
    APPROVE_ESCALATION = "approve_escalation"
    CREATE_STANDARD_TICKET = "create_standard_ticket"
    REJECT_REPORT = "reject_report"


class BugReportInput(StrictBaseModel):
    raw_text: str = Field(..., description="Raw inbound bug report text")

    @field_validator("raw_text")
    @classmethod
    def raw_text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("raw_text must not be empty")
        return v


class PreprocessedBugReport(StrictBaseModel):
    raw_text: str
    normalized_text: str
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    missing_info: list[str] = Field(default_factory=list)
    has_obvious_missing_info: bool = False

    @field_validator("normalized_text")
    @classmethod
    def normalized_text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("normalized_text must not be empty")
        return v

    @field_validator("missing_info")
    @classmethod
    def missing_info_items_must_not_be_blank(cls, v: list[str]) -> list[str]:
        if any(not item or not item.strip() for item in v):
            raise ValueError("missing_info items must not be blank")
        return v

    @model_validator(mode="after")
    def missing_info_flag_must_match_contents(self) -> "PreprocessedBugReport":
        expected_flag = bool(self.missing_info)
        if self.has_obvious_missing_info is not expected_flag:
            raise ValueError(
                "has_obvious_missing_info must be true when missing_info is non-empty "
                "and false when missing_info is empty"
            )
        return self


class TriageClassification(StrictBaseModel):
    category: BugCategory
    urgency: Urgency
    sentiment: Sentiment
    missing_info: list[str] = Field(default_factory=list)
    recommended_route: RouteName
    reasoning: str
    confidence: float = Field(..., description="Confidence between 0.0 and 1.0")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_in_range(cls, v: float) -> float:
        if v is None:
            raise ValueError("confidence is required")
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reasoning must not be empty")
        return v

    @field_validator("missing_info")
    @classmethod
    def missing_info_items_must_not_be_blank(cls, v: list[str]) -> list[str]:
        if any(not item or not item.strip() for item in v):
            raise ValueError("missing_info items must not be blank")
        return v


class RouteDecision(StrictBaseModel):
    selected_route: RouteName
    reason: str | None = None

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank_when_provided(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("reason must not be blank when provided")
        return v


class HumanReviewDecision(StrictBaseModel):
    """Three-option human review decision for escalation requests."""

    required: bool = True
    action: HumanReviewAction
    approver: str
    notes: str | None = None

    @field_validator("approver")
    @classmethod
    def approver_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("approver must not be blank")
        return v

    @field_validator("notes")
    @classmethod
    def notes_must_not_be_blank_when_provided(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("notes must not be blank when provided")
        return v

    @model_validator(mode="after")
    def review_fields_must_be_consistent(self) -> "HumanReviewDecision":
        if self.required is not True:
            raise ValueError("required must be true for a human review decision")

        return self


class WorkflowEvent(StrictBaseModel):
    status: WorkflowStatus
    message: str
    executor: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("message must not be empty")
        return v

    @field_validator("executor")
    @classmethod
    def executor_must_not_be_blank_when_provided(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("executor must not be blank when provided")
        return v


COMPLETED_ROUTES = frozenset(
    {
        RouteName.REQUEST_MORE_INFO,
        RouteName.CREATE_STANDARD_TICKET,
        RouteName.CREATE_ESCALATION_TICKET,
    }
)

ROUTED_ROUTES = frozenset(
    {
        RouteName.REQUEST_MORE_INFO,
        RouteName.CREATE_STANDARD_TICKET,
        RouteName.REQUEST_HUMAN_APPROVAL,
    }
)

INTERMEDIATE_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.RECEIVED,
        WorkflowStatus.PREPROCESSED,
        WorkflowStatus.CLASSIFIED,
        WorkflowStatus.ROUTED,
        WorkflowStatus.ESCALATION_APPROVED,
        WorkflowStatus.STANDARD_TICKET_SELECTED,
    }
)


class _FieldPolicy(Enum):
    UNCONSTRAINED = auto()
    REQUIRED = auto()
    FORBIDDEN = auto()


@dataclass(frozen=True)
class _StatusRules:
    classification: _FieldPolicy = _FieldPolicy.UNCONSTRAINED
    selected_route: _FieldPolicy = _FieldPolicy.UNCONSTRAINED
    error: _FieldPolicy = _FieldPolicy.UNCONSTRAINED
    final_action: _FieldPolicy = _FieldPolicy.UNCONSTRAINED
    allowed_routes: frozenset[RouteName] | None = None


_STATUS_RULES: dict[WorkflowStatus, _StatusRules] = {
    WorkflowStatus.RECEIVED: _StatusRules(
        classification=_FieldPolicy.FORBIDDEN,
        selected_route=_FieldPolicy.FORBIDDEN,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.PREPROCESSED: _StatusRules(
        classification=_FieldPolicy.FORBIDDEN,
        selected_route=_FieldPolicy.FORBIDDEN,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.CLASSIFIED: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
        selected_route=_FieldPolicy.FORBIDDEN,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.ROUTED: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
        selected_route=_FieldPolicy.REQUIRED,
        allowed_routes=ROUTED_ROUTES,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.AWAITING_HUMAN_REVIEW: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
    ),
    WorkflowStatus.ESCALATION_APPROVED: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.STANDARD_TICKET_SELECTED: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.REPORT_REJECTED: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
    WorkflowStatus.COMPLETED: _StatusRules(
        classification=_FieldPolicy.REQUIRED,
        selected_route=_FieldPolicy.REQUIRED,
        allowed_routes=COMPLETED_ROUTES,
        error=_FieldPolicy.FORBIDDEN,
        final_action=_FieldPolicy.REQUIRED,
    ),
    WorkflowStatus.FAILED: _StatusRules(
        error=_FieldPolicy.REQUIRED,
        final_action=_FieldPolicy.FORBIDDEN,
    ),
}


def _check_field_policy(
    status: WorkflowStatus,
    field_name: str,
    value: object,
    policy: _FieldPolicy,
) -> None:
    if policy is _FieldPolicy.REQUIRED and value is None:
        raise ValueError(
            f"{field_name} is required when status is {status.value}"
        )
    if policy is _FieldPolicy.FORBIDDEN and value is not None:
        raise ValueError(
            f"{field_name} must be None when status is {status.value}"
        )


def _check_allowed_routes(
    status: WorkflowStatus,
    selected_route: RouteName | None,
    allowed_routes: frozenset[RouteName] | None,
) -> None:
    if allowed_routes is None:
        return
    if selected_route is not None and selected_route not in allowed_routes:
        raise ValueError(
            f"selected_route {selected_route.value} is not valid "
            f"when status is {status.value}"
        )


class WorkflowResult(StrictBaseModel):
    status: WorkflowStatus
    selected_route: RouteName | None = None
    classification: TriageClassification | None = None
    human_review_required: bool = False
    human_review_action: HumanReviewAction | None = None
    approval_granted: bool | None = None
    final_action: str | None = None
    error: str | None = None
    event_log: list[WorkflowEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("final_action", "error")
    @classmethod
    def optional_text_must_not_be_blank_when_provided(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("optional text fields must not be blank when provided")
        return v

    def _validate_no_review_summary(self, *, route: RouteName) -> None:
        if self.human_review_required:
            raise ValueError(
                f"human_review_required must be false for {route.value}"
            )
        if self.human_review_action is not None:
            raise ValueError(f"human_review_action must be None for {route.value}")
        if self.approval_granted is not None:
            raise ValueError(f"approval_granted must be None for {route.value}")

    def _validate_empty_review_summary_for_status(self) -> None:
        if self.human_review_required:
            raise ValueError(
                f"human_review_required must be false when status is {self.status.value}"
            )
        if self.human_review_action is not None:
            raise ValueError(
                f"human_review_action must be None when status is {self.status.value}"
            )
        if self.approval_granted is not None:
            raise ValueError(
                f"approval_granted must be None when status is {self.status.value}"
            )

    def _validate_review_action(
        self,
        *,
        route: RouteName,
        action: HumanReviewAction,
        approval_granted: bool | None,
    ) -> None:
        if (
            not self.human_review_required
            or self.human_review_action is not action
            or self.approval_granted is not approval_granted
        ):
            raise ValueError(
                f"{route.value} route requires {action.value} human review action"
            )

    def _validate_human_review_summary(self) -> None:
        if self.human_review_action is not None and not self.human_review_required:
            raise ValueError(
                "human_review_required must be true when human_review_action is provided"
            )

        if self.human_review_action is not None:
            expected_approval = {
                HumanReviewAction.APPROVE_ESCALATION: True,
                HumanReviewAction.CREATE_STANDARD_TICKET: None,
                HumanReviewAction.REJECT_REPORT: False,
            }[self.human_review_action]
            if self.approval_granted is not expected_approval:
                raise ValueError(
                    "approval_granted must match the human_review_action summary semantics"
                )

        if self.approval_granted is not None and not self.human_review_required:
            raise ValueError(
                "human_review_required must be true when approval_granted is provided"
            )

    def _validate_route_ownership(self) -> None:
        if self.status is WorkflowStatus.FAILED:
            return

        if (
            self.selected_route is RouteName.REQUEST_HUMAN_APPROVAL
            and self.status
            not in {WorkflowStatus.ROUTED, WorkflowStatus.AWAITING_HUMAN_REVIEW}
        ):
            raise ValueError(
                "selected_route request_human_approval is valid only when "
                "status is routed or awaiting_human_review"
            )

        if (
            self.selected_route is RouteName.LOG_REJECTION
            and self.status is not WorkflowStatus.REPORT_REJECTED
        ):
            raise ValueError(
                "selected_route log_rejection is valid only when status is report_rejected"
            )

    def _validate_failed_status(self) -> None:
        if self.status is not WorkflowStatus.FAILED:
            return

        rules = _STATUS_RULES[self.status]
        _check_field_policy(self.status, "error", self.error, rules.error)
        _check_field_policy(self.status, "final_action", self.final_action, rules.final_action)
        if self.approval_granted is not None:
            raise ValueError("approval_granted must be None when status is failed")
        if self.human_review_action is not None:
            raise ValueError("human_review_action must be None when status is failed")

    def _validate_completed_status(self) -> None:
        if self.status is not WorkflowStatus.COMPLETED:
            return

        rules = _STATUS_RULES[self.status]
        _check_field_policy(self.status, "selected_route", self.selected_route, rules.selected_route)
        _check_field_policy(self.status, "classification", self.classification, rules.classification)
        _check_field_policy(self.status, "final_action", self.final_action, rules.final_action)
        _check_field_policy(self.status, "error", self.error, rules.error)
        _check_allowed_routes(self.status, self.selected_route, rules.allowed_routes)

        if self.selected_route is RouteName.REQUEST_MORE_INFO:
            self._validate_no_review_summary(route=RouteName.REQUEST_MORE_INFO)
        elif self.selected_route is RouteName.CREATE_STANDARD_TICKET:
            if self.human_review_required:
                self._validate_review_action(
                    route=RouteName.CREATE_STANDARD_TICKET,
                    action=HumanReviewAction.CREATE_STANDARD_TICKET,
                    approval_granted=None,
                )
            else:
                self._validate_no_review_summary(
                    route=RouteName.CREATE_STANDARD_TICKET
                )
        elif self.selected_route is RouteName.CREATE_ESCALATION_TICKET:
            if self.human_review_required:
                self._validate_review_action(
                    route=RouteName.CREATE_ESCALATION_TICKET,
                    action=HumanReviewAction.APPROVE_ESCALATION,
                    approval_granted=True,
                )
            else:
                self._validate_no_review_summary(
                    route=RouteName.CREATE_ESCALATION_TICKET
                )

    def _validate_awaiting_human_review_status(self) -> None:
        if self.status is not WorkflowStatus.AWAITING_HUMAN_REVIEW:
            return

        rules = _STATUS_RULES[self.status]
        _check_field_policy(self.status, "classification", self.classification, rules.classification)
        if not self.human_review_required:
            raise ValueError("human_review_required must be true while awaiting human review")
        if self.selected_route is not RouteName.REQUEST_HUMAN_APPROVAL:
            raise ValueError(
                "selected_route must be request_human_approval while awaiting human review"
            )
        if self.approval_granted is not None:
            raise ValueError("approval_granted must be None while awaiting human review")
        if self.human_review_action is not None:
            raise ValueError(
                "human_review_action must be None while awaiting human review"
            )
        if self.final_action is not None:
            raise ValueError("final_action must be None while awaiting human review")
        if self.error is not None:
            raise ValueError("error must be None while awaiting human review")

    def _validate_report_rejected_status(self) -> None:
        if self.status is not WorkflowStatus.REPORT_REJECTED:
            return

        rules = _STATUS_RULES[self.status]
        _check_field_policy(self.status, "classification", self.classification, rules.classification)
        if (
            not self.human_review_required
            or self.human_review_action is not HumanReviewAction.REJECT_REPORT
            or self.approval_granted is not False
            or self.selected_route is not RouteName.LOG_REJECTION
        ):
            raise ValueError("report_rejected status requires report rejection")
        _check_field_policy(self.status, "final_action", self.final_action, rules.final_action)
        _check_field_policy(self.status, "error", self.error, rules.error)

    def _validate_intermediate_status(self) -> None:
        if self.status not in INTERMEDIATE_WORKFLOW_STATUSES:
            return

        rules = _STATUS_RULES[self.status]
        _check_field_policy(self.status, "final_action", self.final_action, rules.final_action)
        _check_field_policy(self.status, "error", self.error, rules.error)

        if self.status in {WorkflowStatus.RECEIVED, WorkflowStatus.PREPROCESSED}:
            _check_field_policy(self.status, "selected_route", self.selected_route, rules.selected_route)
            _check_field_policy(self.status, "classification", self.classification, rules.classification)
            self._validate_empty_review_summary_for_status()
        elif self.status is WorkflowStatus.CLASSIFIED:
            _check_field_policy(self.status, "classification", self.classification, rules.classification)
            _check_field_policy(self.status, "selected_route", self.selected_route, rules.selected_route)
            self._validate_empty_review_summary_for_status()
        elif self.status is WorkflowStatus.ROUTED:
            _check_field_policy(self.status, "classification", self.classification, rules.classification)
            _check_field_policy(self.status, "selected_route", self.selected_route, rules.selected_route)
            _check_allowed_routes(self.status, self.selected_route, rules.allowed_routes)
            if self.human_review_action is not None:
                raise ValueError("human_review_action must be None when status is routed")
            if self.approval_granted is not None:
                raise ValueError("approval_granted must be None when status is routed")
            if self.selected_route is RouteName.REQUEST_HUMAN_APPROVAL:
                if not self.human_review_required:
                    raise ValueError(
                        "human_review_required must be true when status is routed "
                        "to request_human_approval"
                    )
            elif self.human_review_required:
                raise ValueError(
                    "human_review_required must be false when status is routed "
                    f"to {self.selected_route.value}"
                )
        elif self.status is WorkflowStatus.ESCALATION_APPROVED:
            _check_field_policy(self.status, "classification", self.classification, rules.classification)
            if self.selected_route is not RouteName.CREATE_ESCALATION_TICKET:
                raise ValueError(
                    "selected_route must be create_escalation_ticket when "
                    "status is escalation_approved"
                )
            self._validate_review_action(
                route=RouteName.CREATE_ESCALATION_TICKET,
                action=HumanReviewAction.APPROVE_ESCALATION,
                approval_granted=True,
            )
        elif self.status is WorkflowStatus.STANDARD_TICKET_SELECTED:
            _check_field_policy(self.status, "classification", self.classification, rules.classification)
            if self.selected_route is not RouteName.CREATE_STANDARD_TICKET:
                raise ValueError(
                    "selected_route must be create_standard_ticket when "
                    "status is standard_ticket_selected"
                )
            self._validate_review_action(
                route=RouteName.CREATE_STANDARD_TICKET,
                action=HumanReviewAction.CREATE_STANDARD_TICKET,
                approval_granted=None,
            )

    def _validate_final_event_status(self) -> None:
        if self.event_log and self.event_log[-1].status is not self.status:
            raise ValueError(
                "the final workflow event status must match the workflow result status"
            )

    @model_validator(mode="after")
    def workflow_result_fields_must_be_consistent(self) -> "WorkflowResult":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")

        self._validate_human_review_summary()
        self._validate_route_ownership()
        self._validate_failed_status()
        self._validate_completed_status()
        self._validate_awaiting_human_review_status()
        self._validate_report_rejected_status()
        self._validate_intermediate_status()
        self._validate_final_event_status()

        return self
