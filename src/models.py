"""Pydantic models for the Bug Triage Workflow.

This module defines strict Pydantic models and enums used across the
triage workflow. Keep models small and focused; business logic belongs
in executor modules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    """Base model that rejects unexpected fields at all workflow boundaries."""

    model_config = ConfigDict(extra="forbid")


class WorkflowStatus(str, Enum):
    RECEIVED = "received"
    PREPROCESSED = "preprocessed"
    CLASSIFIED = "classified"
    ROUTED = "routed"
    WAITING_FOR_HUMAN_APPROVAL = "waiting_for_human_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
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


class HumanApprovalDecision(StrictBaseModel):
    required: Literal[True] = True
    approval_granted: bool | None = None
    approver: str | None = None
    notes: str | None = None

    @field_validator("approver", "notes")
    @classmethod
    def optional_text_must_not_be_blank_when_provided(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("optional text fields must not be blank when provided")
        return v

    @model_validator(mode="after")
    def approval_fields_must_be_consistent(self) -> "HumanApprovalDecision":
        if self.approval_granted is None:
            raise ValueError("approval_granted is required when approval is required")

        if self.approval_granted is not None and not (self.approver and self.approver.strip()):
            raise ValueError("approver is required when an approval decision is provided")

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


class WorkflowResult(StrictBaseModel):
    status: WorkflowStatus
    selected_route: RouteName | None = None
    classification: TriageClassification | None = None
    human_approval_required: bool = False
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

    @model_validator(mode="after")
    def workflow_result_fields_must_be_consistent(self) -> "WorkflowResult":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")

        if self.status == WorkflowStatus.FAILED and not self.error:
            raise ValueError("error is required when status is failed")

        if self.status == WorkflowStatus.FAILED and self.final_action is not None:
            raise ValueError("final_action must be None when status is failed")

        if self.status == WorkflowStatus.COMPLETED and not self.final_action:
            raise ValueError("final_action is required when status is completed")

        if self.status == WorkflowStatus.COMPLETED and self.error is not None:
            raise ValueError("error must be None when status is completed")

        if self.approval_granted is not None and not self.human_approval_required:
            raise ValueError("human_approval_required must be true when approval_granted is provided")

        if self.status == WorkflowStatus.WAITING_FOR_HUMAN_APPROVAL and not self.human_approval_required:
            raise ValueError("human_approval_required must be true while waiting for approval")

        if self.status == WorkflowStatus.APPROVED:
            if not self.human_approval_required or self.approval_granted is not True:
                raise ValueError("approved status requires human approval to be granted")

        if self.status == WorkflowStatus.REJECTED:
            if not self.human_approval_required or self.approval_granted is not False:
                raise ValueError("rejected status requires human approval to be rejected")

        if (
            self.selected_route == RouteName.CREATE_ESCALATION_TICKET
            and self.human_approval_required
            and self.approval_granted is not True
        ):
            raise ValueError(
                "create_escalation_ticket route requires granted approval "
                "when human approval is required"
            )

        if self.event_log and self.event_log[-1].status != self.status:
            raise ValueError(
                "the final workflow event status must match the workflow result status"
            )

        return self
