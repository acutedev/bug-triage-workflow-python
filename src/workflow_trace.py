"""Workflow event trace helpers for bug triage orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models import WorkflowEvent, WorkflowStatus


def _workflow_event(
    status: WorkflowStatus,
    message: str,
    executor: str,
    **data: Any,
) -> WorkflowEvent:
    """Create one structured workflow trace entry."""

    return WorkflowEvent(
        status=status,
        message=message,
        executor=executor,
        data=data,
    )


@dataclass
class WorkflowTrace:
    """Mutable per-run audit trace shared by workflow executors."""

    events: list[WorkflowEvent] = field(default_factory=list)
    _classifier_provider_boundary_active: bool = field(
        default=False,
        init=False,
        repr=False,
        compare=False,
    )

    def append(
        self,
        status: WorkflowStatus,
        message: str,
        executor: str,
        **data: Any,
    ) -> None:
        self.events.append(
            _workflow_event(
                status,
                message,
                executor,
                **data,
            )
        )

    def snapshot(self) -> list[WorkflowEvent]:
        return list(self.events)

    def enter_classifier_provider_boundary(self) -> None:
        self._classifier_provider_boundary_active = True

    def exit_classifier_provider_boundary(self) -> None:
        self._classifier_provider_boundary_active = False

    def is_classifier_provider_boundary_active(self) -> bool:
        return self._classifier_provider_boundary_active
