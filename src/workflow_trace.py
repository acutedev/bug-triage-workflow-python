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
