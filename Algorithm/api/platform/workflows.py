from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from api.platform.contracts import WorkflowInstance, WorkflowStageState, WorkflowTemplate


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_stage_state(
    template: WorkflowTemplate,
    *,
    stage: Optional[str] = None,
    status: str = "active",
    notes: Optional[list[str]] = None,
) -> WorkflowStageState:
    sequence = list(template.stage_sequence or ["intake"])
    current_stage = str(stage or sequence[0])
    if current_stage not in sequence:
        sequence = [current_stage, *sequence]
    current_index = sequence.index(current_stage)
    completed = sequence[:current_index]
    next_stage = sequence[current_index + 1] if current_index + 1 < len(sequence) else None
    return WorkflowStageState(
        stage=current_stage,
        status=status,
        completed_stages=completed,
        next_stage=next_stage,
        notes=list(notes or []),
        updated_at=now_utc(),
    )


def build_workflow_instance(
    *,
    workflow_id: str,
    workspace_id: str,
    template: WorkflowTemplate,
    title: str,
    status: str = "active",
    stage: Optional[str] = None,
    priority: str = "normal",
    owner_placeholder: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> WorkflowInstance:
    stage_state = build_stage_state(template, stage=stage, status=status)
    return WorkflowInstance(
        workflow_id=workflow_id,
        workspace_id=workspace_id,
        workflow_template_id=template.template_id,
        title=title,
        status=status,
        stage=stage_state.stage,
        priority=priority,
        owner_placeholder=owner_placeholder,
        metadata=dict(metadata or {}),
        stage_state=stage_state,
        created_at=now_utc(),
        updated_at=now_utc(),
    )

