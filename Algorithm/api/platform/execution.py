from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from api.assistant.reports import sanitize_payload
from api.data_providers.premium import probe_premium_connector
from api.platform.contracts import IntegrationExecutionRecord, RenderedExportResult
from api.platform.serializers import content_hash, stable_json_dumps


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_ROOT = ROOT_DIR / "artifacts" / "platform_archive"
DEFAULT_WEBHOOK_OUTBOX = ROOT_DIR / "artifacts" / "platform_webhook_outbox"
DEFAULT_INTERNAL_SINK = ROOT_DIR / "artifacts" / "platform_internal_sink" / "events.jsonl"


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def _base_execution_payload(
    *,
    binding: Dict[str, Any],
    action_type: str,
    dossier_id: Optional[str],
    export_id: Optional[str],
    render_id: Optional[str],
    workspace_id: Optional[str],
    organization_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "execution_id": str(uuid.uuid4()),
        "binding_id": str(binding.get("binding_id")),
        "integration_type": str(binding.get("integration_type")),
        "action_type": action_type,
        "status": "completed",
        "workspace_id": workspace_id or binding.get("workspace_id"),
        "organization_id": organization_id or binding.get("organization_id"),
        "dossier_id": dossier_id,
        "export_id": export_id,
        "render_id": render_id,
        "started_at": now_utc(),
        "completed_at": now_utc(),
        "payload_summary": {},
        "output_summary": {},
        "error_summary": None,
        "metadata": sanitize_payload(metadata or {}),
    }


def execute_local_archive(
    *,
    binding: Dict[str, Any],
    rendered_export: Dict[str, Any] | RenderedExportResult,
    dossier_id: Optional[str],
    export_id: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rendered = (
        rendered_export
        if isinstance(rendered_export, dict)
        else rendered_export.model_dump(mode="python")
    )
    root = Path((binding.get("config") or {}).get("target_root") or DEFAULT_ARCHIVE_ROOT)
    file_path = root / str(rendered.get("file_name_hint") or "platform-export.txt")
    _write_text(file_path, str(rendered.get("rendered_content") or ""))
    payload = _base_execution_payload(
        binding=binding,
        action_type="archive_export",
        dossier_id=dossier_id,
        export_id=export_id,
        render_id=rendered.get("render_id"),
        workspace_id=binding.get("workspace_id"),
        organization_id=binding.get("organization_id"),
        metadata=metadata,
    )
    payload["output_summary"] = {
        "archive_path": str(file_path),
        "bytes_written": len(str(rendered.get("rendered_content") or "").encode("utf-8")),
        "checksum": rendered.get("checksum"),
    }
    return sanitize_payload(payload)


def execute_webhook_outbox(
    *,
    binding: Dict[str, Any],
    rendered_export: Optional[Dict[str, Any]],
    dossier_id: Optional[str],
    export_id: Optional[str],
    event_type: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    outbox_root = Path((binding.get("config") or {}).get("outbox_root") or DEFAULT_WEBHOOK_OUTBOX)
    payload_body = sanitize_payload(
        {
            "event_type": event_type or "platform_event",
            "binding_id": binding.get("binding_id"),
            "dossier_id": dossier_id,
            "export_id": export_id,
            "render_id": (rendered_export or {}).get("render_id"),
            "checksum": (rendered_export or {}).get("checksum"),
            "metadata": metadata or {},
        }
    )
    file_name = f"{uuid.uuid4()}.json"
    file_path = outbox_root / file_name
    _write_text(file_path, json.dumps(payload_body, indent=2, sort_keys=True))
    payload = _base_execution_payload(
        binding=binding,
        action_type="notify_event",
        dossier_id=dossier_id,
        export_id=export_id,
        render_id=(rendered_export or {}).get("render_id"),
        workspace_id=binding.get("workspace_id"),
        organization_id=binding.get("organization_id"),
        metadata=metadata,
    )
    payload["status"] = "queued"
    payload["output_summary"] = {
        "outbox_path": str(file_path),
        "event_checksum": content_hash(payload_body),
        "delivery_mode": "local_webhook_outbox",
    }
    return sanitize_payload(payload)


def execute_internal_sink(
    *,
    binding: Dict[str, Any],
    dossier_id: Optional[str],
    export_id: Optional[str],
    render_id: Optional[str],
    event_type: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sink_path = Path((binding.get("config") or {}).get("sink_path") or DEFAULT_INTERNAL_SINK)
    envelope = sanitize_payload(
        {
            "event_type": event_type or "platform_sink_event",
            "binding_id": binding.get("binding_id"),
            "dossier_id": dossier_id,
            "export_id": export_id,
            "render_id": render_id,
            "metadata": metadata or {},
            "created_at": now_utc(),
        }
    )
    _append_text(sink_path, stable_json_dumps(envelope) + "\n")
    payload = _base_execution_payload(
        binding=binding,
        action_type="sink_event",
        dossier_id=dossier_id,
        export_id=export_id,
        render_id=render_id,
        workspace_id=binding.get("workspace_id"),
        organization_id=binding.get("organization_id"),
        metadata=metadata,
    )
    payload["output_summary"] = {
        "sink_path": str(sink_path),
        "event_checksum": content_hash(envelope),
    }
    return sanitize_payload(payload)


def execute_premium_connector_probe(
    *,
    binding: Dict[str, Any],
    sample_symbol: Optional[str] = None,
    execute_live: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    connector_type = str(binding.get("integration_type") or "")
    result = probe_premium_connector(
        connector_type,
        sample_symbol=str(sample_symbol or "NVDA"),
        execute_live=execute_live,
    )
    readiness_status = str(result.get("readiness_status") or "unknown")
    live_probe_status = str(result.get("live_probe_status") or "not_attempted")
    if live_probe_status == "probe_failed":
        status = "failed"
    elif readiness_status in {"missing_credentials", "misconfigured"} or live_probe_status == "blocked_missing_credentials":
        status = "blocked"
    elif live_probe_status == "probe_succeeded":
        status = "completed"
    else:
        status = "configured"
    payload = _base_execution_payload(
        binding=binding,
        action_type="connector_probe",
        dossier_id=None,
        export_id=None,
        render_id=None,
        workspace_id=binding.get("workspace_id"),
        organization_id=binding.get("organization_id"),
        metadata=metadata,
    )
    payload["status"] = status
    payload["payload_summary"] = sanitize_payload(
        {
            "connector_type": connector_type,
            "sample_symbol": result.get("sample_symbol"),
            "execute_live": bool(execute_live),
        }
    )
    payload["output_summary"] = sanitize_payload(
        {
            "readiness_status": readiness_status,
            "live_probe_status": live_probe_status,
            "configured_vendors": result.get("configured_vendors") or [],
            "capabilities": result.get("capabilities") or [],
            "data_quality_summary": result.get("data_quality_summary"),
            "last_execution_result": result.get("last_execution_result") or {},
            "failure_reason_classification": result.get("failure_reason_classification"),
            "checked_at": result.get("checked_at"),
        }
    )
    payload["error_summary"] = result.get("error_summary")
    return sanitize_payload(payload)
