from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ExportIntegrityResult
from api.platform.serializers import content_hash


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_export_integrity_result(
    *,
    stored_export: Dict[str, Any],
    manifest: Dict[str, Any] | None,
    rendered_export: Dict[str, Any] | None,
    rendered_content: str,
) -> ExportIntegrityResult:
    manifest = sanitize_payload(manifest or {})
    rendered_export = sanitize_payload(rendered_export or {})
    stored_export = sanitize_payload(stored_export or {})

    checksum_expected = (
        stored_export.get("checksum") or rendered_export.get("checksum") or None
    )
    checksum_actual = content_hash(
        {
            "export_id": stored_export.get("export_id"),
            "export_format": stored_export.get("export_format"),
            "rendered_content": rendered_content,
        }
    )
    section_count_expected = int(stored_export.get("section_count") or 0)
    section_count_actual = len(manifest.get("ordered_sections") or [])
    manifest_hash_expected = (
        stored_export.get("source_manifest_hash")
        or stored_export.get("manifest_hash")
        or manifest.get("content_hash")
    )
    manifest_hash_actual = manifest.get("content_hash") or content_hash(manifest)
    tenant_consistent = bool(
        str(stored_export.get("workspace_id") or "")
        == str(manifest.get("workspace_id") or stored_export.get("workspace_id") or "")
        and str(stored_export.get("organization_id") or "")
        == str(
            (manifest.get("organization_context") or {}).get("organization_id")
            or stored_export.get("organization_id")
            or ""
        )
    )
    approval_context_consistent = str(stored_export.get("approval_status") or "") == str(
        manifest.get("approval_status") or stored_export.get("approval_status") or ""
    )
    checks: List[Dict[str, Any]] = [
        {
            "check_key": "checksum_match",
            "passed": checksum_expected == checksum_actual,
            "summary": "Rendered content checksum matches the stored checksum.",
        },
        {
            "check_key": "section_count_match",
            "passed": section_count_expected == section_count_actual,
            "summary": "Stored section count matches the manifest section count.",
        },
        {
            "check_key": "manifest_consistency",
            "passed": manifest_hash_expected == manifest_hash_actual,
            "summary": "Stored manifest hash matches the active manifest hash.",
        },
        {
            "check_key": "tenant_scope_consistency",
            "passed": tenant_consistent,
            "summary": "Stored export tenant scope matches the manifest scope.",
        },
        {
            "check_key": "approval_context_consistency",
            "passed": approval_context_consistent,
            "summary": "Stored approval status matches the manifest approval status snapshot.",
        },
    ]
    warnings = [
        check["summary"] for check in checks if not check["passed"] and check["check_key"] != "checksum_match"
    ]
    errors = [
        check["summary"] for check in checks if not check["passed"] and check["check_key"] == "checksum_match"
    ]
    status = "valid"
    if errors:
        status = "failed"
    elif warnings:
        status = "warning"
    return ExportIntegrityResult(
        stored_export_id=str(stored_export.get("stored_export_id") or ""),
        export_id=str(stored_export.get("export_id") or ""),
        render_id=str(stored_export.get("render_id") or ""),
        version_number=int(stored_export.get("version_number") or 1),
        version_label=str(stored_export.get("version_label") or "v1"),
        status=status,
        checksum_expected=checksum_expected,
        checksum_actual=checksum_actual,
        section_count_expected=section_count_expected,
        section_count_actual=section_count_actual,
        manifest_hash_expected=manifest_hash_expected,
        manifest_hash_actual=manifest_hash_actual,
        tenant_scope_consistent=tenant_consistent,
        approval_context_consistent=approval_context_consistent,
        checks=sanitize_payload(checks),
        warnings=warnings,
        errors=errors,
        verified_at=_now_utc(),
        metadata={
            "pack_type": stored_export.get("pack_type"),
            "export_format": stored_export.get("export_format"),
        },
    )


__all__ = ["build_export_integrity_result"]
