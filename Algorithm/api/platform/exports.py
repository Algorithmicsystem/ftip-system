from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ExportManifest
from api.platform.document_packs import PACK_TYPE_TITLES, build_export_sections, supported_pack_types
from api.platform.serializers import content_hash


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_export_manifest(
    *,
    pack_type: str,
    dossier: Dict[str, Any],
    report: Dict[str, Any],
    workflow: Optional[Dict[str, Any]] = None,
    workspace: Optional[Dict[str, Any]] = None,
    organization: Optional[Dict[str, Any]] = None,
    approval_status: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExportManifest:
    sections = build_export_sections(pack_type=pack_type, report=report, dossier=dossier)
    manifest = ExportManifest(
        export_id=str(uuid.uuid4()),
        dossier_id=str(dossier.get("dossier_id")),
        workflow_id=(workflow or {}).get("workflow_id"),
        workspace_id=(workspace or {}).get("workspace_id"),
        pack_type=pack_type,
        title=PACK_TYPE_TITLES.get(pack_type, pack_type.replace("_", " ").title()),
        subtitle=f"{report.get('symbol') or dossier.get('title') or 'Entity'} export pack",
        generated_at=now_utc(),
        framework_version=report.get("axiom_framework_version") or report.get("report_version"),
        organization_context=sanitize_payload(organization or {}),
        workspace_context=sanitize_payload(workspace or {}),
        entity_context=sanitize_payload(
            {
                "symbol": report.get("symbol") or (dossier.get("current_summary") or {}).get("symbol"),
                "display_name": (dossier.get("current_summary") or {}).get("display_name"),
                "regime_label": dossier.get("latest_regime_label"),
                "trade_family": dossier.get("latest_trade_family"),
                "deployability_tier": dossier.get("latest_deployability_tier"),
            }
        ),
        approval_status=approval_status,
        evidence_summary=report.get("axiom_historical_evidence_summary_text")
        or report.get("axiom_lineage_summary"),
        ordered_sections=sections,
        metadata=sanitize_payload(metadata or {}),
        status="generated",
    )
    payload = manifest.model_dump(mode="python")
    payload["content_hash"] = content_hash(payload)
    return ExportManifest.model_validate(payload)


__all__ = ["build_export_manifest", "supported_pack_types"]
