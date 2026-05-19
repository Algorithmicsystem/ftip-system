from __future__ import annotations

import datetime as dt
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from psycopg.types.json import Json

from api import config, db
from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    AnalysisLink,
    CoverageEntity,
    DossierRecord,
    OrganizationProfile,
    WorkflowInstance,
    WorkspaceRecord,
)


def _db_ready(*, write: bool = False) -> bool:
    if not db.db_enabled():
        return False
    if write and not db.db_write_enabled():
        return False
    if not write and not db.db_read_enabled():
        return False
    return True


class PlatformStore:
    """Persistence foundation for the multi-workflow platform layer.

    This stays intentionally simple: it gives the repository stable domain objects
    and storage seams now, so later auth/org isolation and external workflow
    tooling can attach without redesigning AXIOM or assistant persistence.
    """

    def __init__(self, use_memory: Optional[bool] = None):
        self.use_memory = (
            bool(use_memory) if use_memory is not None else not config.db_enabled()
        )
        self._organizations: Dict[str, Dict[str, Any]] = {}
        self._workspaces: Dict[str, Dict[str, Any]] = {}
        self._entities: Dict[str, Dict[str, Any]] = {}
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._dossiers: Dict[str, Dict[str, Any]] = {}
        self._dossier_links: List[Dict[str, Any]] = []

    def _now(self) -> float:
        return time.time()

    def _now_iso(self) -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _uuid(self) -> str:
        return str(uuid.uuid4())

    def create_organization(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = OrganizationProfile.model_validate(
            {
                "organization_id": payload.get("organization_id") or self._uuid(),
                "name": payload.get("name") or "FTIP Organization",
                "organization_type": payload.get("organization_type") or "research_team",
                "settings": payload.get("settings") or {},
            }
        ).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = self._now_iso()
            record["updated_at"] = self._now_iso()
            self._organizations[record["organization_id"]] = sanitize_payload(record)
            return self._organizations[record["organization_id"]]
        row = db.exec1(
            """
            INSERT INTO organizations (
                organization_id, name, organization_type, settings
            )
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (organization_id)
            DO UPDATE SET
                name = EXCLUDED.name,
                organization_type = EXCLUDED.organization_type,
                settings = EXCLUDED.settings,
                updated_at = now()
            RETURNING organization_id, name, organization_type, settings, created_at, updated_at
            """,
            (
                record["organization_id"],
                record["name"],
                record["organization_type"],
                Json(sanitize_payload(record["settings"])),
            ),
        )
        return {
            "organization_id": str(row[0]),
            "name": row[1],
            "organization_type": row[2],
            "settings": sanitize_payload(row[3]) if row[3] is not None else {},
            "created_at": row[4],
            "updated_at": row[5],
        }

    def list_organizations(self) -> List[Dict[str, Any]]:
        if self.use_memory:
            values = list(self._organizations.values())
            values.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            return [sanitize_payload(item) for item in values]
        rows = db.safe_fetchall(
            """
            SELECT organization_id, name, organization_type, settings, created_at, updated_at
            FROM organizations
            ORDER BY updated_at DESC, created_at DESC
            """
        )
        return [
            {
                "organization_id": str(row[0]),
                "name": row[1],
                "organization_type": row[2],
                "settings": sanitize_payload(row[3]) if row[3] is not None else {},
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]

    def get_organization(self, organization_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            return self._organizations.get(str(organization_id))
        row = db.safe_fetchone(
            """
            SELECT organization_id, name, organization_type, settings, created_at, updated_at
            FROM organizations
            WHERE organization_id=%s
            """,
            (organization_id,),
        )
        if not row:
            return None
        return {
            "organization_id": str(row[0]),
            "name": row[1],
            "organization_type": row[2],
            "settings": sanitize_payload(row[3]) if row[3] is not None else {},
            "created_at": row[4],
            "updated_at": row[5],
        }

    def create_workspace(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = WorkspaceRecord.model_validate(
            {
                "workspace_id": payload.get("workspace_id") or self._uuid(),
                "organization_id": payload["organization_id"],
                "name": payload.get("name") or "FTIP Workspace",
                "audience_type": payload.get("audience_type") or "general",
                "report_profile": payload.get("report_profile") or "trading_focused",
                "default_workflow_template": payload.get("default_workflow_template") or "research_watchlist",
                "platform_profile": payload.get("platform_profile") or "research_core",
                "settings": payload.get("settings") or {},
            }
        ).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = self._now_iso()
            record["updated_at"] = self._now_iso()
            self._workspaces[record["workspace_id"]] = sanitize_payload(record)
            return self._workspaces[record["workspace_id"]]
        row = db.exec1(
            """
            INSERT INTO workspaces (
                workspace_id, organization_id, name, audience_type, report_profile,
                default_workflow_template, platform_profile, settings
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (workspace_id)
            DO UPDATE SET
                organization_id = EXCLUDED.organization_id,
                name = EXCLUDED.name,
                audience_type = EXCLUDED.audience_type,
                report_profile = EXCLUDED.report_profile,
                default_workflow_template = EXCLUDED.default_workflow_template,
                platform_profile = EXCLUDED.platform_profile,
                settings = EXCLUDED.settings,
                updated_at = now()
            RETURNING workspace_id, organization_id, name, audience_type, report_profile,
                default_workflow_template, platform_profile, settings, created_at, updated_at
            """,
            (
                record["workspace_id"],
                record["organization_id"],
                record["name"],
                record["audience_type"],
                record["report_profile"],
                record["default_workflow_template"],
                record["platform_profile"],
                Json(sanitize_payload(record["settings"])),
            ),
        )
        return {
            "workspace_id": str(row[0]),
            "organization_id": str(row[1]),
            "name": row[2],
            "audience_type": row[3],
            "report_profile": row[4],
            "default_workflow_template": row[5],
            "platform_profile": row[6],
            "settings": sanitize_payload(row[7]) if row[7] is not None else {},
            "created_at": row[8],
            "updated_at": row[9],
        }

    def list_workspaces(self, *, organization_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = [
                item
                for item in self._workspaces.values()
                if organization_id is None or str(item.get("organization_id")) == str(organization_id)
            ]
            rows.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if organization_id is not None:
            clauses.append("organization_id=%s")
            params.append(organization_id)
        rows = db.safe_fetchall(
            f"""
            SELECT workspace_id, organization_id, name, audience_type, report_profile,
                   default_workflow_template, platform_profile, settings, created_at, updated_at
            FROM workspaces
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        )
        return [
            {
                "workspace_id": str(row[0]),
                "organization_id": str(row[1]),
                "name": row[2],
                "audience_type": row[3],
                "report_profile": row[4],
                "default_workflow_template": row[5],
                "platform_profile": row[6],
                "settings": sanitize_payload(row[7]) if row[7] is not None else {},
                "created_at": row[8],
                "updated_at": row[9],
            }
            for row in rows
        ]

    def get_workspace(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            return self._workspaces.get(str(workspace_id))
        row = db.safe_fetchone(
            """
            SELECT workspace_id, organization_id, name, audience_type, report_profile,
                   default_workflow_template, platform_profile, settings, created_at, updated_at
            FROM workspaces
            WHERE workspace_id=%s
            """,
            (workspace_id,),
        )
        if not row:
            return None
        return {
            "workspace_id": str(row[0]),
            "organization_id": str(row[1]),
            "name": row[2],
            "audience_type": row[3],
            "report_profile": row[4],
            "default_workflow_template": row[5],
            "platform_profile": row[6],
            "settings": sanitize_payload(row[7]) if row[7] is not None else {},
            "created_at": row[8],
            "updated_at": row[9],
        }

    def upsert_coverage_entity(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        entity = CoverageEntity.model_validate(payload).model_dump(mode="python")
        symbol = str(entity.get("symbol") or "").upper() or None
        if self.use_memory:
            if symbol:
                for existing in self._entities.values():
                    if str(existing.get("symbol") or "").upper() == symbol:
                        merged = {
                            **existing,
                            **sanitize_payload(entity),
                            "updated_at": self._now_iso(),
                        }
                        self._entities[str(existing["entity_id"])] = merged
                        return merged
            entity["created_at"] = entity.get("created_at") or self._now_iso()
            entity["updated_at"] = self._now_iso()
            self._entities[entity["entity_id"]] = sanitize_payload(entity)
            return self._entities[entity["entity_id"]]
        row = db.exec1(
            """
            INSERT INTO coverage_entities (
                entity_id, symbol, external_identifier, entity_type, display_name,
                sector, strategy, theme, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (entity_id)
            DO UPDATE SET
                symbol = EXCLUDED.symbol,
                external_identifier = EXCLUDED.external_identifier,
                entity_type = EXCLUDED.entity_type,
                display_name = EXCLUDED.display_name,
                sector = EXCLUDED.sector,
                strategy = EXCLUDED.strategy,
                theme = EXCLUDED.theme,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING entity_id, symbol, external_identifier, entity_type, display_name,
                sector, strategy, theme, metadata, created_at, updated_at
            """,
            (
                entity["entity_id"],
                entity.get("symbol"),
                entity.get("external_identifier"),
                entity.get("entity_type"),
                entity.get("display_name"),
                entity.get("sector"),
                entity.get("strategy"),
                entity.get("theme"),
                Json(sanitize_payload(entity.get("metadata") or {})),
            ),
        )
        return {
            "entity_id": str(row[0]),
            "symbol": row[1],
            "external_identifier": row[2],
            "entity_type": row[3],
            "display_name": row[4],
            "sector": row[5],
            "strategy": row[6],
            "theme": row[7],
            "metadata": sanitize_payload(row[8]) if row[8] is not None else {},
            "created_at": row[9],
            "updated_at": row[10],
        }

    def find_coverage_entity_by_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            return None
        if self.use_memory:
            for entity in self._entities.values():
                if str(entity.get("symbol") or "").upper() == normalized:
                    return sanitize_payload(entity)
            return None
        row = db.safe_fetchone(
            """
            SELECT entity_id, symbol, external_identifier, entity_type, display_name,
                   sector, strategy, theme, metadata, created_at, updated_at
            FROM coverage_entities
            WHERE symbol=%s
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (normalized,),
        )
        if not row:
            return None
        return {
            "entity_id": str(row[0]),
            "symbol": row[1],
            "external_identifier": row[2],
            "entity_type": row[3],
            "display_name": row[4],
            "sector": row[5],
            "strategy": row[6],
            "theme": row[7],
            "metadata": sanitize_payload(row[8]) if row[8] is not None else {},
            "created_at": row[9],
            "updated_at": row[10],
        }

    def create_workflow(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = WorkflowInstance.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = self._now_iso()
            record["updated_at"] = self._now_iso()
            self._workflows[record["workflow_id"]] = sanitize_payload(record)
            return self._workflows[record["workflow_id"]]
        row = db.exec1(
            """
            INSERT INTO workflow_instances (
                workflow_id, workspace_id, workflow_template_id, title, status, stage,
                priority, owner_placeholder, metadata, stage_state
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (workflow_id)
            DO UPDATE SET
                workspace_id = EXCLUDED.workspace_id,
                workflow_template_id = EXCLUDED.workflow_template_id,
                title = EXCLUDED.title,
                status = EXCLUDED.status,
                stage = EXCLUDED.stage,
                priority = EXCLUDED.priority,
                owner_placeholder = EXCLUDED.owner_placeholder,
                metadata = EXCLUDED.metadata,
                stage_state = EXCLUDED.stage_state,
                updated_at = now()
            RETURNING workflow_id, workspace_id, workflow_template_id, title, status, stage,
                priority, owner_placeholder, metadata, stage_state, created_at, updated_at
            """,
            (
                record["workflow_id"],
                record["workspace_id"],
                record["workflow_template_id"],
                record["title"],
                record["status"],
                record["stage"],
                record["priority"],
                record.get("owner_placeholder"),
                Json(sanitize_payload(record.get("metadata") or {})),
                Json(sanitize_payload(record.get("stage_state") or {})),
            ),
        )
        return {
            "workflow_id": str(row[0]),
            "workspace_id": str(row[1]),
            "workflow_template_id": row[2],
            "title": row[3],
            "status": row[4],
            "stage": row[5],
            "priority": row[6],
            "owner_placeholder": row[7],
            "metadata": sanitize_payload(row[8]) if row[8] is not None else {},
            "stage_state": sanitize_payload(row[9]) if row[9] is not None else {},
            "created_at": row[10],
            "updated_at": row[11],
        }

    def list_workflows(self, *, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = [
                item
                for item in self._workflows.values()
                if workspace_id is None or str(item.get("workspace_id")) == str(workspace_id)
            ]
            rows.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        rows = db.safe_fetchall(
            f"""
            SELECT workflow_id, workspace_id, workflow_template_id, title, status, stage,
                   priority, owner_placeholder, metadata, stage_state, created_at, updated_at
            FROM workflow_instances
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        )
        return [
            {
                "workflow_id": str(row[0]),
                "workspace_id": str(row[1]),
                "workflow_template_id": row[2],
                "title": row[3],
                "status": row[4],
                "stage": row[5],
                "priority": row[6],
                "owner_placeholder": row[7],
                "metadata": sanitize_payload(row[8]) if row[8] is not None else {},
                "stage_state": sanitize_payload(row[9]) if row[9] is not None else {},
                "created_at": row[10],
                "updated_at": row[11],
            }
            for row in rows
        ]

    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            return self._workflows.get(str(workflow_id))
        row = db.safe_fetchone(
            """
            SELECT workflow_id, workspace_id, workflow_template_id, title, status, stage,
                   priority, owner_placeholder, metadata, stage_state, created_at, updated_at
            FROM workflow_instances
            WHERE workflow_id=%s
            """,
            (workflow_id,),
        )
        if not row:
            return None
        return {
            "workflow_id": str(row[0]),
            "workspace_id": str(row[1]),
            "workflow_template_id": row[2],
            "title": row[3],
            "status": row[4],
            "stage": row[5],
            "priority": row[6],
            "owner_placeholder": row[7],
            "metadata": sanitize_payload(row[8]) if row[8] is not None else {},
            "stage_state": sanitize_payload(row[9]) if row[9] is not None else {},
            "created_at": row[10],
            "updated_at": row[11],
        }

    def create_dossier(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = DossierRecord.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = self._now_iso()
            record["updated_at"] = self._now_iso()
            self._dossiers[record["dossier_id"]] = sanitize_payload(record)
            return self._dossiers[record["dossier_id"]]
        row = db.exec1(
            """
            INSERT INTO dossiers (
                dossier_id, workflow_id, entity_id, dossier_type, title, current_summary,
                latest_axiom_analysis_id, latest_deployability_tier, latest_regime_label,
                latest_trade_family, latest_size_band, evidence_status, workflow_stage_state,
                sections, monitoring_state, historical_evidence_summary, lineage_summary, metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
            )
            ON CONFLICT (dossier_id)
            DO UPDATE SET
                workflow_id = EXCLUDED.workflow_id,
                entity_id = EXCLUDED.entity_id,
                dossier_type = EXCLUDED.dossier_type,
                title = EXCLUDED.title,
                current_summary = EXCLUDED.current_summary,
                latest_axiom_analysis_id = EXCLUDED.latest_axiom_analysis_id,
                latest_deployability_tier = EXCLUDED.latest_deployability_tier,
                latest_regime_label = EXCLUDED.latest_regime_label,
                latest_trade_family = EXCLUDED.latest_trade_family,
                latest_size_band = EXCLUDED.latest_size_band,
                evidence_status = EXCLUDED.evidence_status,
                workflow_stage_state = EXCLUDED.workflow_stage_state,
                sections = EXCLUDED.sections,
                monitoring_state = EXCLUDED.monitoring_state,
                historical_evidence_summary = EXCLUDED.historical_evidence_summary,
                lineage_summary = EXCLUDED.lineage_summary,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING dossier_id, workflow_id, entity_id, dossier_type, title, current_summary,
                latest_axiom_analysis_id, latest_deployability_tier, latest_regime_label,
                latest_trade_family, latest_size_band, evidence_status, workflow_stage_state,
                sections, monitoring_state, historical_evidence_summary, lineage_summary, metadata,
                created_at, updated_at
            """,
            (
                record["dossier_id"],
                record["workflow_id"],
                record["entity_id"],
                record["dossier_type"],
                record["title"],
                Json(sanitize_payload(record.get("current_summary") or {})),
                record.get("latest_axiom_analysis_id"),
                record.get("latest_deployability_tier"),
                record.get("latest_regime_label"),
                record.get("latest_trade_family"),
                record.get("latest_size_band"),
                record.get("evidence_status"),
                Json(sanitize_payload(record.get("workflow_stage_state") or {})),
                Json(sanitize_payload(record.get("sections") or [])),
                Json(sanitize_payload(record.get("monitoring_state") or {})),
                Json(sanitize_payload(record.get("historical_evidence_summary") or {})),
                Json(sanitize_payload(record.get("lineage_summary") or {})),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return self._row_to_dossier(row)

    def _row_to_dossier(self, row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "dossier_id": str(row[0]),
            "workflow_id": str(row[1]),
            "entity_id": str(row[2]),
            "dossier_type": row[3],
            "title": row[4],
            "current_summary": sanitize_payload(row[5]) if row[5] is not None else {},
            "latest_axiom_analysis_id": row[6],
            "latest_deployability_tier": row[7],
            "latest_regime_label": row[8],
            "latest_trade_family": row[9],
            "latest_size_band": row[10],
            "evidence_status": row[11],
            "workflow_stage_state": sanitize_payload(row[12]) if row[12] is not None else {},
            "sections": sanitize_payload(row[13]) if row[13] is not None else [],
            "monitoring_state": sanitize_payload(row[14]) if row[14] is not None else {},
            "historical_evidence_summary": sanitize_payload(row[15]) if row[15] is not None else {},
            "lineage_summary": sanitize_payload(row[16]) if row[16] is not None else {},
            "metadata": sanitize_payload(row[17]) if row[17] is not None else {},
            "created_at": row[18],
            "updated_at": row[19],
        }

    def update_dossier(self, dossier_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**(self.get_dossier(dossier_id) or {}), **sanitize_payload(payload)}
        merged["dossier_id"] = dossier_id
        return self.create_dossier(merged)

    def get_dossier(self, dossier_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            return self._dossiers.get(str(dossier_id))
        row = db.safe_fetchone(
            """
            SELECT dossier_id, workflow_id, entity_id, dossier_type, title, current_summary,
                   latest_axiom_analysis_id, latest_deployability_tier, latest_regime_label,
                   latest_trade_family, latest_size_band, evidence_status, workflow_stage_state,
                   sections, monitoring_state, historical_evidence_summary, lineage_summary,
                   metadata, created_at, updated_at
            FROM dossiers
            WHERE dossier_id=%s
            """,
            (dossier_id,),
        )
        if not row:
            return None
        return self._row_to_dossier(row)

    def list_dossiers(
        self,
        *,
        workspace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._dossiers.values())
            if workflow_id is not None:
                rows = [row for row in rows if str(row.get("workflow_id")) == str(workflow_id)]
            if workspace_id is not None:
                workflow_ids = {
                    workflow["workflow_id"]
                    for workflow in self._workflows.values()
                    if str(workflow.get("workspace_id")) == str(workspace_id)
                }
                rows = [row for row in rows if str(row.get("workflow_id")) in workflow_ids]
            rows.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
            return [sanitize_payload(item) for item in rows[:limit]]
        clauses = ["1=1"]
        params: List[Any] = []
        if workflow_id is not None:
            clauses.append("d.workflow_id=%s")
            params.append(workflow_id)
        if workspace_id is not None:
            clauses.append("w.workspace_id=%s")
            params.append(workspace_id)
        params.append(limit)
        rows = db.safe_fetchall(
            f"""
            SELECT d.dossier_id, d.workflow_id, d.entity_id, d.dossier_type, d.title, d.current_summary,
                   d.latest_axiom_analysis_id, d.latest_deployability_tier, d.latest_regime_label,
                   d.latest_trade_family, d.latest_size_band, d.evidence_status, d.workflow_stage_state,
                   d.sections, d.monitoring_state, d.historical_evidence_summary, d.lineage_summary,
                   d.metadata, d.created_at, d.updated_at
            FROM dossiers d
            JOIN workflow_instances w ON w.workflow_id = d.workflow_id
            WHERE {' AND '.join(clauses)}
            ORDER BY d.updated_at DESC, d.created_at DESC
            LIMIT %s
            """,
            params,
        )
        return [self._row_to_dossier(row) for row in rows]

    def add_dossier_analysis_link(self, dossier_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        link = AnalysisLink.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            link["linked_at"] = link.get("linked_at") or self._now_iso()
            self._dossier_links.append({"dossier_id": dossier_id, **sanitize_payload(link)})
            return self._dossier_links[-1]
        row = db.exec1(
            """
            INSERT INTO dossier_analysis_links (
                link_id, dossier_id, report_id, session_id, axiom_artifact_id,
                axiom_report_pack_artifact_id, axiom_lineage_artifact_id,
                axiom_history_artifact_id, axiom_calibration_artifact_id, linked_payload, linked_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, COALESCE(%s::timestamptz, now())
            )
            ON CONFLICT (link_id)
            DO UPDATE SET
                dossier_id = EXCLUDED.dossier_id,
                report_id = EXCLUDED.report_id,
                session_id = EXCLUDED.session_id,
                axiom_artifact_id = EXCLUDED.axiom_artifact_id,
                axiom_report_pack_artifact_id = EXCLUDED.axiom_report_pack_artifact_id,
                axiom_lineage_artifact_id = EXCLUDED.axiom_lineage_artifact_id,
                axiom_history_artifact_id = EXCLUDED.axiom_history_artifact_id,
                axiom_calibration_artifact_id = EXCLUDED.axiom_calibration_artifact_id,
                linked_payload = EXCLUDED.linked_payload,
                linked_at = EXCLUDED.linked_at
            RETURNING link_id, dossier_id, report_id, session_id, axiom_artifact_id,
                axiom_report_pack_artifact_id, axiom_lineage_artifact_id,
                axiom_history_artifact_id, axiom_calibration_artifact_id, linked_payload, linked_at
            """,
            (
                link["link_id"],
                dossier_id,
                link.get("report_id"),
                link.get("session_id"),
                link.get("axiom_artifact_id"),
                link.get("axiom_report_pack_artifact_id"),
                link.get("axiom_lineage_artifact_id"),
                link.get("axiom_history_artifact_id"),
                link.get("axiom_calibration_artifact_id"),
                Json(sanitize_payload(link.get("source_summary") or {})),
                link.get("linked_at"),
            ),
        )
        return {
            "link_id": str(row[0]),
            "dossier_id": str(row[1]),
            "report_id": row[2],
            "session_id": row[3],
            "axiom_artifact_id": row[4],
            "axiom_report_pack_artifact_id": row[5],
            "axiom_lineage_artifact_id": row[6],
            "axiom_history_artifact_id": row[7],
            "axiom_calibration_artifact_id": row[8],
            "source_summary": sanitize_payload(row[9]) if row[9] is not None else {},
            "linked_at": row[10],
        }

    def list_dossier_analysis_links(self, dossier_id: str) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = [row for row in self._dossier_links if str(row.get("dossier_id")) == str(dossier_id)]
            rows.sort(key=lambda item: item.get("linked_at", 0), reverse=True)
            return [sanitize_payload(item) for item in rows]
        rows = db.safe_fetchall(
            """
            SELECT link_id, dossier_id, report_id, session_id, axiom_artifact_id,
                   axiom_report_pack_artifact_id, axiom_lineage_artifact_id,
                   axiom_history_artifact_id, axiom_calibration_artifact_id, linked_payload, linked_at
            FROM dossier_analysis_links
            WHERE dossier_id=%s
            ORDER BY linked_at DESC
            """,
            (dossier_id,),
        )
        return [
            {
                "link_id": str(row[0]),
                "dossier_id": str(row[1]),
                "report_id": row[2],
                "session_id": row[3],
                "axiom_artifact_id": row[4],
                "axiom_report_pack_artifact_id": row[5],
                "axiom_lineage_artifact_id": row[6],
                "axiom_history_artifact_id": row[7],
                "axiom_calibration_artifact_id": row[8],
                "source_summary": sanitize_payload(row[9]) if row[9] is not None else {},
                "linked_at": row[10],
            }
            for row in rows
        ]


platform_store = PlatformStore()
