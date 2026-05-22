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
    ApprovalRequest,
    AuditEvent,
    CoverageEntity,
    DossierRecord,
    ExportIntegrityResult,
    ExportManifest,
    ExportRetrievalResult,
    ExportStorageRef,
    ExportVersionRecord,
    IntegrationBinding,
    IntegrationExecutionRecord,
    MembershipRecord,
    OrganizationProfile,
    RenderedExportResult,
    StoredExportRecord,
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
        self._memberships: Dict[str, Dict[str, Any]] = {}
        self._approval_requests: Dict[str, Dict[str, Any]] = {}
        self._audit_events: List[Dict[str, Any]] = []
        self._export_manifests: Dict[str, Dict[str, Any]] = {}
        self._rendered_exports: Dict[str, Dict[str, Any]] = {}
        self._stored_exports: Dict[str, Dict[str, Any]] = {}
        self._integration_bindings: Dict[str, Dict[str, Any]] = {}
        self._integration_executions: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> float:
        return time.time()

    def _now_iso(self) -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    def _uuid(self) -> str:
        return str(uuid.uuid4())

    def _matches_scope(
        self,
        *,
        organization_id: Optional[str],
        workspace_id: Optional[str],
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> bool:
        scoped_org_ids = {str(item) for item in organization_ids or [] if item is not None}
        scoped_workspace_ids = {
            str(item) for item in workspace_ids or [] if item is not None
        }
        if workspace_ids is not None:
            if workspace_id is not None and str(workspace_id) in scoped_workspace_ids:
                return True
            if organization_ids is not None and organization_id is not None and str(
                organization_id
            ) in scoped_org_ids:
                return True
            return False
        if organization_ids is not None:
            return organization_id is not None and str(organization_id) in scoped_org_ids
        return True

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

    def list_workspaces(
        self,
        *,
        organization_id: Optional[str] = None,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = [
                item
                for item in self._workspaces.values()
                if organization_id is None or str(item.get("organization_id")) == str(organization_id)
            ]
            rows = [
                item
                for item in rows
                if self._matches_scope(
                    organization_id=item.get("organization_id"),
                    workspace_id=item.get("workspace_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                )
            ]
            rows.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if organization_id is not None:
            clauses.append("organization_id=%s")
            params.append(organization_id)
        if organization_ids:
            clauses.append("organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
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

    def get_workspace(
        self,
        workspace_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._workspaces.get(str(workspace_id))
            if item and self._matches_scope(
                organization_id=item.get("organization_id"),
                workspace_id=item.get("workspace_id"),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            ):
                return item
            return None
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
        result = {
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
        if not self._matches_scope(
            organization_id=result.get("organization_id"),
            workspace_id=result.get("workspace_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        ):
            return None
        return result

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

    def list_workflows(
        self,
        *,
        workspace_id: Optional[str] = None,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = [
                item
                for item in self._workflows.values()
                if workspace_id is None or str(item.get("workspace_id")) == str(workspace_id)
            ]
            scoped_workspace_ids = set(str(item) for item in workspace_ids or [])
            if scoped_workspace_ids:
                rows = [
                    item
                    for item in rows
                    if str(item.get("workspace_id")) in scoped_workspace_ids
                ]
            elif organization_ids:
                scoped_org_ids = set(str(item) for item in organization_ids or [])
                rows = [
                    item
                    for item in rows
                    if str(
                        (self._workspaces.get(str(item.get("workspace_id"))) or {}).get(
                            "organization_id"
                        )
                    )
                    in scoped_org_ids
                ]
            rows.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        join = ""
        if organization_ids:
            join = "JOIN workspaces ws ON ws.workspace_id = workflow_instances.workspace_id"
            clauses.append("ws.organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workflow_instances.workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
        rows = db.safe_fetchall(
            f"""
            SELECT workflow_id, workspace_id, workflow_template_id, title, status, stage,
                   priority, owner_placeholder, metadata, stage_state, created_at, updated_at
            FROM workflow_instances
            {join}
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

    def get_workflow(
        self,
        workflow_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._workflows.get(str(workflow_id))
            if item is None:
                return None
            if workspace_ids and str(item.get("workspace_id")) not in {
                str(value) for value in workspace_ids
            }:
                if not organization_ids:
                    return None
                workspace = self._workspaces.get(str(item.get("workspace_id"))) or {}
                if str(workspace.get("organization_id")) not in {
                    str(value) for value in organization_ids
                }:
                    return None
            return item
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
        result = {
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
        if workspace_ids and str(result.get("workspace_id")) not in {
            str(value) for value in workspace_ids
        }:
            if not organization_ids:
                return None
            workspace = self.get_workspace(str(result.get("workspace_id")))
            if str((workspace or {}).get("organization_id")) not in {
                str(value) for value in organization_ids
            }:
                return None
        return result

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

    def get_dossier(
        self,
        dossier_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._dossiers.get(str(dossier_id))
            if item is None:
                return None
            if workspace_ids or organization_ids:
                workflow = self._workflows.get(str(item.get("workflow_id"))) or {}
                workspace = self._workspaces.get(str(workflow.get("workspace_id"))) or {}
                if not self._matches_scope(
                    organization_id=workspace.get("organization_id"),
                    workspace_id=workflow.get("workspace_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                ):
                    return None
            return item
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
        result = self._row_to_dossier(row)
        if workspace_ids or organization_ids:
            workflow = self.get_workflow(
                str(result.get("workflow_id")),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            )
            if workflow is None:
                return None
        return result

    def list_dossiers(
        self,
        *,
        workspace_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
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
            if workspace_ids or organization_ids:
                allowed_workflow_ids = {
                    workflow["workflow_id"]
                    for workflow in self._workflows.values()
                    if self._matches_scope(
                        organization_id=(
                            self._workspaces.get(str(workflow.get("workspace_id"))) or {}
                        ).get("organization_id"),
                        workspace_id=workflow.get("workspace_id"),
                        organization_ids=organization_ids,
                        workspace_ids=workspace_ids,
                    )
                }
                rows = [
                    row for row in rows if str(row.get("workflow_id")) in allowed_workflow_ids
                ]
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
        if organization_ids:
            clauses.append("w.organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("w.workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
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

    def create_membership(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = MembershipRecord.model_validate(
            {
                "membership_id": payload.get("membership_id") or self._uuid(),
                "user_id": payload["user_id"],
                "organization_id": payload.get("organization_id"),
                "workspace_id": payload.get("workspace_id"),
                "role": payload.get("role") or "analyst",
                "permissions": payload.get("permissions") or [],
                "status": payload.get("status") or "active",
                "metadata": payload.get("metadata") or {},
            }
        ).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = self._now_iso()
            record["updated_at"] = self._now_iso()
            self._memberships[record["membership_id"]] = sanitize_payload(record)
            return self._memberships[record["membership_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_memberships (
                membership_id, user_id, organization_id, workspace_id, role, permissions, status, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
            ON CONFLICT (membership_id)
            DO UPDATE SET
                user_id = EXCLUDED.user_id,
                organization_id = EXCLUDED.organization_id,
                workspace_id = EXCLUDED.workspace_id,
                role = EXCLUDED.role,
                permissions = EXCLUDED.permissions,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING membership_id, user_id, organization_id, workspace_id, role, permissions, status, metadata, created_at, updated_at
            """,
            (
                record["membership_id"],
                record["user_id"],
                record.get("organization_id"),
                record.get("workspace_id"),
                record["role"],
                Json(sanitize_payload(record.get("permissions") or [])),
                record["status"],
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return {
            "membership_id": str(row[0]),
            "user_id": row[1],
            "organization_id": str(row[2]) if row[2] is not None else None,
            "workspace_id": str(row[3]) if row[3] is not None else None,
            "role": row[4],
            "permissions": sanitize_payload(row[5]) if row[5] is not None else [],
            "status": row[6],
            "metadata": sanitize_payload(row[7]) if row[7] is not None else {},
            "created_at": row[8],
            "updated_at": row[9],
        }

    def find_membership(
        self,
        *,
        user_id: str,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        include_org_fallback: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            candidates = [
                item
                for item in self._memberships.values()
                if str(item.get("user_id")) == str(user_id)
                and (organization_id is None or str(item.get("organization_id")) == str(organization_id))
                and (workspace_id is None or str(item.get("workspace_id")) == str(workspace_id))
                and str(item.get("status") or "active") == "active"
            ]
            if not candidates and include_org_fallback and organization_id is not None:
                candidates = [
                    item
                    for item in self._memberships.values()
                    if str(item.get("user_id")) == str(user_id)
                    and str(item.get("organization_id")) == str(organization_id)
                    and item.get("workspace_id") in {None, ""}
                    and str(item.get("status") or "active") == "active"
                ]
            candidates.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
            return sanitize_payload(candidates[0]) if candidates else None
        clauses = ["user_id=%s", "status='active'"]
        params: List[Any] = [user_id]
        if organization_id is not None:
            clauses.append("organization_id=%s")
            params.append(organization_id)
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        row = db.safe_fetchone(
            f"""
            SELECT membership_id, user_id, organization_id, workspace_id, role, permissions, status, metadata, created_at, updated_at
            FROM platform_memberships
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            params,
        )
        if not row and include_org_fallback and organization_id is not None and workspace_id is not None:
            row = db.safe_fetchone(
                """
                SELECT membership_id, user_id, organization_id, workspace_id, role, permissions, status, metadata, created_at, updated_at
                FROM platform_memberships
                WHERE user_id=%s AND organization_id=%s AND workspace_id IS NULL AND status='active'
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (user_id, organization_id),
            )
        if not row:
            return None
        return {
            "membership_id": str(row[0]),
            "user_id": row[1],
            "organization_id": str(row[2]) if row[2] is not None else None,
            "workspace_id": str(row[3]) if row[3] is not None else None,
            "role": row[4],
            "permissions": sanitize_payload(row[5]) if row[5] is not None else [],
            "status": row[6],
            "metadata": sanitize_payload(row[7]) if row[7] is not None else {},
            "created_at": row[8],
            "updated_at": row[9],
        }

    def list_memberships(
        self,
        *,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._memberships.values())
            if user_id is not None:
                rows = [row for row in rows if str(row.get("user_id")) == str(user_id)]
            if organization_id is not None:
                rows = [row for row in rows if str(row.get("organization_id")) == str(organization_id)]
            if workspace_id is not None:
                rows = [row for row in rows if str(row.get("workspace_id")) == str(workspace_id)]
            rows.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if user_id is not None:
            clauses.append("user_id=%s")
            params.append(user_id)
        if organization_id is not None:
            clauses.append("organization_id=%s")
            params.append(organization_id)
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        rows = db.safe_fetchall(
            f"""
            SELECT membership_id, user_id, organization_id, workspace_id, role, permissions, status, metadata, created_at, updated_at
            FROM platform_memberships
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        )
        return [
            {
                "membership_id": str(row[0]),
                "user_id": row[1],
                "organization_id": str(row[2]) if row[2] is not None else None,
                "workspace_id": str(row[3]) if row[3] is not None else None,
                "role": row[4],
                "permissions": sanitize_payload(row[5]) if row[5] is not None else [],
                "status": row[6],
                "metadata": sanitize_payload(row[7]) if row[7] is not None else {},
                "created_at": row[8],
                "updated_at": row[9],
            }
            for row in rows
        ]

    def create_approval_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = ApprovalRequest.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = record.get("created_at") or self._now_iso()
            record["updated_at"] = self._now_iso()
            self._approval_requests[record["approval_id"]] = sanitize_payload(record)
            return self._approval_requests[record["approval_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_approval_requests (
                approval_id, workflow_id, dossier_id, requested_role, requested_by, status, stage,
                rationale, required_permissions, decisions, metadata
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            ON CONFLICT (approval_id)
            DO UPDATE SET
                workflow_id = EXCLUDED.workflow_id,
                dossier_id = EXCLUDED.dossier_id,
                requested_role = EXCLUDED.requested_role,
                requested_by = EXCLUDED.requested_by,
                status = EXCLUDED.status,
                stage = EXCLUDED.stage,
                rationale = EXCLUDED.rationale,
                required_permissions = EXCLUDED.required_permissions,
                decisions = EXCLUDED.decisions,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING approval_id, workflow_id, dossier_id, requested_role, requested_by, status, stage,
                rationale, required_permissions, decisions, metadata, created_at, updated_at
            """,
            (
                record["approval_id"],
                record["workflow_id"],
                record.get("dossier_id"),
                record["requested_role"],
                Json(sanitize_payload(record.get("requested_by") or {})),
                record["status"],
                record.get("stage"),
                record.get("rationale"),
                Json(sanitize_payload(record.get("required_permissions") or [])),
                Json(sanitize_payload(record.get("decisions") or [])),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return {
            "approval_id": str(row[0]),
            "workflow_id": str(row[1]),
            "dossier_id": str(row[2]) if row[2] is not None else None,
            "requested_role": row[3],
            "requested_by": sanitize_payload(row[4]) if row[4] is not None else {},
            "status": row[5],
            "stage": row[6],
            "rationale": row[7],
            "required_permissions": sanitize_payload(row[8]) if row[8] is not None else [],
            "decisions": sanitize_payload(row[9]) if row[9] is not None else [],
            "metadata": sanitize_payload(row[10]) if row[10] is not None else {},
            "created_at": row[11],
            "updated_at": row[12],
        }

    def update_approval_request(self, approval_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**(self.get_approval_request(approval_id) or {}), **sanitize_payload(payload)}
        merged["approval_id"] = approval_id
        return self.create_approval_request(merged)

    def get_approval_request(self, approval_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            return self._approval_requests.get(str(approval_id))
        row = db.safe_fetchone(
            """
            SELECT approval_id, workflow_id, dossier_id, requested_role, requested_by, status, stage,
                   rationale, required_permissions, decisions, metadata, created_at, updated_at
            FROM platform_approval_requests
            WHERE approval_id=%s
            """,
            (approval_id,),
        )
        if not row:
            return None
        return {
            "approval_id": str(row[0]),
            "workflow_id": str(row[1]),
            "dossier_id": str(row[2]) if row[2] is not None else None,
            "requested_role": row[3],
            "requested_by": sanitize_payload(row[4]) if row[4] is not None else {},
            "status": row[5],
            "stage": row[6],
            "rationale": row[7],
            "required_permissions": sanitize_payload(row[8]) if row[8] is not None else [],
            "decisions": sanitize_payload(row[9]) if row[9] is not None else [],
            "metadata": sanitize_payload(row[10]) if row[10] is not None else {},
            "created_at": row[11],
            "updated_at": row[12],
        }

    def list_approval_requests(
        self,
        *,
        workflow_id: Optional[str] = None,
        dossier_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._approval_requests.values())
            if workflow_id is not None:
                rows = [row for row in rows if str(row.get("workflow_id")) == str(workflow_id)]
            if dossier_id is not None:
                rows = [row for row in rows if str(row.get("dossier_id")) == str(dossier_id)]
            rows.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if workflow_id is not None:
            clauses.append("workflow_id=%s")
            params.append(workflow_id)
        if dossier_id is not None:
            clauses.append("dossier_id=%s")
            params.append(dossier_id)
        rows = db.safe_fetchall(
            f"""
            SELECT approval_id, workflow_id, dossier_id, requested_role, requested_by, status, stage,
                   rationale, required_permissions, decisions, metadata, created_at, updated_at
            FROM platform_approval_requests
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        )
        return [
            {
                "approval_id": str(row[0]),
                "workflow_id": str(row[1]),
                "dossier_id": str(row[2]) if row[2] is not None else None,
                "requested_role": row[3],
                "requested_by": sanitize_payload(row[4]) if row[4] is not None else {},
                "status": row[5],
                "stage": row[6],
                "rationale": row[7],
                "required_permissions": sanitize_payload(row[8]) if row[8] is not None else [],
                "decisions": sanitize_payload(row[9]) if row[9] is not None else [],
                "metadata": sanitize_payload(row[10]) if row[10] is not None else {},
                "created_at": row[11],
                "updated_at": row[12],
            }
            for row in rows
        ]

    def create_audit_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = AuditEvent.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["timestamp"] = record.get("timestamp") or self._now_iso()
            self._audit_events.append(sanitize_payload(record))
            self._audit_events.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
            return self._audit_events[0]
        row = db.exec1(
            """
            INSERT INTO platform_audit_events (
                event_id, event_type, resource_type, resource_id, organization_id, workspace_id,
                session_id, auth_mode, actor, event_ts, payload, rationale, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, COALESCE(%s::timestamptz, now()), %s::jsonb, %s, %s::jsonb)
            ON CONFLICT (event_id)
            DO UPDATE SET
                event_type = EXCLUDED.event_type,
                resource_type = EXCLUDED.resource_type,
                resource_id = EXCLUDED.resource_id,
                organization_id = EXCLUDED.organization_id,
                workspace_id = EXCLUDED.workspace_id,
                session_id = EXCLUDED.session_id,
                auth_mode = EXCLUDED.auth_mode,
                actor = EXCLUDED.actor,
                event_ts = EXCLUDED.event_ts,
                payload = EXCLUDED.payload,
                rationale = EXCLUDED.rationale,
                metadata = EXCLUDED.metadata
            RETURNING event_id, event_type, resource_type, resource_id, organization_id, workspace_id,
                session_id, auth_mode, actor, event_ts, payload, rationale, metadata
            """,
            (
                record["event_id"],
                record["event_type"],
                record["resource_type"],
                record.get("resource_id"),
                record.get("organization_id"),
                record.get("workspace_id"),
                record.get("session_id"),
                record.get("auth_mode"),
                Json(sanitize_payload(record.get("actor") or {})),
                record.get("timestamp"),
                Json(sanitize_payload(record.get("payload") or {})),
                record.get("rationale"),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return {
            "event_id": str(row[0]),
            "event_type": row[1],
            "resource_type": row[2],
            "resource_id": str(row[3]) if row[3] is not None else None,
            "organization_id": str(row[4]) if row[4] is not None else None,
            "workspace_id": str(row[5]) if row[5] is not None else None,
            "session_id": row[6],
            "auth_mode": row[7],
            "actor": sanitize_payload(row[8]) if row[8] is not None else {},
            "timestamp": row[9],
            "payload": sanitize_payload(row[10]) if row[10] is not None else {},
            "rationale": row[11],
            "metadata": sanitize_payload(row[12]) if row[12] is not None else {},
        }

    def list_audit_events(
        self,
        *,
        organization_id: Optional[str] = None,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_id: Optional[str] = None,
        workspace_ids: Optional[Sequence[str]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._audit_events)
            if organization_id is not None:
                rows = [
                    row for row in rows if str(row.get("organization_id")) == str(organization_id)
                ]
            if workspace_id is not None:
                rows = [row for row in rows if str(row.get("workspace_id")) == str(workspace_id)]
            if organization_ids or workspace_ids:
                rows = [
                    row
                    for row in rows
                    if self._matches_scope(
                        organization_id=row.get("organization_id"),
                        workspace_id=row.get("workspace_id"),
                        organization_ids=organization_ids,
                        workspace_ids=workspace_ids,
                    )
                ]
            if resource_type is not None:
                rows = [row for row in rows if str(row.get("resource_type")) == str(resource_type)]
            if resource_id is not None:
                rows = [row for row in rows if str(row.get("resource_id")) == str(resource_id)]
            rows.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
            return [sanitize_payload(item) for item in rows[:limit]]
        clauses = ["1=1"]
        params: List[Any] = []
        if organization_id is not None:
            clauses.append("organization_id=%s")
            params.append(organization_id)
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        if organization_ids:
            clauses.append("organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
        if resource_type is not None:
            clauses.append("resource_type=%s")
            params.append(resource_type)
        if resource_id is not None:
            clauses.append("resource_id=%s")
            params.append(resource_id)
        params.append(limit)
        rows = db.safe_fetchall(
            f"""
            SELECT event_id, event_type, resource_type, resource_id, organization_id, workspace_id,
                   session_id, auth_mode, actor, event_ts, payload, rationale, metadata
            FROM platform_audit_events
            WHERE {' AND '.join(clauses)}
            ORDER BY event_ts DESC
            LIMIT %s
            """,
            params,
        )
        return [
            {
                "event_id": str(row[0]),
                "event_type": row[1],
                "resource_type": row[2],
                "resource_id": str(row[3]) if row[3] is not None else None,
                "organization_id": str(row[4]) if row[4] is not None else None,
                "workspace_id": str(row[5]) if row[5] is not None else None,
                "session_id": row[6],
                "auth_mode": row[7],
                "actor": sanitize_payload(row[8]) if row[8] is not None else {},
                "timestamp": row[9],
                "payload": sanitize_payload(row[10]) if row[10] is not None else {},
                "rationale": row[11],
                "metadata": sanitize_payload(row[12]) if row[12] is not None else {},
            }
            for row in rows
        ]

    def create_export_manifest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = ExportManifest.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["generated_at"] = record.get("generated_at") or self._now_iso()
            self._export_manifests[record["export_id"]] = sanitize_payload(record)
            return self._export_manifests[record["export_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_export_packs (
                export_id, dossier_id, workflow_id, workspace_id, pack_type, title, subtitle,
                generated_at, framework_version, organization_context, workspace_context, entity_context,
                approval_status, evidence_summary, ordered_sections, metadata, content_hash, status
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now()), %s,
                %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s, %s
            )
            ON CONFLICT (export_id)
            DO UPDATE SET
                dossier_id = EXCLUDED.dossier_id,
                workflow_id = EXCLUDED.workflow_id,
                workspace_id = EXCLUDED.workspace_id,
                pack_type = EXCLUDED.pack_type,
                title = EXCLUDED.title,
                subtitle = EXCLUDED.subtitle,
                generated_at = EXCLUDED.generated_at,
                framework_version = EXCLUDED.framework_version,
                organization_context = EXCLUDED.organization_context,
                workspace_context = EXCLUDED.workspace_context,
                entity_context = EXCLUDED.entity_context,
                approval_status = EXCLUDED.approval_status,
                evidence_summary = EXCLUDED.evidence_summary,
                ordered_sections = EXCLUDED.ordered_sections,
                metadata = EXCLUDED.metadata,
                content_hash = EXCLUDED.content_hash,
                status = EXCLUDED.status
            RETURNING export_id, dossier_id, workflow_id, workspace_id, pack_type, title, subtitle,
                generated_at, framework_version, organization_context, workspace_context, entity_context,
                approval_status, evidence_summary, ordered_sections, metadata, content_hash, status
            """,
            (
                record["export_id"],
                record["dossier_id"],
                record.get("workflow_id"),
                record.get("workspace_id"),
                record["pack_type"],
                record["title"],
                record.get("subtitle"),
                record.get("generated_at"),
                record.get("framework_version"),
                Json(sanitize_payload(record.get("organization_context") or {})),
                Json(sanitize_payload(record.get("workspace_context") or {})),
                Json(sanitize_payload(record.get("entity_context") or {})),
                record.get("approval_status"),
                record.get("evidence_summary"),
                Json(sanitize_payload(record.get("ordered_sections") or [])),
                Json(sanitize_payload(record.get("metadata") or {})),
                record.get("content_hash"),
                record.get("status"),
            ),
        )
        return {
            "export_id": str(row[0]),
            "dossier_id": str(row[1]),
            "workflow_id": str(row[2]) if row[2] is not None else None,
            "workspace_id": str(row[3]) if row[3] is not None else None,
            "pack_type": row[4],
            "title": row[5],
            "subtitle": row[6],
            "generated_at": row[7],
            "framework_version": row[8],
            "organization_context": sanitize_payload(row[9]) if row[9] is not None else {},
            "workspace_context": sanitize_payload(row[10]) if row[10] is not None else {},
            "entity_context": sanitize_payload(row[11]) if row[11] is not None else {},
            "approval_status": row[12],
            "evidence_summary": row[13],
            "ordered_sections": sanitize_payload(row[14]) if row[14] is not None else [],
            "metadata": sanitize_payload(row[15]) if row[15] is not None else {},
            "content_hash": row[16],
            "status": row[17],
        }

    def list_export_manifests(
        self,
        dossier_id: Optional[str] = None,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._export_manifests.values())
            if dossier_id is not None:
                rows = [
                    item
                    for item in rows
                    if str(item.get("dossier_id")) == str(dossier_id)
                ]
            rows = [
                item
                for item in rows
                if self._matches_scope(
                    organization_id=item.get("organization_context", {}).get("organization_id")
                    or item.get("organization_id"),
                    workspace_id=item.get("workspace_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                )
            ]
            rows.sort(key=lambda item: item.get("generated_at", ""), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if dossier_id is not None:
            clauses.append("dossier_id=%s")
            params.append(dossier_id)
        if organization_ids:
            clauses.append("(organization_context->>'organization_id') = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
        rows = db.safe_fetchall(
            f"""
            SELECT export_id, dossier_id, workflow_id, workspace_id, pack_type, title, subtitle,
                   generated_at, framework_version, organization_context, workspace_context, entity_context,
                   approval_status, evidence_summary, ordered_sections, metadata, content_hash, status
            FROM platform_export_packs
            WHERE {' AND '.join(clauses)}
            ORDER BY generated_at DESC
            """,
            params,
        )
        return [
            {
                "export_id": str(row[0]),
                "dossier_id": str(row[1]),
                "workflow_id": str(row[2]) if row[2] is not None else None,
                "workspace_id": str(row[3]) if row[3] is not None else None,
                "pack_type": row[4],
                "title": row[5],
                "subtitle": row[6],
                "generated_at": row[7],
                "framework_version": row[8],
                "organization_context": sanitize_payload(row[9]) if row[9] is not None else {},
                "workspace_context": sanitize_payload(row[10]) if row[10] is not None else {},
                "entity_context": sanitize_payload(row[11]) if row[11] is not None else {},
                "approval_status": row[12],
                "evidence_summary": row[13],
                "ordered_sections": sanitize_payload(row[14]) if row[14] is not None else [],
                "metadata": sanitize_payload(row[15]) if row[15] is not None else {},
                "content_hash": row[16],
                "status": row[17],
            }
            for row in rows
        ]

    def get_export_manifest(
        self,
        export_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._export_manifests.get(str(export_id))
            if item and self._matches_scope(
                organization_id=item.get("organization_context", {}).get("organization_id")
                or item.get("organization_id"),
                workspace_id=item.get("workspace_id"),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            ):
                return item
            return None
        row = db.safe_fetchone(
            """
            SELECT export_id, dossier_id, workflow_id, workspace_id, pack_type, title, subtitle,
                   generated_at, framework_version, organization_context, workspace_context, entity_context,
                   approval_status, evidence_summary, ordered_sections, metadata, content_hash, status
            FROM platform_export_packs
            WHERE export_id=%s
            """,
            (export_id,),
        )
        if not row:
            return None
        result = {
            "export_id": str(row[0]),
            "dossier_id": str(row[1]),
            "workflow_id": str(row[2]) if row[2] is not None else None,
            "workspace_id": str(row[3]) if row[3] is not None else None,
            "pack_type": row[4],
            "title": row[5],
            "subtitle": row[6],
            "generated_at": row[7],
            "framework_version": row[8],
            "organization_context": sanitize_payload(row[9]) if row[9] is not None else {},
            "workspace_context": sanitize_payload(row[10]) if row[10] is not None else {},
            "entity_context": sanitize_payload(row[11]) if row[11] is not None else {},
            "approval_status": row[12],
            "evidence_summary": row[13],
            "ordered_sections": sanitize_payload(row[14]) if row[14] is not None else [],
            "metadata": sanitize_payload(row[15]) if row[15] is not None else {},
            "content_hash": row[16],
            "status": row[17],
        }
        if not self._matches_scope(
            organization_id=result.get("organization_context", {}).get("organization_id"),
            workspace_id=result.get("workspace_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        ):
            return None
        return result

    def create_rendered_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = RenderedExportResult.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["generated_at"] = record.get("generated_at") or self._now_iso()
            self._rendered_exports[record["render_id"]] = sanitize_payload(record)
            return self._rendered_exports[record["render_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_rendered_exports (
                render_id, export_id, export_format, content_type, rendered_content,
                file_name_hint, section_count, checksum, generated_at, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now()), %s::jsonb)
            ON CONFLICT (render_id)
            DO UPDATE SET
                export_id = EXCLUDED.export_id,
                export_format = EXCLUDED.export_format,
                content_type = EXCLUDED.content_type,
                rendered_content = EXCLUDED.rendered_content,
                file_name_hint = EXCLUDED.file_name_hint,
                section_count = EXCLUDED.section_count,
                checksum = EXCLUDED.checksum,
                generated_at = EXCLUDED.generated_at,
                metadata = EXCLUDED.metadata
            RETURNING render_id, export_id, export_format, content_type, rendered_content,
                file_name_hint, section_count, checksum, generated_at, metadata
            """,
            (
                record["render_id"],
                record["export_id"],
                record["export_format"],
                record["content_type"],
                record["rendered_content"],
                record["file_name_hint"],
                record["section_count"],
                record.get("checksum"),
                record.get("generated_at"),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return {
            "render_id": str(row[0]),
            "export_id": str(row[1]),
            "export_format": row[2],
            "content_type": row[3],
            "rendered_content": row[4],
            "file_name_hint": row[5],
            "section_count": row[6],
            "checksum": row[7],
            "generated_at": row[8],
            "metadata": sanitize_payload(row[9]) if row[9] is not None else {},
        }

    def get_rendered_export(
        self,
        render_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._rendered_exports.get(str(render_id))
            if item is None:
                return None
            export = self.get_export_manifest(
                str(item.get("export_id") or ""),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            )
            return item if export is not None else None
        row = db.safe_fetchone(
            """
            SELECT render_id, export_id, export_format, content_type, rendered_content,
                   file_name_hint, section_count, checksum, generated_at, metadata
            FROM platform_rendered_exports
            WHERE render_id=%s
            """,
            (render_id,),
        )
        if not row:
            return None
        result = {
            "render_id": str(row[0]),
            "export_id": str(row[1]),
            "export_format": row[2],
            "content_type": row[3],
            "rendered_content": row[4],
            "file_name_hint": row[5],
            "section_count": row[6],
            "checksum": row[7],
            "generated_at": row[8],
            "metadata": sanitize_payload(row[9]) if row[9] is not None else {},
        }
        export = self.get_export_manifest(
            str(result.get("export_id") or ""),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        return result if export is not None else None

    def list_rendered_exports(
        self,
        *,
        export_id: Optional[str] = None,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._rendered_exports.values())
            if export_id is not None:
                rows = [row for row in rows if str(row.get("export_id")) == str(export_id)]
            if organization_ids or workspace_ids:
                rows = [
                    row
                    for row in rows
                    if self.get_export_manifest(
                        str(row.get("export_id") or ""),
                        organization_ids=organization_ids,
                        workspace_ids=workspace_ids,
                    )
                    is not None
                ]
            rows.sort(key=lambda item: item.get("generated_at", ""), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if export_id is not None:
            clauses.append("export_id=%s")
            params.append(export_id)
        rows = db.safe_fetchall(
            f"""
            SELECT render_id, export_id, export_format, content_type, rendered_content,
                   file_name_hint, section_count, checksum, generated_at, metadata
            FROM platform_rendered_exports
            WHERE {' AND '.join(clauses)}
            ORDER BY generated_at DESC
            """,
            params,
        )
        return [
            {
                "render_id": str(row[0]),
                "export_id": str(row[1]),
                "export_format": row[2],
                "content_type": row[3],
                "rendered_content": row[4],
                "file_name_hint": row[5],
                "section_count": row[6],
                "checksum": row[7],
                "generated_at": row[8],
                "metadata": sanitize_payload(row[9]) if row[9] is not None else {},
            }
            for row in rows
        ]

    def _row_to_stored_export(self, row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "stored_export_id": str(row[0]),
            "export_id": str(row[1]),
            "render_id": str(row[2]),
            "organization_id": str(row[3]) if row[3] is not None else None,
            "workspace_id": str(row[4]) if row[4] is not None else None,
            "dossier_id": str(row[5]) if row[5] is not None else None,
            "workflow_id": str(row[6]) if row[6] is not None else None,
            "pack_type": row[7],
            "export_format": row[8],
            "framework_version": row[9],
            "approval_status": row[10],
            "evidence_status": row[11],
            "checksum": row[12],
            "source_manifest_hash": row[13],
            "content_hash": row[14],
            "manifest_hash": row[15],
            "section_count": row[16],
            "file_name_hint": row[17],
            "content_type": row[18],
            "storage_backend": row[19],
            "storage_key": row[20],
            "storage_ref": sanitize_payload(row[21]) if row[21] is not None else {},
            "version_group_key": row[22],
            "version_number": row[23],
            "version_label": row[24],
            "status": row[25],
            "document_identity": sanitize_payload(row[26]) if row[26] is not None else {},
            "source_context": sanitize_payload(row[27]) if row[27] is not None else {},
            "approval_context": sanitize_payload(row[28]) if row[28] is not None else {},
            "axiom_context": sanitize_payload(row[29]) if row[29] is not None else {},
            "evidence_context": sanitize_payload(row[30]) if row[30] is not None else {},
            "export_context": sanitize_payload(row[31]) if row[31] is not None else {},
            "lineage_summary": sanitize_payload(row[32]) if row[32] is not None else {},
            "created_by": sanitize_payload(row[33]) if row[33] is not None else {},
            "metadata": sanitize_payload(row[34]) if row[34] is not None else {},
            "created_at": row[35],
            "updated_at": row[36],
        }

    def create_stored_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = StoredExportRecord.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = record.get("created_at") or self._now_iso()
            record["updated_at"] = self._now_iso()
            self._stored_exports[record["stored_export_id"]] = sanitize_payload(record)
            return self._stored_exports[record["stored_export_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_stored_exports (
                stored_export_id, export_id, render_id, organization_id, workspace_id, dossier_id,
                workflow_id, pack_type, export_format, framework_version, approval_status,
                evidence_status, checksum, source_manifest_hash, content_hash, manifest_hash,
                section_count, file_name_hint, content_type, storage_backend, storage_key,
                storage_ref, version_group_key, version_number, version_label, status,
                document_identity, source_context, approval_context, axiom_context,
                evidence_context, export_context, lineage_summary, created_by, metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb
            )
            ON CONFLICT (stored_export_id)
            DO UPDATE SET
                export_id = EXCLUDED.export_id,
                render_id = EXCLUDED.render_id,
                organization_id = EXCLUDED.organization_id,
                workspace_id = EXCLUDED.workspace_id,
                dossier_id = EXCLUDED.dossier_id,
                workflow_id = EXCLUDED.workflow_id,
                pack_type = EXCLUDED.pack_type,
                export_format = EXCLUDED.export_format,
                framework_version = EXCLUDED.framework_version,
                approval_status = EXCLUDED.approval_status,
                evidence_status = EXCLUDED.evidence_status,
                checksum = EXCLUDED.checksum,
                source_manifest_hash = EXCLUDED.source_manifest_hash,
                content_hash = EXCLUDED.content_hash,
                manifest_hash = EXCLUDED.manifest_hash,
                section_count = EXCLUDED.section_count,
                file_name_hint = EXCLUDED.file_name_hint,
                content_type = EXCLUDED.content_type,
                storage_backend = EXCLUDED.storage_backend,
                storage_key = EXCLUDED.storage_key,
                storage_ref = EXCLUDED.storage_ref,
                version_group_key = EXCLUDED.version_group_key,
                version_number = EXCLUDED.version_number,
                version_label = EXCLUDED.version_label,
                status = EXCLUDED.status,
                document_identity = EXCLUDED.document_identity,
                source_context = EXCLUDED.source_context,
                approval_context = EXCLUDED.approval_context,
                axiom_context = EXCLUDED.axiom_context,
                evidence_context = EXCLUDED.evidence_context,
                export_context = EXCLUDED.export_context,
                lineage_summary = EXCLUDED.lineage_summary,
                created_by = EXCLUDED.created_by,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING stored_export_id, export_id, render_id, organization_id, workspace_id, dossier_id,
                workflow_id, pack_type, export_format, framework_version, approval_status,
                evidence_status, checksum, source_manifest_hash, content_hash, manifest_hash,
                section_count, file_name_hint, content_type, storage_backend, storage_key,
                storage_ref, version_group_key, version_number, version_label, status,
                document_identity, source_context, approval_context, axiom_context,
                evidence_context, export_context, lineage_summary, created_by, metadata,
                created_at, updated_at
            """,
            (
                record["stored_export_id"],
                record["export_id"],
                record["render_id"],
                record.get("organization_id"),
                record.get("workspace_id"),
                record.get("dossier_id"),
                record.get("workflow_id"),
                record["pack_type"],
                record["export_format"],
                record.get("framework_version"),
                record.get("approval_status"),
                record.get("evidence_status"),
                record.get("checksum"),
                record.get("source_manifest_hash"),
                record.get("content_hash"),
                record.get("manifest_hash"),
                record.get("section_count"),
                record["file_name_hint"],
                record.get("content_type"),
                record["storage_backend"],
                record["storage_key"],
                Json(sanitize_payload(record.get("storage_ref") or {})),
                record["version_group_key"],
                record.get("version_number"),
                record.get("version_label"),
                record.get("status"),
                Json(sanitize_payload(record.get("document_identity") or {})),
                Json(sanitize_payload(record.get("source_context") or {})),
                Json(sanitize_payload(record.get("approval_context") or {})),
                Json(sanitize_payload(record.get("axiom_context") or {})),
                Json(sanitize_payload(record.get("evidence_context") or {})),
                Json(sanitize_payload(record.get("export_context") or {})),
                Json(sanitize_payload(record.get("lineage_summary") or {})),
                Json(sanitize_payload(record.get("created_by") or {})),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return self._row_to_stored_export(row)

    def update_stored_export(
        self,
        stored_export_id: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        current = self.get_stored_export(stored_export_id)
        if current is None:
            raise KeyError(stored_export_id)
        merged = {**current, **sanitize_payload(payload), "stored_export_id": stored_export_id}
        return self.create_stored_export(merged)

    def supersede_stored_exports(
        self,
        *,
        version_group_key: str,
        except_stored_export_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            updated: List[Dict[str, Any]] = []
            for key, item in list(self._stored_exports.items()):
                if str(item.get("version_group_key")) != str(version_group_key):
                    continue
                if except_stored_export_id and str(key) == str(except_stored_export_id):
                    continue
                if str(item.get("status")) == "stored":
                    item = {**item, "status": "superseded", "updated_at": self._now_iso()}
                    self._stored_exports[key] = sanitize_payload(item)
                    updated.append(self._stored_exports[key])
            return updated
        clauses = ["version_group_key=%s", "status=%s"]
        params: List[Any] = [version_group_key, "stored"]
        if except_stored_export_id:
            clauses.append("stored_export_id<>%s")
            params.append(except_stored_export_id)
        rows = db.safe_fetchall(
            f"""
            UPDATE platform_stored_exports
            SET status='superseded', updated_at=now()
            WHERE {' AND '.join(clauses)}
            RETURNING stored_export_id, export_id, render_id, organization_id, workspace_id, dossier_id,
                workflow_id, pack_type, export_format, framework_version, approval_status,
                evidence_status, checksum, source_manifest_hash, content_hash, manifest_hash,
                section_count, file_name_hint, content_type, storage_backend, storage_key,
                storage_ref, version_group_key, version_number, version_label, status,
                document_identity, source_context, approval_context, axiom_context,
                evidence_context, export_context, lineage_summary, created_by, metadata,
                created_at, updated_at
            """,
            params,
        )
        return [self._row_to_stored_export(row) for row in rows]

    def get_stored_export(
        self,
        stored_export_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._stored_exports.get(str(stored_export_id))
            if item and self._matches_scope(
                organization_id=item.get("organization_id"),
                workspace_id=item.get("workspace_id"),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            ):
                return sanitize_payload(item)
            return None
        row = db.safe_fetchone(
            """
            SELECT stored_export_id, export_id, render_id, organization_id, workspace_id, dossier_id,
                   workflow_id, pack_type, export_format, framework_version, approval_status,
                   evidence_status, checksum, source_manifest_hash, content_hash, manifest_hash,
                   section_count, file_name_hint, content_type, storage_backend, storage_key,
                   storage_ref, version_group_key, version_number, version_label, status,
                   document_identity, source_context, approval_context, axiom_context,
                   evidence_context, export_context, lineage_summary, created_by, metadata,
                   created_at, updated_at
            FROM platform_stored_exports
            WHERE stored_export_id=%s
            """,
            (stored_export_id,),
        )
        if not row:
            return None
        result = self._row_to_stored_export(row)
        if not self._matches_scope(
            organization_id=result.get("organization_id"),
            workspace_id=result.get("workspace_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        ):
            return None
        return result

    def list_stored_exports(
        self,
        *,
        dossier_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        export_id: Optional[str] = None,
        pack_type: Optional[str] = None,
        export_format: Optional[str] = None,
        version_group_key: Optional[str] = None,
        include_superseded: bool = True,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._stored_exports.values())
            if dossier_id is not None:
                rows = [item for item in rows if str(item.get("dossier_id")) == str(dossier_id)]
            if workspace_id is not None:
                rows = [item for item in rows if str(item.get("workspace_id")) == str(workspace_id)]
            if export_id is not None:
                rows = [item for item in rows if str(item.get("export_id")) == str(export_id)]
            if pack_type is not None:
                rows = [item for item in rows if str(item.get("pack_type")) == str(pack_type)]
            if export_format is not None:
                rows = [item for item in rows if str(item.get("export_format")) == str(export_format)]
            if version_group_key is not None:
                rows = [
                    item for item in rows if str(item.get("version_group_key")) == str(version_group_key)
                ]
            if not include_superseded:
                rows = [item for item in rows if str(item.get("status")) != "superseded"]
            rows = [
                item
                for item in rows
                if self._matches_scope(
                    organization_id=item.get("organization_id"),
                    workspace_id=item.get("workspace_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                )
            ]
            rows.sort(
                key=lambda item: (
                    str(item.get("created_at") or ""),
                    int(item.get("version_number") or 0),
                ),
                reverse=True,
            )
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if dossier_id is not None:
            clauses.append("dossier_id=%s")
            params.append(dossier_id)
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        if export_id is not None:
            clauses.append("export_id=%s")
            params.append(export_id)
        if pack_type is not None:
            clauses.append("pack_type=%s")
            params.append(pack_type)
        if export_format is not None:
            clauses.append("export_format=%s")
            params.append(export_format)
        if version_group_key is not None:
            clauses.append("version_group_key=%s")
            params.append(version_group_key)
        if not include_superseded:
            clauses.append("status<>%s")
            params.append("superseded")
        if organization_ids:
            clauses.append("organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
        rows = db.safe_fetchall(
            f"""
            SELECT stored_export_id, export_id, render_id, organization_id, workspace_id, dossier_id,
                   workflow_id, pack_type, export_format, framework_version, approval_status,
                   evidence_status, checksum, source_manifest_hash, content_hash, manifest_hash,
                   section_count, file_name_hint, content_type, storage_backend, storage_key,
                   storage_ref, version_group_key, version_number, version_label, status,
                   document_identity, source_context, approval_context, axiom_context,
                   evidence_context, export_context, lineage_summary, created_by, metadata,
                   created_at, updated_at
            FROM platform_stored_exports
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, version_number DESC
            """,
            params,
        )
        return [self._row_to_stored_export(row) for row in rows]

    def list_stored_export_versions(
        self,
        stored_export_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        record = self.get_stored_export(
            stored_export_id,
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        )
        if record is None:
            return []
        return [
            ExportVersionRecord(
                stored_export_id=str(item.get("stored_export_id") or ""),
                export_id=str(item.get("export_id") or ""),
                render_id=str(item.get("render_id") or ""),
                pack_type=str(item.get("pack_type") or ""),
                export_format=str(item.get("export_format") or ""),
                version_group_key=str(item.get("version_group_key") or ""),
                version_number=int(item.get("version_number") or 1),
                version_label=str(item.get("version_label") or "v1"),
                status=str(item.get("status") or "stored"),
                approval_status=item.get("approval_status"),
                evidence_status=item.get("evidence_status"),
                checksum=item.get("checksum"),
                storage_backend=str(item.get("storage_backend") or ""),
                storage_key=str(item.get("storage_key") or ""),
                file_name_hint=str(item.get("file_name_hint") or ""),
                created_at=item.get("created_at"),
            ).model_dump(mode="python")
            for item in self.list_stored_exports(
                version_group_key=str(record.get("version_group_key") or ""),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            )
        ]

    def create_integration_binding(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = IntegrationBinding.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["created_at"] = record.get("created_at") or self._now_iso()
            record["updated_at"] = self._now_iso()
            self._integration_bindings[record["binding_id"]] = sanitize_payload(record)
            return self._integration_bindings[record["binding_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_integration_bindings (
                binding_id, integration_type, organization_id, workspace_id, status, config, health, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            ON CONFLICT (binding_id)
            DO UPDATE SET
                integration_type = EXCLUDED.integration_type,
                organization_id = EXCLUDED.organization_id,
                workspace_id = EXCLUDED.workspace_id,
                status = EXCLUDED.status,
                config = EXCLUDED.config,
                health = EXCLUDED.health,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            RETURNING binding_id, integration_type, organization_id, workspace_id, status, config, health, metadata, created_at, updated_at
            """,
            (
                record["binding_id"],
                record["integration_type"],
                record.get("organization_id"),
                record.get("workspace_id"),
                record["status"],
                Json(sanitize_payload(record.get("config") or {})),
                Json(sanitize_payload(record.get("health") or {})),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return {
            "binding_id": str(row[0]),
            "integration_type": row[1],
            "organization_id": str(row[2]) if row[2] is not None else None,
            "workspace_id": str(row[3]) if row[3] is not None else None,
            "status": row[4],
            "config": sanitize_payload(row[5]) if row[5] is not None else {},
            "health": sanitize_payload(row[6]) if row[6] is not None else {},
            "metadata": sanitize_payload(row[7]) if row[7] is not None else {},
            "created_at": row[8],
            "updated_at": row[9],
        }

    def list_integration_bindings(
        self,
        *,
        organization_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = list(self._integration_bindings.values())
            if organization_id is not None:
                rows = [row for row in rows if str(row.get("organization_id")) == str(organization_id)]
            if workspace_id is not None:
                rows = [row for row in rows if str(row.get("workspace_id")) == str(workspace_id)]
            rows = [
                row
                for row in rows
                if self._matches_scope(
                    organization_id=row.get("organization_id"),
                    workspace_id=row.get("workspace_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                )
            ]
            rows.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["1=1"]
        params: List[Any] = []
        if organization_id is not None:
            clauses.append("organization_id=%s")
            params.append(organization_id)
        if workspace_id is not None:
            clauses.append("workspace_id=%s")
            params.append(workspace_id)
        if organization_ids:
            clauses.append("organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
        rows = db.safe_fetchall(
            f"""
            SELECT binding_id, integration_type, organization_id, workspace_id, status, config, health, metadata, created_at, updated_at
            FROM platform_integration_bindings
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC, created_at DESC
            """,
            params,
        )
        return [
            {
                "binding_id": str(row[0]),
                "integration_type": row[1],
                "organization_id": str(row[2]) if row[2] is not None else None,
                "workspace_id": str(row[3]) if row[3] is not None else None,
                "status": row[4],
                "config": sanitize_payload(row[5]) if row[5] is not None else {},
                "health": sanitize_payload(row[6]) if row[6] is not None else {},
                "metadata": sanitize_payload(row[7]) if row[7] is not None else {},
                "created_at": row[8],
                "updated_at": row[9],
            }
            for row in rows
        ]

    def get_integration_binding(
        self,
        binding_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            item = self._integration_bindings.get(str(binding_id))
            if item and self._matches_scope(
                organization_id=item.get("organization_id"),
                workspace_id=item.get("workspace_id"),
                organization_ids=organization_ids,
                workspace_ids=workspace_ids,
            ):
                return item
            return None
        row = db.safe_fetchone(
            """
            SELECT binding_id, integration_type, organization_id, workspace_id, status, config, health, metadata, created_at, updated_at
            FROM platform_integration_bindings
            WHERE binding_id=%s
            """,
            (binding_id,),
        )
        if not row:
            return None
        result = {
            "binding_id": str(row[0]),
            "integration_type": row[1],
            "organization_id": str(row[2]) if row[2] is not None else None,
            "workspace_id": str(row[3]) if row[3] is not None else None,
            "status": row[4],
            "config": sanitize_payload(row[5]) if row[5] is not None else {},
            "health": sanitize_payload(row[6]) if row[6] is not None else {},
            "metadata": sanitize_payload(row[7]) if row[7] is not None else {},
            "created_at": row[8],
            "updated_at": row[9],
        }
        if not self._matches_scope(
            organization_id=result.get("organization_id"),
            workspace_id=result.get("workspace_id"),
            organization_ids=organization_ids,
            workspace_ids=workspace_ids,
        ):
            return None
        return result

    def update_integration_binding(self, binding_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**(self.get_integration_binding(binding_id) or {}), **sanitize_payload(payload)}
        merged["binding_id"] = binding_id
        return self.create_integration_binding(merged)

    def create_integration_execution(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = IntegrationExecutionRecord.model_validate(payload).model_dump(mode="python")
        if self.use_memory:
            record["started_at"] = record.get("started_at") or self._now_iso()
            record["completed_at"] = record.get("completed_at") or self._now_iso()
            self._integration_executions[record["execution_id"]] = sanitize_payload(record)
            return self._integration_executions[record["execution_id"]]
        row = db.exec1(
            """
            INSERT INTO platform_integration_executions (
                execution_id, binding_id, integration_type, action_type, status, workspace_id,
                organization_id, dossier_id, export_id, render_id, started_at, completed_at,
                payload_summary, output_summary, error_summary, metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                COALESCE(%s::timestamptz, now()), COALESCE(%s::timestamptz, now()),
                %s::jsonb, %s::jsonb, %s, %s::jsonb
            )
            ON CONFLICT (execution_id)
            DO UPDATE SET
                binding_id = EXCLUDED.binding_id,
                integration_type = EXCLUDED.integration_type,
                action_type = EXCLUDED.action_type,
                status = EXCLUDED.status,
                workspace_id = EXCLUDED.workspace_id,
                organization_id = EXCLUDED.organization_id,
                dossier_id = EXCLUDED.dossier_id,
                export_id = EXCLUDED.export_id,
                render_id = EXCLUDED.render_id,
                started_at = EXCLUDED.started_at,
                completed_at = EXCLUDED.completed_at,
                payload_summary = EXCLUDED.payload_summary,
                output_summary = EXCLUDED.output_summary,
                error_summary = EXCLUDED.error_summary,
                metadata = EXCLUDED.metadata
            RETURNING execution_id, binding_id, integration_type, action_type, status, workspace_id,
                organization_id, dossier_id, export_id, render_id, started_at, completed_at,
                payload_summary, output_summary, error_summary, metadata
            """,
            (
                record["execution_id"],
                record["binding_id"],
                record["integration_type"],
                record["action_type"],
                record["status"],
                record.get("workspace_id"),
                record.get("organization_id"),
                record.get("dossier_id"),
                record.get("export_id"),
                record.get("render_id"),
                record.get("started_at"),
                record.get("completed_at"),
                Json(sanitize_payload(record.get("payload_summary") or {})),
                Json(sanitize_payload(record.get("output_summary") or {})),
                record.get("error_summary"),
                Json(sanitize_payload(record.get("metadata") or {})),
            ),
        )
        return {
            "execution_id": str(row[0]),
            "binding_id": str(row[1]),
            "integration_type": row[2],
            "action_type": row[3],
            "status": row[4],
            "workspace_id": str(row[5]) if row[5] is not None else None,
            "organization_id": str(row[6]) if row[6] is not None else None,
            "dossier_id": str(row[7]) if row[7] is not None else None,
            "export_id": str(row[8]) if row[8] is not None else None,
            "render_id": str(row[9]) if row[9] is not None else None,
            "started_at": row[10],
            "completed_at": row[11],
            "payload_summary": sanitize_payload(row[12]) if row[12] is not None else {},
            "output_summary": sanitize_payload(row[13]) if row[13] is not None else {},
            "error_summary": row[14],
            "metadata": sanitize_payload(row[15]) if row[15] is not None else {},
        }

    def list_integration_executions(
        self,
        binding_id: str,
        *,
        organization_ids: Optional[Sequence[str]] = None,
        workspace_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.use_memory:
            rows = [
                item
                for item in self._integration_executions.values()
                if str(item.get("binding_id")) == str(binding_id)
            ]
            rows = [
                item
                for item in rows
                if self._matches_scope(
                    organization_id=item.get("organization_id"),
                    workspace_id=item.get("workspace_id"),
                    organization_ids=organization_ids,
                    workspace_ids=workspace_ids,
                )
            ]
            rows.sort(key=lambda item: item.get("completed_at", ""), reverse=True)
            return [sanitize_payload(item) for item in rows]
        clauses = ["binding_id=%s"]
        params: List[Any] = [binding_id]
        if organization_ids:
            clauses.append("organization_id = ANY(%s)")
            params.append(list(organization_ids))
        if workspace_ids:
            clauses.append("workspace_id = ANY(%s)")
            params.append(list(workspace_ids))
        rows = db.safe_fetchall(
            f"""
            SELECT execution_id, binding_id, integration_type, action_type, status, workspace_id,
                   organization_id, dossier_id, export_id, render_id, started_at, completed_at,
                   payload_summary, output_summary, error_summary, metadata
            FROM platform_integration_executions
            WHERE {' AND '.join(clauses)}
            ORDER BY completed_at DESC
            """,
            params,
        )
        return [
            {
                "execution_id": str(row[0]),
                "binding_id": str(row[1]),
                "integration_type": row[2],
                "action_type": row[3],
                "status": row[4],
                "workspace_id": str(row[5]) if row[5] is not None else None,
                "organization_id": str(row[6]) if row[6] is not None else None,
                "dossier_id": str(row[7]) if row[7] is not None else None,
                "export_id": str(row[8]) if row[8] is not None else None,
                "render_id": str(row[9]) if row[9] is not None else None,
                "started_at": row[10],
                "completed_at": row[11],
                "payload_summary": sanitize_payload(row[12]) if row[12] is not None else {},
                "output_summary": sanitize_payload(row[13]) if row[13] is not None else {},
                "error_summary": row[14],
                "metadata": sanitize_payload(row[15]) if row[15] is not None else {},
            }
            for row in rows
        ]


platform_store = PlatformStore()
