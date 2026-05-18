from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from psycopg.types.json import Json

from api import db
from api.assistant.reports import sanitize_payload
from api.assistant.storage import AssistantStorage
from api.axiom.history import (
    AXIOM_CALIBRATION_ARTIFACT_KIND,
    AXIOM_SCORE_HISTORY_ARTIFACT_KIND,
)


def _db_ready(*, write: bool = False) -> bool:
    if not db.db_enabled():
        return False
    if write and not db.db_write_enabled():
        return False
    if not write and not db.db_read_enabled():
        return False
    return True


def persist_axiom_score_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not _db_ready(write=True):
        return None
    payload = sanitize_payload(record)
    evidence = payload.get("evidence_backed_deployability") or {}
    coverage = payload.get("coverage_summary") or {}
    build_meta = payload.get("build_metadata") or {}
    row = db.exec1(
        """
        INSERT INTO axiom_scores_daily (
            symbol,
            as_of_date,
            framework_version,
            snapshot_id,
            snapshot_version,
            feature_version,
            signal_version,
            regime_label,
            trade_family,
            deployability_tier,
            evidence_backed_deployability_tier,
            size_band,
            gross_opportunity,
            friction_burden,
            validated_edge,
            deployable_alpha_utility,
            overall_coverage,
            overall_confidence,
            payload,
            outcome_payload,
            build_meta
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb
        )
        ON CONFLICT (symbol, as_of_date, framework_version)
        DO UPDATE SET
            snapshot_id = EXCLUDED.snapshot_id,
            snapshot_version = EXCLUDED.snapshot_version,
            feature_version = EXCLUDED.feature_version,
            signal_version = EXCLUDED.signal_version,
            regime_label = EXCLUDED.regime_label,
            trade_family = EXCLUDED.trade_family,
            deployability_tier = EXCLUDED.deployability_tier,
            evidence_backed_deployability_tier = EXCLUDED.evidence_backed_deployability_tier,
            size_band = EXCLUDED.size_band,
            gross_opportunity = EXCLUDED.gross_opportunity,
            friction_burden = EXCLUDED.friction_burden,
            validated_edge = EXCLUDED.validated_edge,
            deployable_alpha_utility = EXCLUDED.deployable_alpha_utility,
            overall_coverage = EXCLUDED.overall_coverage,
            overall_confidence = EXCLUDED.overall_confidence,
            payload = EXCLUDED.payload,
            outcome_payload = EXCLUDED.outcome_payload,
            build_meta = EXCLUDED.build_meta,
            updated_at = now()
        RETURNING symbol, as_of_date, framework_version
        """,
        (
            payload.get("symbol"),
            payload.get("as_of_date"),
            payload.get("framework_version"),
            payload.get("snapshot_id"),
            payload.get("snapshot_version"),
            payload.get("feature_version"),
            payload.get("signal_version"),
            payload.get("regime_label"),
            payload.get("trade_family"),
            payload.get("deployability_tier"),
            evidence.get("deployability_tier"),
            evidence.get("size_band") or payload.get("size_band_recommendation"),
            payload.get("gross_opportunity"),
            payload.get("friction_burden"),
            payload.get("validated_edge"),
            payload.get("deployable_alpha_utility"),
            coverage.get("overall_coverage"),
            coverage.get("overall_confidence"),
            Json(payload),
            Json(payload.get("forward_outcomes") or {}),
            Json(build_meta),
        ),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of_date": str(row[1]),
        "framework_version": row[2],
    }


def persist_axiom_calibration_snapshot(
    snapshot: Dict[str, Any],
    *,
    snapshot_key: Optional[str] = None,
) -> Optional[str]:
    if not _db_ready(write=True):
        return None
    payload = sanitize_payload(snapshot)
    key = snapshot_key or str(
        payload.get("snapshot_key")
        or f"{payload.get('framework_version')}:{payload.get('horizon_label')}:{payload.get('as_of_date') or 'latest'}"
    )
    row = db.exec1(
        """
        INSERT INTO axiom_calibration_snapshots (
            snapshot_key,
            as_of_date,
            horizon_label,
            framework_version,
            payload
        )
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (snapshot_key)
        DO UPDATE SET
            as_of_date = EXCLUDED.as_of_date,
            horizon_label = EXCLUDED.horizon_label,
            framework_version = EXCLUDED.framework_version,
            payload = EXCLUDED.payload,
            updated_at = now()
        RETURNING snapshot_key
        """,
        (
            key,
            payload.get("as_of_date"),
            payload.get("horizon_label"),
            payload.get("framework_version"),
            Json(payload),
        ),
    )
    return str(row[0]) if row else None


def persist_axiom_replay_run(run_payload: Dict[str, Any]) -> Optional[str]:
    if not _db_ready(write=True):
        return None
    payload = sanitize_payload(run_payload)
    run_id = str(payload.get("run_id") or uuid.uuid4())
    row = db.exec1(
        """
        INSERT INTO axiom_replay_runs (
            run_id,
            symbols,
            date_start,
            date_end,
            lookback,
            framework_version,
            record_count,
            payload
        )
        VALUES (%s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (run_id)
        DO UPDATE SET
            symbols = EXCLUDED.symbols,
            date_start = EXCLUDED.date_start,
            date_end = EXCLUDED.date_end,
            lookback = EXCLUDED.lookback,
            framework_version = EXCLUDED.framework_version,
            record_count = EXCLUDED.record_count,
            payload = EXCLUDED.payload,
            updated_at = now()
        RETURNING run_id
        """,
        (
            run_id,
            Json(payload.get("symbols") or []),
            payload.get("date_start"),
            payload.get("date_end"),
            payload.get("lookback"),
            payload.get("framework_version"),
            payload.get("record_count"),
            Json(payload),
        ),
    )
    return str(row[0]) if row else None


def load_axiom_history_records(
    *,
    symbols: Optional[Sequence[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    store: Optional[AssistantStorage] = None,
    session_id: Optional[str] = None,
    limit: int = 2500,
) -> List[Dict[str, Any]]:
    if store is not None and getattr(store, "use_memory", False):
        artifacts = store.list_artifacts(
            kind=AXIOM_SCORE_HISTORY_ARTIFACT_KIND,
            session_id=session_id,
            limit=limit,
        )
        records = [sanitize_payload(artifact.get("payload") or {}) for artifact in artifacts]
        if symbols:
            symbol_set = {str(symbol).upper() for symbol in symbols}
            records = [record for record in records if str(record.get("symbol") or "").upper() in symbol_set]
        if start_date:
            records = [record for record in records if str(record.get("as_of_date") or "") >= str(start_date)]
        if end_date:
            records = [record for record in records if str(record.get("as_of_date") or "") <= str(end_date)]
        records.sort(key=lambda item: (str(item.get("as_of_date") or ""), str(item.get("symbol") or "")))
        return records

    if _db_ready(write=False):
        clauses = ["1=1"]
        params: List[Any] = []
        if symbols:
            clauses.append("symbol = ANY(%s)")
            params.append(list(symbols))
        if start_date:
            clauses.append("as_of_date >= %s")
            params.append(start_date)
        if end_date:
            clauses.append("as_of_date <= %s")
            params.append(end_date)
        params.append(limit)
        rows = db.safe_fetchall(
            f"""
            SELECT payload
            FROM axiom_scores_daily
            WHERE {' AND '.join(clauses)}
            ORDER BY as_of_date ASC, symbol ASC
            LIMIT %s
            """,
            params,
        )
        return [sanitize_payload(row[0]) for row in rows if row and row[0] is not None]

    if store is None:
        return []
    artifacts = store.list_artifacts(
        kind=AXIOM_SCORE_HISTORY_ARTIFACT_KIND,
        session_id=session_id,
        limit=limit,
    )
    records = [sanitize_payload(artifact.get("payload") or {}) for artifact in artifacts]
    if symbols:
        symbol_set = {str(symbol).upper() for symbol in symbols}
        records = [record for record in records if str(record.get("symbol") or "").upper() in symbol_set]
    if start_date:
        records = [record for record in records if str(record.get("as_of_date") or "") >= str(start_date)]
    if end_date:
        records = [record for record in records if str(record.get("as_of_date") or "") <= str(end_date)]
    records.sort(key=lambda item: (str(item.get("as_of_date") or ""), str(item.get("symbol") or "")))
    return records


def persist_axiom_artifacts_to_store(
    *,
    store: AssistantStorage,
    session_id: str,
    history_record: Dict[str, Any],
    calibration_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    history_id = store.save_artifact(session_id, AXIOM_SCORE_HISTORY_ARTIFACT_KIND, history_record)
    calibration_id = None
    if calibration_artifact is not None:
        calibration_id = store.save_artifact(
            session_id,
            AXIOM_CALIBRATION_ARTIFACT_KIND,
            calibration_artifact,
        )
    return {
        "history_artifact_id": history_id,
        "calibration_artifact_id": calibration_id,
    }
