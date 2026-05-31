"""Feature provenance tracking — persists which data sources and coverage status
produced each feature set, enabling per-signal audit and debugging.

Usage::

    # After calling build_canonical_features():
    from api.alpha.provenance import record_feature_provenance
    record_feature_provenance("AAPL", dt.date(2024, 1, 1), feature_payload)

    # Query:
    from api.alpha.provenance import get_feature_provenance
    rec = get_feature_provenance("AAPL", dt.date(2024, 1, 1))
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from api import db


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FeatureProvenanceRecord(BaseModel):
    symbol: str
    as_of: dt.date
    feature_version: str
    lookback: int = 63
    coverage_status: str = "unknown"   # available | partial | insufficient_history | unavailable | unknown
    price_source: Optional[str] = None
    event_source: Optional[str] = None
    breadth_source: Optional[str] = None
    sentiment_source: Optional[str] = None
    null_feature_count: int = 0
    total_feature_count: int = 0
    missing_features: List[str] = []
    data_warnings: List[str] = []
    feature_hash: Optional[str] = None
    snapshot_id: Optional[str] = None
    meta: Dict[str, Any] = {}

    @property
    def completeness_pct(self) -> Optional[float]:
        if self.total_feature_count == 0:
            return None
        present = self.total_feature_count - self.null_feature_count
        return round(present / self.total_feature_count, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_nulls(features: Dict[str, Any]) -> tuple:
    """Return (null_count, total_count, missing_names)."""
    total = len(features)
    missing = [k for k, v in features.items() if v is None]
    return len(missing), total, missing


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_feature_provenance(
    symbol: str,
    as_of: dt.date,
    feature_payload: Dict[str, Any],
) -> bool:
    """Persist provenance extracted from a build_canonical_features() result.

    Returns True if the record was written, False if DB unavailable.
    """
    if not db.db_enabled():
        return False

    meta = feature_payload.get("meta") or {}
    features = feature_payload.get("features") or {}
    null_count, total_count, missing = _count_nulls(features)

    try:
        db.safe_execute(
            """
            INSERT INTO feature_provenance_daily (
                symbol, as_of, feature_version, lookback,
                coverage_status,
                price_source, event_source, breadth_source, sentiment_source,
                null_feature_count, total_feature_count, missing_features,
                data_warnings, feature_hash, snapshot_id, meta
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, as_of, feature_version)
            DO UPDATE SET
                lookback            = EXCLUDED.lookback,
                coverage_status     = EXCLUDED.coverage_status,
                price_source        = EXCLUDED.price_source,
                event_source        = EXCLUDED.event_source,
                breadth_source      = EXCLUDED.breadth_source,
                sentiment_source    = EXCLUDED.sentiment_source,
                null_feature_count  = EXCLUDED.null_feature_count,
                total_feature_count = EXCLUDED.total_feature_count,
                missing_features    = EXCLUDED.missing_features,
                data_warnings       = EXCLUDED.data_warnings,
                feature_hash        = EXCLUDED.feature_hash,
                snapshot_id         = EXCLUDED.snapshot_id,
                meta                = EXCLUDED.meta,
                recorded_at         = now()
            """,
            (
                str(symbol).upper(),
                as_of,
                str(feature_payload.get("feature_version") or "unknown"),
                int(feature_payload.get("effective_lookback") or 63),
                str(meta.get("coverage_status") or "unknown"),
                meta.get("price_source"),
                meta.get("event_source"),
                meta.get("breadth_source"),
                meta.get("sentiment_source"),
                null_count,
                total_count,
                missing,
                list(meta.get("data_warnings") or []),
                meta.get("feature_hash"),
                feature_payload.get("snapshot_id"),
                {k: v for k, v in meta.items()
                 if k not in ("coverage_status", "price_source", "event_source",
                              "breadth_source", "sentiment_source", "feature_hash",
                              "data_warnings")},
            ),
        )
        return True
    except Exception:  # pragma: no cover — DB unavailable in tests
        return False


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_feature_provenance(
    symbol: str,
    as_of: dt.date,
) -> Optional[FeatureProvenanceRecord]:
    """Return the most recent provenance record for (symbol, as_of)."""
    if not db.db_read_enabled():
        return None
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of, feature_version, lookback,
               coverage_status,
               price_source, event_source, breadth_source, sentiment_source,
               null_feature_count, total_feature_count, missing_features,
               data_warnings, feature_hash, snapshot_id
        FROM feature_provenance_daily
        WHERE symbol = %s AND as_of = %s
        ORDER BY recorded_at DESC
        LIMIT 1
        """,
        (str(symbol).upper(), as_of),
    )
    if not row:
        return None
    return FeatureProvenanceRecord(
        symbol=row[0],
        as_of=row[1],
        feature_version=row[2],
        lookback=row[3] or 63,
        coverage_status=row[4] or "unknown",
        price_source=row[5],
        event_source=row[6],
        breadth_source=row[7],
        sentiment_source=row[8],
        null_feature_count=row[9] or 0,
        total_feature_count=row[10] or 0,
        missing_features=list(row[11] or []),
        data_warnings=list(row[12] or []),
        feature_hash=row[13],
        snapshot_id=row[14],
    )


def get_feature_coverage_summary(
    days: int = 30,
) -> Dict[str, Any]:
    """Return coverage-status counts across all symbols for the last N days."""
    if not db.db_read_enabled():
        return {}
    cutoff = dt.date.today() - dt.timedelta(days=days)
    rows = db.safe_fetchall(
        """
        SELECT coverage_status, COUNT(*) AS n
        FROM feature_provenance_daily
        WHERE as_of >= %s
        GROUP BY coverage_status
        ORDER BY n DESC
        """,
        (cutoff,),
    )
    total = sum(int(r[1]) for r in (rows or []))
    counts = {str(r[0]): int(r[1]) for r in (rows or [])}
    return {
        "window_days": days,
        "total_records": total,
        "by_status": counts,
        "available_pct": round(counts.get("available", 0) / total, 4) if total > 0 else None,
    }
