"""Regression tests for Phase 11 — feature provenance tracking table."""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feature_payload(
    feature_version="phase9_canonical_features_v1",
    coverage_status="available",
    price_source="yfinance",
    event_source="computed",
    breadth_source="market_breadth_daily",
    feature_hash="abc123",
    snapshot_id="snap_001",
    features=None,
):
    if features is None:
        features = {"mom_21": 0.12, "rsi14": 65.0, "vol_21d": 0.18, "trend_r2_63d": 0.87}
    return {
        "feature_version": feature_version,
        "effective_lookback": 63,
        "snapshot_id": snapshot_id,
        "features": features,
        "meta": {
            "coverage_status": coverage_status,
            "price_source": price_source,
            "event_source": event_source,
            "breadth_source": breadth_source,
            "feature_hash": feature_hash,
        },
    }


# ---------------------------------------------------------------------------
# 1. FeatureProvenanceRecord model
# ---------------------------------------------------------------------------

def test_feature_provenance_record_importable():
    from api.alpha.provenance import FeatureProvenanceRecord
    assert FeatureProvenanceRecord


def test_feature_provenance_record_fields():
    from api.alpha.provenance import FeatureProvenanceRecord
    rec = FeatureProvenanceRecord(
        symbol="AAPL",
        as_of=dt.date(2024, 6, 1),
        feature_version="phase9_canonical_features_v1",
        coverage_status="available",
        price_source="yfinance",
        null_feature_count=1,
        total_feature_count=50,
    )
    assert rec.symbol == "AAPL"
    assert rec.coverage_status == "available"
    assert rec.price_source == "yfinance"


def test_completeness_pct_computed():
    from api.alpha.provenance import FeatureProvenanceRecord
    rec = FeatureProvenanceRecord(
        symbol="AAPL",
        as_of=dt.date(2024, 6, 1),
        feature_version="v1",
        null_feature_count=5,
        total_feature_count=50,
    )
    assert abs(rec.completeness_pct - 0.9) < 1e-6


def test_completeness_pct_zero_total():
    from api.alpha.provenance import FeatureProvenanceRecord
    rec = FeatureProvenanceRecord(
        symbol="AAPL",
        as_of=dt.date(2024, 6, 1),
        feature_version="v1",
        null_feature_count=0,
        total_feature_count=0,
    )
    assert rec.completeness_pct is None


# ---------------------------------------------------------------------------
# 2. _count_nulls helper
# ---------------------------------------------------------------------------

def test_count_nulls_counts_none_values():
    from api.alpha.provenance import _count_nulls
    features = {"mom_21": 0.12, "rsi14": None, "vol_21d": None, "trend_r2_63d": 0.87}
    null_count, total_count, missing = _count_nulls(features)
    assert null_count == 2
    assert total_count == 4
    assert set(missing) == {"rsi14", "vol_21d"}


def test_count_nulls_all_present():
    from api.alpha.provenance import _count_nulls
    features = {"a": 1.0, "b": 2.0, "c": 3.0}
    null_count, total_count, missing = _count_nulls(features)
    assert null_count == 0
    assert total_count == 3
    assert missing == []


# ---------------------------------------------------------------------------
# 3. record_feature_provenance
# ---------------------------------------------------------------------------

def test_record_returns_false_when_db_disabled():
    from api.alpha.provenance import record_feature_provenance
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_enabled.return_value = False
        result = record_feature_provenance("AAPL", dt.date(2024, 6, 1), _make_feature_payload())
    assert result is False


def test_record_writes_to_db():
    from api.alpha.provenance import record_feature_provenance
    payload = _make_feature_payload()
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        result = record_feature_provenance("AAPL", dt.date(2024, 6, 1), payload)
    assert result is True
    mock_db.safe_execute.assert_called_once()


def test_record_extracts_provider_sources():
    from api.alpha.provenance import record_feature_provenance
    payload = _make_feature_payload(price_source="provided_market_bars", event_source="sec_filings")
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        record_feature_provenance("TSLA", dt.date(2024, 6, 1), payload)
    args = mock_db.safe_execute.call_args[0][1]
    # Index 5 = price_source, 6 = event_source in the INSERT params
    assert args[5] == "provided_market_bars"
    assert args[6] == "sec_filings"


def test_record_counts_null_features():
    from api.alpha.provenance import record_feature_provenance
    payload = _make_feature_payload(features={"a": 1.0, "b": None, "c": None, "d": 0.5})
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        record_feature_provenance("AAPL", dt.date(2024, 6, 1), payload)
    args = mock_db.safe_execute.call_args[0][1]
    # null_feature_count at index 9, total_feature_count at index 10
    assert args[9] == 2   # null_feature_count
    assert args[10] == 4  # total_feature_count


def test_record_upserts_on_conflict():
    """INSERT must use ON CONFLICT DO UPDATE."""
    from api.alpha.provenance import record_feature_provenance
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        record_feature_provenance("AAPL", dt.date(2024, 6, 1), _make_feature_payload())
    sql = mock_db.safe_execute.call_args[0][0]
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql


# ---------------------------------------------------------------------------
# 4. get_feature_provenance
# ---------------------------------------------------------------------------

def test_get_provenance_returns_none_when_db_disabled():
    from api.alpha.provenance import get_feature_provenance
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_read_enabled.return_value = False
        result = get_feature_provenance("AAPL", dt.date(2024, 6, 1))
    assert result is None


def test_get_provenance_returns_none_when_no_row():
    from api.alpha.provenance import get_feature_provenance
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = None
        result = get_feature_provenance("AAPL", dt.date(2024, 6, 1))
    assert result is None


def test_get_provenance_returns_record():
    from api.alpha.provenance import get_feature_provenance, FeatureProvenanceRecord
    row = (
        "AAPL", dt.date(2024, 6, 1), "phase9_canonical_features_v1", 63,
        "available", "yfinance", "computed", "market_breadth_daily", None,
        2, 50, ["rsi14", "vol_21d"], [], "abc123", "snap_001",
    )
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchone.return_value = row
        result = get_feature_provenance("AAPL", dt.date(2024, 6, 1))
    assert isinstance(result, FeatureProvenanceRecord)
    assert result.symbol == "AAPL"
    assert result.coverage_status == "available"
    assert result.null_feature_count == 2
    assert result.total_feature_count == 50
    assert "rsi14" in result.missing_features
    assert result.feature_hash == "abc123"
    assert abs(result.completeness_pct - 0.96) < 1e-6


# ---------------------------------------------------------------------------
# 5. get_feature_coverage_summary
# ---------------------------------------------------------------------------

def test_coverage_summary_empty_when_db_disabled():
    from api.alpha.provenance import get_feature_coverage_summary
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_read_enabled.return_value = False
        result = get_feature_coverage_summary()
    assert result == {}


def test_coverage_summary_computes_available_pct():
    from api.alpha.provenance import get_feature_coverage_summary
    rows = [("available", 80), ("partial", 15), ("insufficient_history", 5)]
    with patch("api.alpha.provenance.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        summary = get_feature_coverage_summary(days=30)
    assert summary["total_records"] == 100
    assert summary["by_status"]["available"] == 80
    assert abs(summary["available_pct"] - 0.8) < 1e-6
    assert summary["window_days"] == 30


# ---------------------------------------------------------------------------
# 6. Routes
# ---------------------------------------------------------------------------

def test_feature_provenance_route_registered():
    from api.signals import routes
    paths = [r.path for r in routes.router.routes]
    assert "/signals/features/provenance" in paths


def test_feature_coverage_route_registered():
    from api.signals import routes
    paths = [r.path for r in routes.router.routes]
    assert "/signals/features/coverage" in paths


def test_feature_provenance_endpoint_404_when_not_found():
    import asyncio
    from fastapi import HTTPException
    from api.signals.routes import feature_provenance
    with patch("api.signals.routes.db") as mock_db, \
         patch("api.signals.routes.get_feature_provenance", return_value=None):
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        try:
            asyncio.run(feature_provenance("ZZZZ", dt.date(2024, 6, 1)))
            assert False, "should have raised"
        except HTTPException as exc:
            assert exc.status_code == 404


def test_feature_provenance_endpoint_returns_record():
    import asyncio
    from api.alpha.provenance import FeatureProvenanceRecord
    from api.signals.routes import feature_provenance
    rec = FeatureProvenanceRecord(
        symbol="AAPL", as_of=dt.date(2024, 6, 1),
        feature_version="phase9_v1", coverage_status="available",
        price_source="yfinance", null_feature_count=0, total_feature_count=50,
    )
    with patch("api.signals.routes.db") as mock_db, \
         patch("api.signals.routes.get_feature_provenance", return_value=rec):
        mock_db.db_enabled.return_value = True
        mock_db.db_read_enabled.return_value = True
        result = asyncio.run(feature_provenance("AAPL", dt.date(2024, 6, 1)))
    assert result["symbol"] == "AAPL"
    assert result["coverage_status"] == "available"
    assert result["price_source"] == "yfinance"
