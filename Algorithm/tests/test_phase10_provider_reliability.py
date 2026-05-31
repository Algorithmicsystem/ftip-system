"""Regression tests for Phase 10 — provider reliability time-series."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. ProviderReliabilityRecord model
# ---------------------------------------------------------------------------

def test_provider_reliability_record_importable():
    from api.providers.reliability import ProviderReliabilityRecord
    assert ProviderReliabilityRecord


def test_provider_reliability_record_fields():
    from api.providers.reliability import ProviderReliabilityRecord
    rec = ProviderReliabilityRecord(
        as_of_date=dt.date(2024, 6, 1),
        provider="finnhub",
        is_enabled=True,
        status="ok",
        message="",
    )
    assert rec.provider == "finnhub"
    assert rec.status == "ok"
    assert rec.is_enabled is True
    assert rec.as_of_date == dt.date(2024, 6, 1)


def test_provider_reliability_record_defaults():
    from api.providers.reliability import ProviderReliabilityRecord
    rec = ProviderReliabilityRecord(
        as_of_date=dt.date(2024, 6, 1),
        provider="fred",
        is_enabled=False,
        status="down",
    )
    assert rec.message == ""
    assert rec.meta == {}


# ---------------------------------------------------------------------------
# 2. snapshot_provider_reliability
# ---------------------------------------------------------------------------

def _make_health_response(providers):
    """Build a mock ProvidersHealthResponse-like object."""
    health = MagicMock()
    health.status = "ok"
    mock_providers = []
    for name, status, enabled, msg in providers:
        p = MagicMock()
        p.name = name
        p.status = status
        p.enabled = enabled
        p.message = msg
        mock_providers.append(p)
    health.providers = mock_providers
    return health


def test_snapshot_returns_zero_when_db_disabled():
    from api.providers.reliability import snapshot_provider_reliability
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_enabled.return_value = False
        health = _make_health_response([("finnhub", "ok", True, "")])
        result = snapshot_provider_reliability(health)
    assert result == 0


def test_snapshot_writes_one_row_per_provider():
    from api.providers.reliability import snapshot_provider_reliability
    providers = [
        ("finnhub", "ok", True, ""),
        ("fred", "degraded", True, "slow"),
        ("sec_edgar", "down", True, "timeout"),
    ]
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        health = _make_health_response(providers)
        count = snapshot_provider_reliability(health, as_of_date=dt.date(2024, 6, 1))
    assert count == 3
    assert mock_db.safe_execute.call_count == 3


def test_snapshot_uses_today_when_date_not_provided():
    from api.providers.reliability import snapshot_provider_reliability
    with patch("api.providers.reliability.db") as mock_db, \
         patch("api.providers.reliability.dt") as mock_dt:
        mock_db.db_enabled.return_value = True
        mock_db.safe_execute.return_value = None
        mock_dt.date.today.return_value = dt.date(2024, 6, 15)
        mock_dt.timedelta = dt.timedelta
        mock_dt.date.side_effect = lambda *a, **k: dt.date(*a, **k)

        health = _make_health_response([("finnhub", "ok", True, "")])
        snapshot_provider_reliability(health)

    call_args = mock_db.safe_execute.call_args[0][1]
    assert call_args[0] == dt.date(2024, 6, 15)


# ---------------------------------------------------------------------------
# 3. get_provider_reliability_window
# ---------------------------------------------------------------------------

def test_get_window_returns_empty_when_db_disabled():
    from api.providers.reliability import get_provider_reliability_window
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = False
        result = get_provider_reliability_window(days=30)
    assert result == []


def test_get_window_returns_records():
    from api.providers.reliability import get_provider_reliability_window, ProviderReliabilityRecord
    rows = [
        (dt.date(2024, 6, 1), "finnhub", True, "ok", ""),
        (dt.date(2024, 6, 1), "fred", True, "degraded", "slow"),
    ]
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        result = get_provider_reliability_window(days=30)
    assert len(result) == 2
    for rec in result:
        assert isinstance(rec, ProviderReliabilityRecord)
    assert result[0].provider == "finnhub"
    assert result[1].status == "degraded"


def test_get_window_filters_by_provider():
    from api.providers.reliability import get_provider_reliability_window
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = [
            (dt.date(2024, 6, 1), "finnhub", True, "ok", ""),
        ]
        result = get_provider_reliability_window(days=7, provider="finnhub")
    query_args = mock_db.safe_fetchall.call_args[0]
    assert "provider = %s" in query_args[0]
    assert "finnhub" in query_args[1]


# ---------------------------------------------------------------------------
# 4. get_provider_reliability_summary
# ---------------------------------------------------------------------------

def test_summary_computes_uptime_pct():
    from api.providers.reliability import get_provider_reliability_summary
    rows = [
        (dt.date(2024, 6, 1), "finnhub", True, "ok", ""),
        (dt.date(2024, 6, 2), "finnhub", True, "ok", ""),
        (dt.date(2024, 6, 3), "finnhub", True, "down", "timeout"),
        (dt.date(2024, 6, 4), "finnhub", True, "ok", ""),
    ]
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        summary = get_provider_reliability_summary(days=7)

    finnhub = summary["finnhub"]
    assert finnhub["total_days"] == 4
    assert finnhub["ok_days"] == 3
    assert finnhub["down_days"] == 1
    # 3*1.0 + 0*0.5 + 1*0.0 = 3.0 / 4 = 0.75
    assert abs(finnhub["uptime_pct"] - 0.75) < 1e-6
    assert finnhub["status_label"] == "intermittent"


def test_summary_degraded_counts_half():
    from api.providers.reliability import get_provider_reliability_summary
    rows = [
        (dt.date(2024, 6, 1), "fred", True, "degraded", ""),
        (dt.date(2024, 6, 2), "fred", True, "degraded", ""),
    ]
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        summary = get_provider_reliability_summary(days=7)

    fred = summary["fred"]
    # 2 * 0.5 / 2 = 0.5
    assert abs(fred["uptime_pct"] - 0.5) < 1e-6
    assert fred["status_label"] == "unreliable"


def test_summary_reliable_label_at_full_uptime():
    from api.providers.reliability import get_provider_reliability_summary
    rows = [(dt.date(2024, 6, i), "sec_edgar", True, "ok", "") for i in range(1, 8)]
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = rows
        summary = get_provider_reliability_summary(days=7)

    assert summary["sec_edgar"]["uptime_pct"] == 1.0
    assert summary["sec_edgar"]["status_label"] == "reliable"


def test_summary_returns_empty_when_no_data():
    from api.providers.reliability import get_provider_reliability_summary
    with patch("api.providers.reliability.db") as mock_db:
        mock_db.db_read_enabled.return_value = True
        mock_db.safe_fetchall.return_value = []
        summary = get_provider_reliability_summary(days=7)
    assert summary == {}


# ---------------------------------------------------------------------------
# 5. GET /providers/reliability endpoint
# ---------------------------------------------------------------------------

def test_providers_reliability_route_exists():
    from api.providers import routes
    paths = [r.path for r in routes.router.routes]
    assert "/providers/reliability" in paths


def test_providers_reliability_endpoint_structure():
    from api.providers.routes import providers_reliability
    rows = [
        (dt.date(2024, 6, 1), "finnhub", True, "ok", ""),
        (dt.date(2024, 6, 2), "finnhub", True, "degraded", "slow response"),
    ]
    with patch("api.providers.routes.get_provider_reliability_summary") as mock_summary, \
         patch("api.providers.routes.get_provider_reliability_window") as mock_window:
        mock_summary.return_value = {
            "finnhub": {"total_days": 2, "ok_days": 1, "degraded_days": 1,
                        "uptime_pct": 0.75, "status_label": "intermittent", "window_days": 7}
        }
        from api.providers.reliability import ProviderReliabilityRecord
        mock_window.return_value = [
            ProviderReliabilityRecord(as_of_date=dt.date(2024, 6, 2), provider="finnhub",
                                      is_enabled=True, status="degraded", message="slow response"),
        ]
        result = providers_reliability(days=7, provider=None)

    assert result["window_days"] == 7
    assert "summary" in result
    assert "records" in result
    assert isinstance(result["records"], list)
    assert result["records"][0]["provider"] == "finnhub"


def test_providers_health_snapshots_on_call():
    """GET /providers/health must call snapshot_provider_reliability as side-effect."""
    from api.providers.routes import providers_health
    mock_health = MagicMock()
    mock_health.status = "ok"
    mock_health.providers = []
    with patch("api.providers.routes.get_providers_health", return_value=mock_health), \
         patch("api.providers.routes.snapshot_provider_reliability") as mock_snap:
        providers_health()
    mock_snap.assert_called_once_with(mock_health)
