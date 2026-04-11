import datetime as dt
import math
from decimal import Decimal

from fastapi.testclient import TestClient

from api.main import app


def test_assistant_analyze_returns_schema(monkeypatch):
    from api.assistant import orchestrator

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": dt.date(2024, 1, 2),
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        }

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(orchestrator, "run_features", _noop)
    monkeypatch.setattr(orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "score": 0.7,
            "confidence": 0.6,
            "entry_low": 100,
            "entry_high": 110,
            "stop_loss": 90,
            "take_profit_1": 130,
            "take_profit_2": 150,
            "horizon_days": 10,
            "reason_codes": ["MOMO_UP"],
            "reason_details": {"MOMO_UP": "Momentum rising"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {"ret_5d": 0.12, "vol_21d": 0.3},
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "warnings": [],
        },
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )
        assert resp.status_code == 200
        data = resp.json()

    assert {
        "symbol",
        "as_of_date",
        "horizon",
        "risk_mode",
        "analysis_job",
        "freshness_summary",
        "signal",
        "key_features",
        "quality",
        "evidence",
        "data_bundle",
        "feature_factor_bundle",
        "strategy",
        "why_this_signal",
        "signal_summary",
        "technical_analysis",
        "fundamental_analysis",
        "statistical_analysis",
        "sentiment_analysis",
        "macro_geopolitical_analysis",
        "risk_quality_analysis",
        "overall_analysis",
        "strategy_view",
        "risks_weaknesses_invalidators",
        "evidence_provenance",
        "session_id",
        "report_id",
        "active_analysis",
    }.issubset(data.keys())
    assert data["signal"]["action"] == "BUY"
    assert data["active_analysis"]["symbol"] == "NVDA"
    assert data["strategy"]["final_signal"] in {"BUY", "HOLD", "SELL"}
    assert data["overall_analysis"]


def test_assistant_analyze_surfaces_enriched_data_fabric(monkeypatch):
    from api.assistant import orchestrator
    from api.assistant import intelligence

    monkeypatch.setenv("FTIP_DATA_FABRIC_ENABLED", "1")

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": dt.date(2024, 1, 2),
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        }

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(orchestrator, "run_features", _noop)
    monkeypatch.setattr(orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "HOLD",
            "score": 0.1,
            "confidence": 0.55,
            "entry_low": 100,
            "entry_high": 110,
            "stop_loss": 95,
            "take_profit_1": 115,
            "take_profit_2": 120,
            "horizon_days": 21,
            "reason_codes": ["MIXED"],
            "reason_details": {"MIXED": "Mixed setup"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {"ret_5d": 0.01, "vol_21d": 0.3},
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        intelligence.data_fabric,
        "enrich_data_bundle",
        lambda **_kwargs: {
            "enabled": True,
            "status": "ok",
            "domains": {
                "fundamental_filing": {
                    "filing_backbone": {
                        "latest_form": "10-Q",
                        "latest_filing_date": "2024-01-01",
                        "latest_10k": {"filing_date": "2023-03-01"},
                        "latest_10q": {"filing_date": "2024-01-01"},
                    },
                    "statement_snapshot": {
                        "latest_quarter": {"revenue": 1200, "report_date": "2024-01-01"},
                    },
                    "normalized_metrics": {
                        "revenue_growth_yoy": 0.18,
                        "operating_margin": 0.24,
                        "net_margin": 0.2,
                        "current_ratio": 2.1,
                        "debt_to_equity": 0.4,
                        "free_cash_flow_margin": 0.16,
                    },
                    "quality_proxies": {
                        "reporting_quality_proxy": 82.0,
                        "business_quality_durability": 79.0,
                    },
                    "coverage_score": 0.88,
                    "strength_summary": ["Quarterly revenue growth remains strong."],
                    "weakness_summary": ["Cash-flow detail is based on quarterly facts only."],
                    "coverage_caveats": ["Leverage detail is partly supported by Finnhub metrics."],
                    "filing_recency_days": 45,
                    "meta": {
                        "sources": ["sec_edgar", "finnhub_basic_financials"],
                        "status": "fresh",
                    },
                },
                "sentiment_narrative_flow": {
                    "source_breakdown": {"gnews": 3, "gdelt": 2},
                    "aggregated_sentiment_bias": 0.12,
                    "meta": {"sources": ["gnews", "gdelt"]},
                },
                "macro_cross_asset": {
                    "macro_regime_context": {"regime": "growth_supportive"},
                    "fred_series": {"rates": {"latest": 4.2}},
                    "meta": {"sources": ["fred", "world_bank"]},
                },
                "geopolitical_policy": {
                    "event_buckets": {"policy_regulation": 2},
                    "event_intensity_score": 0.4,
                    "meta": {"sources": ["gdelt"]},
                },
                "quality_provenance": {
                    "source_map": {
                        "fundamental_filing": ["sec_edgar", "finnhub_basic_financials"],
                        "sentiment_narrative_flow": ["gnews", "gdelt"],
                    },
                    "provider_notes": [],
                    "meta": {"status": "fresh"},
                },
            },
        },
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["data_bundle"]["fundamental_filing"]["filing_recency_days"] == 45
    assert data["data_bundle"]["sentiment_narrative_flow"]["source_breakdown"]["gnews"] == 3
    assert data["data_bundle"]["macro_cross_asset"]["macro_regime_context"]["regime"] == "growth_supportive"
    assert data["data_bundle"]["quality_provenance"]["source_map"]["fundamental_filing"] == [
        "sec_edgar",
        "finnhub_basic_financials",
    ]
    assert "Latest periodic filing is 10-Q dated 2024-01-01" in data["fundamental_analysis"]
    assert "external data fabric status" in data["evidence_provenance"]


def test_assistant_top_picks_schema(monkeypatch):
    from api.assistant import orchestrator

    monkeypatch.setattr(
        orchestrator,
        "fetch_top_picks",
        lambda limit: (
            dt.date(2024, 1, 2),
            [
                {
                    "symbol": "NVDA",
                    "direction": "long",
                    "score": 0.7,
                    "confidence": 0.6,
                    "reason_codes": ["MOMO_UP"],
                }
            ],
        ),
    )
    monkeypatch.setattr(
        orchestrator, "universe_coverage", lambda *_args, **_kwargs: 0.95
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/top-picks",
            json={
                "universe": "sp500",
                "horizon": "swing",
                "risk_mode": "balanced",
                "limit": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

    assert "picks" in data
    assert data["picks"][0]["symbol"] == "NVDA"


def test_assistant_analyze_sanitizes_non_finite_numeric_values(monkeypatch):
    from api.assistant import orchestrator

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": dt.date(2024, 1, 2),
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        }

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(orchestrator, "run_features", _noop)
    monkeypatch.setattr(orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "score": Decimal("0.7"),
            "confidence": Decimal("Infinity"),
            "entry_low": 100.0,
            "entry_high": 110.0,
            "stop_loss": 90.0,
            "take_profit_1": 130.0,
            "take_profit_2": 150.0,
            "horizon_days": 10,
            "reason_codes": ["MOMO_UP"],
            "reason_details": {"MOMO_UP": "Momentum rising"},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {
            "ret_5d": Decimal("-Infinity"),
            "finite_decimal": Decimal("1.25"),
            "vol_21d": Decimal("0.3"),
            "nested": {
                "feature": float("nan"),
                "decimal_feature": Decimal("NaN"),
                "items": [Decimal("1.5"), float("-inf"), {"z": Decimal("Infinity")}],
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "news_ok": True,
            "sentiment_ok": True,
            "risk": {"drawdown": float("inf"), "sharpe": Decimal("1.2")},
            "trace": [0.1, math.nan, {"z": Decimal("NaN"), "w": math.inf}],
            "warnings": [],
        },
    )

    with TestClient(app) as client:
        resp = client.post(
            "/assistant/analyze",
            json={"symbol": "NVDA", "horizon": "swing", "risk_mode": "balanced"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["signal"]["score"] == 0.7
    assert data["signal"]["confidence"] is None
    assert data["signal"]["entry_low"] == 100.0
    assert data["key_features"]["ret_5d"] is None
    assert data["key_features"]["vol_21d"] == 0.3
    assert data["key_features"]["finite_decimal"] == 1.25
    assert data["key_features"]["nested"]["feature"] is None
    assert data["key_features"]["nested"]["decimal_feature"] is None
    assert data["key_features"]["nested"]["items"] == [1.5, None, {"z": None}]
    assert data["quality"]["risk"]["drawdown"] is None
    assert data["quality"]["risk"]["sharpe"] == 1.2
    assert data["quality"]["trace"] == [0.1, None, {"z": None, "w": None}]
    assert data["signal_summary"]
    assert data["strategy_view"]
    assert data["freshness_summary"]["overall_status"]
    assert data["why_this_signal"]["top_positive_drivers"] is not None


def test_fetch_signal_falls_back_to_prosperity_row(monkeypatch):
    from api.assistant import orchestrator

    calls = {"count": 0}

    def _fake_fetchone(query, params):
        calls["count"] += 1
        if "FROM signals_daily" in query:
            return None
        if "FROM prosperity_signals_daily" in query:
            assert "as_of = %s" in query
            assert params == ("AAPL", dt.date(2024, 1, 2))
            return ("BUY", 0.81, 0.64)
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(orchestrator.db, "safe_fetchone", _fake_fetchone)

    signal = orchestrator.fetch_signal("AAPL", dt.date(2024, 1, 2))

    assert calls["count"] == 2
    assert signal == {
        "action": "BUY",
        "score": 0.81,
        "confidence": 0.64,
        "entry_low": None,
        "entry_high": None,
        "stop_loss": None,
        "take_profit_1": None,
        "take_profit_2": None,
        "horizon_days": None,
        "reason_codes": [],
        "reason_details": {},
    }


def test_fetch_signal_falls_back_to_prosperity_row_with_as_of_date(monkeypatch):
    from api.assistant import orchestrator

    orchestrator._PROSPERITY_SIGNAL_ASOF_COLUMN = None
    orchestrator._PROSPERITY_SIGNAL_ACTION_COLUMN = None

    def _fake_fetchone(query, params=None):
        if "FROM signals_daily" in query:
            return None
        if "FROM information_schema.columns" in query and "as_of_date" in query:
            return ("as_of_date",)
        if "FROM information_schema.columns" in query and "signal', 'action" in query:
            return ("signal",)
        if "FROM prosperity_signals_daily" in query:
            assert "as_of_date = %s" in query
            return ("SELL", -0.22, 0.51)
        raise AssertionError(f"unexpected query: {query}")

    monkeypatch.setattr(orchestrator.db, "safe_fetchone", _fake_fetchone)

    signal = orchestrator.fetch_signal("AAPL", dt.date(2024, 1, 2))

    assert signal is not None
    assert signal["action"] == "SELL"
    assert signal["score"] == -0.22
    assert signal["confidence"] == 0.51
