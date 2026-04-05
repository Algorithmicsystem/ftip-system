import asyncio
from typing import Any

from fastapi.testclient import TestClient

from api.assistant import intelligence, reports, service, strategy
from api.assistant.storage import AssistantStorage
from api.main import app


class DummySignal:
    def __init__(self, symbol: str = "AAPL"):
        self.symbol = symbol

    def model_dump(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": "2024-01-01",
            "lookback": 10,
            "signal": "BUY",
            "score": 1.0,
            "confidence": 0.8,
            "thresholds": {"buy": 0.5},
            "notes": ["test"],
        }


class DummyBacktest:
    def model_dump(self) -> dict[str, Any]:
        return {
            "total_return": 0.1,
            "sharpe": 1.0,
            "max_drawdown": -0.05,
            "volatility": 0.2,
            "lookback": 252,
        }


def test_chat_returns_503_when_disabled(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/assistant/chat", json={"message": "hi"})
    assert resp.status_code == 503


def test_chat_missing_api_key(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = TestClient(app)
    resp = client.post("/assistant/chat", json={"message": "hi"})
    assert resp.status_code == 500
    assert "LLM API key not configured" in resp.json()["error"]["message"]


def test_storage_memory_roundtrip():
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session(metadata={"foo": "bar"})
    store.add_message(session_id, "user", "hello")
    store.upsert_session_metadata(session_id, {"title": "Test"})
    artifact_id = store.save_artifact(
        session_id,
        reports.ANALYSIS_REPORT_KIND,
        {
          "symbol": "AAPL",
          "as_of_date": "2024-01-01",
          "horizon": "swing",
          "risk_mode": "balanced",
          "overall_analysis": "Stored report.",
        },
    )

    session = store.get_session(session_id)
    assert session is not None
    assert session["metadata"].get("foo") == "bar"
    assert session["metadata"].get("title") == "Test"

    messages = store.get_messages(session_id)
    assert len(messages) == 1
    assert messages[0]["content"] == "hello"
    report = store.get_latest_analysis_report(
        session_id=session_id,
        symbol="AAPL",
        horizon="swing",
        risk_mode="balanced",
    )
    assert report is not None
    assert report["report_id"] == artifact_id
    assert report["overall_analysis"] == "Stored report."


def test_generate_analysis_report_persists_artifact(monkeypatch):
    store = AssistantStorage(use_memory=True)

    async def _fake_freshness(symbol: str, refresh: bool = True):
        return {
            "as_of_date": "2024-01-02",
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

    monkeypatch.setattr(service.orchestrator, "ensure_freshness", _fake_freshness)
    monkeypatch.setattr(service.orchestrator, "run_features", _noop)
    monkeypatch.setattr(service.orchestrator, "run_signals", _noop)
    monkeypatch.setattr(
        service.orchestrator,
        "fetch_signal",
        lambda *_args, **_kwargs: {
            "action": "BUY",
            "score": 0.9,
            "confidence": 0.7,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 112,
            "take_profit_2": 120,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is rising"},
        },
    )
    monkeypatch.setattr(
        service.orchestrator,
        "fetch_key_features",
        lambda *_args, **_kwargs: {
            "ret_1d": 0.01,
            "ret_5d": 0.03,
            "ret_21d": 0.12,
            "vol_21d": 0.25,
            "vol_63d": 0.22,
            "atr_pct": 0.04,
            "trend_slope_21d": 0.2,
            "trend_slope_63d": 0.1,
            "mom_vol_adj_21d": 0.8,
            "sentiment_score": 0.4,
            "sentiment_surprise": 0.1,
            "regime_label": "trend",
            "regime_strength": 0.6,
        },
    )
    monkeypatch.setattr(
        service.orchestrator,
        "fetch_quality",
        lambda *_args, **_kwargs: {
            "bars_ok": True,
            "fundamentals_ok": False,
            "sentiment_ok": True,
            "intraday_ok": False,
            "missingness": 0.02,
            "anomaly_flags": [],
            "quality_score": 88,
            "news_ok": True,
            "bars_updated_at": "2024-01-02T00:00:00Z",
            "news_updated_at": "2024-01-02T00:00:00Z",
            "sentiment_updated_at": "2024-01-02T00:00:00Z",
            "warnings": [],
        },
    )

    result = asyncio.run(
        service.generate_analysis_report(
            {
                "symbol": "NVDA",
                "horizon": "swing",
                "risk_mode": "balanced",
            },
            store=store,
        )
    )

    assert result["report_id"]
    assert result["session_id"]
    assert result["overall_analysis"]
    assert result["strategy_view"]
    assert result["analysis_job"]["trace_id"]
    assert result["freshness_summary"]["overall_status"]
    assert result["data_bundle"]["quality_provenance"]
    assert result["feature_factor_bundle"]["composite_intelligence"]
    assert result["strategy"]["final_signal"]
    assert result["why_this_signal"]["top_positive_drivers"] is not None
    assert result["evidence_provenance"]

    session = store.get_session(result["session_id"])
    assert session is not None
    assert session["metadata"]["active_analysis"]["report_id"] == result["report_id"]
    assert session["metadata"]["active_analysis"]["signal"]

    report = store.get_latest_analysis_report(session_id=result["session_id"], symbol="NVDA")
    assert report is not None
    assert report["report_id"] == result["report_id"]
    assert report["signal_summary"] == result["signal_summary"]
    assert report["overall_analysis"] == result["overall_analysis"]
    assert (
        store.get_latest_artifact(
            kind=intelligence.ANALYSIS_JOB_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=intelligence.DATA_BUNDLE_KIND, session_id=result["session_id"]
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=intelligence.FEATURE_FACTOR_BUNDLE_KIND,
            session_id=result["session_id"],
        )
        is not None
    )
    assert (
        store.get_latest_artifact(
            kind=strategy.STRATEGY_ARTIFACT_KIND, session_id=result["session_id"]
        )
        is not None
    )


def test_chat_uses_persisted_analysis_report(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.8,
            "confidence": 0.7,
            "entry_low": 100,
            "entry_high": 104,
            "stop_loss": 95,
            "take_profit_1": 110,
            "take_profit_2": 118,
            "horizon_days": 21,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is rising"},
        },
        key_features={
            "ret_1d": 0.01,
            "ret_5d": 0.03,
            "ret_21d": 0.08,
            "vol_21d": 0.25,
            "vol_63d": 0.22,
            "atr_pct": 0.04,
            "trend_slope_21d": 0.2,
            "trend_slope_63d": 0.1,
            "mom_vol_adj_21d": 0.8,
            "sentiment_score": 0.4,
            "sentiment_surprise": 0.1,
            "regime_label": "trend",
            "regime_strength": 0.6,
        },
        quality={
            "bars_ok": True,
            "fundamentals_ok": False,
            "sentiment_ok": True,
            "intraday_ok": False,
            "missingness": 0.02,
            "anomaly_flags": [],
            "quality_score": 88,
            "news_ok": True,
            "warnings": [],
        },
        evidence={
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is rising"},
            "sources": ["market_bars_daily", "news_raw", "sentiment_daily"],
        },
    )
    report_id = store.save_artifact(session_id, reports.ANALYSIS_REPORT_KIND, report)
    store.upsert_session_metadata(
        session_id,
        {
            "active_analysis": reports.build_active_analysis_reference(
                report, session_id=session_id, report_id=report_id
            )
        },
    )

    def _fake_completion(messages):
        combined = "\n".join(message["content"] for message in messages)
        assert "NVDA" in combined
        assert report["overall_analysis"] in combined
        assert report["strategy_view"] in combined
        assert report["evidence_provenance"] in combined
        return (
            "Grounded reply about the stored NVDA analysis.",
            "model",
            {"prompt_tokens": 10, "completion_tokens": 12},
        )

    monkeypatch.setattr(service, "_safe_completion", _fake_completion)
    result = service.chat_with_assistant(
        {"session_id": session_id, "message": "Should I buy NVDA based on the analysis?"},
        store=store,
    )
    assert result["reply"] == "Grounded reply about the stored NVDA analysis."
    assert result["report_found"] is True
    assert result["active_analysis"]["symbol"] == "NVDA"
    assert (
        store.get_latest_artifact(kind=strategy.CHAT_GROUNDING_CONTEXT_KIND, session_id=session_id)
        is not None
    )


def test_chat_returns_no_analysis_message_when_report_absent(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    store = AssistantStorage(use_memory=True)

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda _messages: (_ for _ in ()).throw(AssertionError("completion should not run")),
    )

    result = service.chat_with_assistant(
        {"message": "Explain NVDA based on the analysis."},
        store=store,
    )
    assert result["report_found"] is False
    assert "No stored analysis report exists for NVDA yet." in result["reply"]


def test_explain_signal_mocked(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "mocked reply",
            "model",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )
    result = service.explain_signal(
        {"symbol": "AAPL", "as_of": "2024-01-01", "lookback": 10},
        signal_fetcher=lambda symbol, as_of, lookback: DummySignal(symbol),
        store=AssistantStorage(use_memory=True),
    )
    assert result["reply"] == "mocked reply"


def test_explain_backtest_mocked(monkeypatch):
    monkeypatch.setenv("FTIP_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setattr(
        service,
        "_safe_completion",
        lambda messages: (
            "backtest reply",
            "model",
            {"prompt_tokens": 1, "completion_tokens": 1},
        ),
    )
    result = service.explain_backtest(
        {
            "symbols": ["AAPL"],
            "from_date": "2023-01-01",
            "to_date": "2023-12-31",
            "lookback": 252,
            "rebalance_every": 21,
            "trading_cost_bps": 10.0,
            "slippage_bps": 5.0,
            "max_weight": None,
            "min_trade_delta": 0.0005,
            "max_turnover_per_rebalance": 0.25,
            "allow_shorts": False,
        },
        backtest_runner=lambda req: DummyBacktest(),
        store=AssistantStorage(use_memory=True),
    )
    assert result["reply"] == "backtest reply"


def test_feature_factor_and_strategy_layers_produce_structured_outputs():
    data_bundle = {
        "market_price_volume": {
            "day_return": 0.01,
            "ret_5d": 0.04,
            "ret_10d": 0.05,
            "ret_21d": 0.1,
            "ret_63d": 0.2,
            "ret_126d": 0.3,
            "ret_252d": 0.45,
            "realized_vol_5d": 0.22,
            "realized_vol_21d": 0.25,
            "realized_vol_63d": 0.24,
            "atr_pct": 0.04,
            "gap_pct": 0.01,
            "breakout_distance_63d": 0.03,
            "volume_anomaly": 1.4,
            "support_21d": 100.0,
            "resistance_21d": 118.0,
            "compression_ratio": 0.07,
        },
        "technical_market_structure": {
            "moving_averages": {"ma_10": 112, "ma_21": 109, "ma_63": 101, "ma_126": 92},
            "trend_slope_21d": 0.12,
            "trend_slope_63d": 0.08,
            "trend_curvature": 0.04,
            "mean_reversion_gap": 0.05,
            "breakout_state": "trend_extension",
            "volume_price_alignment": 0.09,
            "regime_label": "trend",
            "regime_strength": 0.6,
        },
        "fundamental_filing": {
            "latest_quarter": {"revenue": 1000, "op_margin": 0.24, "gross_margin": 0.61},
            "revenue_growth_yoy": 0.18,
            "margin_stability": 0.78,
            "positive_fcf_ratio": 0.75,
            "filing_recency_days": 54,
        },
        "sentiment_narrative_flow": {
            "sentiment_score": 0.32,
            "sentiment_surprise": 0.08,
            "sentiment_trend": 0.02,
            "headline_count": 14,
            "attention_crowding": 1.4,
            "novelty_ratio": 0.7,
            "narrative_concentration": 0.32,
            "disagreement_score": 0.18,
            "hype_price_divergence": 0.04,
        },
        "macro_cross_asset": {
            "benchmark_proxy": "XLK",
            "benchmark_ret_21d": 0.07,
            "benchmark_vol_21d": 0.19,
            "inferred_market_regime": "risk_on",
            "macro_alignment_score": 72.0,
            "risk_on_score": 0.06,
            "stress_overlay": -0.01,
            "meta": {"status": "fresh", "coverage_score": 0.85},
        },
        "geopolitical_policy": {
            "category_counts": {"rates_policy": 1, "regulation_policy": 0},
            "exogenous_event_score": 0.12,
        },
        "relative_context": {
            "sector": "Technology",
            "peer_count": 8,
            "relative_ret_21d": 0.05,
            "relative_momentum": 0.09,
            "relative_strength_percentile": 0.82,
            "peer_dispersion_score": 0.11,
            "meta": {"status": "fresh", "coverage_score": 0.9},
        },
        "quality_provenance": {
            "quality_score": 88,
            "missingness": 0.03,
            "warnings": [],
            "freshness_summary": {
                "bars": {"status": "fresh", "updated_at": "2026-04-05T00:00:00Z"},
                "news": {"status": "fresh", "updated_at": "2026-04-05T00:00:00Z"},
                "sentiment": {"status": "fresh", "updated_at": "2026-04-05T00:00:00Z"},
            },
        },
    }
    feature_bundle = intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal={"action": "BUY", "score": 0.8, "confidence": 0.74},
        key_features={
            "ret_1d": 0.01,
            "ret_5d": 0.04,
            "ret_21d": 0.1,
            "mom_vol_adj_21d": 0.75,
            "sentiment_score": 0.32,
            "regime_label": "trend",
        },
        quality={
            "missingness": 0.03,
            "fundamentals_ok": True,
        },
    )
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={
            "horizon": "swing",
            "scenario": "base",
        },
        signal={"action": "BUY", "score": 0.8, "confidence": 0.74},
        data_bundle=data_bundle,
        feature_factor_bundle=feature_bundle,
    )

    assert "multi_horizon_price_momentum" in feature_bundle
    assert "composite_intelligence" in feature_bundle
    assert "Opportunity Quality Score" in feature_bundle["composite_intelligence"]
    assert strategy_bundle["final_signal"] in {"BUY", "HOLD", "SELL"}
    assert strategy_bundle["component_scores"]["trend_following"]["weight"] > 0
    assert strategy_bundle["top_contributors"]


def test_providers_health_is_registered_and_in_openapi() -> None:
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    r = client.get("/providers/health")
    assert r.status_code == 200
    data = r.json()
    assert "providers" in data
    for k in ("openai", "massive", "finnhub", "fred", "secedgar"):
        assert k in data["providers"]
    openapi = client.get("/openapi.json").json()
    assert "/providers/health" in openapi.get("paths", {})
