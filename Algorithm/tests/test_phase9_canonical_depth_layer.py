import datetime as dt
import math

import pytest

from api.alpha import build_canonical_features, build_canonical_signal
from api.assistant.phase5.context import build_narrator_context
from api.assistant.phase5.routing import route_question
from api.assistant import reports
from api.backtest import service as backtest_service
from api.main import Candle, compute_signal_for_symbol_from_candles
from api.prosperity import routes as prosperity_routes
from api.research import build_research_snapshot_from_bars


def _trend_bars(count: int = 320) -> list[dict]:
    base_date = dt.date(2024, 1, 1)
    bars: list[dict] = []
    prev_close = 100.0
    for idx in range(count):
        drift = 0.42 + math.sin(idx / 11.0) * 0.35
        close = prev_close + drift
        gap_bump = 0.0
        if idx > 0 and idx % 9 == 0:
            gap_bump = prev_close * 0.03
        open_px = prev_close + gap_bump
        high = max(open_px, close) + 2.2 + (0.8 if idx % 13 == 0 else 0.0)
        low = min(open_px, close) - 2.0 - (0.9 if idx % 17 == 0 else 0.0)
        volume = 800_000 + (idx % 7) * 650_000 + (250_000 if idx % 10 == 0 else 0)
        bars.append(
            {
                "symbol": "NVDA",
                "as_of_date": base_date + dt.timedelta(days=idx),
                "open": open_px,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "source": "unit_test",
            }
        )
        prev_close = close
    return bars


def _proxy_bars(
    symbol: str,
    *,
    start: dt.date,
    count: int,
    daily_drift: float,
    wave: float,
) -> list[dict]:
    bars: list[dict] = []
    close = 100.0
    for idx in range(count):
        close = close * (1.0 + daily_drift + math.sin(idx / 6.0) * wave)
        bars.append(
            {
                "as_of_date": (start + dt.timedelta(days=idx)).isoformat(),
                "open": close * 0.997,
                "high": close * 1.012,
                "low": close * 0.988,
                "close": close,
                "volume": 1_000_000 + idx * 1000,
                "source": f"{symbol.lower()}_proxy",
            }
        )
    return bars


def _sample_candles(bars: list[dict]) -> list[Candle]:
    return [
        Candle(
            timestamp=row["as_of_date"].isoformat(),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
        )
        for row in bars
    ]


def _reference_context(start: dt.date, count: int) -> dict:
    return {
        "SPY": {"bars": _proxy_bars("SPY", start=start, count=count, daily_drift=-0.0014, wave=0.0018)},
        "QQQ": {"bars": _proxy_bars("QQQ", start=start, count=count, daily_drift=-0.0011, wave=0.0020)},
        "IWM": {"bars": _proxy_bars("IWM", start=start, count=count, daily_drift=-0.0017, wave=0.0022)},
        "XLK": {"bars": _proxy_bars("XLK", start=start, count=count, daily_drift=-0.0013, wave=0.0019)},
        "TLT": {"bars": _proxy_bars("TLT", start=start, count=count, daily_drift=0.0011, wave=0.0010)},
        "GLD": {"bars": _proxy_bars("GLD", start=start, count=count, daily_drift=0.0008, wave=0.0010)},
        "USO": {"bars": _proxy_bars("USO", start=start, count=count, daily_drift=-0.0002, wave=0.0015)},
        "UUP": {"bars": _proxy_bars("UUP", start=start, count=count, daily_drift=0.0007, wave=0.0008)},
    }


def test_snapshot_depth_context_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _trend_bars(260)
    as_of_date = bars[-1]["as_of_date"]

    monkeypatch.setattr(
        "api.research.snapshot._load_fundamentals",
        lambda *_args, **_kwargs: (
            [
                {
                    "period_end": (as_of_date - dt.timedelta(days=35)).isoformat(),
                    "report_date": (as_of_date - dt.timedelta(days=6)).isoformat(),
                }
            ],
            "unit_test",
        ),
    )
    monkeypatch.setattr(
        "api.research.snapshot._load_news",
        lambda *_args, **_kwargs: (
            [
                {
                    "published_at": dt.datetime.combine(
                        as_of_date - dt.timedelta(days=1),
                        dt.time(14, 0),
                        tzinfo=dt.timezone.utc,
                    ).isoformat(),
                    "title": "NVDA earnings guidance update triggers repricing",
                    "content_snippet": "Guidance and margin commentary remain volatile.",
                },
                {
                    "published_at": dt.datetime.combine(
                        as_of_date - dt.timedelta(days=4),
                        dt.time(10, 0),
                        tzinfo=dt.timezone.utc,
                    ).isoformat(),
                    "title": "NVDA files quarter results and outlook",
                    "content_snippet": "Quarter results and outlook drive catalyst attention.",
                },
            ],
            "unit_test",
        ),
    )
    monkeypatch.setattr("api.research.snapshot._load_sentiment_history", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("api.research.snapshot._load_quality", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("api.research.snapshot._load_intraday", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "api.research.snapshot._load_breadth_context",
        lambda *_args, **_kwargs: {
            "universe_count": 120,
            "advancing_1d_ratio": 0.42,
            "advancing_21d_ratio": 0.47,
            "above_trend_ratio": 0.45,
            "sector_participation_ratio": 0.41,
            "cross_sectional_dispersion": 0.038,
            "sector_dispersion": 0.032,
            "leader_strength": 0.11,
            "laggard_pressure": -0.08,
            "leadership_concentration": 0.72,
            "leadership_rotation": 0.63,
            "leadership_instability": 0.74,
            "source": "unit_test",
        },
    )

    first = build_research_snapshot_from_bars("NVDA", as_of_date, 252, bars)
    second = build_research_snapshot_from_bars("NVDA", as_of_date, 252, list(reversed(bars)))

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["event_context"]["earnings_window_flag"] is True
    assert first["event_context"]["event_match_count_7d"] >= 2
    assert first["breadth_context"]["leadership_concentration"] == pytest.approx(0.72)
    assert first["coverage"]["event_matches"] >= 2
    assert first["coverage"]["breadth_universe_count"] == 120


def test_canonical_depth_features_and_suppression_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.research.snapshot._db_ready", lambda: False)
    bars = _trend_bars()
    as_of_date = bars[-1]["as_of_date"]
    snapshot = build_research_snapshot_from_bars(
        "NVDA",
        as_of_date,
        252,
        bars,
        include_reference_context=True,
    )
    snapshot["symbol_meta"] = {"symbol": "NVDA", "sector": "technology"}
    snapshot["event_context"] = {
        "estimated_next_event_date": (as_of_date + dt.timedelta(days=2)).isoformat(),
        "latest_event_date": (as_of_date - dt.timedelta(days=1)).isoformat(),
        "days_to_next_event": 2,
        "days_since_last_major_event": 1,
        "earnings_window_flag": True,
        "post_event_instability_flag": True,
        "event_match_count_3d": 3,
        "event_match_count_7d": 5,
        "event_match_count_21d": 7,
        "event_density_score": 88.0,
        "catalyst_burst_score": 92.0,
        "major_event_matches": [
            {"title": "NVDA earnings guidance update"},
            {"title": "NVDA quarter results surprise"},
        ],
    }
    snapshot["breadth_context"] = {
        "universe_count": 140,
        "advancing_1d_ratio": 0.36,
        "advancing_21d_ratio": 0.41,
        "above_trend_ratio": 0.39,
        "sector_participation_ratio": 0.37,
        "cross_sectional_dispersion": 0.043,
        "sector_dispersion": 0.039,
        "leader_strength": 0.09,
        "laggard_pressure": -0.10,
        "leadership_concentration": 0.78,
        "leadership_rotation": 0.71,
        "leadership_instability": 0.82,
        "source": "unit_test",
    }
    snapshot["reference_context"] = _reference_context(bars[0]["as_of_date"], len(bars))

    feature_payload = build_canonical_features(snapshot)
    features = feature_payload["features"]
    signal = build_canonical_signal(snapshot, feature_payload, quality_score=70)

    assert features["event_overhang_score"] >= 70
    assert features["implementation_fragility_score"] >= 55
    assert features["breadth_confirmation_score"] <= 50
    assert features["cross_asset_conflict_score"] >= 55
    assert features["market_stress_score"] >= 55
    assert "event_overhang" in signal["suppression_flags"]
    assert "market_stress" in signal["suppression_flags"]
    assert signal["adjusted_confidence_notes"]
    assert signal["signal"] == "HOLD"


def test_wrappers_share_depth_adjusted_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _trend_bars()
    candles = _sample_candles(bars)
    as_of_date = bars[-1]["as_of_date"]

    monkeypatch.setattr("api.research.snapshot._db_ready", lambda: False)
    monkeypatch.setattr(
        "api.research.snapshot._load_event_context",
        lambda **_kwargs: {
            "estimated_next_event_date": (as_of_date + dt.timedelta(days=2)).isoformat(),
            "latest_event_date": (as_of_date - dt.timedelta(days=1)).isoformat(),
            "days_to_next_event": 2,
            "days_since_last_major_event": 1,
            "earnings_window_flag": True,
            "post_event_instability_flag": True,
            "event_match_count_3d": 3,
            "event_match_count_7d": 5,
            "event_match_count_21d": 7,
            "event_density_score": 88.0,
            "catalyst_burst_score": 92.0,
            "major_event_matches": [{"title": "NVDA earnings guidance update"}],
        },
    )
    monkeypatch.setattr(
        "api.research.snapshot._load_breadth_context",
        lambda *_args, **_kwargs: {
            "universe_count": 140,
            "advancing_1d_ratio": 0.36,
            "advancing_21d_ratio": 0.41,
            "above_trend_ratio": 0.39,
            "sector_participation_ratio": 0.37,
            "cross_sectional_dispersion": 0.043,
            "sector_dispersion": 0.039,
            "leader_strength": 0.09,
            "laggard_pressure": -0.10,
            "leadership_concentration": 0.78,
            "leadership_rotation": 0.71,
            "leadership_instability": 0.82,
            "source": "unit_test",
        },
    )
    monkeypatch.setattr(
        "api.research.snapshot._query_reference_bars",
        lambda *_args, **_kwargs: _reference_context(bars[0]["as_of_date"], len(bars)),
    )

    snapshot = build_research_snapshot_from_bars(
        "NVDA",
        as_of_date,
        252,
        bars,
        include_reference_context=True,
    )
    feature_payload = build_canonical_features(snapshot)
    canonical_signal = build_canonical_signal(snapshot, feature_payload, quality_score=55)
    main_signal = compute_signal_for_symbol_from_candles(
        "NVDA",
        as_of_date.isoformat(),
        252,
        candles,
    ).model_dump()
    prosperity_signal = prosperity_routes._compute_signal_payload(
        "NVDA",
        as_of_date,
        252,
        candles,
    )["signal_dict"]

    monkeypatch.setattr(backtest_service, "build_research_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(backtest_service, "_fetch_quality_score", lambda *args, **kwargs: 55)
    backtest_signal = backtest_service._compute_canonical_signal_for_date("NVDA", as_of_date, 252)

    assert main_signal["signal"] == canonical_signal["signal"]
    assert main_signal["score"] == pytest.approx(canonical_signal["score"], abs=1e-12)
    assert main_signal["confidence"] == pytest.approx(canonical_signal["confidence"], abs=1e-12)
    assert prosperity_signal["signal"] == canonical_signal["signal"]
    assert prosperity_signal["score"] == pytest.approx(canonical_signal["score"], abs=1e-12)
    assert prosperity_signal["confidence"] == pytest.approx(canonical_signal["confidence"], abs=1e-12)
    assert backtest_signal is not None
    assert backtest_signal["action"] == canonical_signal["signal"]
    assert backtest_signal["score"] == pytest.approx(canonical_signal["score"], abs=1e-12)
    assert backtest_signal["confidence"] == pytest.approx(canonical_signal["confidence"], abs=1e-12)
    assert main_signal["suppression_flags"] == canonical_signal["suppression_flags"]
    assert prosperity_signal["suppression_flags"] == canonical_signal["suppression_flags"]
    assert backtest_signal["payload"]["suppression_flags"] == canonical_signal["suppression_flags"]


def test_report_and_narrator_include_depth_sections() -> None:
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.52,
            "confidence": 0.44,
            "reason_codes": ["TREND_UP", "EVENT_OVERHANG"],
            "reason_details": {"EVENT_OVERHANG": "Event window is elevated."},
        },
        key_features={"ret_21d": 0.09, "regime_label": "trend"},
        quality={"bars_ok": True, "warnings": []},
        evidence={"reason_codes": ["TREND_UP"], "reason_details": {}},
        job_context={
            "scenario": "base",
            "analysis_depth": "standard",
            "refresh_mode": "refresh_stale",
            "canonical_lineage": {
                "snapshot_id": "snap-depth",
                "snapshot_version": "phase9_canonical_snapshot_v1",
                "feature_version": "phase9_canonical_features_v1",
                "signal_version": "phase9_canonical_signal_v1",
            },
        },
        data_bundle={
            "quality_provenance": {"quality_score": 82, "freshness_summary": {}},
            "canonical_alpha_core": {
                "feature_vector": {
                    "event_overhang_score": 81.0,
                    "implementation_fragility_score": 66.0,
                    "breadth_confirmation_score": 41.0,
                    "cross_asset_conflict_score": 63.0,
                    "market_stress_score": 69.0,
                },
                "signal_payload": {
                    "suppression_flags": ["event_overhang", "market_stress"],
                    "adjusted_confidence_notes": [
                        "Confidence reduced because event proximity and catalyst density are elevated."
                    ],
                },
            },
            "event_catalyst_risk": {
                "event_risk_classification": "event_distorted",
                "event_overhang_score": 81.0,
                "event_uncertainty_score": 78.0,
                "catalyst_burst_score": 84.0,
                "days_to_next_event": 2,
                "days_since_last_major_event": 1,
                "earnings_window_flag": True,
                "post_event_instability_flag": True,
                "major_event_titles": ["NVDA earnings guidance update"],
                "meta": {"coverage_status": "available", "data_quality_note": "Event context is available."},
            },
            "liquidity_execution_fragility": {
                "liquidity_quality_score": 49.0,
                "execution_cleanliness_score": 38.0,
                "implementation_fragility_score": 66.0,
                "gap_instability_score": 71.0,
                "range_instability_score": 62.0,
                "turnover_stability_score": 43.0,
                "friction_proxy_score": 68.0,
                "tradability_caution_score": 70.0,
                "overnight_gap_risk_score": 74.0,
                "tradability_state": "implementation_fragile",
                "meta": {"coverage_status": "available", "data_quality_note": "Liquidity context is available."},
            },
            "market_breadth_internals": {
                "breadth_state": "narrow_leadership",
                "breadth_confirmation_score": 41.0,
                "participation_breadth_score": 44.0,
                "breadth_thrust_proxy": 43.0,
                "cross_sectional_dispersion_proxy": 76.0,
                "sector_dispersion_proxy": 68.0,
                "leadership_concentration_score": 79.0,
                "internal_market_divergence_score": 73.0,
                "leader_strength_score": 56.0,
                "laggard_pressure_score": 71.0,
                "leadership_rotation_score": 65.0,
                "leadership_instability_score": 77.0,
                "meta": {"coverage_status": "available", "data_quality_note": "Breadth context is available."},
            },
            "cross_asset_confirmation": {
                "benchmark_proxy": "SPY",
                "sector_proxy": "XLK",
                "benchmark_confirmation_score": 36.0,
                "sector_confirmation_score": 33.0,
                "macro_asset_alignment_score": 39.0,
                "cross_asset_conflict_score": 63.0,
                "cross_asset_divergence_score": 59.0,
                "beta_context_score": 37.0,
                "idiosyncratic_strength_score": 42.0,
                "idiosyncratic_weakness_score": 58.0,
                "meta": {"coverage_status": "available", "data_quality_note": "Cross-asset context is available."},
            },
            "stress_spillover_conditions": {
                "market_stress_score": 69.0,
                "spillover_risk_score": 66.0,
                "contagion_risk_proxy": 64.0,
                "correlation_breakdown_proxy": 61.0,
                "volatility_shock_score": 63.0,
                "stress_transition_score": 68.0,
                "defensive_regime_flag": True,
                "unstable_environment_flag": True,
                "meta": {"coverage_status": "available", "data_quality_note": "Stress context is available."},
            },
        },
        feature_factor_bundle={},
        strategy={
            "final_signal": "HOLD",
            "strategy_posture": "watchlist_positive",
            "confidence": 0.36,
            "confidence_score": 36.0,
            "conviction_tier": "low",
            "fragility_tier": "elevated",
            "actionability_score": 33.0,
            "strategy_version": "phase4_institutional_v1",
        },
    )

    assert report["event_catalyst_risk_analysis"]
    assert report["liquidity_execution_fragility_analysis"]
    assert report["market_breadth_internal_state_analysis"]
    assert report["cross_asset_confirmation_analysis"]
    assert report["stress_spillover_analysis"]

    route = route_question("Is this setup distorted by earnings, weak breadth, or market stress?")
    assert route["intent"] == "market_depth"
    narrator_context = build_narrator_context(
        report,
        active_analysis=reports.build_active_analysis_reference(report, session_id="s1", report_id="r1"),
        route=route,
        user_message="Is this setup distorted by earnings, weak breadth, or market stress?",
        caller_context=None,
    )
    assert narrator_context["market_depth_snapshot"]["suppression_flags"]
    assert "event_catalyst_risk_analysis" in narrator_context["selected_sections"]
    assert "stress_spillover_analysis" in narrator_context["selected_sections"]
