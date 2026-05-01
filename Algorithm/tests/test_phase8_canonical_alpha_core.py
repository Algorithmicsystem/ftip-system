import datetime as dt
import math

import pytest

from api.alpha import build_canonical_features, build_canonical_signal
from api.assistant import reports
from api.backtest import service as backtest_service
from api.feature_engine import compute_daily_features
from api.main import Candle, compute_features, compute_signal_for_symbol_from_candles
from api.prosperity import routes as prosperity_routes
from api.research import build_research_snapshot_from_bars
from api.signal_engine import compute_daily_signal


def _sample_bars(count: int = 320) -> list[dict]:
    base_date = dt.date(2024, 1, 1)
    bars: list[dict] = []
    for idx in range(count):
        close = 100.0 + idx * 0.42 + math.sin(idx / 9.0) * 1.8
        bars.append(
            {
                "symbol": "NVDA",
                "as_of_date": base_date + dt.timedelta(days=idx),
                "open": close - 0.8,
                "high": close + 1.1,
                "low": close - 1.2,
                "close": close,
                "volume": 1_000_000 + idx * 1_500,
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


def _assert_feature_keys_match(
    actual: dict, expected: dict, keys: list[str], *, tol: float = 1e-12
) -> None:
    for key in keys:
        assert key in actual
        assert key in expected
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, str) or expected_value is None:
            assert actual_value == expected_value
        else:
            assert actual_value == pytest.approx(expected_value, abs=tol)


def test_snapshot_is_deterministic_and_point_in_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.research.snapshot._db_ready", lambda: False)

    bars = _sample_bars()
    as_of_date = bars[-2]["as_of_date"]
    future_bar = {
        "symbol": "NVDA",
        "as_of_date": as_of_date + dt.timedelta(days=1),
        "open": 999.0,
        "high": 1001.0,
        "low": 998.0,
        "close": 1000.0,
        "volume": 999_999,
    }

    first = build_research_snapshot_from_bars("NVDA", as_of_date, 252, bars + [future_bar])
    second = build_research_snapshot_from_bars(
        "NVDA", as_of_date, 252, list(reversed(bars)) + [future_bar]
    )

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["snapshot_version"] == second["snapshot_version"]
    assert first["requested_lookback"] == 252
    assert all(
        dt.date.fromisoformat(row["as_of_date"]) <= as_of_date
        for row in first["price_bars"]
    )
    assert first["available_history_bars"] == len(bars) - 1


def test_canonical_features_are_equivalent_across_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api.research.snapshot._db_ready", lambda: False)

    bars = _sample_bars()
    as_of_date = bars[-1]["as_of_date"]
    snapshot = build_research_snapshot_from_bars("NVDA", as_of_date, 252, bars)
    snapshot["sentiment_history"] = [
        {
            "as_of_date": as_of_date.isoformat(),
            "sentiment_score": 0.12,
            "sentiment_mean": 0.05,
        }
    ]
    canonical_payload = build_canonical_features(snapshot)
    expected = canonical_payload["features"]

    feature_engine_features = compute_daily_features(
        bars, sentiment_score=0.12, sentiment_mean=0.05
    )
    candles = _sample_candles(bars)
    main_features = compute_features(candles)
    prosperity_features, prosperity_meta, regime = prosperity_routes._compute_features_payload(
        "NVDA", as_of_date, 252, candles
    )

    shared_market_keys = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_21d",
        "ret_63d",
        "ret_126d",
        "ret_252d",
        "vol_21d",
        "vol_63d",
        "vol_126d",
        "atr_14",
        "atr_pct",
        "trend_slope_21d",
        "trend_r2_21d",
        "trend_slope_63d",
        "trend_r2_63d",
        "mom_vol_adj_21d",
        "maxdd_63d",
        "maxdd_252d",
        "dollar_vol_21d",
        "mom_5",
        "mom_21",
        "mom_63",
        "mom_126",
        "mom_252",
        "trend_sma20_50",
        "volatility_ann",
        "rsi14",
        "volume_z20",
        "last_close",
        "regime_label",
        "regime_strength",
        "signal_regime",
    ]
    _assert_feature_keys_match(feature_engine_features, expected, shared_market_keys)
    _assert_feature_keys_match(main_features, expected, shared_market_keys)
    _assert_feature_keys_match(prosperity_features, expected, shared_market_keys)

    assert regime == expected["signal_regime"]
    assert prosperity_meta["snapshot_id"] == canonical_payload["snapshot_id"]
    assert prosperity_meta["snapshot_version"] == canonical_payload["snapshot_version"]
    assert prosperity_meta["feature_version"] == canonical_payload["feature_version"]
    assert prosperity_meta["effective_lookback"] == canonical_payload["effective_lookback"]


def test_canonical_signal_is_equivalent_across_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("api.research.snapshot._db_ready", lambda: False)

    bars = _sample_bars()
    as_of_date = bars[-1]["as_of_date"]
    snapshot = build_research_snapshot_from_bars("NVDA", as_of_date, 252, bars)
    feature_payload = build_canonical_features(snapshot)
    canonical_signal = build_canonical_signal(snapshot, feature_payload, quality_score=55)

    feature_map = feature_payload["features"]
    latest_close = feature_map["last_close"]
    legacy_signal = compute_daily_signal(
        feature_map,
        quality_score=55,
        latest_close=latest_close,
    )
    candles = _sample_candles(bars)
    main_signal = compute_signal_for_symbol_from_candles(
        "NVDA", as_of_date.isoformat(), 252, candles
    ).model_dump()
    prosperity_signal = prosperity_routes._compute_signal_payload(
        "NVDA", as_of_date, 252, candles
    )["signal_dict"]

    monkeypatch.setattr(backtest_service, "build_research_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(backtest_service, "_fetch_quality_score", lambda *args, **kwargs: 55)
    backtest_signal = backtest_service._compute_canonical_signal_for_date(
        "NVDA", as_of_date, 252
    )

    for payload in (legacy_signal, main_signal, prosperity_signal):
        action_key = "action" if "action" in payload else "signal"
        assert payload[action_key] == canonical_signal["signal"]
        assert payload["score"] == pytest.approx(canonical_signal["score"], abs=1e-12)
        assert payload["confidence"] == pytest.approx(
            canonical_signal["confidence"], abs=1e-12
        )
        assert payload["regime"] == canonical_signal["regime"]
        assert payload["score_mode"] == canonical_signal["score_mode"]
        assert payload["base_score"] == pytest.approx(
            canonical_signal["base_score"], abs=1e-12
        )
        assert payload["stacked_score"] == pytest.approx(
            canonical_signal["stacked_score"], abs=1e-12
        )

    assert main_signal["snapshot_id"] == canonical_signal["meta"]["snapshot_id"]
    assert main_signal["snapshot_version"] == canonical_signal["meta"]["snapshot_version"]
    assert prosperity_signal["meta"]["snapshot_id"] == canonical_signal["meta"]["snapshot_id"]
    assert prosperity_signal["meta"]["snapshot_version"] == canonical_signal["meta"]["snapshot_version"]
    assert backtest_signal is not None
    assert backtest_signal["action"] == canonical_signal["signal"]
    assert backtest_signal["score"] == pytest.approx(canonical_signal["score"], abs=1e-12)
    assert backtest_signal["confidence"] == pytest.approx(
        canonical_signal["confidence"], abs=1e-12
    )


def test_canonical_lineage_propagates_into_report_and_active_context() -> None:
    report = reports.build_analysis_report(
        symbol="NVDA",
        as_of_date="2024-01-02",
        horizon="swing",
        risk_mode="balanced",
        signal={
            "action": "BUY",
            "score": 0.61,
            "confidence": 0.42,
            "reason_codes": ["TREND_UP"],
            "reason_details": {"TREND_UP": "Trend is constructive."},
        },
        key_features={"ret_21d": 0.08, "regime_label": "trend"},
        quality={"bars_ok": True, "warnings": []},
        evidence={"reason_codes": ["TREND_UP"], "reason_details": {}},
        job_context={
            "scenario": "base",
            "analysis_depth": "standard",
            "refresh_mode": "refresh_stale",
            "canonical_lineage": {
                "snapshot_id": "snap-test",
                "snapshot_version": "phase8_canonical_snapshot_v1",
                "feature_version": "phase8_canonical_features_v1",
                "signal_version": "phase8_canonical_signal_v1",
            },
        },
        data_bundle={"quality_provenance": {"quality_score": 80}},
        feature_factor_bundle={},
        strategy={"strategy_version": "phase4_institutional_v1"},
    )
    active_analysis = reports.build_active_analysis_reference(report)

    assert report["snapshot_id"] == "snap-test"
    assert report["snapshot_version"] == "phase8_canonical_snapshot_v1"
    assert report["feature_version"] == "phase8_canonical_features_v1"
    assert report["signal_version"] == "phase8_canonical_signal_v1"
    assert active_analysis["snapshot_id"] == "snap-test"
    assert active_analysis["snapshot_version"] == "phase8_canonical_snapshot_v1"
    assert active_analysis["feature_version"] == "phase8_canonical_features_v1"
    assert active_analysis["signal_version"] == "phase8_canonical_signal_v1"
