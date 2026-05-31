"""Regression tests for Phase 4 — intelligence quality and API consistency."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. FEATURE_HINTS completeness
# ---------------------------------------------------------------------------

def test_feature_hints_covers_canonical_features():
    """FEATURE_HINTS must cover all ret_*, vol_*, mom_*, trend_*, atr_*, maxdd_*."""
    from api.llm.prompts import FEATURE_HINTS
    required_prefixes = ["ret_", "vol_", "mom_", "trend_", "atr_", "maxdd_"]
    for prefix in required_prefixes:
        matching = [k for k in FEATURE_HINTS if k.startswith(prefix)]
        assert matching, f"FEATURE_HINTS missing any entry with prefix '{prefix}'"


def test_feature_hints_covers_score_features():
    """FEATURE_HINTS must describe depth score features used by canonical_signal."""
    from api.llm.prompts import FEATURE_HINTS
    score_features = [
        "event_overhang_score",
        "implementation_fragility_score",
        "liquidity_quality_score",
        "cross_asset_conflict_score",
        "market_stress_score",
        "breadth_confirmation_score",
        "earnings_window_flag",
    ]
    for feat in score_features:
        assert feat in FEATURE_HINTS, f"FEATURE_HINTS missing '{feat}'"


def test_feature_hints_legacy_keys_preserved():
    """Original 8 keys must still be present after expansion."""
    from api.llm.prompts import FEATURE_HINTS
    legacy = ["mom_5", "mom_21", "mom_63", "trend_sma20_50",
               "volatility_ann", "rsi14", "volume_z20", "last_close"]
    for key in legacy:
        assert key in FEATURE_HINTS, f"Legacy key '{key}' removed from FEATURE_HINTS"


def test_feature_driver_lines_uses_expanded_hints():
    """feature_driver_lines must return human-readable desc for canonical features."""
    from api.llm.prompts import feature_driver_lines
    features = {"ret_21d": 0.12, "market_stress_score": 65.0, "atr_pct": 0.03}
    lines = feature_driver_lines(features, top_n=3)
    assert len(lines) == 3
    # All lines must contain the feature name and value
    for line in lines:
        assert ":" in line


# ---------------------------------------------------------------------------
# 2. api/errors.py canonical envelope
# ---------------------------------------------------------------------------

def test_err_response_shape():
    """err_response must return canonical envelope with error.type/message/trace_id."""
    from api.errors import err_response
    resp = err_response("validation_error", "symbol required", status_code=400)
    body = resp.body
    import json
    payload = json.loads(body)
    assert "error" in payload
    assert payload["error"]["type"] == "validation_error"
    assert payload["error"]["message"] == "symbol required"
    assert "trace_id" in payload["error"]
    assert "trace_id" in payload
    assert resp.status_code == 400


def test_err_response_trace_id_propagated():
    """Provided trace_id must appear in both error.trace_id and top-level trace_id."""
    from api.errors import err_response
    import json
    resp = err_response("not_found", "symbol not found", 404, trace_id="abc123")
    payload = json.loads(resp.body)
    assert payload["error"]["trace_id"] == "abc123"
    assert payload["trace_id"] == "abc123"
    assert resp.headers.get("x-trace-id") == "abc123"


def test_err_response_auto_trace_id_when_omitted():
    """When trace_id is omitted, a non-empty trace_id is generated."""
    from api.errors import err_response
    import json
    resp = err_response("internal_error", "oops")
    payload = json.loads(resp.body)
    assert payload["trace_id"]
    assert len(payload["trace_id"]) > 8


def test_simple_error_shape():
    """simple_error must return a flat error dict without trace overhead."""
    from api.errors import simple_error
    import json
    resp = simple_error("locked", "job is already running", 409)
    payload = json.loads(resp.body)
    assert payload["error"] == "locked"
    assert payload["detail"] == "job is already running"
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 3. system_confidence deterministic computation
# ---------------------------------------------------------------------------

def _make_signal_response(score=0.6, signal="BUY", regime="TRENDING",
                           notes=None, penalties=None):
    from api.alpha.signal_runner import SignalResponse
    return SignalResponse(
        symbol="AAPL",
        as_of="2024-06-01",
        lookback=63,
        effective_lookback=63,
        regime=regime,
        thresholds={"buy": 0.3, "sell": -0.3},
        score=score,
        signal=signal,
        confidence=0.7,
        features={"mom_21": 0.08, "rsi14": 62.0, "trend_sma20_50": 0.03,
                  "volume_z20": 1.2, "sentiment_score": 0.15},
        adjusted_confidence_notes=notes or [],
        environment_penalties=penalties or {},
    )


def test_system_confidence_high_for_strong_trending_signal():
    """Strong trending BUY must yield system_confidence >= 50."""
    sig = _make_signal_response(score=0.75, signal="BUY", regime="TRENDING")
    payload = sig.external_payload()
    assert payload["system_confidence"] >= 50.0


def test_system_confidence_lower_with_suppression_notes():
    """More adjusted_confidence_notes must reduce system_confidence."""
    sig_clean = _make_signal_response(notes=[])
    sig_noisy = _make_signal_response(notes=["note1", "note2", "note3"])
    clean_conf = sig_clean.external_payload()["system_confidence"]
    noisy_conf = sig_noisy.external_payload()["system_confidence"]
    assert clean_conf > noisy_conf


def test_system_confidence_lower_with_high_penalties():
    """High penalty values must reduce system_confidence vs zero penalties."""
    sig_penalized = _make_signal_response(
        penalties={"event_drag": 0.8, "stress_drag": 0.6}
    )
    sig_clean = _make_signal_response(penalties={})
    penalized_conf = sig_penalized.external_payload()["system_confidence"]
    clean_conf = sig_clean.external_payload()["system_confidence"]
    assert penalized_conf < clean_conf


def test_system_confidence_zero_score_gives_low_confidence():
    """Zero score (no conviction) must produce near-zero system_confidence."""
    sig = _make_signal_response(score=0.0)
    conf = sig.external_payload()["system_confidence"]
    assert conf < 15.0


# ---------------------------------------------------------------------------
# 4. evidence_for / evidence_against structure
# ---------------------------------------------------------------------------

def test_evidence_for_present_in_external_payload():
    """external_payload must always include evidence_for and evidence_against."""
    sig = _make_signal_response()
    payload = sig.external_payload()
    assert "evidence_for" in payload
    assert "evidence_against" in payload
    assert isinstance(payload["evidence_for"], list)
    assert isinstance(payload["evidence_against"], list)


def test_evidence_items_have_required_keys():
    """Every evidence item must have feature, description, value, strength_label."""
    sig = _make_signal_response(
        score=0.7, signal="BUY",
        # provide enough features to generate evidence
    )
    payload = sig.external_payload()
    for item in payload["evidence_for"] + payload["evidence_against"]:
        assert "feature" in item, f"Missing 'feature' in {item}"
        assert "description" in item, f"Missing 'description' in {item}"
        assert "value" in item, f"Missing 'value' in {item}"
        assert "strength_label" in item, f"Missing 'strength_label' in {item}"


def test_evidence_strength_label_values():
    """strength_label must be one of strong/moderate/weak."""
    sig = _make_signal_response()
    payload = sig.external_payload()
    valid = {"strong", "moderate", "weak"}
    for item in payload["evidence_for"] + payload["evidence_against"]:
        assert item["strength_label"] in valid, f"Invalid strength_label: {item['strength_label']}"


def test_evidence_buy_signal_rsi_above_60_goes_to_for():
    """RSI > 60 on a BUY signal must appear in evidence_for."""
    from api.alpha.signal_runner import SignalResponse
    sig = SignalResponse(
        symbol="TEST", as_of="2024-06-01", lookback=63, effective_lookback=63,
        regime="TRENDING", thresholds={}, score=0.7, signal="BUY",
        confidence=0.7, features={"rsi14": 72.0, "mom_21": 0.1},
    )
    payload = sig.external_payload()
    for_features = [item["feature"] for item in payload["evidence_for"]]
    assert "rsi14" in for_features


def test_evidence_sell_signal_positive_mom_goes_to_against():
    """Positive momentum on a SELL signal must appear in evidence_against."""
    from api.alpha.signal_runner import SignalResponse
    sig = SignalResponse(
        symbol="TEST", as_of="2024-06-01", lookback=63, effective_lookback=63,
        regime="CHOPPY", thresholds={}, score=-0.6, signal="SELL",
        confidence=0.6, features={"mom_21": 0.12, "rsi14": 35.0},
    )
    payload = sig.external_payload()
    against_features = [item["feature"] for item in payload["evidence_against"]]
    assert "mom_21" in against_features


def test_evidence_capped_at_six_items():
    """evidence_for and evidence_against must not exceed 6 items each."""
    from api.alpha.signal_runner import SignalResponse
    sig = SignalResponse(
        symbol="TEST", as_of="2024-06-01", lookback=63, effective_lookback=63,
        regime="TRENDING", thresholds={}, score=0.8, signal="BUY",
        confidence=0.8,
        features={
            "mom_5": 0.04, "mom_21": 0.1, "mom_63": 0.2, "mom_126": 0.3,
            "mom_252": 0.4, "rsi14": 65.0, "trend_sma20_50": 0.05,
            "volume_z20": 2.0, "sentiment_score": 0.4, "trend_slope_63d": 0.003,
        },
    )
    payload = sig.external_payload()
    assert len(payload["evidence_for"]) <= 6
    assert len(payload["evidence_against"]) <= 6
