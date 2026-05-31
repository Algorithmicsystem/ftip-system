"""Regression tests for Phase 8 — unified ExplanationPayload and signal staleness."""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 1. ExplanationPayload struct
# ---------------------------------------------------------------------------

def test_explanation_payload_importable():
    """ExplanationPayload and DriverItem must be importable."""
    from api.assistant.explanation import ExplanationPayload, DriverItem
    assert ExplanationPayload
    assert DriverItem


def test_build_explanation_payload_minimal():
    """build_explanation_payload must work with an empty signal dict."""
    from api.assistant.explanation import build_explanation_payload
    payload = build_explanation_payload({})
    assert payload.top_drivers == []
    assert payload.reason_codes == []
    assert payload.headline is None


def test_build_explanation_payload_maps_evidence_to_drivers():
    """evidence_for/against from signal dict must become DriverItems."""
    from api.assistant.explanation import build_explanation_payload
    signal = {
        "evidence_for": [
            {"feature": "mom_21", "description": "1-month momentum: 0.1",
             "value": 0.1, "strength_label": "strong"},
        ],
        "evidence_against": [
            {"feature": "market_stress_score", "description": "market stress: 72.0",
             "value": 72.0, "strength_label": "moderate"},
        ],
        "reason_codes": ["STRONG_TREND"],
        "regime": "TRENDING",
        "as_of": "2024-01-01",
    }
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 3)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        payload = build_explanation_payload(signal)

    assert len(payload.top_drivers) == 2
    supporters = [d for d in payload.top_drivers if d.direction == "supports"]
    opposers = [d for d in payload.top_drivers if d.direction == "opposes"]
    assert len(supporters) == 1
    assert supporters[0].feature == "mom_21"
    assert supporters[0].strength_label == "strong"
    assert len(opposers) == 1
    assert opposers[0].feature == "market_stress_score"
    assert payload.reason_codes == ["STRONG_TREND"]
    assert payload.regime_context == "TRENDING"


def test_build_explanation_payload_driver_sort_order():
    """Supporters must come before opposers; within each group: strong before weak."""
    from api.assistant.explanation import build_explanation_payload
    signal = {
        "evidence_for": [
            {"feature": "weak_sup", "description": "w", "value": 0.1, "strength_label": "weak"},
            {"feature": "strong_sup", "description": "s", "value": 0.8, "strength_label": "strong"},
        ],
        "evidence_against": [
            {"feature": "strong_opp", "description": "o", "value": -0.7, "strength_label": "strong"},
        ],
        "as_of": "2024-01-01",
    }
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 2)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        payload = build_explanation_payload(signal)

    directions = [d.direction for d in payload.top_drivers]
    # All supporters before all opposers
    last_supporter_idx = max(i for i, d in enumerate(directions) if d == "supports")
    first_opposer_idx = min(i for i, d in enumerate(directions) if d == "opposes")
    assert last_supporter_idx < first_opposer_idx
    # Strong supporter before weak supporter
    labels_for = [d.strength_label for d in payload.top_drivers if d.direction == "supports"]
    assert labels_for[0] == "strong"


def test_build_explanation_payload_capped_at_8_drivers():
    """top_drivers must not exceed 8 items."""
    from api.assistant.explanation import build_explanation_payload
    signal = {
        "evidence_for": [
            {"feature": f"f{i}", "description": "d", "value": 0.1, "strength_label": "weak"}
            for i in range(6)
        ],
        "evidence_against": [
            {"feature": f"a{i}", "description": "d", "value": -0.1, "strength_label": "weak"}
            for i in range(6)
        ],
        "as_of": "2024-01-01",
    }
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 2)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        payload = build_explanation_payload(signal)

    assert len(payload.top_drivers) <= 8


# ---------------------------------------------------------------------------
# 2. Staleness scoring
# ---------------------------------------------------------------------------

def test_staleness_fresh_for_same_day():
    """Signal generated today must have staleness_label='fresh' and age=0."""
    from api.assistant.explanation import compute_signal_staleness
    today = dt.date.today().isoformat()
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date.today()
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        result = compute_signal_staleness(today)
    assert result["signal_age_days"] == 0
    assert result["staleness_label"] == "fresh"
    assert result["staleness_score"] == 0.0


def test_staleness_aging_at_4_days():
    """Signal 4 days old must have staleness_label='aging'."""
    from api.assistant.explanation import compute_signal_staleness
    as_of = dt.date(2024, 1, 1).isoformat()
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 5)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        result = compute_signal_staleness(as_of)
    assert result["signal_age_days"] == 4
    assert result["staleness_label"] == "aging"


def test_staleness_stale_at_8_days():
    """Signal 8 days old must have staleness_label='stale'."""
    from api.assistant.explanation import compute_signal_staleness
    as_of = dt.date(2024, 1, 1).isoformat()
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 9)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        result = compute_signal_staleness(as_of)
    assert result["signal_age_days"] == 8
    assert result["staleness_label"] == "stale"


def test_staleness_expired_after_10_days():
    """Signal 15 days old must have staleness_label='expired' and score=100."""
    from api.assistant.explanation import compute_signal_staleness
    as_of = dt.date(2024, 1, 1).isoformat()
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 16)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        result = compute_signal_staleness(as_of)
    assert result["signal_age_days"] == 15
    assert result["staleness_label"] == "expired"
    assert result["staleness_score"] == 100.0


def test_staleness_none_for_missing_as_of():
    """compute_signal_staleness returns None fields when as_of is None."""
    from api.assistant.explanation import compute_signal_staleness
    result = compute_signal_staleness(None)
    assert result["signal_age_days"] is None
    assert result["staleness_label"] is None
    assert result["staleness_score"] is None


# ---------------------------------------------------------------------------
# 3. Staleness injected into external_payload
# ---------------------------------------------------------------------------

def test_signal_response_external_payload_has_staleness_fields():
    """external_payload must include signal_age_days, staleness_label, staleness_score."""
    from api.alpha.signal_runner import SignalResponse
    sig = SignalResponse(
        symbol="AAPL",
        as_of="2024-06-01",
        lookback=63,
        effective_lookback=63,
        regime="TRENDING",
        thresholds={},
        score=0.7,
        signal="BUY",
        confidence=0.7,
        features={"mom_21": 0.1, "rsi14": 62.0},
    )
    payload = sig.external_payload()
    assert "signal_age_days" in payload
    assert "staleness_label" in payload
    assert "staleness_score" in payload
    # staleness_label must be one of the valid values
    assert payload["staleness_label"] in ("fresh", "aging", "stale", "expired", None)


# ---------------------------------------------------------------------------
# 4. LLM narration overlay
# ---------------------------------------------------------------------------

def test_narration_payload_as_explanation_returns_explanation_payload():
    """NarrationPayload.as_explanation() must return an ExplanationPayload."""
    from api.assistant.narration import NarrationPayload
    from api.assistant.explanation import ExplanationPayload
    narration = NarrationPayload(
        headline="AAPL looks bullish",
        summary="Strong short-term momentum.",
        bullets=["RSI above 60", "Volume surge"],
        disclaimer="For research only.",
        followups=["What is the stop loss?"],
    )
    signal = {"evidence_for": [], "evidence_against": [], "reason_codes": [], "as_of": "2024-01-01"}
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 2)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        result = narration.as_explanation(signal)
    assert isinstance(result, ExplanationPayload)
    assert result.headline == "AAPL looks bullish"
    assert result.summary == "Strong short-term momentum."
    assert result.bullets == ["RSI above 60", "Volume surge"]
    assert result.disclaimer == "For research only."


def test_narration_overlay_preserves_deterministic_fields():
    """LLM overlay must not erase deterministic fields (reason_codes, staleness, etc.)."""
    from api.assistant.narration import NarrationPayload
    narration = NarrationPayload(
        headline="Bearish",
        summary="Declining momentum.",
        bullets=["RSI below 40"],
        disclaimer="Research only.",
        followups=[],
    )
    signal = {
        "evidence_for": [],
        "evidence_against": [
            {"feature": "mom_21", "description": "negative momentum",
             "value": -0.1, "strength_label": "moderate"}
        ],
        "reason_codes": ["LOW_MOMENTUM"],
        "as_of": "2024-01-01",
    }
    with patch("api.assistant.explanation.dt") as mock_dt:
        mock_dt.date.today.return_value = dt.date(2024, 1, 4)
        mock_dt.date.fromisoformat = dt.date.fromisoformat
        result = narration.as_explanation(signal)

    assert result.reason_codes == ["LOW_MOMENTUM"]
    assert len(result.top_drivers) == 1
    assert result.top_drivers[0].direction == "opposes"
    assert result.signal_age_days == 3
    assert result.staleness_label == "aging"
    # LLM fields also present
    assert result.headline == "Bearish"
