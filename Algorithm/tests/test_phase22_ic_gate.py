"""Phase 22: IC Auto-Gate tests."""
from __future__ import annotations

import datetime as dt
import pytest

from api.jobs.ic_gate import load_ic_gate_state, apply_ic_gate, MIN_IC_SAMPLES


# ---------------------------------------------------------------------------
# load_ic_gate_state tests
# ---------------------------------------------------------------------------

def test_gate_open_for_strong_ic(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(
        mod.db, "safe_fetchone",
        lambda *_a, **_k: ("STRONG", 50, 0.08),
    )
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is False
    assert gate["confidence_mult"] == 1.0
    assert gate["ic_state"] == "STRONG"


def test_gate_open_for_moderate_ic(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: ("MODERATE", 30, 0.04))
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is False
    assert gate["confidence_mult"] == 1.0


def test_gate_active_for_weak_ic(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: ("WEAK", 30, 0.015))
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is True
    assert gate["confidence_mult"] == 0.70
    assert gate["gate_note"] == "ic_quality_gate_weak"


def test_gate_active_for_degraded_ic(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: ("DEGRADED", 20, -0.02))
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is True
    assert gate["confidence_mult"] == 0.40


def test_gate_active_for_insufficient_ic_with_low_sample_count(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    # sample_count below MIN_IC_SAMPLES threshold
    monkeypatch.setattr(
        mod.db, "safe_fetchone",
        lambda *_a, **_k: ("INSUFFICIENT", MIN_IC_SAMPLES - 1, None),
    )
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is True
    assert gate["confidence_mult"] == 0.85


def test_gate_open_for_insufficient_ic_with_sufficient_samples(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(
        mod.db, "safe_fetchone",
        lambda *_a, **_k: ("INSUFFICIENT", MIN_IC_SAMPLES + 5, None),
    )
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is False
    assert gate["confidence_mult"] == 1.0


def test_gate_open_when_db_disabled(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is False


def test_gate_open_when_no_ic_row(monkeypatch):
    import api.jobs.ic_gate as mod
    monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
    monkeypatch.setattr(mod.db, "safe_fetchone", lambda *_a, **_k: None)
    gate = load_ic_gate_state(dt.date(2026, 1, 5))
    assert gate["gate_active"] is False


# ---------------------------------------------------------------------------
# apply_ic_gate tests
# ---------------------------------------------------------------------------

def test_apply_gate_reduces_confidence(monkeypatch):
    gate = {
        "gate_active": True,
        "confidence_mult": 0.70,
        "ic_state": "WEAK",
        "gate_note": "ic_quality_gate_weak",
    }
    signal = {"action": "BUY", "confidence": 80.0, "signal_meta": {}}
    result = apply_ic_gate(signal, gate)
    assert result["confidence"] == pytest.approx(56.0, abs=0.01)
    assert result["signal_meta"]["quality_gate_applied"] is True
    assert result["signal_meta"]["ic_gate_state"] == "WEAK"
    assert result["signal_meta"]["ic_gate_original_confidence"] == 80.0


def test_apply_gate_does_not_mutate_original():
    gate = {
        "gate_active": True,
        "confidence_mult": 0.40,
        "ic_state": "DEGRADED",
        "gate_note": "ic_quality_gate_degraded",
    }
    original = {"action": "BUY", "confidence": 75.0, "signal_meta": {"foo": "bar"}}
    result = apply_ic_gate(original, gate)
    assert original["confidence"] == 75.0
    assert "quality_gate_applied" not in original["signal_meta"]
    assert result is not original


def test_apply_gate_passthrough_when_inactive():
    gate = {"gate_active": False, "confidence_mult": 1.0, "ic_state": "STRONG", "gate_note": ""}
    signal = {"action": "BUY", "confidence": 80.0, "signal_meta": {}}
    result = apply_ic_gate(signal, gate)
    assert result is signal   # same object, no copy needed
    assert result["confidence"] == 80.0


def test_apply_gate_handles_missing_signal_meta():
    gate = {
        "gate_active": True,
        "confidence_mult": 0.70,
        "ic_state": "WEAK",
        "gate_note": "ic_quality_gate_weak",
    }
    signal = {"action": "HOLD", "confidence": 50.0}
    result = apply_ic_gate(signal, gate)
    assert result["signal_meta"]["quality_gate_applied"] is True
