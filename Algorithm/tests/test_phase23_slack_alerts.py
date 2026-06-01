"""Phase 23: Slack Alert Format tests."""
from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from api.jobs.alerts import format_slack_blocks, run_alert_scan, _load_active_rules


# ---------------------------------------------------------------------------
# format_slack_blocks tests
# ---------------------------------------------------------------------------

def _event(signal="BUY", conviction=72.5):
    return {
        "symbol": "NVDA",
        "signal": signal,
        "dau": 88.5,
        "regime": "fundamental_convergence",
        "breadth_state": "EXPANDING",
        "ic_state": "STRONG",
        "conviction_score": conviction,
        "as_of_date": "2026-01-05",
        "rule_id": "rule-abc",
    }


def test_slack_blocks_top_level_keys():
    payload = format_slack_blocks(_event())
    assert "text" in payload
    assert "blocks" in payload
    assert isinstance(payload["blocks"], list)
    assert len(payload["blocks"]) >= 2


def test_slack_blocks_header_contains_symbol_and_signal():
    payload = format_slack_blocks(_event("SELL"))
    header_block = payload["blocks"][0]
    assert header_block["type"] == "header"
    text = header_block["text"]["text"]
    assert "NVDA" in text
    assert "SELL" in text


def test_slack_blocks_fields_contain_expected_values():
    payload = format_slack_blocks(_event("BUY", conviction=65.0))
    section = next(b for b in payload["blocks"] if b["type"] == "section")
    field_texts = " ".join(f["text"] for f in section["fields"])
    assert "NVDA" in field_texts
    assert "BUY" in field_texts
    assert "STRONG" in field_texts
    assert "88" in field_texts  # DAU


def test_slack_blocks_text_fallback_includes_conviction():
    payload = format_slack_blocks(_event(conviction=55.3))
    assert "55.3" in payload["text"]


def test_slack_blocks_sell_signal_has_red_emoji():
    payload = format_slack_blocks(_event("SELL"))
    assert ":red_circle:" in payload["text"]


def test_slack_blocks_buy_signal_has_green_emoji():
    payload = format_slack_blocks(_event("BUY"))
    assert ":large_green_circle:" in payload["text"]


def test_slack_blocks_hold_signal_has_yellow_emoji():
    payload = format_slack_blocks(_event("HOLD"))
    assert ":large_yellow_circle:" in payload["text"]


# ---------------------------------------------------------------------------
# run_alert_scan channel_type routing tests
# ---------------------------------------------------------------------------

def _make_rule(channel_type="generic", webhook_url="https://hooks.example.com"):
    return {
        "rule_id": "r1",
        "symbol": "AAPL",
        "min_dau": 50.0,
        "signal_filter": [],
        "favorable_regimes": None,
        "require_breadth_alignment": False,
        "min_conviction_score": 0.0,
        "webhook_url": webhook_url,
        "meta": {},
        "channel_type": channel_type,
    }


def _patch_scan(monkeypatch, rule):
    import api.jobs.alerts as mod
    monkeypatch.setattr(mod, "_load_active_rules", lambda: [rule])
    monkeypatch.setattr(mod, "_load_axiom_scores_batch", lambda *_a: {
        "AAPL": {
            "dau": 80.0,
            "regime_label": "fundamental_convergence",
            "deployability_tier": "live_candidate",
            "signal_label": "BUY",
            "confidence": 70.0,
        }
    })
    monkeypatch.setattr(mod, "_load_breadth_state", lambda _d: "EXPANDING")
    monkeypatch.setattr(mod, "_load_ic_state", lambda _d: "STRONG")
    monkeypatch.setattr(mod, "_already_fired", lambda *_a: False)
    monkeypatch.setattr(mod, "_store_event", lambda _e: None)


def test_generic_channel_sends_plain_payload(monkeypatch):
    import api.jobs.alerts as mod
    rule = _make_rule("generic")
    _patch_scan(monkeypatch, rule)

    captured = {}

    def fake_deliver(url, payload):
        captured["payload"] = payload
        return 200

    monkeypatch.setattr(mod, "deliver_webhook", fake_deliver)
    run_alert_scan(dt.date(2026, 1, 5))

    assert "blocks" not in captured.get("payload", {})
    assert "symbol" in captured["payload"]


def test_slack_channel_sends_blocks_payload(monkeypatch):
    import api.jobs.alerts as mod
    rule = _make_rule("slack")
    _patch_scan(monkeypatch, rule)

    captured = {}

    def fake_deliver(url, payload):
        captured["payload"] = payload
        return 200

    monkeypatch.setattr(mod, "deliver_webhook", fake_deliver)
    run_alert_scan(dt.date(2026, 1, 5))

    assert "blocks" in captured.get("payload", {})
    assert "text" in captured["payload"]


def test_no_webhook_url_skips_delivery(monkeypatch):
    import api.jobs.alerts as mod
    rule = _make_rule("slack", webhook_url=None)
    _patch_scan(monkeypatch, rule)

    delivered = []
    monkeypatch.setattr(mod, "deliver_webhook", lambda u, p: delivered.append(p) or 200)
    run_alert_scan(dt.date(2026, 1, 5))

    assert delivered == []
