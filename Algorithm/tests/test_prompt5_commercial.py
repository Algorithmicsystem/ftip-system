"""Prompt 5 tests: Competitive intelligence, cross-asset amplifier, billing, revenue share, developer platform."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api.main import app

WEBAPP = Path(__file__).resolve().parents[1] / "api" / "webapp"


# ---------------------------------------------------------------------------
# SYSTEM 1: Competitive Intelligence
# ---------------------------------------------------------------------------

class TestAxiomSectorGroups:

    def test_sector_groups_exist(self):
        from api.competitive.competitive_intelligence import AXIOM_SECTOR_GROUPS
        assert isinstance(AXIOM_SECTOR_GROUPS, dict)
        assert len(AXIOM_SECTOR_GROUPS) >= 5

    def test_technology_sector_has_aapl(self):
        from api.competitive.competitive_intelligence import AXIOM_SECTOR_GROUPS
        assert "AAPL" in AXIOM_SECTOR_GROUPS["Technology"]

    def test_healthcare_sector_has_jnj(self):
        from api.competitive.competitive_intelligence import AXIOM_SECTOR_GROUPS
        assert "JNJ" in AXIOM_SECTOR_GROUPS["Healthcare"]

    def test_all_30_universe_symbols_covered(self):
        from api.competitive.competitive_intelligence import AXIOM_SECTOR_GROUPS, _SYMBOL_TO_SECTOR
        from api.universe import AXIOM_UNIVERSE
        for sym in AXIOM_UNIVERSE:
            assert sym in _SYMBOL_TO_SECTOR, f"{sym} not in any sector group"

    def test_get_sector_for_symbol(self):
        from api.competitive.competitive_intelligence import get_sector_for_symbol
        assert get_sector_for_symbol("AAPL") == "Technology"
        assert get_sector_for_symbol("JPM") == "Financials"
        assert get_sector_for_symbol("UNKNOWN_XYZ") == "unknown"

    def test_get_static_competitors_no_db(self):
        from api.competitive.competitive_intelligence import get_static_competitors
        peers = get_static_competitors("AAPL")
        assert len(peers) >= 2
        assert "AAPL" not in peers

    def test_get_static_competitors_msft(self):
        from api.competitive.competitive_intelligence import get_static_competitors
        peers = get_static_competitors("MSFT")
        assert "AAPL" in peers

    def test_identify_competitors_returns_without_db(self):
        from api.competitive.competitive_intelligence import identify_competitors
        import datetime as dt
        peers = identify_competitors("AAPL", "Technology", dt.date.today())
        assert len(peers) >= 2

    def test_identify_competitors_excludes_self(self):
        from api.competitive.competitive_intelligence import identify_competitors
        import datetime as dt
        peers = identify_competitors("AAPL", "Technology", dt.date.today())
        assert "AAPL" not in peers

    def test_generate_report_no_db_returns_sector(self):
        from api.competitive.competitive_intelligence import generate_competitive_intelligence_report
        report = generate_competitive_intelligence_report("AAPL")
        assert report.sector == "Technology"
        assert report.competitor_count >= 2

    def test_generate_report_no_db_has_profiles(self):
        from api.competitive.competitive_intelligence import generate_competitive_intelligence_report
        report = generate_competitive_intelligence_report("AAPL")
        assert len(report.competitors) >= 2

    def test_sector_ranking_endpoint_exists(self):
        with TestClient(app) as client:
            r = client.get("/competitive/sector/Technology/ranking")
        assert r.status_code == 200

    def test_sector_ranking_has_rankings_list(self):
        with TestClient(app) as client:
            r = client.get("/competitive/sector/Technology/ranking")
        data = r.json()
        assert "rankings" in data
        assert len(data["rankings"]) >= 2

    def test_sector_ranking_sorted_descending(self):
        with TestClient(app) as client:
            r = client.get("/competitive/sector/Technology/ranking")
        data = r.json()
        rankings = data["rankings"]
        daUs = [x["dau"] for x in rankings]
        assert daUs == sorted(daUs, reverse=True)

    def test_sector_ranking_has_rank_field(self):
        with TestClient(app) as client:
            r = client.get("/competitive/sector/Technology/ranking")
        data = r.json()
        assert data["rankings"][0]["rank"] == 1

    def test_head_to_head_has_narrative(self):
        with TestClient(app) as client:
            r = client.get("/competitive/AAPL/vs/MSFT")
        assert r.status_code == 200
        data = r.json()
        assert "head_to_head_narrative" in data
        assert isinstance(data["head_to_head_narrative"], str)
        assert len(data["head_to_head_narrative"]) > 10

    def test_head_to_head_narrative_mentions_symbols(self):
        with TestClient(app) as client:
            r = client.get("/competitive/AAPL/vs/MSFT")
        data = r.json()
        narrative = data["head_to_head_narrative"]
        assert "AAPL" in narrative or "MSFT" in narrative


# ---------------------------------------------------------------------------
# SYSTEM 2: Cross-Asset Intelligence
# ---------------------------------------------------------------------------

class TestCrossAssetEngine:

    def test_amplifier_max_positive_is_015(self):
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        snap = compute_cross_asset_snapshot({}, "TRENDING",
                                            vix_level=12.0, yield_curve_slope=2.0,
                                            copper_return_90d=0.10, dxy_return_30d=-0.03)
        assert snap.equity_signal_amplifier <= 0.15, \
            f"Amplifier {snap.equity_signal_amplifier} exceeds +0.15 cap"

    def test_amplifier_max_negative_is_030(self):
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        snap = compute_cross_asset_snapshot({}, "CHOPPY",
                                            vix_level=40.0, yield_curve_slope=-0.5,
                                            copper_return_90d=-0.10, dxy_return_30d=0.04)
        assert snap.equity_signal_amplifier >= -0.30, \
            f"Amplifier {snap.equity_signal_amplifier} below -0.30 floor"

    def test_amplifier_range_consistent(self):
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        for vix in [10, 20, 30, 40]:
            for slope in [-1.0, 0.0, 1.5]:
                snap = compute_cross_asset_snapshot({}, "TRENDING", vix_level=float(vix),
                                                    yield_curve_slope=slope)
                assert -0.30 <= snap.equity_signal_amplifier <= 0.15

    def test_macro_narrative_is_string(self):
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        snap = compute_cross_asset_snapshot({}, "TRENDING")
        assert isinstance(snap.macro_narrative, str)
        assert len(snap.macro_narrative) > 10

    def test_equity_regime_confirmed_field(self):
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        snap = compute_cross_asset_snapshot({}, "TRENDING")
        assert isinstance(snap.equity_regime_confirmed, bool)

    def test_macro_headwind_tailwind_sum_100(self):
        from api.macro.cross_asset_engine import compute_cross_asset_snapshot
        snap = compute_cross_asset_snapshot({}, "TRENDING")
        assert abs(snap.macro_headwind_score + snap.macro_tailwind_score - 100.0) < 0.01

    def test_equity_implications_endpoint(self):
        with TestClient(app) as client:
            r = client.get("/macro/equity-implications/AAPL")
        assert r.status_code == 200

    def test_equity_implications_has_amplifier(self):
        with TestClient(app) as client:
            r = client.get("/macro/equity-implications/AAPL")
        data = r.json()
        assert "amplifier" in data

    def test_universal_intelligence_has_cross_asset_fields(self):
        with TestClient(app) as client:
            r = client.get("/intelligence/universal/AAPL")
        assert r.status_code == 200
        data = r.json()
        assert "cross_asset_amplifier" in data
        assert "cross_asset_adjusted_dau" in data
        assert "macro_narrative" in data


# ---------------------------------------------------------------------------
# SYSTEM 3A: Billing
# ---------------------------------------------------------------------------

class TestBillingTiers:

    def test_billing_tiers_exist(self):
        from api.developer.billing import BILLING_TIERS
        assert "free" in BILLING_TIERS
        assert "starter" in BILLING_TIERS
        assert "professional" in BILLING_TIERS
        assert "institutional" in BILLING_TIERS

    def test_free_tier_1k_calls(self):
        from api.developer.billing import BILLING_TIERS
        assert BILLING_TIERS["free"]["monthly_calls"] == 1_000

    def test_starter_tier_price(self):
        from api.developer.billing import BILLING_TIERS
        assert BILLING_TIERS["starter"]["price_usd"] == 199

    def test_professional_tier_price(self):
        from api.developer.billing import BILLING_TIERS
        assert BILLING_TIERS["professional"]["price_usd"] == 999

    def test_institutional_unlimited(self):
        from api.developer.billing import BILLING_TIERS
        assert BILLING_TIERS["institutional"]["monthly_calls"] is None

    def test_institutional_1k_rpm(self):
        from api.developer.billing import BILLING_TIERS
        assert BILLING_TIERS["institutional"]["rpm"] == 1_000

    def test_quota_enforcer_rate_limit_allows_under_limit(self):
        from api.developer.billing import QuotaEnforcer
        qe = QuotaEnforcer()
        allowed, _ = qe.check_rate_limit("test-key", "free")
        assert allowed is True

    def test_quota_enforcer_blocks_above_rpm(self):
        from api.developer.billing import QuotaEnforcer
        qe = QuotaEnforcer()
        for _ in range(10):
            qe.check_rate_limit("test-key-2", "free")
        allowed, retry = qe.check_rate_limit("test-key-2", "free")
        assert allowed is False
        assert retry is not None

    def test_quota_enforcer_monthly_quota(self):
        from api.developer.billing import QuotaEnforcer
        qe = QuotaEnforcer()
        for _ in range(1_000):
            qe.check_monthly_quota("heavy-user", "free")
        allowed, remaining = qe.check_monthly_quota("heavy-user", "free")
        assert allowed is False
        assert remaining == 0

    def test_quota_enforcer_unlimited_institutional(self):
        from api.developer.billing import QuotaEnforcer
        qe = QuotaEnforcer()
        for _ in range(10_000):
            qe.check_monthly_quota("institutional-key", "institutional")
        allowed, remaining = qe.check_monthly_quota("institutional-key", "institutional")
        assert allowed is True
        assert remaining == -1

    def test_billing_tiers_endpoint(self):
        with TestClient(app) as client:
            r = client.get("/developer/billing/tiers")
        assert r.status_code == 200
        data = r.json()
        assert "tiers" in data
        assert "free" in data["tiers"]


# ---------------------------------------------------------------------------
# SYSTEM 3B: Revenue Share
# ---------------------------------------------------------------------------

class TestRevenueShare:

    def test_partner_revenue_share_dict_exists(self):
        from api.developer.revenue_share import PARTNER_REVENUE_SHARE
        assert "referral" in PARTNER_REVENUE_SHARE
        assert "reseller" in PARTNER_REVENUE_SHARE
        assert "oem" in PARTNER_REVENUE_SHARE
        assert "white_label" in PARTNER_REVENUE_SHARE

    def test_referral_10pct(self):
        from api.developer.revenue_share import PARTNER_REVENUE_SHARE
        assert PARTNER_REVENUE_SHARE["referral"]["revenue_share_pct"] == 10.0

    def test_reseller_25pct(self):
        from api.developer.revenue_share import PARTNER_REVENUE_SHARE
        assert PARTNER_REVENUE_SHARE["reseller"]["revenue_share_pct"] == 25.0

    def test_oem_30pct(self):
        from api.developer.revenue_share import PARTNER_REVENUE_SHARE
        assert PARTNER_REVENUE_SHARE["oem"]["revenue_share_pct"] == 30.0

    def test_white_label_40pct(self):
        from api.developer.revenue_share import PARTNER_REVENUE_SHARE
        assert PARTNER_REVENUE_SHARE["white_label"]["revenue_share_pct"] == 40.0

    def test_compute_revenue_share_math(self):
        from api.developer.revenue_share import compute_partner_revenue_share
        result = compute_partner_revenue_share("p1", "reseller", gross_revenue_usd=1_000.0)
        assert result["revenue_share_usd"] == pytest.approx(250.0)

    def test_compute_revenue_share_has_payout_eligible(self):
        from api.developer.revenue_share import compute_partner_revenue_share
        result = compute_partner_revenue_share("p2", "white_label", gross_revenue_usd=2_000.0)
        assert "payout_eligible" in result
        assert result["payout_eligible"] is True

    def test_revenue_share_endpoint_referral(self):
        with TestClient(app) as client:
            r = client.get("/developer/revenue-share/referral")
        assert r.status_code == 200
        data = r.json()
        assert data["revenue_share_pct"] == 10.0

    def test_revenue_share_endpoint_invalid(self):
        with TestClient(app) as client:
            r = client.get("/developer/revenue-share/nonexistent_tier")
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Developer API endpoints
# ---------------------------------------------------------------------------

class TestDeveloperEndpoints:

    def test_keys_endpoint_returns_200(self):
        with TestClient(app) as client:
            r = client.get("/developer/keys")
        assert r.status_code == 200

    def test_keys_endpoint_has_key_count(self):
        with TestClient(app) as client:
            r = client.get("/developer/keys")
        data = r.json()
        assert "key_count" in data
        assert "keys" in data

    def test_keys_rotate_returns_new_key_id(self):
        with TestClient(app) as client:
            r = client.post("/developer/keys/rotate")
        assert r.status_code == 200
        data = r.json()
        assert "new_key_id" in data
        assert data["new_key_id"].startswith("ax_")

    def test_billing_usage_endpoint(self):
        with TestClient(app) as client:
            r = client.get("/developer/billing/usage")
        assert r.status_code == 200
        data = r.json()
        assert "tier" in data
        assert "rpm" in data

    def test_webhook_test_endpoint_valid_event(self):
        with TestClient(app) as client:
            r = client.post("/developer/webhooks/test?event_type=signal.buy")
        assert r.status_code == 200
        data = r.json()
        assert data["event_type"] == "signal.buy"
        assert "deliveries_attempted" in data

    def test_webhook_test_endpoint_invalid_event(self):
        with TestClient(app) as client:
            r = client.post("/developer/webhooks/test?event_type=invalid.event")
        assert r.status_code == 400

    def test_webhook_events_includes_signal_buy(self):
        from api.developer.webhooks import WEBHOOK_EVENTS
        assert "signal.buy" in WEBHOOK_EVENTS

    def test_webhook_events_includes_risk_sri_alert(self):
        from api.developer.webhooks import WEBHOOK_EVENTS
        assert "risk.sri_alert" in WEBHOOK_EVENTS

    def test_developer_dashboard_page_serves(self):
        with TestClient(app) as client:
            r = client.get("/app/developer")
        assert r.status_code == 200

    def test_version_is_30(self):
        with TestClient(app) as client:
            r = client.get("/config/client")
        assert r.status_code == 200
        assert r.json()["version"] in ("30.0.0", "31.0.0", "32.0.0", "33.0.0")


# ---------------------------------------------------------------------------
# Frontend checks
# ---------------------------------------------------------------------------

class TestFrontendPrompt5:

    def test_peers_tab_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "tab-peers" in html

    def test_peers_tab_nav_item_in_html(self):
        html = (WEBAPP / "index.html").read_text()
        assert "Peers" in html

    def test_load_peers_tab_function_in_js(self):
        src = (WEBAPP / "js" / "panels" / "symbol_intelligence.js").read_text()
        assert "loadPeersTab" in src

    def test_factor_env_shows_amplifier(self):
        src = (WEBAPP / "js" / "panels" / "factor_environment.js").read_text()
        assert "equity_signal_amplifier" in src
        assert "Signal Amplifier" in src

    def test_developer_link_in_sidebar(self):
        html = (WEBAPP / "index.html").read_text()
        assert "Developer" in html
        assert "/app/developer" in html

    def test_developer_html_exists(self):
        assert (WEBAPP / "pages" / "developer.html").exists()

    def test_developer_html_has_tier_grid(self):
        html = (WEBAPP / "pages" / "developer.html").read_text()
        assert "tier-grid" in html

    def test_developer_html_has_partner_grid(self):
        html = (WEBAPP / "pages" / "developer.html").read_text()
        assert "partner-grid" in html
