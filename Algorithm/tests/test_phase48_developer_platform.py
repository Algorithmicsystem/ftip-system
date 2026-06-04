"""Phase 19 tests: Developer API Platform."""
from __future__ import annotations

import json

import pytest

from api.developer.api_versioning import (
    V1_ENDPOINTS,
    get_api_documentation,
    get_endpoint_info,
    get_endpoints_by_category,
)
from api.developer.sdk_generator import (
    generate_javascript_sdk,
    generate_python_sdk,
    generate_r_sdk,
)
from api.developer.webhooks import (
    WEBHOOK_EVENTS,
    WebhookDelivery,
    WebhookSubscription,
    create_subscription,
    sign_webhook_payload,
)
from api.developer.usage_analytics import (
    UsageMetrics,
    compute_platform_analytics,
    compute_usage_metrics,
    detect_churn_risk,
)
from api.developer.partner_program import (
    PARTNER_TIERS,
    PartnerProfile,
    generate_white_label_config,
    register_partner,
)


# ===========================================================================
# TestAPIVersioning
# ===========================================================================

class TestAPIVersioning:
    def test_v1_endpoints_not_empty(self):
        assert len(V1_ENDPOINTS) >= 30

    def test_all_endpoints_have_required_fields(self):
        required = {"path", "method", "category", "tier_required",
                    "rate_limit_per_minute", "description", "response_schema", "deprecated"}
        for ep in V1_ENDPOINTS:
            for field in required:
                assert field in ep, f"Missing field '{field}' in endpoint {ep.get('path')}"

    def test_get_api_documentation_structure(self):
        doc = get_api_documentation()
        assert "version" in doc
        assert doc["version"] == "v1"
        assert "total_endpoints" in doc
        assert doc["total_endpoints"] == len(V1_ENDPOINTS)
        assert "categories" in doc
        assert "endpoints_by_category" in doc

    def test_documentation_active_endpoint_count(self):
        doc = get_api_documentation()
        assert doc["active_endpoints"] == doc["total_endpoints"] - doc["deprecated_endpoints"]

    def test_get_endpoint_info_existing(self):
        ep = get_endpoint_info("/prosperity/signal/{symbol}", "GET")
        assert ep["tier_required"] == "free"
        assert ep["category"] == "signals"

    def test_get_endpoint_info_missing(self):
        ep = get_endpoint_info("/nonexistent/path", "GET")
        assert ep == {}

    def test_get_endpoints_by_category_signals(self):
        eps = get_endpoints_by_category("signals")
        assert len(eps) >= 1
        for ep in eps:
            assert ep["category"] == "signals"

    def test_get_endpoints_by_category_competitive(self):
        eps = get_endpoints_by_category("competitive")
        assert len(eps) >= 2

    def test_get_endpoints_by_category_developer(self):
        eps = get_endpoints_by_category("developer")
        assert len(eps) >= 5

    def test_tier_distribution_covers_all_tiers(self):
        tiers = {ep["tier_required"] for ep in V1_ENDPOINTS}
        assert "free" in tiers
        assert "pro" in tiers
        assert "enterprise" in tiers

    def test_enterprise_endpoints_include_intelligence(self):
        intel_eps = [ep for ep in V1_ENDPOINTS
                     if ep["category"] == "intelligence" and ep["tier_required"] == "enterprise"]
        assert len(intel_eps) >= 1


# ===========================================================================
# TestSDKGenerator
# ===========================================================================

class TestSDKGenerator:
    def test_python_sdk_compiles(self):
        code = generate_python_sdk()
        compile(code, "<python_sdk>", "exec")

    def test_python_sdk_has_axiom_client(self):
        code = generate_python_sdk()
        assert "class AXIOMClient" in code

    def test_python_sdk_has_axiom_error(self):
        code = generate_python_sdk()
        assert "class AXIOMError" in code

    def test_python_sdk_core_methods_present(self):
        code = generate_python_sdk()
        for method in ("get_signal", "get_snapshot", "get_universe",
                        "get_sri", "get_macro_regime", "explain_signal",
                        "subscribe_webhook", "get_usage_metrics"):
            assert method in code, f"Missing method: {method}"

    def test_python_sdk_custom_base_url(self):
        code = generate_python_sdk(base_url="https://custom.example.com")
        assert "https://custom.example.com" in code

    def test_python_sdk_version_embedded(self):
        code = generate_python_sdk(version="v2")
        assert "v2" in code

    def test_javascript_sdk_has_axiom_client(self):
        code = generate_javascript_sdk()
        assert "class AXIOMClient" in code

    def test_javascript_sdk_has_async_methods(self):
        code = generate_javascript_sdk()
        assert "async _request" in code

    def test_javascript_sdk_has_core_methods(self):
        code = generate_javascript_sdk()
        for method in ("getSignal", "getSnapshot", "getSRI",
                        "getMacroRegime", "subscribeWebhook"):
            assert method in code, f"Missing JS method: {method}"

    def test_javascript_sdk_has_module_export(self):
        code = generate_javascript_sdk()
        assert "module.exports" in code

    def test_r_sdk_has_core_functions(self):
        code = generate_r_sdk()
        for fn in ("axiom_client", "axiom_get_signal", "axiom_get_snapshot",
                    "axiom_get_sri", "axiom_get_macro_regime"):
            assert fn in code, f"Missing R function: {fn}"

    def test_r_sdk_has_library_calls(self):
        code = generate_r_sdk()
        assert "library(httr)" in code
        assert "library(jsonlite)" in code


# ===========================================================================
# TestWebhookSystem
# ===========================================================================

class TestWebhookSystem:
    def test_webhook_events_defined(self):
        assert len(WEBHOOK_EVENTS) == 9

    def test_webhook_events_include_signal_types(self):
        assert "signal.buy" in WEBHOOK_EVENTS
        assert "signal.sell" in WEBHOOK_EVENTS
        assert "signal.regime_change" in WEBHOOK_EVENTS

    def test_webhook_events_include_risk_types(self):
        assert "risk.sri_alert" in WEBHOOK_EVENTS
        assert "risk.sornette_warning" in WEBHOOK_EVENTS

    def test_webhook_events_include_system_types(self):
        assert "system.health_degraded" in WEBHOOK_EVENTS
        assert "ml.model_updated" in WEBHOOK_EVENTS

    def test_sign_webhook_payload_deterministic(self):
        payload = {"symbol": "AAPL", "signal": "BUY", "dau": 80.0}
        sig1 = sign_webhook_payload(payload, "secret123")
        sig2 = sign_webhook_payload(payload, "secret123")
        assert sig1 == sig2

    def test_sign_webhook_payload_different_secrets(self):
        payload = {"event": "signal.buy"}
        sig1 = sign_webhook_payload(payload, "secret_a")
        sig2 = sign_webhook_payload(payload, "secret_b")
        assert sig1 != sig2

    def test_sign_webhook_payload_is_hex_string(self):
        sig = sign_webhook_payload({"x": 1}, "secret")
        assert len(sig) == 64
        int(sig, 16)  # Should not raise

    def test_create_subscription_returns_correct_type(self):
        sub = create_subscription(
            tenant_id="tenant_001",
            event_type="signal.buy",
            callback_url="https://example.com/webhook",
            secret="test_secret",
        )
        assert isinstance(sub, WebhookSubscription)
        assert sub.event_type == "signal.buy"
        assert sub.tenant_id == "tenant_001"
        assert sub.is_active is True
        assert len(sub.subscription_id) > 0

    def test_create_subscription_invalid_event_raises(self):
        with pytest.raises(ValueError, match="Unknown event_type"):
            create_subscription(
                tenant_id="t1",
                event_type="invalid.event",
                callback_url="https://example.com",
                secret="s",
            )

    def test_create_subscription_with_filter(self):
        sub = create_subscription(
            tenant_id="t1",
            event_type="signal.buy",
            callback_url="https://example.com",
            secret="s",
            filter_config={"min_dau": 70},
        )
        assert sub.filter == {"min_dau": 70}

    def test_webhook_delivery_dataclass_fields(self):
        import datetime as dt
        delivery = WebhookDelivery(
            delivery_id="d1",
            subscription_id="s1",
            event_type="signal.buy",
            payload={"test": True},
            delivered_at=dt.datetime.utcnow(),
            status="delivered",
            http_status_code=200,
            retry_count=0,
        )
        assert delivery.status == "delivered"
        assert delivery.http_status_code == 200


# ===========================================================================
# TestUsageAnalytics
# ===========================================================================

class TestUsageAnalytics:
    def test_compute_usage_metrics_returns_correct_type(self):
        metrics = compute_usage_metrics("tenant_001", "7d")
        assert isinstance(metrics, UsageMetrics)

    def test_compute_usage_metrics_no_db(self):
        metrics = compute_usage_metrics("tenant_001", "7d")
        assert metrics.tenant_id == "tenant_001"
        assert metrics.period == "7d"
        assert isinstance(metrics.total_requests, int)

    def test_compute_usage_metrics_all_periods(self):
        for period in ("1d", "7d", "30d", "90d"):
            m = compute_usage_metrics("t1", period)
            assert m.period == period

    def test_compute_platform_analytics_structure(self):
        result = compute_platform_analytics("7d")
        assert "period" in result
        assert "total_requests" in result
        assert "active_tenants" in result
        assert "top_endpoints" in result
        assert "top_symbols" in result
        assert "error_rate" in result

    def test_compute_platform_analytics_period_matches(self):
        result = compute_platform_analytics("30d")
        assert result["period"] == "30d"

    def test_detect_churn_risk_structure(self):
        result = detect_churn_risk("tenant_001")
        assert "tenant_id" in result
        assert "churn_risk" in result
        assert "pct_change" in result

    def test_detect_churn_risk_tenant_id_preserved(self):
        result = detect_churn_risk("my_tenant")
        assert result["tenant_id"] == "my_tenant"

    def test_detect_churn_risk_valid_labels(self):
        result = detect_churn_risk("tenant_001")
        assert result["churn_risk"] in ("low", "medium", "high", "unknown")

    def test_usage_metrics_dataclass_defaults(self):
        m = UsageMetrics(tenant_id="t1", period="7d")
        assert m.total_requests == 0
        assert m.error_rate == 0.0
        assert m.most_used_symbols == []
        assert m.requests_by_endpoint == {}


# ===========================================================================
# TestPartnerProgram
# ===========================================================================

class TestPartnerProgram:
    def test_partner_tiers_defined(self):
        assert len(PARTNER_TIERS) == 4
        assert "reseller" in PARTNER_TIERS
        assert "white_label" in PARTNER_TIERS
        assert "academic" in PARTNER_TIERS
        assert "integration_partner" in PARTNER_TIERS

    def test_reseller_revenue_share(self):
        assert PARTNER_TIERS["reseller"]["revenue_share_pct"] == 20.0

    def test_white_label_revenue_share(self):
        assert PARTNER_TIERS["white_label"]["revenue_share_pct"] == 30.0

    def test_academic_no_revenue_share(self):
        assert PARTNER_TIERS["academic"]["revenue_share_pct"] == 0.0

    def test_register_partner_returns_profile(self):
        partner = register_partner("Acme Corp", "reseller", "contact@acme.com")
        assert isinstance(partner, PartnerProfile)
        assert partner.org_name == "Acme Corp"
        assert partner.partner_tier == "reseller"
        assert partner.contact_email == "contact@acme.com"

    def test_register_partner_id_generated(self):
        partner = register_partner("Org", "academic", "email@org.com")
        assert len(partner.partner_id) > 0

    def test_register_partner_sets_revenue_share(self):
        partner = register_partner("WL Corp", "white_label", "wl@corp.com")
        assert partner.revenue_share_pct == 30.0

    def test_register_partner_sets_api_prefix(self):
        partner = register_partner("RS Corp", "reseller", "rs@corp.com")
        assert partner.api_key_prefix == "ax_rs_"

    def test_register_partner_invalid_tier_raises(self):
        with pytest.raises(ValueError, match="Unknown partner_tier"):
            register_partner("Bad Corp", "unknown_tier", "bad@corp.com")

    def test_generate_white_label_config_structure(self):
        partner = register_partner("Brand Co", "white_label", "brand@co.com")
        config = generate_white_label_config(partner, {
            "logo_url": "https://brand.co/logo.png",
            "primary_color": "#FF5733",
            "product_name": "Brand Intelligence",
        })
        assert "partner_id" in config
        assert "branding" in config
        assert config["branding"]["primary_color"] == "#FF5733"
        assert config["branding"]["product_name"] == "Brand Intelligence"
        assert config["revenue_share_pct"] == 30.0

    def test_generate_white_label_config_defaults(self):
        partner = register_partner("Min Corp", "white_label", "m@corp.com")
        config = generate_white_label_config(partner, {})
        assert config["branding"]["primary_color"] == "#000000"
        assert config["branding"]["secondary_color"] == "#FFFFFF"

    def test_partner_profile_dataclass_fields(self):
        partner = PartnerProfile(
            partner_id="pid",
            org_name="Test Org",
            partner_tier="academic",
            contact_email="test@org.com",
        )
        assert partner.agreement_signed is False
        assert partner.revenue_share_pct == 0.0
        assert partner.custom_branding == {}
        assert partner.endpoints_allowed == []
