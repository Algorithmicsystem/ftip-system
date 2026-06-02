"""Phase 35: Commercial Readiness tests.

Covers:
  6.1  Tier definitions and limits
  6.2  API key hashing
  6.3  register_tenant / resolve_tenant
  6.4  check_sector_access
  6.5  tier_has_access ordering
  6.6  require_tier FastAPI dependency
  6.7  GET /auth/tenant endpoint
  6.8  GET /api-docs endpoint
"""
from __future__ import annotations

import datetime as dt
import hashlib
import pytest


# ---------------------------------------------------------------------------
# 6.1 — Tier definitions
# ---------------------------------------------------------------------------

class TestTierLimits:
    def test_free_tier_has_lower_rpm_than_pro(self):
        from api.jobs.tenant_auth import get_tier_limits
        assert get_tier_limits("free")["rpm"] < get_tier_limits("pro")["rpm"]

    def test_enterprise_tier_has_unlimited_rpm(self):
        from api.jobs.tenant_auth import get_tier_limits
        assert get_tier_limits("enterprise")["rpm"] == 0

    def test_free_tier_missing_pe_prefix(self):
        from api.jobs.tenant_auth import get_tier_limits
        prefixes = get_tier_limits("free")["allowed_prefixes"]
        assert "/pe" not in prefixes
        assert "/smb" not in prefixes

    def test_pro_tier_missing_pe_prefix(self):
        from api.jobs.tenant_auth import get_tier_limits
        prefixes = get_tier_limits("pro")["allowed_prefixes"]
        assert "/pe" not in prefixes
        assert "/smb" not in prefixes

    def test_enterprise_tier_includes_all_prefixes(self):
        from api.jobs.tenant_auth import get_tier_limits
        prefixes = get_tier_limits("enterprise")["allowed_prefixes"]
        for p in ["/prosperity", "/axiom", "/linkage", "/ops", "/pe", "/smb"]:
            assert p in prefixes

    def test_unknown_tier_falls_back_to_free(self):
        from api.jobs.tenant_auth import get_tier_limits
        limits = get_tier_limits("unknown_tier")
        assert limits["rpm"] == get_tier_limits("free")["rpm"]


# ---------------------------------------------------------------------------
# 6.2 — API key hashing
# ---------------------------------------------------------------------------

class TestKeyHashing:
    def test_hash_is_sha256_hex(self):
        from api.jobs.tenant_auth import _hash_key
        raw = "test-key-abc-123"
        result = _hash_key(raw)
        expected = hashlib.sha256(raw.encode()).hexdigest()
        assert result == expected

    def test_hash_is_64_chars(self):
        from api.jobs.tenant_auth import _hash_key
        assert len(_hash_key("any-key")) == 64

    def test_different_keys_have_different_hashes(self):
        from api.jobs.tenant_auth import _hash_key
        assert _hash_key("key-A") != _hash_key("key-B")

    def test_same_key_always_same_hash(self):
        from api.jobs.tenant_auth import _hash_key
        key = "stable-key-xyz"
        assert _hash_key(key) == _hash_key(key)


# ---------------------------------------------------------------------------
# 6.3 — register_tenant / resolve_tenant
# ---------------------------------------------------------------------------

class TestRegisterResolveTenant:
    def test_register_returns_false_when_write_disabled(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: False)
        result = mod.register_tenant("T001", "Acme Fund", "raw-key-123")
        assert result is False

    def test_register_returns_true_on_success(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        result = mod.register_tenant("T001", "Acme Fund", "raw-key-123", tier="pro")
        assert result is True
        assert executed[0][0] == "T001"
        assert executed[0][2] == mod._hash_key("raw-key-123")
        assert executed[0][3] == "pro"

    def test_register_stores_sector_restriction(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        executed = []
        monkeypatch.setattr(mod.db, "db_write_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_execute", lambda sql, params: executed.append(params))
        mod.register_tenant("T001", "Acme", "key", allowed_sectors=["Technology", "Healthcare"])
        import json
        sectors = json.loads(executed[0][4])
        assert "Technology" in sectors

    def test_resolve_returns_none_when_db_disabled(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        result = mod.resolve_tenant("any-key")
        assert result is None

    def test_resolve_returns_none_for_unknown_key(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: None)
        result = mod.resolve_tenant("unknown-key")
        assert result is None

    def test_resolve_returns_tenant_for_valid_key(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        row = ("T001", "Acme Fund", "pro", ["Technology"], 120, None)
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: row)
        result = mod.resolve_tenant("valid-key")
        assert result is not None
        assert result["tenant_id"] == "T001"
        assert result["tier"] == "pro"
        assert result["allowed_sectors"] == ["Technology"]

    def test_resolve_empty_key_returns_none(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        result = mod.resolve_tenant("")
        assert result is None

    def test_resolve_uses_hash_not_raw_key(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        captured = []
        def fake_fetchone(sql, params):
            captured.append(params)
            return None
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", fake_fetchone)
        raw_key = "my-secret-key"
        mod.resolve_tenant(raw_key)
        # The query should use the hash, not the raw key
        assert captured[0][0] == mod._hash_key(raw_key)
        assert raw_key not in captured[0]


# ---------------------------------------------------------------------------
# 6.4 — check_sector_access
# ---------------------------------------------------------------------------

class TestCheckSectorAccess:
    def test_none_allowed_sectors_grants_all(self):
        from api.jobs.tenant_auth import check_sector_access
        tenant = {"allowed_sectors": None}
        assert check_sector_access(tenant, "Technology") is True
        assert check_sector_access(tenant, "Healthcare") is True

    def test_allowed_sector_returns_true(self):
        from api.jobs.tenant_auth import check_sector_access
        tenant = {"allowed_sectors": ["Technology", "Energy"]}
        assert check_sector_access(tenant, "Technology") is True

    def test_restricted_sector_returns_false(self):
        from api.jobs.tenant_auth import check_sector_access
        tenant = {"allowed_sectors": ["Technology", "Energy"]}
        assert check_sector_access(tenant, "Healthcare") is False

    def test_empty_allowed_sectors_restricts_all(self):
        from api.jobs.tenant_auth import check_sector_access
        tenant = {"allowed_sectors": []}
        assert check_sector_access(tenant, "Technology") is False


# ---------------------------------------------------------------------------
# 6.5 — tier_has_access ordering
# ---------------------------------------------------------------------------

class TestTierHasAccess:
    def test_enterprise_has_access_to_all_tiers(self):
        from api.jobs.tenant_auth import tier_has_access
        assert tier_has_access("enterprise", "free") is True
        assert tier_has_access("enterprise", "pro") is True
        assert tier_has_access("enterprise", "enterprise") is True

    def test_free_cannot_access_pro_or_enterprise(self):
        from api.jobs.tenant_auth import tier_has_access
        assert tier_has_access("free", "pro") is False
        assert tier_has_access("free", "enterprise") is False

    def test_pro_cannot_access_enterprise(self):
        from api.jobs.tenant_auth import tier_has_access
        assert tier_has_access("pro", "enterprise") is False

    def test_same_tier_is_allowed(self):
        from api.jobs.tenant_auth import tier_has_access
        for tier in ("free", "pro", "enterprise"):
            assert tier_has_access(tier, tier) is True


# ---------------------------------------------------------------------------
# 6.6 — require_tier dependency
# ---------------------------------------------------------------------------

class TestRequireTierDependency:
    @pytest.mark.anyio
    async def test_passes_when_db_disabled(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: False)
        dep = mod.require_tier("pro")
        req = MagicMock()
        req.headers = {}
        result = await dep(req)
        assert result is None

    @pytest.mark.anyio
    async def test_passes_when_no_key(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        dep = mod.require_tier("pro")
        req = MagicMock()
        req.headers = {}
        result = await dep(req)
        assert result is None

    @pytest.mark.anyio
    async def test_raises_401_for_unknown_key(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: None)
        dep = mod.require_tier("pro")
        req = MagicMock()
        req.headers = {"x-api-key": "bad-key"}
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await dep(req)
        assert exc_info.value.status_code == 401

    @pytest.mark.anyio
    async def test_raises_403_for_insufficient_tier(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        row = ("T001", "Acme", "free", None, 30, None)
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: row)
        dep = mod.require_tier("enterprise")
        req = MagicMock()
        req.headers = {"x-api-key": "free-key"}
        req.state = MagicMock()
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await dep(req)
        assert exc_info.value.status_code == 403

    @pytest.mark.anyio
    async def test_passes_for_sufficient_tier(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        row = ("T001", "Acme", "enterprise", None, 0, None)
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: row)
        dep = mod.require_tier("pro")
        req = MagicMock()
        req.headers = {"x-api-key": "ent-key"}
        req.state = MagicMock()
        result = await dep(req)
        assert result is not None
        assert result["tier"] == "enterprise"


# ---------------------------------------------------------------------------
# 6.7 — /auth/tenant endpoint
# ---------------------------------------------------------------------------

class TestAuthTenantEndpoint:
    def test_returns_no_key_when_no_header(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        from api.jobs.platform_routes import get_tenant_info
        req = MagicMock()
        req.headers = {}
        result = get_tenant_info(req)
        assert result["status"] == "no_key"

    def test_returns_invalid_key_for_unknown(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: None)
        from api.jobs.platform_routes import get_tenant_info
        req = MagicMock()
        req.headers = {"x-api-key": "bad-key"}
        result = get_tenant_info(req)
        assert result["status"] == "invalid_key"

    def test_returns_tenant_info_for_valid_key(self, monkeypatch):
        import api.jobs.tenant_auth as mod
        from unittest.mock import MagicMock
        row = ("T001", "Acme Fund", "pro", ["Technology"], 120, None)
        monkeypatch.setattr(mod.db, "db_read_enabled", lambda: True)
        monkeypatch.setattr(mod.db, "safe_fetchone", lambda *a, **k: row)
        from api.jobs.platform_routes import get_tenant_info
        req = MagicMock()
        req.headers = {"x-api-key": "valid-key"}
        result = get_tenant_info(req)
        assert result["status"] == "ok"
        assert result["tier"] == "pro"
        assert result["tenant_id"] == "T001"
        assert "allowed_endpoint_prefixes" in result
        assert "rpm_limit" in result


# ---------------------------------------------------------------------------
# 6.8 — /api-docs endpoint
# ---------------------------------------------------------------------------

class TestApiDocsEndpoint:
    def test_returns_endpoint_catalogue(self):
        from api.jobs.platform_routes import api_documentation
        result = api_documentation()
        assert "endpoints" in result
        assert result["endpoint_count"] > 0

    def test_has_all_three_tiers(self):
        from api.jobs.platform_routes import api_documentation
        result = api_documentation()
        tiers = set(result["tiers"].keys())
        assert "free" in tiers
        assert "pro" in tiers
        assert "enterprise" in tiers

    def test_has_authentication_docs(self):
        from api.jobs.platform_routes import api_documentation
        result = api_documentation()
        assert "authentication" in result
        assert "header" in result["authentication"]

    def test_enterprise_endpoints_present(self):
        from api.jobs.platform_routes import api_documentation
        result = api_documentation()
        ent_eps = [e for e in result["endpoints"] if e["tier"] == "enterprise"]
        paths = [e["path"] for e in ent_eps]
        assert any("/pe/" in p for p in paths)
        assert any("/smb/" in p for p in paths)

    def test_free_endpoints_present(self):
        from api.jobs.platform_routes import api_documentation
        result = api_documentation()
        free_eps = [e for e in result["endpoints"] if e["tier"] == "free"]
        paths = [e["path"] for e in free_eps]
        assert any("/prosperity" in p for p in paths)

    def test_endpoint_has_required_fields(self):
        from api.jobs.platform_routes import api_documentation
        result = api_documentation()
        for ep in result["endpoints"]:
            assert "path" in ep
            assert "method" in ep
            assert "tier" in ep
            assert "description" in ep
            assert ep["tier"] in ("free", "pro", "enterprise")

    def test_router_has_expected_paths(self):
        from api.jobs.platform_routes import router
        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/auth/tenant" in paths
        assert "/api-docs" in paths
