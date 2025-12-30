import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from api import security
from api.main import app


def _clear_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ["FTIP_API_KEY", "FTIP_API_KEYS", "FTIP_API_KEY_PRIMARY", "FTIP_ALLOW_QUERY_KEY"]:
        monkeypatch.delenv(var, raising=False)
    security.reset_auth_cache()


def _make_request(headers=None, query_string: str = "") -> Request:
    scope = {
        "type": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": query_string.encode(),
        "method": "GET",
        "path": "/prosperity/health",
        "client": ("test", 123),
    }
    return Request(scope)


@pytest.fixture(autouse=True)
def reset_auth(monkeypatch: pytest.MonkeyPatch):
    _clear_auth(monkeypatch)
    yield
    security.reset_auth_cache()


def test_get_allowed_api_keys_merge_and_trim(monkeypatch: pytest.MonkeyPatch):
    _clear_auth(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", " primary ")
    monkeypatch.setenv("FTIP_API_KEYS", " one , two , ")
    monkeypatch.setenv("FTIP_API_KEY_PRIMARY", "final")

    keys = security.get_allowed_api_keys()
    assert keys == ["primary", "one", "two", "final"]
    assert security.auth_enabled() is True


def test_get_allowed_api_keys_disabled_when_empty(monkeypatch: pytest.MonkeyPatch):
    _clear_auth(monkeypatch)
    keys = security.get_allowed_api_keys()
    assert keys == []
    assert security.auth_enabled() is False


def test_header_and_bearer_parsing(monkeypatch: pytest.MonkeyPatch):
    _clear_auth(monkeypatch)
    req_header = _make_request(headers={"X-FTIP-API-Key": " demo "})
    assert security.get_provided_api_key(req_header) == "demo"

    req_bearer = _make_request(headers={"Authorization": "Bearer secret-token"})
    assert security.get_provided_api_key(req_bearer) == "secret-token"

    monkeypatch.setenv("FTIP_ALLOW_QUERY_KEY", "1")
    security.reset_auth_cache()
    req_query = _make_request(query_string="api_key=query-token")
    assert security.get_provided_api_key(req_query) == "query-token"


def test_prosperity_routes_require_key_when_enabled(monkeypatch: pytest.MonkeyPatch):
    _clear_auth(monkeypatch)
    monkeypatch.setenv("FTIP_API_KEY", "demo-key")
    security.reset_auth_cache()

    with TestClient(app) as client:
        res = client.get("/prosperity/health")
        assert res.status_code == 401
        body = res.json()
        assert body["error"]["message"] == "unauthorized"

        res_ok = client.get("/prosperity/health", headers={"X-FTIP-API-Key": "demo-key"})
        assert res_ok.status_code == 200
        assert res_ok.json()["status"] == "ok"

        res_health = client.get("/health")
        assert res_health.status_code == 200
