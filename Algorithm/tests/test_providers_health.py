import os

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.providers import ProviderHealth
from api.providers.finnhub import FinnhubProvider
from api.providers.fred import FREDProvider
from api.providers.sec_edgar import SecEdgarProvider


def _stub_health(self):
    if not self.enabled():
        return ProviderHealth(
            name=self.name,
            enabled=False,
            status="down",
            message="disabled",
        )
    return ProviderHealth(name=self.name, enabled=True, status="ok", message="stubbed")


def test_providers_health_shape_and_entries(monkeypatch):
    monkeypatch.setattr(FinnhubProvider, "health_check", _stub_health)
    monkeypatch.setattr(FREDProvider, "health_check", _stub_health)
    monkeypatch.setattr(SecEdgarProvider, "health_check", _stub_health)

    client = TestClient(app)
    response = client.get("/providers/health")

    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "providers" in payload

    providers = {item["name"]: item for item in payload["providers"]}
    assert {"finnhub", "fred", "sec_edgar"}.issubset(providers.keys())

    finnhub_enabled = bool(os.getenv("FINNHUB_API_KEY"))
    fred_enabled = bool(os.getenv("FRED_API_KEY"))

    assert providers["finnhub"]["enabled"] is finnhub_enabled
    assert providers["fred"]["enabled"] is fred_enabled
    assert providers["sec_edgar"]["enabled"] is True


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_finnhub_health_integration_runs_only_with_key():
    if not os.getenv("FINNHUB_API_KEY"):
        pytest.skip("FINNHUB_API_KEY not set")

    health = FinnhubProvider().health_check()
    assert health.enabled is True
    assert health.status in {"ok", "down"}


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_fred_health_integration_runs_only_with_key():
    if not os.getenv("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set")

    health = FREDProvider().health_check()
    assert health.enabled is True
    assert health.status in {"ok", "down"}


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_sec_edgar_health_integration():
    health = SecEdgarProvider().health_check()
    assert health.enabled is True
    assert health.status in {"ok", "degraded", "down"}
