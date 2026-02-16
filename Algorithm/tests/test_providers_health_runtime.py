import inspect

from fastapi.testclient import TestClient

import api.main
from api.main import app


def test_providers_health_runtime_registration_and_openapi() -> None:
    assert inspect.getfile(api.main).endswith("Algorithm/api/main.py")
    assert "/providers/health" in {route.path for route in app.router.routes}

    client = TestClient(app)

    health_response = client.get("/providers/health")
    assert health_response.status_code == 200

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    assert "/providers/health" in openapi_response.json().get("paths", {})

    payload = health_response.json()
    assert "providers" in payload

    providers = payload["providers"]
    assert "openai" in providers
    assert "massive" in providers
    assert "finnhub" in providers
    assert "fred" in providers
    assert "secedgar" in providers
