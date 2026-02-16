import os

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 to run"
)
def test_providers_health_endpoint_integration() -> None:
    client = TestClient(app)

    response = client.get("/providers/health")

    assert response.status_code in {200, 401}

    payload = response.json()
    assert isinstance(payload, dict)

    providers = payload.get("providers", [])
    assert isinstance(providers, list)

    for provider in providers:
        assert isinstance(provider, dict)
        if "enabled" in provider:
            assert isinstance(provider["enabled"], bool)
