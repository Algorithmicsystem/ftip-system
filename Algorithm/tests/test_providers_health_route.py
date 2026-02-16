from fastapi.testclient import TestClient

from api.main import app


def test_providers_health_route_is_mounted() -> None:
    client = TestClient(app)

    response = client.get("/providers/health")

    assert response.status_code != 404

    if response.status_code == 200:
        payload = response.json()
        assert isinstance(payload, dict)
        assert "providers" in payload
