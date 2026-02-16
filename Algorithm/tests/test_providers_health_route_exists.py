from fastapi.testclient import TestClient

from api.main import app


def test_providers_health_route_is_mounted() -> None:
    client = TestClient(app)

    response = client.get("/providers/health")

    assert response.status_code != 404
    assert response.status_code in {200, 401}
