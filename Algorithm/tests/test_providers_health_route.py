from fastapi.testclient import TestClient

from api.main import app


def test_providers_health_route_registered_on_runtime_app() -> None:
    assert any(route.path == "/providers/health" for route in app.router.routes)


def test_providers_health_route_in_openapi() -> None:
    openapi = app.openapi()
    assert "/providers/health" in openapi.get("paths", {})


def test_providers_health_route_returns_non_404_and_expected_shape() -> None:
    client = TestClient(app)
    response = client.get("/providers/health")

    assert response.status_code != 404
    assert response.status_code in {200, 401}

    if response.status_code == 200:
        payload = response.json()
        assert isinstance(payload.get("providers"), dict)
