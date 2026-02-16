from pathlib import Path

from fastapi.testclient import TestClient

import api.main
from api.main import app


def test_imports_project_api_main_module() -> None:
    module_path = Path(api.main.__file__).resolve()
    print(f"api.main file: {module_path}")
    assert str(module_path).endswith("/Algorithm/api/main.py")


def test_providers_health_route_is_mounted_and_documented() -> None:
    client = TestClient(app)

    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    paths = openapi_response.json().get("paths", {})
    assert "/providers/health" in paths

    response = client.get("/providers/health")
    assert response.status_code != 404
    assert response.status_code in {200, 401}

    if response.status_code == 200:
        payload = response.json()
        assert isinstance(payload, dict)
        assert "providers" in payload
