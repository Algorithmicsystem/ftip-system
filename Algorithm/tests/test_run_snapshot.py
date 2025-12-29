import os

import pytest
from fastapi.testclient import TestClient

from api.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_run_snapshot_route_registered() -> None:
    client = TestClient(app)
    paths = [route.path for route in client.app.routes]
    assert "/db/run_snapshot" in paths
