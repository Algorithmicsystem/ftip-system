import os

import pytest
from fastapi.testclient import TestClient

from api import db
from api.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_universe_load_default_endpoint_smoke() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    client = TestClient(app)

    resp = client.post("/db/universe/load_default")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["upserted"] > 0

    list_resp = client.get("/db/universe")
    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["count"] >= body["upserted"]
    assert len(payload.get("symbols", [])) == payload["count"]
