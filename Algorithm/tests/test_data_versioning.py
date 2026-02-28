import os

import pytest

from api import db
from api.data import service

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_create_data_version_stored_fields() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    version = service.record_data_version(
        source_name="unit_test_source",
        source_snapshot_hash="snapshot-abc-001",
        notes="phase1-test",
    )

    assert version["id"] > 0
    assert version["source_name"] == "unit_test_source"
    assert version["source_snapshot_hash"] == "snapshot-abc-001"
    assert "code_sha" in version
    assert version["created_at"] is not None
