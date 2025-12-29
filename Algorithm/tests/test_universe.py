import os
import uuid

import pytest

from api import db

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_upsert_and_get_universe() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")

    db.ensure_schema()

    symbols = [f"U{uuid.uuid4().hex[:6]}{i}" for i in range(3)]
    received, upserted = db.upsert_universe(symbols, source="test_case")

    assert received == len(symbols)
    assert upserted == len(symbols)

    retrieved = db.get_universe(active_only=True)
    for sym in symbols:
        assert sym in retrieved
