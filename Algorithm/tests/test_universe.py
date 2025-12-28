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

    symbols = [f"ZZ{uuid.uuid4().hex[:4]}" for _ in range(3)]
    received, upserted = db.upsert_universe(symbols, source="pytest")

    assert received == 3
    assert upserted == 3

    universe = db.get_universe(active_only=True)
    for sym in symbols:
        assert sym in universe
