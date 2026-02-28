import datetime as dt
import os
import uuid

import pytest

from api import db
from api.data import service

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_prices_time_travel_latest_as_of() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    symbol = "TST" + uuid.uuid4().hex[:5].upper()
    service.upsert_symbols([{"symbol": symbol}])

    d1 = service.record_data_version("prices_test", "prices-snap-1")
    d2 = service.record_data_version("prices_test", "prices-snap-2")

    day = dt.date(2024, 4, 1)
    asof1 = dt.datetime(2024, 4, 1, 22, 0, tzinfo=dt.timezone.utc)
    asof2 = dt.datetime(2024, 4, 2, 22, 0, tzinfo=dt.timezone.utc)

    service.ingest_prices_daily(
        d1["id"],
        [
            {
                "symbol": symbol,
                "date": day,
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
                "as_of_ts": asof1,
            }
        ],
    )
    service.ingest_prices_daily(
        d2["id"],
        [
            {
                "symbol": symbol,
                "date": day,
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11.2,
                "volume": 120,
                "as_of_ts": asof2,
            }
        ],
    )

    q1 = service.query_prices_daily(symbol, day, day, asof1)
    assert len(q1) == 1
    assert q1[0]["close"] == 10.5

    q2 = service.query_prices_daily(
        symbol, day, day, dt.datetime(2024, 4, 3, tzinfo=dt.timezone.utc)
    )
    assert len(q2) == 1
    assert q2[0]["close"] == 11.2
