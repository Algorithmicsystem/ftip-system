import datetime as dt
import os
import uuid

import pytest

from api import db
from api.data import service

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_fundamentals_time_travel_as_of() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    symbol = "TST" + uuid.uuid4().hex[:5].upper()
    service.upsert_symbols([{"symbol": symbol, "country": "US"}])

    dv1 = service.record_data_version("fund_test", "fund-snap-1")
    dv2 = service.record_data_version("fund_test", "fund-snap-2")

    pub1 = dt.datetime(2024, 1, 10, 12, 0, tzinfo=dt.timezone.utc)
    pub2 = dt.datetime(2024, 2, 10, 12, 0, tzinfo=dt.timezone.utc)

    service.ingest_fundamentals(
        dv1["id"],
        [
            {
                "symbol": symbol,
                "metric_key": "EPS",
                "metric_value": 1.25,
                "period_end": dt.date(2023, 12, 31),
                "published_ts": pub1,
                "as_of_ts": pub1,
            }
        ],
    )

    service.ingest_fundamentals(
        dv2["id"],
        [
            {
                "symbol": symbol,
                "metric_key": "EPS",
                "metric_value": 1.6,
                "period_end": dt.date(2023, 12, 31),
                "published_ts": pub2,
                "as_of_ts": pub2,
            }
        ],
    )

    between = service.query_latest_fundamentals(
        symbol, dt.datetime(2024, 1, 20, tzinfo=dt.timezone.utc)
    )
    assert len(between) == 1
    assert between[0]["metric_value"] == 1.25

    after = service.query_latest_fundamentals(
        symbol, dt.datetime(2024, 2, 20, tzinfo=dt.timezone.utc)
    )
    assert len(after) == 1
    assert after[0]["metric_value"] == 1.6
