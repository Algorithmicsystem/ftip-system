import datetime as dt
import os
import uuid

import pytest

from api import db
from api.data import service

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_no_lookahead_on_fundamentals_and_news() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    symbol = "TST" + uuid.uuid4().hex[:5].upper()
    service.upsert_symbols([{"symbol": symbol}])
    version = service.record_data_version("lookahead_test", "lookahead-snap-1")

    future_pub = dt.datetime(2025, 1, 10, tzinfo=dt.timezone.utc)
    as_of = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)

    service.ingest_fundamentals(
        version["id"],
        [
            {
                "symbol": symbol,
                "metric_key": "REV",
                "metric_value": 100.0,
                "period_end": dt.date(2024, 12, 31),
                "published_ts": future_pub,
                "as_of_ts": future_pub,
            }
        ],
    )
    service.ingest_news(
        version["id"],
        [
            {
                "symbol": symbol,
                "published_ts": future_pub,
                "source": "wire",
                "credibility": 0.5,
                "headline": "Future item",
            }
        ],
    )

    fundamentals = service.query_latest_fundamentals(symbol, as_of)
    news = service.query_news(symbol, as_of)

    assert fundamentals == []
    assert news == []
