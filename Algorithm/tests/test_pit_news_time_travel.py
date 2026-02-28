import datetime as dt
import os
import uuid

import pytest

from api import db
from api.data import service

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_news_time_travel_as_of() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    symbol = "TST" + uuid.uuid4().hex[:5].upper()
    service.upsert_symbols([{"symbol": symbol, "country": "CA"}])

    dv1 = service.record_data_version("news_test", "news-snap-1")
    dv2 = service.record_data_version("news_test", "news-snap-2")

    pub1 = dt.datetime(2024, 3, 1, 14, 0, tzinfo=dt.timezone.utc)
    pub2 = dt.datetime(2024, 3, 2, 14, 0, tzinfo=dt.timezone.utc)

    service.ingest_news(
        dv1["id"],
        [
            {
                "symbol": symbol,
                "published_ts": pub1,
                "source": "wire",
                "credibility": 0.7,
                "headline": "First headline",
                "full_text": "first",
            }
        ],
    )
    service.ingest_news(
        dv2["id"],
        [
            {
                "symbol": symbol,
                "published_ts": pub2,
                "source": "wire",
                "credibility": 0.9,
                "headline": "Second headline",
                "full_text": "second",
            }
        ],
    )

    before_second = service.query_news(
        symbol,
        dt.datetime(2024, 3, 1, 18, 0, tzinfo=dt.timezone.utc),
        limit=10,
    )
    assert len(before_second) == 1
    assert before_second[0]["headline"] == "First headline"

    after_second = service.query_news(
        symbol,
        dt.datetime(2024, 3, 3, tzinfo=dt.timezone.utc),
        limit=10,
    )
    assert len(after_second) >= 2
    headlines = {row["headline"] for row in after_second}
    assert "Second headline" in headlines
