import datetime as dt
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from api import db
from api.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_data_router_endpoints_smoke() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")
    db.ensure_schema()

    client = TestClient(app)
    symbol = "TST" + uuid.uuid4().hex[:5].upper()

    version_resp = client.post(
        "/data/version/create",
        json={
            "source_name": "api-smoke",
            "source_snapshot_hash": "api-smoke-snap-1",
            "notes": "smoke",
        },
    )
    assert version_resp.status_code == 200
    data_version_id = version_resp.json()["id"]

    sym_resp = client.post(
        "/data/symbols/upsert",
        json={"items": [{"symbol": symbol, "country": "US", "exchange": "NYSE"}]},
    )
    assert sym_resp.status_code == 200

    universe_resp = client.post(
        "/data/universe/set",
        json={"universe_name": "default", "symbols": [symbol]},
    )
    assert universe_resp.status_code == 200

    prices_ingest = client.post(
        "/data/prices/ingest_daily",
        json={
            "data_version_id": data_version_id,
            "items": [
                {
                    "symbol": symbol,
                    "date": "2024-05-01",
                    "open": 10,
                    "high": 11,
                    "low": 9,
                    "close": 10.5,
                    "volume": 1000,
                }
            ],
        },
    )
    assert prices_ingest.status_code == 200

    prices_query = client.get(
        "/data/prices/query_daily",
        params={
            "symbol": symbol,
            "start_date": "2024-05-01",
            "end_date": "2024-05-01",
            "as_of_ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "adjusted": False,
        },
    )
    assert prices_query.status_code == 200
    assert isinstance(prices_query.json()["items"], list)

    fund_ingest = client.post(
        "/data/fundamentals/ingest_pit",
        json={
            "data_version_id": data_version_id,
            "items": [
                {
                    "symbol": symbol,
                    "metric_key": "EPS",
                    "metric_value": 2.1,
                    "period_end": "2024-03-31",
                    "published_ts": "2024-04-30T00:00:00Z",
                }
            ],
        },
    )
    assert fund_ingest.status_code == 200

    fund_query = client.get(
        "/data/fundamentals/query_pit",
        params={
            "symbol": symbol,
            "as_of_ts": "2024-06-01T00:00:00Z",
            "metric_keys": "EPS",
        },
    )
    assert fund_query.status_code == 200

    news_ingest = client.post(
        "/data/news/ingest",
        json={
            "data_version_id": data_version_id,
            "items": [
                {
                    "symbol": symbol,
                    "published_ts": "2024-05-15T10:00:00Z",
                    "source": "wire",
                    "credibility": 0.88,
                    "headline": "API smoke headline",
                }
            ],
        },
    )
    assert news_ingest.status_code == 200

    news_query = client.get(
        "/data/news/query",
        params={"symbol": symbol, "as_of_ts": "2024-06-01T00:00:00Z", "limit": 50},
    )
    assert news_query.status_code == 200
    assert isinstance(news_query.json()["items"], list)

    corp_resp = client.post(
        "/data/corp_actions/ingest",
        json={
            "data_version_id": data_version_id,
            "items": [
                {
                    "symbol": symbol,
                    "action_type": "split",
                    "effective_date": "2024-05-20",
                    "factor": 2.0,
                    "announced_ts": "2024-05-10T00:00:00Z",
                }
            ],
        },
    )
    assert corp_resp.status_code == 200
