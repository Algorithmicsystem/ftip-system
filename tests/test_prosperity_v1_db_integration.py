import datetime as dt
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pytest
from fastapi.testclient import TestClient

# Ensure imports work when this test is run from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "Algorithm"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from api import db  # noqa: E402
from api.main import app  # noqa: E402


@pytest.mark.skipif(os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set")
class TestProsperityV1DBIntegration:
    @pytest.fixture(autouse=True)
    def _enable_db(self) -> Iterable[None]:
        os.environ["FTIP_DB_ENABLED"] = "1"
        os.environ["FTIP_DB_WRITE_ENABLED"] = "1"
        os.environ["FTIP_DB_READ_ENABLED"] = "1"
        yield

    @staticmethod
    def _admin_headers() -> dict:
        token = os.getenv("PROSPERITY_ADMIN_TOKEN")
        return {"x-admin-token": token} if token else {}

    @staticmethod
    def _new_symbol(prefix: str = "IT") -> str:
        return f"{prefix}{uuid.uuid4().hex[:6]}".upper()

    @staticmethod
    def _weekday_dates(from_date: dt.date, to_date: dt.date) -> List[dt.date]:
        dates: List[dt.date] = []
        cursor = from_date
        while cursor <= to_date:
            if cursor.weekday() < 5:
                dates.append(cursor)
            cursor += dt.timedelta(days=1)
        return dates

    @staticmethod
    def _clean_symbol_rows(symbol: str) -> None:
        db.safe_execute("DELETE FROM prosperity_daily_bars WHERE symbol=%s", (symbol,))
        db.safe_execute("DELETE FROM prosperity_signals_daily WHERE symbol=%s", (symbol,))
        db.safe_execute("DELETE FROM prosperity_features_daily WHERE symbol=%s", (symbol,))
        db.safe_execute("DELETE FROM prosperity_universe WHERE symbol=%s", (symbol,))

    def test_v1_round_trip_snapshot_latest_endpoints_with_real_db(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @dataclass
        class FakeBar:
            timestamp: str
            close: float
            volume: float
            open: float
            high: float
            low: float
            adj_close: float

        def fake_massive_fetch_daily_bars(
            symbol: str,
            from_date_iso: str,
            to_date_iso: str,
        ) -> List[FakeBar]:
            from_date = dt.date.fromisoformat(from_date_iso)
            to_date = dt.date.fromisoformat(to_date_iso)
            bars: List[FakeBar] = []
            for idx, day in enumerate(self._weekday_dates(from_date, to_date)):
                price = 100.0 + (idx * 0.2)
                bars.append(
                    FakeBar(
                        timestamp=day.isoformat(),
                        close=price,
                        volume=1_000_000 + idx,
                        open=price - 0.1,
                        high=price + 0.2,
                        low=price - 0.2,
                        adj_close=price,
                    )
                )
            return bars

        # Patch only provider boundary.
        monkeypatch.setattr("api.main.massive_fetch_daily_bars", fake_massive_fetch_daily_bars)

        symbol = self._new_symbol("V1")
        as_of = dt.date(2025, 12, 12)
        from_date = as_of - dt.timedelta(days=420)
        lookback = 252

        self._clean_symbol_rows(symbol)

        with TestClient(app) as client:
            bootstrap = client.post("/prosperity/bootstrap", headers=self._admin_headers())
            assert bootstrap.status_code == 200
            assert bootstrap.json()["status"] == "ok"

            snapshot = client.post(
                "/prosperity/snapshot/run",
                json={
                    "symbols": [symbol],
                    "from_date": from_date.isoformat(),
                    "to_date": as_of.isoformat(),
                    "as_of_date": as_of.isoformat(),
                    "lookback": lookback,
                    "concurrency": 2,
                },
            )
            assert snapshot.status_code == 200
            payload = snapshot.json()
            assert payload["status"] == "ok"
            assert payload["result"]["rows_written"]["signals"] == 1
            assert payload["result"]["rows_written"]["features"] == 1

            latest_signal = client.get(
                "/prosperity/latest/signal",
                params={"symbol": symbol, "lookback": lookback},
            )
            assert latest_signal.status_code == 200
            signal_body = latest_signal.json()
            assert signal_body["symbol"] == symbol
            assert signal_body["as_of"] == as_of.isoformat()

            latest_features = client.get(
                "/prosperity/latest/features",
                params={"symbol": symbol, "lookback": lookback},
            )
            assert latest_features.status_code == 200
            features_body = latest_features.json()
            assert features_body["symbol"] == symbol
            assert features_body["as_of"] == as_of.isoformat()

        signal_rows = db.safe_fetchone(
            "SELECT COUNT(*) FROM prosperity_signals_daily WHERE symbol=%s AND as_of=%s AND lookback=%s",
            (symbol, as_of, lookback),
        )
        feature_rows = db.safe_fetchone(
            "SELECT COUNT(*) FROM prosperity_features_daily WHERE symbol=%s AND as_of=%s AND lookback=%s",
            (symbol, as_of, lookback),
        )
        assert signal_rows and int(signal_rows[0]) >= 1
        assert feature_rows and int(feature_rows[0]) >= 1

        self._clean_symbol_rows(symbol)

    def test_v1_daily_snapshot_job_writes_and_reads_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @dataclass
        class FakeBar:
            timestamp: str
            close: float
            volume: float
            open: float
            high: float
            low: float
            adj_close: float

        def fake_massive_fetch_daily_bars(
            symbol: str,
            from_date_iso: str,
            to_date_iso: str,
        ) -> List[FakeBar]:
            from_date = dt.date.fromisoformat(from_date_iso)
            to_date = dt.date.fromisoformat(to_date_iso)
            bars: List[FakeBar] = []
            for idx, day in enumerate(self._weekday_dates(from_date, to_date)):
                price = 120.0 + (idx * 0.15)
                bars.append(
                    FakeBar(
                        timestamp=day.isoformat(),
                        close=price,
                        volume=2_000_000 + idx,
                        open=price - 0.1,
                        high=price + 0.2,
                        low=price - 0.2,
                        adj_close=price,
                    )
                )
            return bars

        # Patch only provider boundary.
        monkeypatch.setattr("api.main.massive_fetch_daily_bars", fake_massive_fetch_daily_bars)

        symbol = self._new_symbol("JB")
        monkeypatch.setenv("FTIP_UNIVERSE", symbol)
        monkeypatch.setenv("FTIP_LOOKBACK", "252")
        monkeypatch.setenv("FTIP_SNAPSHOT_WINDOW_DAYS", "420")

        self._clean_symbol_rows(symbol)

        with TestClient(app) as client:
            bootstrap = client.post("/prosperity/bootstrap", headers=self._admin_headers())
            assert bootstrap.status_code == 200

            job = client.post("/jobs/prosperity/daily-snapshot")
            assert job.status_code == 200
            body = job.json()
            assert body["status"] in {"ok", "partial"}
            assert symbol in body["symbols_ok"]
            as_of = dt.date.fromisoformat(body["as_of_date"])

            latest_signal = client.get(
                "/prosperity/latest/signal",
                params={"symbol": symbol, "lookback": 252},
            )
            assert latest_signal.status_code == 200

            latest_features = client.get(
                "/prosperity/latest/features",
                params={"symbol": symbol, "lookback": 252},
            )
            assert latest_features.status_code == 200

        signal_rows = db.safe_fetchone(
            "SELECT COUNT(*) FROM prosperity_signals_daily WHERE symbol=%s AND as_of=%s AND lookback=%s",
            (symbol, as_of, 252),
        )
        feature_rows = db.safe_fetchone(
            "SELECT COUNT(*) FROM prosperity_features_daily WHERE symbol=%s AND as_of=%s AND lookback=%s",
            (symbol, as_of, 252),
        )
        assert signal_rows and int(signal_rows[0]) >= 1
        assert feature_rows and int(feature_rows[0]) >= 1

        self._clean_symbol_rows(symbol)
