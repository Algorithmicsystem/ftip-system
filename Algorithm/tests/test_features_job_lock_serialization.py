import datetime as dt
from contextlib import contextmanager

from psycopg.types.json import Json

from api import db
from api.jobs import features


class _FakeCursor:
    def __init__(self):
        self.insert_params = None
        self._fetchone_values = [None, None]

    def execute(self, sql, params=None):
        if "INSERT INTO ftip_job_runs" in sql:
            self.insert_params = params

    def fetchone(self):
        if self._fetchone_values:
            return self._fetchone_values.pop(0)
        return None


class _FakeConn:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def test_features_acquire_job_lock_serializes_date_like_values(monkeypatch):
    conn = _FakeConn()
    cur = _FakeCursor()

    @contextmanager
    def fake_with_connection():
        yield conn, cur

    monkeypatch.setattr(db, "with_connection", fake_with_connection)

    requested = {
        "as_of_date": dt.date(2024, 1, 2),
        "symbols": ["AAPL"],
        "window": {
            "start_ts": dt.datetime(2024, 1, 1, 12, 30, tzinfo=dt.timezone.utc)
        },
    }

    acquired, info = features._acquire_job_lock(
        "run-id",
        "features.daily",
        dt.date(2024, 1, 2),
        requested,
        60,
        "tester",
    )

    assert acquired is True
    assert info["run_id"] == "run-id"
    assert conn.commits == 1
    assert cur.insert_params is not None

    requested_json = cur.insert_params[3]
    assert isinstance(requested_json, Json)
    assert requested_json.obj["as_of_date"] == "2024-01-02"
    assert requested_json.obj["window"]["start_ts"] == "2024-01-01T12:30:00+00:00"
