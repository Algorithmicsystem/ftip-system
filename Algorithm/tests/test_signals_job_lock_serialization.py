import datetime as dt
from contextlib import contextmanager

from psycopg.types.json import Json

from api import db
from api.jobs import signals


class _FakeCursor:
    def __init__(self):
        self.insert_params = None
        self._fetchone_values = [None, None]
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
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


def test_signals_acquire_job_lock_serializes_date_like_values(monkeypatch):
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

    acquired, info = signals._acquire_job_lock(
        "run-id",
        "signals.daily",
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


def test_signals_acquire_job_lock_ignores_recent_finished_runs(monkeypatch):
    conn = _FakeConn()
    cur = _FakeCursor()

    @contextmanager
    def fake_with_connection():
        yield conn, cur

    monkeypatch.setattr(db, "with_connection", fake_with_connection)

    signals._acquire_job_lock(
        "run-id",
        "signals.daily",
        dt.date(2024, 1, 2),
        {"as_of_date": dt.date(2024, 1, 2)},
        60,
        "tester",
    )

    lock_selects = [
        sql
        for sql, _ in cur.executed
        if "FROM ftip_job_runs" in sql and "SELECT run_id" in sql
    ]
    assert lock_selects
    for sql in lock_selects:
        assert "finished_at IS NULL" in sql
        assert "finished_at > now() -" not in sql
