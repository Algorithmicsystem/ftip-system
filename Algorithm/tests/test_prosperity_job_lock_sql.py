import datetime as dt
from contextlib import contextmanager

from api import db
from api.jobs import prosperity


class _FakeCursor:
    def __init__(self, fetchone_values):
        self.executed = []
        self._fetchone_values = list(fetchone_values)
        self._fetchall_values = [[]]

    def execute(self, sql, params=None):
        self.executed.append(sql)

    def fetchall(self):
        if self._fetchall_values:
            return self._fetchall_values.pop(0)
        return []

    def fetchone(self):
        if self._fetchone_values:
            return self._fetchone_values.pop(0)
        return None


class _FakeConn:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def test_acquire_job_lock_sql_uses_row_locking(monkeypatch):
    inserted_started_at = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    inserted_lock_time = inserted_started_at
    inserted_expires_at = inserted_started_at + dt.timedelta(seconds=30)

    fetchone_values = [
        None,  # no locked active rows
        None,  # no pending rows
        ("run-id", inserted_started_at, "tester", inserted_lock_time, inserted_expires_at),
    ]
    captured = {}

    @contextmanager
    def fake_with_connection():
        conn = _FakeConn()
        cur = _FakeCursor(fetchone_values)
        captured["conn"] = conn
        captured["cur"] = cur
        yield conn, cur

    monkeypatch.setattr(db, "with_connection", fake_with_connection)

    acquired, info = prosperity._acquire_job_lock(
        "run-id",
        prosperity.JOB_NAME,
        dt.date(2024, 1, 2),
        {"symbols": ["AAPL"]},
        30,
        "tester",
    )

    assert acquired is True
    assert info["run_id"] == "run-id"
    executed_sql = "\n".join(captured["cur"].executed).upper()
    assert "ON CONFLICT" not in executed_sql
    assert "FOR UPDATE SKIP LOCKED" in executed_sql
    assert captured["conn"].commits == 1
