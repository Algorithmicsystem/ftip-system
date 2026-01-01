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


class _CooldownCursor:
    def __init__(self, rows):
        self.rows = rows
        self.fetchone_result = None
        self.executed = []

    def _row_tuple(self, row):
        return (
            row["run_id"],
            row.get("started_at"),
            row.get("lock_owner"),
            row.get("lock_acquired_at"),
            row.get("lock_expires_at"),
        )

    def _matching_rows(self, job_name: str, window_seconds: int):
        now = dt.datetime.now(dt.timezone.utc)
        cutoff = now - dt.timedelta(seconds=window_seconds)
        return [
            row
            for row in self.rows
            if row["job_name"] == job_name
            and (row.get("finished_at") is None or row.get("finished_at") > cutoff)
        ]

    def execute(self, sql, params=None):
        self.executed.append(sql)
        normalized = " ".join(sql.split()).lower()
        if "select run_id" in normalized and "from ftip_job_runs" in normalized:
            job_name, window_seconds = params
            matches = self._matching_rows(job_name, window_seconds)
            self.fetchone_result = self._row_tuple(matches[0]) if matches else None
        elif "insert into ftip_job_runs" in normalized:
            (
                run_id,
                job_name,
                as_of_date,
                _requested,
                lock_owner,
                ttl_seconds,
            ) = params
            now = dt.datetime.now(dt.timezone.utc)
            row = {
                "run_id": run_id,
                "job_name": job_name,
                "as_of_date": as_of_date,
                "started_at": now,
                "finished_at": None,
                "lock_owner": lock_owner,
                "lock_acquired_at": now,
                "lock_expires_at": now + dt.timedelta(seconds=ttl_seconds),
            }
            self.rows.append(row)
            self.fetchone_result = self._row_tuple(row)
        else:
            self.fetchone_result = None

    def fetchall(self):
        return []

    def fetchone(self):
        result = self.fetchone_result
        self.fetchone_result = None
        return result


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


def test_acquire_job_lock_enforces_cooldown_window(monkeypatch):
    lock_window = 120
    finished_recently = dt.datetime.now(dt.timezone.utc)
    rows = [
        {
            "run_id": "existing",
            "job_name": prosperity.JOB_NAME,
            "as_of_date": dt.date(2024, 1, 1),
            "started_at": finished_recently - dt.timedelta(seconds=10),
            "finished_at": finished_recently,
            "lock_owner": "tester",
            "lock_acquired_at": finished_recently - dt.timedelta(seconds=20),
            "lock_expires_at": finished_recently + dt.timedelta(seconds=60),
        }
    ]

    conn = _FakeConn()
    cur = _CooldownCursor(rows)

    @contextmanager
    def fake_with_connection():
        yield conn, cur

    monkeypatch.setenv("FTIP_JOB_LOCK_WINDOW_SEC", str(lock_window))
    monkeypatch.setattr(db, "with_connection", fake_with_connection)
    monkeypatch.setattr(prosperity, "_cleanup_stale_job_runs", lambda *args, **kwargs: [])

    acquired, info = prosperity._acquire_job_lock(
        "run-id",
        prosperity.JOB_NAME,
        dt.date(2024, 1, 2),
        {"symbols": ["AAPL"]},
        30,
        "tester",
    )

    assert acquired is False
    assert info["run_id"] == "existing"

    rows[0]["finished_at"] = finished_recently - dt.timedelta(seconds=lock_window + 5)

    acquired_after_window, info_after_window = prosperity._acquire_job_lock(
        "new-run",
        prosperity.JOB_NAME,
        dt.date(2024, 1, 2),
        {"symbols": ["MSFT"]},
        30,
        "tester2",
    )

    assert acquired_after_window is True
    assert info_after_window["run_id"] == "new-run"
