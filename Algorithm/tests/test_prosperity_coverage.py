import datetime as dt

from api import db
from api.jobs.prosperity import _coverage_response


def test_coverage_response(monkeypatch):
    monkeypatch.setattr(db, "safe_fetchone", lambda *_args, **_kwargs: (3,))

    def fake_fetchall(sql, params):  # type: ignore[override]
        if "GROUP BY" in sql:
            return [("OK", "API_ERROR", 1), ("FAILED", "INSUFFICIENT_BARS", 2)]
        if "status='FAILED'" in sql:
            return [("AAPL", "INSUFFICIENT_BARS", "required=252 returned=1", 252, 1)]
        return []

    monkeypatch.setattr(db, "safe_fetchall", fake_fetchall)

    res = _coverage_response(as_of_date=dt.date(2024, 1, 1))
    assert res["attempted"] == 3
    assert res["ok"] == 1
    assert res["failed"] == 2
    assert res["skipped"] == 0
    assert res["by_reason_code"]["INSUFFICIENT_BARS"] == 2
    assert res["failed_symbols"][0]["symbol"] == "AAPL"
