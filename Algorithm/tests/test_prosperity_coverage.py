import datetime as dt

from api import db
from api.jobs.prosperity import _coverage_response
from api.prosperity import query


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


def test_universe_coverage_handles_symbols_with_no_data(monkeypatch):
    monkeypatch.setattr(
        db,
        "safe_fetchall",
        lambda *_args, **_kwargs: [
            ("AAPL", dt.date(2024, 1, 2), dt.date(2024, 1, 5), 4),
            ("MSFT", None, None, 0),
        ],
    )

    res = query.coverage_for_universe()
    assert len(res) == 2
    assert res[0]["missing_days_estimate"] == 0
    assert res[1]["bars"] == 0
    assert res[1]["missing_days_estimate"] is None
