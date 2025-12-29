from api import db


class _FakeCursor:
    def __init__(self) -> None:
        self.executed = []

    def execute(self, sql: str) -> None:  # type: ignore[override]
        self.executed.append(sql)


def test_set_statement_timeout_inlines_validated_value() -> None:
    cur = _FakeCursor()
    value = db.set_statement_timeout(cur, "2500")

    assert value == 2500
    assert cur.executed == ["SET LOCAL statement_timeout TO 2500"]


def test_set_statement_timeout_clamps_and_avoids_params() -> None:
    cur = _FakeCursor()
    value = db.set_statement_timeout(cur, -5)

    assert value == 1000
    assert cur.executed[0].startswith("SET LOCAL statement_timeout TO ")
    assert "%s" not in cur.executed[0]
    assert "$1" not in cur.executed[0]
