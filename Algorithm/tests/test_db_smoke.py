import datetime as dt
import os
import uuid

import pytest

from api import db

pytestmark = pytest.mark.skipif(
    os.getenv("DATABASE_URL") is None, reason="DATABASE_URL not set"
)


def test_ensure_schema_and_insert_signal() -> None:
    os.environ.setdefault("FTIP_DB_ENABLED", "1")

    db.ensure_schema()

    symbol = f"TST{uuid.uuid4().hex[:6]}".upper()
    as_of = dt.date.today()

    row = db.exec1(
        """
        INSERT INTO signals (
            symbol, as_of, lookback, regime, score, signal, confidence, thresholds,
            features, notes, score_mode, base_score, stacked_score, stacked_meta,
            calibration_loaded, calibration_meta
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (symbol, as_of, lookback, score_mode)
        DO UPDATE SET score=EXCLUDED.score
        RETURNING id
        """,
        (
            symbol,
            as_of,
            252,
            "TEST",
            1.23,
            "BUY",
            0.9,
            {"hi": 1},
            {"feature": 2.0},
            ["note"],
            "stacked",
            0.5,
            0.5,
            {"stack": True},
            True,
            {"cal": "yes"},
        ),
    )

    assert row is not None
    assert int(row[0]) > 0
