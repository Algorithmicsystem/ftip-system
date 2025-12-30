from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, Iterable, List, Optional

from api import db


def upsert_strategy_rows(rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        db.safe_execute(
            """
            INSERT INTO prosperity_strategy_signals_daily(
                symbol, as_of_date, lookback, strategy_id, strategy_version, regime,
                raw_score, normalized_score, signal, confidence, rationale, feature_contributions, meta
            ) VALUES (
                %(symbol)s, %(as_of_date)s, %(lookback)s, %(strategy_id)s, %(strategy_version)s, %(regime)s,
                %(raw_score)s, %(normalized_score)s, %(signal)s, %(confidence)s,
                %(rationale)s::jsonb, %(feature_contributions)s::jsonb, %(meta)s::jsonb
            )
            ON CONFLICT(symbol, as_of_date, lookback, strategy_id, strategy_version) DO UPDATE SET
                regime=EXCLUDED.regime,
                raw_score=EXCLUDED.raw_score,
                normalized_score=EXCLUDED.normalized_score,
                signal=EXCLUDED.signal,
                confidence=EXCLUDED.confidence,
                rationale=EXCLUDED.rationale,
                feature_contributions=EXCLUDED.feature_contributions,
                meta=EXCLUDED.meta,
                updated_at=now()
            """,
            {
                "symbol": row["symbol"],
                "as_of_date": row["as_of_date"],
                "lookback": row["lookback"],
                "strategy_id": row["strategy_id"],
                "strategy_version": row.get("strategy_version", "v1"),
                "regime": row.get("regime"),
                "raw_score": row.get("raw_score"),
                "normalized_score": row.get("normalized_score"),
                "signal": row.get("signal"),
                "confidence": row.get("confidence"),
                "rationale": json.dumps(row.get("rationale") or []),
                "feature_contributions": json.dumps(row.get("feature_contributions") or {}),
                "meta": json.dumps(row.get("meta") or {}),
            },
        )
        count += 1
    return count


def upsert_ensemble_row(row: Dict[str, Any]) -> None:
    db.safe_execute(
        """
        INSERT INTO prosperity_ensemble_signals_daily(
            symbol, as_of_date, lookback, regime, ensemble_method, final_signal, final_score,
            final_confidence, thresholds, risk_overlay_applied, strategies_used, audit, hashes
        ) VALUES (
            %(symbol)s, %(as_of_date)s, %(lookback)s, %(regime)s, %(ensemble_method)s, %(final_signal)s, %(final_score)s,
            %(final_confidence)s, %(thresholds)s::jsonb, %(risk_overlay_applied)s,
            %(strategies_used)s::jsonb, %(audit)s::jsonb, %(hashes)s::jsonb
        )
        ON CONFLICT(symbol, as_of_date, lookback) DO UPDATE SET
            regime=EXCLUDED.regime,
            ensemble_method=EXCLUDED.ensemble_method,
            final_signal=EXCLUDED.final_signal,
            final_score=EXCLUDED.final_score,
            final_confidence=EXCLUDED.final_confidence,
            thresholds=EXCLUDED.thresholds,
            risk_overlay_applied=EXCLUDED.risk_overlay_applied,
            strategies_used=EXCLUDED.strategies_used,
            audit=EXCLUDED.audit,
            hashes=EXCLUDED.hashes,
            updated_at=now()
        """,
        {
            "symbol": row["symbol"],
            "as_of_date": row["as_of_date"],
            "lookback": row["lookback"],
            "regime": row.get("regime"),
            "ensemble_method": row.get("ensemble_method"),
            "final_signal": row.get("final_signal"),
            "final_score": row.get("final_score"),
            "final_confidence": row.get("final_confidence"),
            "thresholds": json.dumps(row.get("thresholds") or {}),
            "risk_overlay_applied": bool(row.get("risk_overlay_applied", False)),
            "strategies_used": json.dumps(row.get("strategies_used") or []),
            "audit": json.dumps(row.get("audit") or {}),
            "hashes": json.dumps(row.get("hashes") or {}),
        },
    )


def latest_ensemble(symbol: str, lookback: int) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of_date, lookback, regime, ensemble_method, final_signal, final_score,
               final_confidence, thresholds, risk_overlay_applied, strategies_used, audit, hashes
        FROM prosperity_ensemble_signals_daily
        WHERE symbol=%s AND lookback=%s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (symbol, lookback),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of_date": row[1].isoformat(),
        "lookback": row[2],
        "regime": row[3],
        "ensemble_method": row[4],
        "final_signal": row[5],
        "final_score": row[6],
        "final_confidence": row[7],
        "thresholds": row[8],
        "risk_overlay_applied": row[9],
        "strategies_used": row[10],
        "audit": row[11],
        "hashes": row[12],
    }


def ensemble_as_of(symbol: str, lookback: int, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    row = db.safe_fetchone(
        """
        SELECT symbol, as_of_date, lookback, regime, ensemble_method, final_signal, final_score,
               final_confidence, thresholds, risk_overlay_applied, strategies_used, audit, hashes
        FROM prosperity_ensemble_signals_daily
        WHERE symbol=%s AND lookback=%s AND as_of_date<=%s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (symbol, lookback, as_of_date),
    )
    if not row:
        return None
    return {
        "symbol": row[0],
        "as_of_date": row[1].isoformat(),
        "lookback": row[2],
        "regime": row[3],
        "ensemble_method": row[4],
        "final_signal": row[5],
        "final_score": row[6],
        "final_confidence": row[7],
        "thresholds": row[8],
        "risk_overlay_applied": row[9],
        "strategies_used": row[10],
        "audit": row[11],
        "hashes": row[12],
    }


def latest_strategies(symbol: str, lookback: int) -> List[Dict[str, Any]]:
    rows = db.safe_fetchall(
        """
        SELECT symbol, as_of_date, lookback, strategy_id, strategy_version, regime, raw_score, normalized_score,
               signal, confidence, rationale, feature_contributions, meta
        FROM prosperity_strategy_signals_daily
        WHERE symbol=%s AND lookback=%s
        ORDER BY as_of_date DESC
        LIMIT 25
        """,
        (symbol, lookback),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "symbol": r[0],
                "as_of_date": r[1].isoformat(),
                "lookback": r[2],
                "strategy_id": r[3],
                "strategy_version": r[4],
                "regime": r[5],
                "raw_score": r[6],
                "normalized_score": r[7],
                "signal": r[8],
                "confidence": r[9],
                "rationale": r[10],
                "feature_contributions": r[11],
                "meta": r[12],
            }
        )
    return out


def strategies_as_of(symbol: str, lookback: int, as_of_date: dt.date) -> List[Dict[str, Any]]:
    rows = db.safe_fetchall(
        """
        SELECT symbol, as_of_date, lookback, strategy_id, strategy_version, regime, raw_score, normalized_score,
               signal, confidence, rationale, feature_contributions, meta
        FROM prosperity_strategy_signals_daily
        WHERE symbol=%s AND lookback=%s AND as_of_date<=%s
        ORDER BY as_of_date DESC
        LIMIT 25
        """,
        (symbol, lookback, as_of_date),
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "symbol": r[0],
                "as_of_date": r[1].isoformat(),
                "lookback": r[2],
                "strategy_id": r[3],
                "strategy_version": r[4],
                "regime": r[5],
                "raw_score": r[6],
                "normalized_score": r[7],
                "signal": r[8],
                "confidence": r[9],
                "rationale": r[10],
                "feature_contributions": r[11],
                "meta": r[12],
            }
        )
    return out


__all__ = [
    "upsert_strategy_rows",
    "upsert_ensemble_row",
    "latest_ensemble",
    "latest_strategies",
    "ensemble_as_of",
    "strategies_as_of",
]
