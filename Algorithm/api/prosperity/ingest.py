from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any, Dict, List, Tuple

from fastapi import HTTPException

from api import db


# Utility hashing

def _hash_dict(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def upsert_universe(symbols: List[str]) -> Tuple[int, List[str]]:
    cleaned = sorted({(s or "").strip().upper() for s in symbols if s})
    if not cleaned:
        return 0, []
    for sym in cleaned:
        db.safe_execute(
            """
            INSERT INTO prosperity_universe(symbol, active)
            VALUES (%s, TRUE)
            ON CONFLICT(symbol) DO UPDATE SET active=EXCLUDED.active, updated_at=now()
            """,
            (sym,),
        )
    return len(cleaned), cleaned


def _existing_bars(symbol: str, from_date: dt.date, to_date: dt.date) -> List[dt.date]:
    rows = db.safe_fetchall(
        """
        SELECT date FROM prosperity_daily_bars WHERE symbol=%s AND date BETWEEN %s AND %s
        """,
        (symbol, from_date, to_date),
    )
    return [r[0] for r in rows]


def ingest_bars(symbol: str, from_date: dt.date, to_date: dt.date, *, source: str = "massive", force_refresh: bool = False) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol required")
    upsert_universe([sym])

    missing: List[dt.date] = []
    if not force_refresh:
        have = set(_existing_bars(sym, from_date, to_date))
        cur = from_date
        while cur <= to_date:
            if cur not in have:
                missing.append(cur)
            cur += dt.timedelta(days=1)
    else:
        cur = from_date
        while cur <= to_date:
            missing.append(cur)
            cur += dt.timedelta(days=1)

    from api.main import massive_fetch_daily_bars  # type: ignore

    inserted = 0
    updated = 0
    if missing:
        bars = massive_fetch_daily_bars(sym, from_date.isoformat(), to_date.isoformat())
        for b in bars:
            day = dt.date.fromisoformat(b.timestamp)
            db.safe_execute(
                """
                INSERT INTO prosperity_daily_bars(symbol, date, open, high, low, close, adj_close, volume, source, raw)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(symbol, date) DO UPDATE SET
                    open=EXCLUDED.open,
                    high=EXCLUDED.high,
                    low=EXCLUDED.low,
                    close=EXCLUDED.close,
                    adj_close=EXCLUDED.adj_close,
                    volume=EXCLUDED.volume,
                    source=EXCLUDED.source,
                    raw=EXCLUDED.raw,
                    updated_at=now()
                """,
                (
                    sym,
                    day,
                    getattr(b, "open", None),
                    getattr(b, "high", None),
                    getattr(b, "low", None),
                    b.close,
                    getattr(b, "adj_close", None),
                    getattr(b, "volume", None),
                    source,
                    json.dumps(b.__dict__),
                ),
            )
            if day in missing:
                inserted += 1
            else:
                updated += 1
    return {
        "symbol": sym,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "inserted": inserted,
        "updated": updated,
        "source": source,
    }


def compute_and_store_features(symbol: str, as_of_date: dt.date, lookback: int) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    rows = db.safe_fetchall(
        """
        SELECT date, close, volume FROM prosperity_daily_bars
        WHERE symbol=%s AND date<=%s ORDER BY date ASC
        """,
        (sym, as_of_date),
    )
    if len(rows) < lookback:
        raise HTTPException(status_code=400, detail="insufficient bars for lookback")
    window = rows[-lookback:]
    from api.main import Candle, compute_features, detect_regime  # type: ignore

    candles = [Candle(timestamp=r[0].isoformat(), close=float(r[1]), volume=float(r[2]) if r[2] is not None else None) for r in window]
    feats = compute_features(candles)
    regime = detect_regime(feats)
    payload = {**feats, "regime": regime, "as_of": as_of_date.isoformat(), "lookback": int(lookback)}
    features_hash = _hash_dict(payload)

    db.safe_execute(
        """
        INSERT INTO prosperity_features_daily(
            symbol, as_of, lookback, features, meta
        ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT(symbol, as_of, lookback) DO UPDATE SET
            features=EXCLUDED.features,
            meta=EXCLUDED.meta,
            updated_at=now()
        """,
        (
            sym,
            as_of_date,
            lookback,
            json.dumps(feats),
            json.dumps({"regime": regime, "features_hash": features_hash}),
        ),
    )
    return {
        "symbol": sym,
        "as_of": as_of_date.isoformat(),
        "lookback": lookback,
        "stored": True,
        "features": feats,
        "regime": regime,
    }


def compute_and_store_signal(symbol: str, as_of_date: dt.date, lookback: int) -> Dict[str, Any]:
    sym = (symbol or "").strip().upper()
    rows = db.safe_fetchall(
        """
        SELECT date, close, volume FROM prosperity_daily_bars
        WHERE symbol=%s AND date<=%s ORDER BY date ASC
        """,
        (sym, as_of_date),
    )
    from api.main import Candle, compute_signal_for_symbol_from_candles, _score_mode  # type: ignore

    candles_all = [
        Candle(timestamp=r[0].isoformat(), close=float(r[1]), volume=float(r[2]) if r[2] is not None else None)
        for r in rows
    ]
    signal_payload = compute_signal_for_symbol_from_candles(sym, as_of_date.isoformat(), lookback, candles_all)
    signal_dict = signal_payload.model_dump()
    signal_hash = _hash_dict(signal_dict)
    score_mode = _score_mode()

    db.safe_execute(
        """
        INSERT INTO prosperity_signals_daily(
            symbol, as_of, lookback, score, signal, thresholds, regime, confidence, notes, features, meta
        ) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
        ON CONFLICT(symbol, as_of, lookback) DO UPDATE SET
            score=EXCLUDED.score,
            signal=EXCLUDED.signal,
            thresholds=EXCLUDED.thresholds,
            regime=EXCLUDED.regime,
            confidence=EXCLUDED.confidence,
            notes=EXCLUDED.notes,
            features=EXCLUDED.features,
            meta=EXCLUDED.meta,
            updated_at=now()
        """,
        (
            sym,
            as_of_date,
            lookback,
            signal_dict.get("score"),
            signal_dict.get("signal"),
            json.dumps(signal_dict.get("thresholds")),
            signal_dict.get("regime"),
            signal_dict.get("confidence"),
            json.dumps(signal_dict.get("notes")),
            json.dumps(signal_dict.get("features")),
            json.dumps(
                {
                    "score_mode": score_mode,
                    "base_score": signal_dict.get("base_score"),
                    "stacked_score": signal_dict.get("stacked_score"),
                    "calibration_meta": signal_dict.get("calibration_meta"),
                    "signal_hash": signal_hash,
                }
            ),
        ),
    )
    return signal_dict


def ingest_bars_bulk(symbols: List[str], from_date: dt.date, to_date: dt.date, *, concurrency: int = 3, force_refresh: bool = False) -> Dict[str, Any]:
    ok = 0
    errors: Dict[str, str] = {}
    for sym in symbols:
        try:
            ingest_bars(sym, from_date, to_date, force_refresh=force_refresh)
            ok += 1
        except Exception as e:  # pragma: no cover - passthrough
            errors[sym] = str(e)
    return {"ok": ok, "errors": errors}


def compute_features_bulk(symbols: List[str], as_of_date: dt.date, lookback: int) -> Dict[str, Any]:
    ok = 0
    errors: Dict[str, str] = {}
    for sym in symbols:
        try:
            compute_and_store_features(sym, as_of_date, lookback)
            ok += 1
        except Exception as e:
            errors[sym] = str(e)
    return {"ok": ok, "errors": errors}


def compute_signals_bulk(symbols: List[str], as_of_date: dt.date, lookback: int) -> Dict[str, Any]:
    ok = 0
    errors: Dict[str, str] = {}
    for sym in symbols:
        try:
            compute_and_store_signal(sym, as_of_date, lookback)
            ok += 1
        except Exception as e:
            errors[sym] = str(e)
    return {"ok": ok, "errors": errors}
