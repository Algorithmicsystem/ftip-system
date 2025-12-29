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
    payload = {**feats, "regime": regime, "as_of_date": as_of_date.isoformat(), "lookback": int(lookback)}
    features_hash = _hash_dict(payload)

    db.safe_execute(
        """
        INSERT INTO prosperity_features_daily(
            symbol, as_of_date, lookback, mom_5, mom_21, mom_63, trend_sma20_50, volatility_ann, rsi14, volume_z20, last_close, regime, features_hash
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT(symbol, as_of_date, lookback) DO UPDATE SET
            mom_5=EXCLUDED.mom_5,
            mom_21=EXCLUDED.mom_21,
            mom_63=EXCLUDED.mom_63,
            trend_sma20_50=EXCLUDED.trend_sma20_50,
            volatility_ann=EXCLUDED.volatility_ann,
            rsi14=EXCLUDED.rsi14,
            volume_z20=EXCLUDED.volume_z20,
            last_close=EXCLUDED.last_close,
            regime=EXCLUDED.regime,
            features_hash=EXCLUDED.features_hash,
            updated_at=now()
        """,
        (
            sym,
            as_of_date,
            lookback,
            feats.get("mom_5"),
            feats.get("mom_21"),
            feats.get("mom_63"),
            feats.get("trend_sma20_50"),
            feats.get("volatility_ann"),
            feats.get("rsi14"),
            feats.get("volume_z20"),
            feats.get("last_close"),
            regime,
            features_hash,
        ),
    )
    return {
        "symbol": sym,
        "as_of_date": as_of_date.isoformat(),
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
    if len(rows) < lookback:
        raise HTTPException(status_code=400, detail="insufficient bars for lookback")
    from api.main import Candle, compute_signal_for_symbol_from_candles, _score_mode  # type: ignore

    candles = [Candle(timestamp=r[0].isoformat(), close=float(r[1]), volume=float(r[2]) if r[2] is not None else None) for r in rows[-lookback:]]
    signal_payload = compute_signal_for_symbol_from_candles(sym, candles, lookback)
    signal_hash = _hash_dict(signal_payload)
    score_mode = _score_mode()

    db.safe_execute(
        """
        INSERT INTO prosperity_signals_daily(
            symbol, as_of_date, lookback, score_mode, score, base_score, stacked_score, thresholds, signal, confidence, regime, calibration_meta, notes, signal_hash
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
        ON CONFLICT(symbol, as_of_date, lookback, score_mode) DO UPDATE SET
            score=EXCLUDED.score,
            base_score=EXCLUDED.base_score,
            stacked_score=EXCLUDED.stacked_score,
            thresholds=EXCLUDED.thresholds,
            signal=EXCLUDED.signal,
            confidence=EXCLUDED.confidence,
            regime=EXCLUDED.regime,
            calibration_meta=EXCLUDED.calibration_meta,
            notes=EXCLUDED.notes,
            signal_hash=EXCLUDED.signal_hash,
            updated_at=now()
        """,
        (
            sym,
            as_of_date,
            lookback,
            score_mode,
            signal_payload.get("score"),
            signal_payload.get("base_score"),
            signal_payload.get("stacked_score"),
            json.dumps(signal_payload.get("thresholds")),
            signal_payload.get("signal"),
            signal_payload.get("confidence"),
            signal_payload.get("regime"),
            json.dumps(signal_payload.get("calibration_meta")),
            json.dumps(signal_payload.get("notes")),
            signal_hash,
        ),
    )
    return signal_payload


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
