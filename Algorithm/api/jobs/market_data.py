from __future__ import annotations

import datetime as dt
import logging
import socket
import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from psycopg.types.json import Json

from api import config, db, security
from api.data_providers import (
    ProviderError,
    ProviderUnavailable,
    SymbolNoData,
    canonical_symbol,
    detect_country_exchange,
    fetch_daily_bars,
    fetch_fundamentals_quarterly,
    fetch_intraday_bars,
    fetch_news_items,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "TSLA",
    "SPY",
    "SHOP.TO",
    "CSU.TO",
    "BAM.TO",
    "RY.TO",
    "TD.TO",
]


class UniverseRequest(BaseModel):
    symbols: Optional[List[str]] = None
    mode: Optional[str] = None


class DailyBarsRequest(BaseModel):
    as_of_date: Optional[dt.date] = None
    from_date: Optional[dt.date] = None
    to_date: Optional[dt.date] = None
    symbols: Optional[List[str]] = None
    source: Optional[str] = "auto"


class IntradayBarsRequest(BaseModel):
    start_ts: dt.datetime
    end_ts: dt.datetime
    timeframe: str = "5m"
    symbols: Optional[List[str]] = None


class NewsRequest(BaseModel):
    from_ts: dt.datetime
    to_ts: dt.datetime
    symbols: Optional[List[str]] = None


class FundamentalsRequest(BaseModel):
    symbols: Optional[List[str]] = None


class SentimentRequest(BaseModel):
    as_of_date: dt.date
    symbols: Optional[List[str]] = None


class SymbolFailure(Exception):
    def __init__(self, reason_code: str, reason_detail: str) -> None:
        super().__init__(reason_detail)
        self.reason_code = reason_code
        self.reason_detail = reason_detail


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


def _lock_ttl_seconds() -> int:
    ttl = config.env_int("FTIP_JOB_LOCK_TTL_SECONDS", 1200)
    return max(ttl, 1)


def _lock_window_seconds() -> int:
    window = config.env_int("FTIP_JOB_LOCK_WINDOW_SEC", 120)
    return max(window, 0)


def _job_lock_owner() -> str:
    return (
        config.env("FTIP_JOB_LOCK_OWNER") or socket.gethostname() or "ftip-job-runner"
    )


def _cleanup_stale_job_runs(cur, job_name: str, ttl_seconds: int) -> List[str]:
    cur.execute(
        """
        UPDATE ftip_job_runs
        SET status = 'FAILED',
            error = 'stale lock cleared',
            finished_at = now(),
            updated_at = now()
        WHERE job_name = %s
          AND finished_at IS NULL
          AND started_at < now() - (%s || ' seconds')::interval
        RETURNING run_id
        """,
        (job_name, ttl_seconds),
    )
    rows = cur.fetchall() or []
    return [str(row[0]) for row in rows]


def _acquire_job_lock(
    run_id: str,
    job_name: str,
    as_of_date: dt.date,
    requested: Dict[str, object],
    ttl_seconds: int,
    lock_owner: str,
) -> Tuple[bool, Dict[str, Optional[str]]]:
    with db.with_connection() as (conn, cur):
        _cleanup_stale_job_runs(cur, job_name, ttl_seconds)
        lock_window_seconds = _lock_window_seconds()

        cur.execute(
            """
            SELECT run_id, started_at, lock_owner, lock_acquired_at, lock_expires_at
            FROM ftip_job_runs
            WHERE job_name = %s
              AND (finished_at IS NULL OR finished_at > now() - (%s || ' seconds')::interval)
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            (job_name, lock_window_seconds),
        )
        existing_locked = cur.fetchone()
        if existing_locked:
            conn.commit()
            existing_run_id, started_at, owner, lock_acquired_at, lock_expires_at = (
                existing_locked
            )
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": (
                    lock_acquired_at.isoformat() if lock_acquired_at else None
                ),
                "lock_expires_at": (
                    lock_expires_at.isoformat() if lock_expires_at else None
                ),
            }

        cur.execute(
            """
            SELECT run_id, started_at, lock_owner, lock_acquired_at, lock_expires_at
            FROM ftip_job_runs
            WHERE job_name = %s
              AND (finished_at IS NULL OR finished_at > now() - (%s || ' seconds')::interval)
            LIMIT 1
            """,
            (job_name, lock_window_seconds),
        )
        existing_pending = cur.fetchone()
        if existing_pending:
            conn.commit()
            existing_run_id, started_at, owner, lock_acquired_at, lock_expires_at = (
                existing_pending
            )
            return False, {
                "run_id": str(existing_run_id),
                "started_at": started_at.isoformat() if started_at else None,
                "lock_owner": owner,
                "lock_acquired_at": (
                    lock_acquired_at.isoformat() if lock_acquired_at else None
                ),
                "lock_expires_at": (
                    lock_expires_at.isoformat() if lock_expires_at else None
                ),
            }

        cur.execute(
            """
            INSERT INTO ftip_job_runs (
                run_id,
                job_name,
                as_of_date,
                started_at,
                requested,
                lock_owner,
                lock_acquired_at,
                lock_expires_at,
                status,
                created_at,
                updated_at,
                finished_at
            )
            VALUES (
                %s, %s, %s, now(), %s, %s, now(), now() + (%s || ' seconds')::interval,
                'RUNNING', now(), now(), NULL
            )
            """,
            (run_id, job_name, as_of_date, Json(requested), lock_owner, ttl_seconds),
        )
        conn.commit()
        return True, {"run_id": run_id}


def _update_job_run(
    run_id: str, status: str, result: Dict[str, object], error: Optional[str]
) -> None:
    with db.with_connection() as (conn, cur):
        cur.execute(
            """
            UPDATE ftip_job_runs
            SET finished_at = now(), status = %s, result = %s, error = %s, updated_at = now()
            WHERE run_id = %s
            """,
            (status, Json(result), error, run_id),
        )
        conn.commit()


def _log_symbol_coverage(
    run_id: str,
    job_name: str,
    as_of_date: dt.date,
    symbol: str,
    status: str,
    *,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
) -> None:
    db.safe_execute(
        """
        INSERT INTO prosperity_symbol_coverage(
            run_id, job_name, as_of_date, symbol, status, reason_code, reason_detail
        ) VALUES (%s,%s,%s,%s,%s,%s,%s)
        """,
        (run_id, job_name, as_of_date, symbol, status, reason_code, reason_detail),
    )


def _default_as_of_date() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _load_active_symbols() -> List[str]:
    rows = db.safe_fetchall("SELECT symbol FROM market_symbols WHERE is_active = TRUE")
    symbols = [row[0] for row in rows]
    return symbols or [canonical_symbol(s) for s in DEFAULT_UNIVERSE]


def _upsert_symbols(symbols: List[str]) -> None:
    with db.with_connection() as (conn, cur):
        for symbol in symbols:
            info = detect_country_exchange(symbol)
            cur.execute(
                """
                INSERT INTO market_symbols(symbol, exchange, country, currency, is_active, created_at, updated_at)
                VALUES (%s,%s,%s,%s,TRUE,now(),now())
                ON CONFLICT (symbol)
                DO UPDATE SET
                    exchange = EXCLUDED.exchange,
                    country = EXCLUDED.country,
                    currency = EXCLUDED.currency,
                    is_active = TRUE,
                    updated_at = now()
                """,
                (
                    symbol,
                    info.get("exchange"),
                    info.get("country"),
                    info.get("currency"),
                ),
            )
        conn.commit()


def _sentiment_lexicon() -> Dict[str, set[str]]:
    pos = {
        "beat",
        "strong",
        "growth",
        "upgrade",
        "surge",
        "record",
        "wins",
        "profit",
        "positive",
    }
    neg = {
        "miss",
        "weak",
        "downgrade",
        "drop",
        "loss",
        "negative",
        "cut",
        "lawsuit",
        "decline",
    }
    return {"pos": pos, "neg": neg}


def _score_headline(text: str, lexicon: Dict[str, set[str]]) -> float:
    tokens = [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split()]
    if not tokens:
        return 0.0
    pos_hits = sum(1 for tok in tokens if tok in lexicon["pos"])
    neg_hits = sum(1 for tok in tokens if tok in lexicon["neg"])
    return (pos_hits - neg_hits) / max(1, len(tokens))


@router.post("/data/universe")
async def upsert_universe(req: UniverseRequest):
    _require_db_enabled(write=True, read=True)
    symbols = req.symbols or []
    if req.mode == "default" or not symbols:
        symbols = DEFAULT_UNIVERSE
    cleaned = [canonical_symbol(sym) for sym in symbols]
    _upsert_symbols(cleaned)
    return {"status": "ok", "symbols": cleaned}


@router.post("/data/bars-daily")
async def ingest_bars_daily(req: DailyBarsRequest):
    _require_db_enabled(write=True, read=True)
    if not req.from_date or not req.to_date:
        raise HTTPException(status_code=400, detail="from_date and to_date required")
    if req.from_date > req.to_date:
        raise HTTPException(status_code=400, detail="from_date must be <= to_date")

    as_of_date = req.as_of_date or req.to_date
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    _upsert_symbols(symbols)

    run_id = uuid.uuid4().hex
    job_name = "data.bars_daily"
    ttl_seconds = _lock_ttl_seconds()
    lock_owner = _job_lock_owner()

    acquired, lock_info = _acquire_job_lock(
        run_id, job_name, as_of_date, req.model_dump(), ttl_seconds, lock_owner
    )
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            bars = fetch_daily_bars(symbol, req.from_date, req.to_date)
            with db.with_connection() as (conn, cur):
                for bar in bars:
                    cur.execute(
                        """
                        INSERT INTO market_bars_daily(
                            symbol, as_of_date, open, high, low, close, volume, source, ingested_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
                        ON CONFLICT (symbol, as_of_date)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source,
                            ingested_at = now()
                        """,
                        (
                            bar["symbol"],
                            bar["as_of_date"],
                            bar.get("open"),
                            bar.get("high"),
                            bar.get("low"),
                            bar.get("close"),
                            bar.get("volume"),
                            bar.get("source") or "unknown",
                        ),
                    )
                    rows_written += 1
                conn.commit()

            db.safe_execute(
                """
                INSERT INTO quality_daily(symbol, as_of_date, bars_ok, updated_at)
                VALUES (%s,%s,TRUE,now())
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET bars_ok = TRUE, updated_at = now()
                """,
                (symbol, as_of_date),
            )
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except ProviderError as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code=exc.reason_code,
                reason_detail=exc.reason_detail,
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": exc.reason_detail,
                    "reason_code": exc.reason_code,
                }
            )
        except Exception as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code="UNEXPECTED_ERROR",
                reason_detail=str(exc),
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": str(exc),
                    "reason_code": "UNEXPECTED_ERROR",
                }
            )

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"market_bars_daily": rows_written},
        "timings": {},
    }
    _update_job_run(
        run_id, status.upper(), payload, None if status == "ok" else "partial failures"
    )
    return payload


@router.post("/data/bars-intraday")
async def ingest_bars_intraday(req: IntradayBarsRequest):
    _require_db_enabled(write=True, read=True)
    if req.start_ts >= req.end_ts:
        raise HTTPException(status_code=400, detail="start_ts must be before end_ts")

    as_of_date = req.end_ts.astimezone(dt.timezone.utc).date()
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    _upsert_symbols(symbols)

    run_id = uuid.uuid4().hex
    job_name = "data.bars_intraday"
    acquired, lock_info = _acquire_job_lock(
        run_id,
        job_name,
        as_of_date,
        req.model_dump(),
        _lock_ttl_seconds(),
        _job_lock_owner(),
    )
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            bars = fetch_intraday_bars(symbol, req.start_ts, req.end_ts, req.timeframe)
            with db.with_connection() as (conn, cur):
                for bar in bars:
                    cur.execute(
                        """
                        INSERT INTO market_bars_intraday(
                            symbol, ts, timeframe, open, high, low, close, volume, source, ingested_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                        ON CONFLICT (symbol, ts, timeframe)
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source,
                            ingested_at = now()
                        """,
                        (
                            bar["symbol"],
                            bar["ts"],
                            bar["timeframe"],
                            bar.get("open"),
                            bar.get("high"),
                            bar.get("low"),
                            bar.get("close"),
                            bar.get("volume"),
                            bar.get("source") or "unknown",
                        ),
                    )
                    rows_written += 1
                conn.commit()

            if bars:
                db.safe_execute(
                    """
                    INSERT INTO quality_daily(symbol, as_of_date, intraday_ok, updated_at)
                    VALUES (%s,%s,TRUE,now())
                    ON CONFLICT (symbol, as_of_date)
                    DO UPDATE SET intraday_ok = TRUE, updated_at = now()
                    """,
                    (symbol, as_of_date),
                )
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except ProviderUnavailable as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code=exc.reason_code,
                reason_detail=exc.reason_detail,
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": exc.reason_detail,
                    "reason_code": exc.reason_code,
                }
            )
        except SymbolNoData as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code=exc.reason_code,
                reason_detail=exc.reason_detail,
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": exc.reason_detail,
                    "reason_code": exc.reason_code,
                }
            )
        except Exception as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code="UNEXPECTED_ERROR",
                reason_detail=str(exc),
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": str(exc),
                    "reason_code": "UNEXPECTED_ERROR",
                }
            )

    if not symbols_failed:
        status = "ok"
    elif symbols_ok:
        status = "partial"
    else:
        provider_unavailable = any(
            item.get("reason_code") in {"PROVIDER_UNAVAILABLE", "PROVIDER_UNSUPPORTED"}
            for item in symbols_failed
        )
        status = "partial" if provider_unavailable else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"market_bars_intraday": rows_written},
        "timings": {},
    }
    _update_job_run(
        run_id, status.upper(), payload, None if status == "ok" else "partial failures"
    )
    return payload


@router.post("/data/news")
async def ingest_news(req: NewsRequest):
    _require_db_enabled(write=True, read=True)
    if req.from_ts >= req.to_ts:
        raise HTTPException(status_code=400, detail="from_ts must be before to_ts")

    as_of_date = req.to_ts.astimezone(dt.timezone.utc).date()
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    _upsert_symbols(symbols)

    run_id = uuid.uuid4().hex
    job_name = "data.news"
    acquired, lock_info = _acquire_job_lock(
        run_id,
        job_name,
        as_of_date,
        req.model_dump(),
        _lock_ttl_seconds(),
        _job_lock_owner(),
    )
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            items = fetch_news_items(symbol, req.from_ts, req.to_ts)
            with db.with_connection() as (conn, cur):
                for item in items:
                    cur.execute(
                        """
                        INSERT INTO news_raw(
                            symbol, published_at, source, title, url, url_hash, content_snippet, ingested_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,now())
                        ON CONFLICT (url_hash)
                        DO NOTHING
                        """,
                        (
                            item["symbol"],
                            item["published_at"],
                            item["source"],
                            item["title"],
                            item["url"],
                            item["url_hash"],
                            item.get("content_snippet"),
                        ),
                    )
                    rows_written += 1
                conn.commit()
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except Exception as exc:
            reason_code = "PROVIDER_UNAVAILABLE"
            reason_detail = str(exc)
            if isinstance(exc, ProviderError):
                reason_code = exc.reason_code
                reason_detail = exc.reason_detail
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code=reason_code,
                reason_detail=reason_detail,
            )
            symbols_failed.append(
                {"symbol": symbol, "reason": reason_detail, "reason_code": reason_code}
            )

    if not symbols_failed:
        status = "ok"
    elif symbols_ok:
        status = "partial"
    else:
        provider_unavailable = any(
            item.get("reason_code") == "PROVIDER_UNAVAILABLE" for item in symbols_failed
        )
        status = "partial" if provider_unavailable else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"news_raw": rows_written},
        "timings": {},
    }
    _update_job_run(
        run_id, status.upper(), payload, None if status == "ok" else "partial failures"
    )
    return payload


@router.post("/data/fundamentals")
async def ingest_fundamentals(req: FundamentalsRequest):
    _require_db_enabled(write=True, read=True)
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    _upsert_symbols(symbols)

    run_id = uuid.uuid4().hex
    job_name = "data.fundamentals"
    as_of_date = _default_as_of_date()
    acquired, lock_info = _acquire_job_lock(
        run_id,
        job_name,
        as_of_date,
        req.model_dump(),
        _lock_ttl_seconds(),
        _job_lock_owner(),
    )
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            fundamentals = fetch_fundamentals_quarterly(symbol)
            with db.with_connection() as (conn, cur):
                for row in fundamentals:
                    cur.execute(
                        """
                        INSERT INTO fundamentals_quarterly(
                            symbol, fiscal_period_end, report_date, revenue, eps, gross_margin, op_margin, fcf, source, ingested_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                        ON CONFLICT (symbol, fiscal_period_end)
                        DO UPDATE SET
                            report_date = EXCLUDED.report_date,
                            revenue = EXCLUDED.revenue,
                            eps = EXCLUDED.eps,
                            gross_margin = EXCLUDED.gross_margin,
                            op_margin = EXCLUDED.op_margin,
                            fcf = EXCLUDED.fcf,
                            source = EXCLUDED.source,
                            ingested_at = now()
                        """,
                        (
                            row["symbol"],
                            row["fiscal_period_end"],
                            row.get("report_date"),
                            row.get("revenue"),
                            row.get("eps"),
                            row.get("gross_margin"),
                            row.get("op_margin"),
                            row.get("fcf"),
                            row.get("source") or "unknown",
                        ),
                    )
                    rows_written += 1
                conn.commit()

            db.safe_execute(
                """
                INSERT INTO quality_daily(symbol, as_of_date, fundamentals_ok, updated_at)
                VALUES (%s,%s,TRUE,now())
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET fundamentals_ok = TRUE, updated_at = now()
                """,
                (symbol, as_of_date),
            )
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            symbols_ok.append(symbol)
        except ProviderError as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code=exc.reason_code,
                reason_detail=exc.reason_detail,
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": exc.reason_detail,
                    "reason_code": exc.reason_code,
                }
            )
        except Exception as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code="UNEXPECTED_ERROR",
                reason_detail=str(exc),
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": str(exc),
                    "reason_code": "UNEXPECTED_ERROR",
                }
            )

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"fundamentals_quarterly": rows_written},
        "timings": {},
    }
    _update_job_run(
        run_id, status.upper(), payload, None if status == "ok" else "partial failures"
    )
    return payload


@router.post("/data/sentiment-daily")
async def compute_sentiment(req: SentimentRequest):
    _require_db_enabled(write=True, read=True)
    as_of_date = req.as_of_date
    start_ts = dt.datetime.combine(as_of_date, dt.time.min).replace(
        tzinfo=dt.timezone.utc
    )
    end_ts = dt.datetime.combine(as_of_date, dt.time.max).replace(
        tzinfo=dt.timezone.utc
    )
    symbols = [canonical_symbol(sym) for sym in (req.symbols or _load_active_symbols())]
    _upsert_symbols(symbols)

    run_id = uuid.uuid4().hex
    job_name = "data.sentiment_daily"
    acquired, lock_info = _acquire_job_lock(
        run_id,
        job_name,
        as_of_date,
        req.model_dump(),
        _lock_ttl_seconds(),
        _job_lock_owner(),
    )
    if not acquired:
        return JSONResponse(status_code=409, content={"error": "locked", **lock_info})

    lexicon = _sentiment_lexicon()
    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []
    rows_written = 0

    for symbol in symbols:
        try:
            rows = db.safe_fetchall(
                """
                SELECT title
                FROM news_raw
                WHERE symbol = %s
                  AND published_at >= %s
                  AND published_at <= %s
                """,
                (symbol, start_ts, end_ts),
            )
            scores = [_score_headline(row[0], lexicon) for row in rows]
            headline_count = len(scores)
            pos = sum(1 for score in scores if score > 0)
            neg = sum(1 for score in scores if score < 0)
            neu = headline_count - pos - neg
            sentiment_mean = (
                float(sum(scores) / headline_count) if headline_count else None
            )
            sentiment_score = sentiment_mean
            db.safe_execute(
                """
                INSERT INTO sentiment_daily(
                    symbol, as_of_date, headline_count, sentiment_mean, sentiment_pos,
                    sentiment_neg, sentiment_neu, sentiment_score, source, computed_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET
                    headline_count = EXCLUDED.headline_count,
                    sentiment_mean = EXCLUDED.sentiment_mean,
                    sentiment_pos = EXCLUDED.sentiment_pos,
                    sentiment_neg = EXCLUDED.sentiment_neg,
                    sentiment_neu = EXCLUDED.sentiment_neu,
                    sentiment_score = EXCLUDED.sentiment_score,
                    source = EXCLUDED.source,
                    computed_at = now()
                """,
                (
                    symbol,
                    as_of_date,
                    headline_count,
                    sentiment_mean,
                    pos,
                    neg,
                    neu,
                    sentiment_score,
                    "lexicon_v1",
                ),
            )
            db.safe_execute(
                """
                INSERT INTO quality_daily(symbol, as_of_date, sentiment_ok, updated_at)
                VALUES (%s,%s,TRUE,now())
                ON CONFLICT (symbol, as_of_date)
                DO UPDATE SET sentiment_ok = TRUE, updated_at = now()
                """,
                (symbol, as_of_date),
            )
            _log_symbol_coverage(run_id, job_name, as_of_date, symbol, "OK")
            rows_written += 1
            symbols_ok.append(symbol)
        except Exception as exc:
            _log_symbol_coverage(
                run_id,
                job_name,
                as_of_date,
                symbol,
                "FAILED",
                reason_code="UNEXPECTED_ERROR",
                reason_detail=str(exc),
            )
            symbols_failed.append(
                {
                    "symbol": symbol,
                    "reason": str(exc),
                    "reason_code": "UNEXPECTED_ERROR",
                }
            )

    status = "ok" if not symbols_failed else "partial" if symbols_ok else "failed"
    payload = {
        "status": status,
        "run_id": run_id,
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": {"sentiment_daily": rows_written},
        "timings": {},
    }
    _update_job_run(
        run_id, status.upper(), payload, None if status == "ok" else "partial failures"
    )
    return payload
