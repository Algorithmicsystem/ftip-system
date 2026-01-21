from __future__ import annotations

import datetime as dt
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import APIRouter, Depends, HTTPException, Request

from api import config, db, security
from api.ops import metrics_tracker
from api.prosperity import ingest, query
from api.prosperity.narrator import router as narrator_router
from api.prosperity.strategy_graph import router as strategy_graph_router
from api.prosperity.models import (
    BarsIngestBulkRequest,
    BarsIngestRequest,
    BarsResponse,
    FeaturesComputeRequest,
    HealthResponse,
    SignalsComputeRequest,
    SnapshotRunRequest,
    UniverseUpsertRequest,
)

router = APIRouter(dependencies=[Depends(security.require_prosperity_api_key)])
logger = logging.getLogger(__name__)

router.include_router(strategy_graph_router, prefix="/strategy_graph")
router.include_router(narrator_router, prefix="/narrator")


class SymbolFailure(Exception):
    def __init__(
        self,
        reason_code: str,
        reason_detail: str = "",
        *,
        bars_required: Optional[int] = None,
        bars_returned: Optional[int] = None,
    ):
        super().__init__(reason_detail or reason_code)
        self.reason_code = reason_code
        self.reason_detail = reason_detail or reason_code
        self.bars_required = bars_required
        self.bars_returned = bars_returned


def _require_db_enabled(write: bool = False, read: bool = False) -> None:
    if not db.db_enabled():
        raise HTTPException(status_code=503, detail="database disabled")
    if write and not db.db_write_enabled():
        raise HTTPException(status_code=503, detail="database writes disabled")
    if read and not db.db_read_enabled():
        raise HTTPException(status_code=503, detail="database reads disabled")


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        db_enabled=db.db_enabled(),
        db_write_enabled=db.db_write_enabled(),
        db_read_enabled=db.db_read_enabled(),
    )


@router.post("/bootstrap")
async def bootstrap(request: Request):
    token = config.env("PROSPERITY_ADMIN_TOKEN")
    if token and request.headers.get("x-admin-token") != token:
        raise HTTPException(status_code=403, detail="forbidden")

    if not db.db_enabled():
        return {"status": "ok", "db_enabled": False, "migrated": False, "versions": []}

    from api import migrations

    versions = migrations.ensure_schema()
    db.ensure_schema()
    return {"status": "ok", "db_enabled": True, "migrated": bool(versions), "versions": versions}


@router.post("/universe/upsert")
async def universe_upsert(req: UniverseUpsertRequest):
    _require_db_enabled(write=True)
    count, symbols = ingest.upsert_universe(req.symbols)
    return {"count": count, "symbols": symbols}


@router.post("/bars/ingest")
async def bars_ingest(req: BarsIngestRequest):
    _require_db_enabled(write=True)
    return ingest.ingest_bars(req.symbol, req.from_date, req.to_date, force_refresh=req.force_refresh)


@router.post("/bars/ingest_bulk")
async def bars_ingest_bulk(req: BarsIngestBulkRequest):
    _require_db_enabled(write=True)
    return ingest.ingest_bars_bulk(req.symbols, req.from_date, req.to_date, concurrency=req.concurrency, force_refresh=req.force_refresh)


@router.get("/bars", response_model=BarsResponse)
async def bars(symbol: str, from_date: dt.date, to_date: dt.date):
    _require_db_enabled(read=True)
    data = query.fetch_bars(symbol.upper(), from_date, to_date)
    return BarsResponse(symbol=symbol.upper(), from_date=from_date, to_date=to_date, data=data)


@router.post("/features/compute")
async def features_compute(req: FeaturesComputeRequest):
    _require_db_enabled(write=True)
    return ingest.compute_and_store_features(req.symbol, req.as_of_date, req.lookback)


@router.post("/signals/compute")
async def signals_compute(req: SignalsComputeRequest):
    _require_db_enabled(write=True)
    return ingest.compute_and_store_signal(req.symbol, req.as_of_date, req.lookback)


def _normalize_symbols(symbols: List[str]) -> List[str]:
    cleaned = sorted({(s or "").strip().upper() for s in symbols if s and s.strip()})
    if not cleaned:
        raise HTTPException(status_code=400, detail="symbols required")
    return cleaned


def _bars_required(lookback: int) -> int:
    return max(int(lookback), 252)


def _validate_bars(
    symbol: str,
    bars: List[Dict[str, Any]],
    required: int,
    *,
    window_start: Optional[dt.date] = None,
    window_end: Optional[dt.date] = None,
) -> None:
    window_detail = ""
    if window_start and window_end:
        window_detail = f" window={window_start.isoformat()}..{window_end.isoformat()}"
    if not bars:
        raise SymbolFailure(
            "NO_DATA",
            f"no bars returned for {symbol}{window_detail}",
            bars_required=required,
            bars_returned=0,
        )
    if len(bars) < required:
        detail = f"required={required} returned={len(bars)}{window_detail}"
        raise SymbolFailure(
            "INSUFFICIENT_BARS",
            detail,
            bars_required=required,
            bars_returned=len(bars),
        )


def _candles_from_bars(bars: List[Dict[str, Any]]):
    from api.main import Candle  # type: ignore

    candles = [
        Candle(
            timestamp=b["date"],
            close=float(b["close"]),
            volume=float(b["volume"]) if b.get("volume") is not None else None,
        )
        for b in bars
    ]
    return candles


def _compute_features_payload(symbol: str, as_of_date: dt.date, lookback: int, candles):
    from api.main import compute_features, detect_regime  # type: ignore

    feats = compute_features(candles[-lookback:])
    regime = detect_regime(feats)
    payload = {**feats, "regime": regime, "as_of": as_of_date.isoformat(), "lookback": int(lookback)}
    meta = {"regime": regime, "features_hash": ingest._hash_dict(payload)}
    return feats, meta, regime


def _compute_signal_payload(symbol: str, as_of_date: dt.date, lookback: int, candles_all):
    from api.main import compute_signal_for_symbol_from_candles, _score_mode  # type: ignore

    signal_payload = compute_signal_for_symbol_from_candles(
        symbol, as_of_date.isoformat(), lookback, candles_all
    )
    signal_dict = signal_payload.model_dump()
    signal_hash = ingest._hash_dict(signal_dict)

    preferred_score_mode = signal_dict.get("score_mode")
    calibration_meta = signal_dict.get("calibration_meta") or {}
    if not preferred_score_mode:
        preferred_score_mode = calibration_meta.get("score_mode")
    if not preferred_score_mode:
        notes = signal_dict.get("notes") or []
        preferred_score_mode = "stacked" if any(isinstance(n, str) and "STACKED" in n.upper() for n in notes) else None
    score_mode = preferred_score_mode or _score_mode() or "single"

    base_score = signal_dict.get("base_score")
    if base_score is None:
        base_score = calibration_meta.get("base_score")
    if base_score is None:
        base_score = signal_dict.get("score")

    stacked_score = signal_dict.get("stacked_score")
    thresholds = signal_dict.get("thresholds") or {}
    notes = signal_dict.get("notes") or []
    features = signal_dict.get("features") or {}
    meta = signal_dict.get("meta") or {}

    regime = signal_dict.get("regime")
    confidence = signal_dict.get("confidence")

    return {
        "signal_dict": signal_dict,
        "signal_hash": signal_hash,
        "score_mode": score_mode,
        "base_score": base_score,
        "stacked_score": stacked_score,
        "thresholds": thresholds,
        "notes": notes,
        "features": features,
        "meta": meta,
        "regime": regime,
        "confidence": confidence,
    }


def _compute_strategy_graph(symbol: str, as_of_date: dt.date, lookback: int, candles):
    from api.main import Candle  # type: ignore  # noqa: F401
    from ftip.strategy_graph import compute_strategy_graph

    res = compute_strategy_graph(symbol, as_of_date, lookback, candles)
    strat_rows = []
    for strat in res.get("strategies", []):
        strat_rows.append(
            {
                "symbol": symbol,
                "as_of_date": as_of_date,
                "lookback": lookback,
                "strategy_id": strat.get("strategy_id"),
                "strategy_version": strat.get("version"),
                "regime": res.get("regime"),
                "raw_score": strat.get("raw_score"),
                "normalized_score": strat.get("normalized_score"),
                "signal": strat.get("signal"),
                "confidence": strat.get("confidence"),
                "rationale": strat.get("rationale"),
                "feature_contributions": strat.get("feature_contributions"),
                "meta": {"regime_meta": res.get("regime_meta")},
            }
        )
    ensemble_row = res.get("ensemble") or {}
    ensemble_payload = {
        "symbol": symbol,
        "as_of_date": as_of_date,
        "lookback": lookback,
        "regime": res.get("regime"),
        "ensemble_method": ensemble_row.get("ensemble_method"),
        "final_signal": ensemble_row.get("final_signal"),
        "final_score": ensemble_row.get("final_score"),
        "final_confidence": ensemble_row.get("final_confidence"),
        "thresholds": ensemble_row.get("thresholds"),
        "risk_overlay_applied": ensemble_row.get("risk_overlay_applied"),
        "strategies_used": ensemble_row.get("strategies_used"),
        "audit": res.get("audit"),
        "hashes": res.get("hashes"),
    }
    return strat_rows, ensemble_payload


def _persist_symbol_outputs(
    symbol: str,
    as_of_date: dt.date,
    lookback: int,
    features: Dict[str, Any],
    feature_meta: Dict[str, Any],
    signal_payload: Dict[str, Any],
    *,
    strategy_rows: Optional[List[Dict[str, Any]]] = None,
    ensemble_row: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    rows_written = {"features": 0, "signals": 0, "strategies": 0, "ensembles": 0}
    with db.with_connection() as (conn, cur):
        try:
            cur.execute(
                """
                INSERT INTO prosperity_features_daily(
                    symbol, as_of, lookback, features, meta
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT(symbol, as_of, lookback) DO UPDATE SET
                    features=EXCLUDED.features,
                    meta=EXCLUDED.meta,
                    updated_at=now()
                """,
                (symbol, as_of_date, lookback, json.dumps(features), json.dumps(feature_meta)),
            )
            rows_written["features"] += 1

            cur.execute(
                """
                INSERT INTO prosperity_signals_daily(
                    symbol, as_of, lookback, score_mode, score, base_score, stacked_score, signal, thresholds, regime, confidence, notes, features, calibration_meta, meta, signal_hash
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s)
                ON CONFLICT(symbol, as_of, lookback) DO UPDATE SET
                    score_mode=EXCLUDED.score_mode,
                    score=EXCLUDED.score,
                    base_score=EXCLUDED.base_score,
                    stacked_score=EXCLUDED.stacked_score,
                    signal=EXCLUDED.signal,
                    thresholds=EXCLUDED.thresholds,
                    regime=EXCLUDED.regime,
                    confidence=EXCLUDED.confidence,
                    notes=EXCLUDED.notes,
                    features=EXCLUDED.features,
                    calibration_meta=EXCLUDED.calibration_meta,
                    meta=EXCLUDED.meta,
                    signal_hash=EXCLUDED.signal_hash,
                    updated_at=now()
                """,
                (
                    symbol,
                    as_of_date,
                    lookback,
                    signal_payload["score_mode"],
                    signal_payload["signal_dict"].get("score"),
                    signal_payload["base_score"],
                    signal_payload["stacked_score"],
                    signal_payload["signal_dict"].get("signal"),
                    json.dumps(signal_payload["thresholds"]),
                    signal_payload["regime"],
                    signal_payload["confidence"],
                    json.dumps(signal_payload["notes"]),
                    json.dumps(signal_payload["features"]),
                    json.dumps(signal_payload["signal_dict"].get("calibration_meta") or {}),
                    json.dumps(signal_payload["meta"]),
                    signal_payload["signal_hash"],
                ),
            )
            rows_written["signals"] += 1

            if strategy_rows is not None:
                for row in strategy_rows:
                    cur.execute(
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
                    rows_written["strategies"] += 1

            if ensemble_row is not None:
                cur.execute(
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
                        "symbol": ensemble_row["symbol"],
                        "as_of_date": ensemble_row["as_of_date"],
                        "lookback": ensemble_row["lookback"],
                        "regime": ensemble_row.get("regime"),
                        "ensemble_method": ensemble_row.get("ensemble_method"),
                        "final_signal": ensemble_row.get("final_signal"),
                        "final_score": ensemble_row.get("final_score"),
                        "final_confidence": ensemble_row.get("final_confidence"),
                        "thresholds": json.dumps(ensemble_row.get("thresholds") or {}),
                        "risk_overlay_applied": bool(ensemble_row.get("risk_overlay_applied", False)),
                        "strategies_used": json.dumps(ensemble_row.get("strategies_used") or []),
                        "audit": json.dumps(ensemble_row.get("audit") or {}),
                        "hashes": json.dumps(ensemble_row.get("hashes") or {}),
                    },
                )
                rows_written["ensembles"] += 1

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return rows_written


def _log_symbol_coverage(
    run_id: Optional[str],
    job_name: Optional[str],
    as_of_date: dt.date,
    symbol: str,
    status: str,
    *,
    reason_code: Optional[str] = None,
    reason_detail: Optional[str] = None,
    bars_required: Optional[int] = None,
    bars_returned: Optional[int] = None,
    lock_owner: Optional[str] = None,
) -> None:
    if not run_id or not job_name:
        return
    try:
        db.safe_execute(
            """
            INSERT INTO prosperity_symbol_coverage(
                run_id, job_name, as_of_date, symbol, status, reason_code, reason_detail, bars_returned, bars_required
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id,
                job_name,
                as_of_date,
                symbol,
                status,
                reason_code,
                reason_detail,
                bars_returned,
                bars_required,
            ),
        )
        logger.info(
            "daily_snapshot.coverage_logged",
            extra={
                "run_id": run_id,
                "as_of_date": as_of_date.isoformat(),
                "lock_owner": lock_owner,
                "symbol": symbol,
                "status": status,
                "reason_code": reason_code,
            },
        )
    except Exception:
        logger.warning(
            "jobs.prosperity.daily_snapshot.coverage_failed",
            extra={"run_id": run_id, "symbol": symbol, "status": status},
        )


def _classify_error(exc: Exception) -> Tuple[str, str, bool]:
    if isinstance(exc, SymbolFailure):
        return exc.reason_code, exc.reason_detail or exc.reason_code, False
    if isinstance(exc, HTTPException):
        detail = str(exc.detail)
        if exc.status_code >= 500:
            return "UPSTREAM_5XX", detail, True
        return "VALIDATION_ERROR", detail, False
    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout)):
        return "TIMEOUT", str(exc), True
    if isinstance(exc, requests.exceptions.RequestException):
        return "NETWORK_ERROR", str(exc), True
    return "UNEXPECTED_ERROR", str(exc), False


def _retry_sleep(attempt: int) -> None:
    base_sleep = 1.0
    jitter = random.uniform(0.0, 0.5)
    time.sleep(base_sleep * (2 ** (attempt - 1)) + jitter)


@router.post("/snapshot/run")
async def snapshot_run(
    req: SnapshotRunRequest,
    request: Request,
    run_id: Optional[str] = None,
    job_name: Optional[str] = None,
    lock_owner: Optional[str] = None,
):
    _require_db_enabled(write=True, read=True)
    trace_id = getattr(request.state, "trace_id", None)
    if req.from_date > req.to_date:
        raise HTTPException(status_code=400, detail="from_date must be on/before to_date")
    if req.to_date > req.as_of_date:
        raise HTTPException(status_code=400, detail="as_of_date must be on/after to_date")

    symbols = req.symbols or ((config.env("FTIP_UNIVERSE_DEFAULT", "") or "AAPL,MSFT").split(","))
    symbols = _normalize_symbols(symbols)
    concurrency = min(max(req.concurrency, 1), 5)

    requested = {
        "symbols": symbols,
        "from_date": req.from_date.isoformat(),
        "to_date": req.to_date.isoformat(),
        "as_of_date": req.as_of_date.isoformat(),
        "lookback": req.lookback,
        "concurrency": concurrency,
        "force_refresh": bool(req.force_refresh),
    }

    timings: Dict[str, float] = {}
    rows_written = {"signals": 0, "features": 0}
    strategy_graph_rows = {"strategies": 0, "ensembles": 0}
    symbols_ok: List[str] = []
    symbols_failed: List[Dict[str, str]] = []

    t0 = time.time()
    try:
        ingest.upsert_universe(symbols)
    except Exception as exc:  # pragma: no cover - guarded via validation
        raise HTTPException(status_code=503, detail=f"failed to upsert universe: {exc}")
    timings["upsert_universe"] = time.time() - t0

    max_attempts = 3

    for sym in symbols:
        sym_start = time.time()
        bars_returned: Optional[int] = None
        bars_required = _bars_required(req.lookback)
        effective_as_of: Optional[dt.date] = None
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            try:
                ingest.ingest_bars(sym, req.from_date, req.to_date, force_refresh=req.force_refresh)
                initial_bars, effective_as_of = query.fetch_bars_with_latest(sym, req.from_date, req.to_date)
                if effective_as_of is None:
                    bars_returned = len(initial_bars)
                    _validate_bars(sym, initial_bars, bars_required)

                # Over-fetch to cover weekend/holiday gaps, then slice to the required lookback window.
                window_start = effective_as_of - dt.timedelta(days=420)
                bars = query.fetch_bars(sym, window_start, effective_as_of)
                bars_returned = len(bars)
                _validate_bars(
                    sym,
                    bars,
                    bars_required,
                    window_start=window_start,
                    window_end=effective_as_of,
                )
                bars = bars[-bars_required:]

                candles = _candles_from_bars(bars)
                feats, feat_meta, regime = _compute_features_payload(sym, effective_as_of, req.lookback, candles)
                signal_payload = _compute_signal_payload(sym, effective_as_of, req.lookback, candles)
                strategy_rows = None
                ensemble_row = None
                if req.compute_strategy_graph:
                    strategy_rows, ensemble_row = _compute_strategy_graph(
                        sym, effective_as_of, req.lookback, candles
                    )

                written = _persist_symbol_outputs(
                    sym,
                    effective_as_of,
                    req.lookback,
                    feats,
                    {**feat_meta, "regime": regime},
                    signal_payload,
                    strategy_rows=strategy_rows if req.compute_strategy_graph else None,
                    ensemble_row=ensemble_row if req.compute_strategy_graph else None,
                )
                rows_written["features"] += written.get("features", 0)
                rows_written["signals"] += written.get("signals", 0)
                strategy_graph_rows["strategies"] += written.get("strategies", 0)
                strategy_graph_rows["ensembles"] += written.get("ensembles", 0)
                symbols_ok.append(sym)
                _log_symbol_coverage(
                    run_id,
                    job_name,
                    effective_as_of,
                    sym,
                    "OK",
                    bars_required=bars_required,
                    bars_returned=bars_returned,
                    lock_owner=lock_owner,
                )
                logger.info(
                    "jobs.prosperity.daily_snapshot.symbol_ok",
                    extra={"symbol": sym, "duration_sec": time.time() - sym_start, "trace_id": trace_id},
                )
                break
            except Exception as exc:
                reason_code, reason_detail, retryable = _classify_error(exc)
                if retryable and attempts < max_attempts:
                    logger.info(
                        "daily_snapshot.retry",
                        extra={
                            "run_id": run_id,
                            "as_of_date": req.as_of_date.isoformat(),
                            "lock_owner": lock_owner,
                            "symbol": sym,
                            "attempt": attempts + 1,
                            "reason_code": reason_code,
                        },
                    )
                    _retry_sleep(attempts)
                    continue

                symbols_failed.append(
                    {
                        "symbol": sym,
                        "reason": reason_detail or reason_code,
                        "reason_code": reason_code,
                        "reason_detail": reason_detail or reason_code,
                        "attempts": attempts,
                        "retryable": bool(retryable),
                    }
                )
                if isinstance(exc, SymbolFailure):
                    bars_required = exc.bars_required or bars_required
                    bars_returned = exc.bars_returned if exc.bars_returned is not None else bars_returned
                _log_symbol_coverage(
                    run_id,
                    job_name,
                    effective_as_of or req.as_of_date,
                    sym,
                    "FAILED",
                    reason_code=reason_code,
                    reason_detail=reason_detail,
                    bars_required=bars_required,
                    bars_returned=bars_returned,
                    lock_owner=lock_owner,
                )
                logger.warning(
                    "jobs.prosperity.daily_snapshot.symbol_failed",
                    extra={
                        "symbol": sym,
                        "reason_code": reason_code,
                        "reason_detail": reason_detail,
                        "duration_sec": time.time() - sym_start,
                        "trace_id": trace_id,
                    },
                )
                break

    timings["total"] = time.time() - t0

    result_payload = {
        "symbols_ok": symbols_ok,
        "symbols_failed": symbols_failed,
        "rows_written": rows_written,
    }
    if req.compute_strategy_graph:
        result_payload["strategy_graph_rows"] = strategy_graph_rows

    status = "ok" if not symbols_failed else "partial"
    metrics_tracker.record_run(
        "snapshot",
        trace_id,
        status,
        timings=timings,
        rows_written={"signals": rows_written.get("signals"), "features": rows_written.get("features")},
    )

    return {
        "status": status,
        "trace_id": trace_id,
        "requested": requested,
        "result": result_payload,
        "timings": timings,
    }


@router.get("/latest/signal")
async def latest_signal(symbol: str, lookback: int = 252):
    _require_db_enabled(read=True)
    try:
        res = query.latest_signal(symbol.upper(), lookback)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return res


@router.get("/latest/features")
async def latest_features(symbol: str, lookback: int = 252):
    _require_db_enabled(read=True)
    try:
        res = query.latest_features(symbol.upper(), lookback)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    if not res:
        raise HTTPException(status_code=404, detail="not found")
    return res


@router.get("/graph/strategy")
async def strategy_graph(symbol: str, lookback: int = 252, days: int = 365):
    _require_db_enabled(read=True)
    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol required")
    window_days = max(1, int(days))

    try:
        latest_row = db.safe_fetchone(
            """
            SELECT max(as_of)
            FROM prosperity_signals_daily
            WHERE symbol=%s AND lookback=%s
            """,
            (sym, lookback),
        )
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    latest_as_of = latest_row[0] if latest_row else None
    if latest_as_of is None:
        raise HTTPException(status_code=404, detail="not found")

    window_start = latest_as_of - dt.timedelta(days=window_days)

    try:
        rows = db.safe_fetchall(
            """
            SELECT as_of, signal, score, regime, confidence
            FROM prosperity_signals_daily
            WHERE symbol=%s AND lookback=%s AND as_of BETWEEN %s AND %s
            ORDER BY as_of ASC
            """,
            (sym, lookback, window_start, latest_as_of),
        )
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    if not rows:
        raise HTTPException(status_code=404, detail="not found")

    nodes: Dict[str, int] = {}
    edges: Dict[tuple[str, str], int] = {}
    series: List[Dict[str, Any]] = []

    for as_of, signal, score, regime, confidence in rows:
        sig = (signal or "UNKNOWN").upper()
        nodes[sig] = nodes.get(sig, 0) + 1
        series.append(
            {
                "as_of": as_of.isoformat(),
                "signal": sig,
                "score": float(score) if score is not None else None,
                "regime": regime,
                "confidence": confidence,
            }
        )

    for idx in range(len(rows) - 1):
        from_sig = (rows[idx][1] or "UNKNOWN").upper()
        to_sig = (rows[idx + 1][1] or "UNKNOWN").upper()
        edges[(from_sig, to_sig)] = edges.get((from_sig, to_sig), 0) + 1

    nodes_list = [{"id": key, "count": count} for key, count in sorted(nodes.items())]
    edges_list = [
        {"from": pair[0], "to": pair[1], "count": count} for pair, count in sorted(edges.items())
    ]

    return {
        "symbol": sym,
        "lookback": lookback,
        "window": {"days": window_days, "from": window_start.isoformat(), "to": latest_as_of.isoformat()},
        "nodes": nodes_list,
        "edges": edges_list,
        "series_sample": series[:50],
    }


@router.get("/graph/universe")
async def graph_universe():
    _require_db_enabled(read=True)
    try:
        rows = db.safe_fetchall(
            """
            SELECT u.symbol, COUNT(s.as_of) AS row_count
            FROM prosperity_universe u
            LEFT JOIN prosperity_signals_daily s ON s.symbol = u.symbol
            WHERE u.active = TRUE
            GROUP BY u.symbol
            ORDER BY u.symbol ASC
            """
        )
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    data = [{"symbol": row[0], "rows": int(row[1]) if row[1] is not None else 0} for row in rows]
    return {"symbols": data}
