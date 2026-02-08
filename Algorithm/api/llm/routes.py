from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from api import config, db
from api.llm import client, prompts
from api.llm.schemas import (
    NarratorAskRequest,
    NarratorAskResponse,
    NarratorPortfolioRequest,
    NarratorPortfolioResponse,
    NarratorSignalRequest,
    NarratorSignalResponse,
)
from api.prosperity import query

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if not trace_id:
        trace_id = uuid.uuid4().hex
        request.state.trace_id = trace_id
    return trace_id


def _ensure_api_key(trace_id: str) -> None:
    api_key = config.openai_api_key()
    if not api_key:
        raise client._missing_key_exc(trace_id)


def _symbol_clean(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _fetch_signal_from_db(
    symbol: str, as_of: dt.date, lookback: int
) -> Optional[Dict[str, Any]]:
    if not (db.db_enabled() and db.db_read_enabled()):
        return None
    try:
        return query.signal_as_of(symbol, lookback, as_of)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _fetch_features_from_db(
    symbol: str, as_of: dt.date, lookback: int
) -> Optional[Dict[str, Any]]:
    if not (db.db_enabled() and db.db_read_enabled()):
        return None
    try:
        return query.features_as_of(symbol, lookback, as_of)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _fetch_history(
    symbol: str, as_of: dt.date, lookback: int, days: int
) -> List[Dict[str, Any]]:
    if days <= 0 or not (db.db_enabled() and db.db_read_enabled()):
        return []
    try:
        return query.signal_history(symbol, lookback, as_of, days)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _compute_signal_live(symbol: str, as_of: dt.date, lookback: int) -> Dict[str, Any]:
    from api import main as api_main

    result = api_main.compute_signal_for_symbol(symbol, as_of.isoformat(), lookback)
    return result.model_dump() if hasattr(result, "model_dump") else dict(result)


def _resolve_signal(
    symbol: str,
    as_of: dt.date,
    lookback: int,
    *,
    include_features: bool,
    include_history_days: int,
) -> Dict[str, Any]:
    clean_symbol = _symbol_clean(symbol)
    sig = _fetch_signal_from_db(clean_symbol, as_of, lookback)
    history = _fetch_history(clean_symbol, as_of, lookback, include_history_days)

    if sig is None:
        sig = _compute_signal_live(clean_symbol, as_of, lookback)

    features = sig.get("features") if isinstance(sig, dict) else None
    if include_features and (not features):
        db_feats = _fetch_features_from_db(clean_symbol, as_of, lookback)
        if db_feats:
            features = db_feats.get("features")

    sig_dict = dict(sig)
    sig_dict["features"] = features or {}
    sig_dict["history"] = history
    return sig_dict


def _market_stats_from_features(features: Dict[str, Any]) -> Dict[str, float]:
    return {
        "last_close": float(features.get("last_close", 0.0)),
        "mom_21": float(features.get("mom_21", 0.0)),
        "volatility_ann": float(features.get("volatility_ann", 0.0)),
        "rsi14": float(features.get("rsi14", 0.0)),
    }


def _prepare_citations(symbols: List[str]) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    for sym in symbols:
        citations.append({"symbol": sym, "field": "signal"})
    return citations


def _safe_perf_value(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _performance_defaults(
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    base = {
        "return": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
    }
    if overrides:
        for key in base.keys():
            base[key] = _safe_perf_value(overrides.get(key, base[key]))
    return base


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/narrator/signal", response_model=NarratorSignalResponse)
async def narrator_signal(
    req: NarratorSignalRequest, request: Request
) -> NarratorSignalResponse:
    trace_id = _trace_id(request)
    _ensure_api_key(trace_id)

    signal = _resolve_signal(
        req.symbol,
        req.as_of,
        req.lookback,
        include_features=req.include_features,
        include_history_days=req.include_db_history_days,
    )

    drivers = (
        prompts.feature_driver_lines(signal.get("features", {}))
        if req.include_features
        else []
    )
    messages = prompts.build_signal_prompt(
        signal, drivers, signal.get("history", []), req.style
    )
    reply, _, _ = client.complete_chat(messages, trace_id=trace_id, max_tokens=600)
    bullets = prompts.extract_bullets(reply)

    as_of_value = signal.get("as_of")
    as_of_date = (
        as_of_value
        if isinstance(as_of_value, dt.date)
        else dt.date.fromisoformat(str(as_of_value))
    )

    return NarratorSignalResponse(
        symbol=signal.get("symbol"),
        as_of=as_of_date,
        signal=signal.get("signal"),
        narrative=reply,
        bullets=bullets,
        disclaimer=prompts.make_disclaimer(),
    )


@router.post("/narrator/portfolio", response_model=NarratorPortfolioResponse)
async def narrator_portfolio(
    req: NarratorPortfolioRequest, request: Request
) -> NarratorPortfolioResponse:
    trace_id = _trace_id(request)
    _ensure_api_key(trace_id)

    summary: Dict[str, Any] = {
        "performance": _performance_defaults(),
        "contributors": [],
        "exposures": [],
    }

    if req.include_backtest:
        from api import main as api_main

        bt_req = api_main.PortfolioBacktestRequest(
            symbols=req.symbols,
            from_date=req.from_date.isoformat(),
            to_date=req.to_date.isoformat(),
            lookback=req.lookback,
            rebalance_every=req.rebalance_every,
            max_weight=req.max_weight,
        )
        backtest = api_main.backtest_portfolio(bt_req)
        perf = _performance_defaults(
            {
                "return": getattr(backtest, "total_return", None),
                "sharpe": getattr(backtest, "sharpe", None),
                "max_drawdown": getattr(backtest, "max_drawdown", None),
                "turnover": getattr(backtest, "turnover", None),
            }
        )
        summary["performance"] = perf
        audit = backtest.audit or {}
        skipped = audit.get("skipped_symbols") or []
        if skipped:
            summary["exposures"].append(
                f"Skipped symbols: {[item.get('symbol') for item in skipped]}"
            )
    else:
        summary["performance"] = _performance_defaults()
        summary["exposures"].append("Backtest skipped per request; metrics limited.")

    messages = prompts.build_portfolio_prompt(summary, req.style)
    reply, _, _ = client.complete_chat(messages, trace_id=trace_id, max_tokens=500)
    bullets = prompts.extract_bullets(reply)

    return NarratorPortfolioResponse(
        narrative=reply,
        bullets=bullets,
        disclaimer=prompts.make_disclaimer(),
        performance=summary.get("performance"),
    )


@router.post("/narrator/ask/legacy", response_model=NarratorAskResponse)
async def narrator_ask(
    req: NarratorAskRequest, request: Request
) -> NarratorAskResponse:
    trace_id = _trace_id(request)
    _ensure_api_key(trace_id)

    symbols_context: Dict[str, Any] = {}
    for raw_sym in req.symbols:
        sym = _symbol_clean(raw_sym)
        sig = _resolve_signal(
            sym, req.as_of, req.lookback, include_features=True, include_history_days=0
        )
        market_stats = _market_stats_from_features(sig.get("features", {}))
        symbols_context[sym] = {"signal": sig, "market_stats": market_stats}

    context = {
        "symbols": symbols_context,
        "user_intent": (req.context or {}).get("user_intent") if req.context else None,
    }
    messages = prompts.build_ask_prompt(req.question, context)
    reply, _, _ = client.complete_chat(messages, trace_id=trace_id, max_tokens=600)

    return NarratorAskResponse(
        answer=reply,
        citations=_prepare_citations(list(symbols_context.keys())),
        disclaimer=prompts.make_disclaimer(),
    )


__all__ = ["router"]
