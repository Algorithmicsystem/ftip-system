from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api import config, db
from api.llm import client as llm_client
from api.ops import metrics_tracker
from api.prosperity import query
from api.prosperity.strategy_graph_db import (
    ensemble_as_of,
    strategies_as_of,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prosperity-narrator"])


class MemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.hits: Dict[str, List[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = [ts for ts in self.hits.get(key, []) if ts > now - self.window_seconds]
        if self.max_requests > 0 and len(bucket) >= self.max_requests:
            self.hits[key] = bucket
            return False
        bucket.append(now)
        self.hits[key] = bucket
        return True


_limiter = MemoryRateLimiter(
    config.narrator_rate_limit(), config.narrator_rate_window_seconds()
)


SYSTEM_PROMPT = (
    "You are the FTIP Narrator. Explain model outputs using only the DB context provided. "
    "If information is missing, say you do not know and suggest running the appropriate compute endpoint. "
    "Stay concise, factual, and avoid financial advice. Output 6-12 sentences for narrations."
)

DISCLAIMER = "Not financial advice; research preview only."


class NarratorAskRequest(BaseModel):
    question: str
    symbols: Optional[List[str]] = None
    as_of_date: Optional[dt.date] = None
    lookback: int = 252


class NarratorAskResponse(BaseModel):
    answer: str
    trace_id: str
    model: str
    disclaimer: str
    grounding: Dict[str, Any]


class NarratorExplainResponse(BaseModel):
    symbol: str
    as_of_date: dt.date
    lookback: int
    final_signal: Optional[str]
    final_score: Optional[float]
    final_confidence: Optional[float]
    regime: Optional[str]
    thresholds: Dict[str, Any]
    risk_overlay_applied: Optional[bool]
    top_strategies: List[Dict[str, Any]]
    narration: str
    disclaimer: str
    grounding: Dict[str, Any]
    trace_id: str


class HealthResponse(BaseModel):
    status: str
    llm_enabled: bool
    has_api_key: bool
    model: Optional[str]
    trace_id: str


class NarratorExplainParams(BaseModel):
    symbol: str
    as_of_date: dt.date
    lookback: int = 252


def _trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if not trace_id:
        trace_id = uuid.uuid4().hex
        request.state.trace_id = trace_id
    return trace_id


async def _rate_limit(request: Request) -> None:
    if config.narrator_rate_limit() <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    if not _limiter.allow(ip):
        raise HTTPException(
            status_code=429, detail="Too many narrator requests; please slow down."
        )


async def _ensure_enabled(trace_id: str) -> None:
    if not config.llm_enabled():
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Narrator disabled (set FTIP_LLM_ENABLED=1)",
                "trace_id": trace_id,
            },
        )
    api_key = config.openai_api_key()
    if not api_key:
        raise llm_client._missing_key_exc(trace_id)
    if not (db.db_enabled() and db.db_read_enabled()):
        raise HTTPException(
            status_code=503,
            detail={"message": "database disabled", "trace_id": trace_id},
        )


SYSTEM_CONTEXT = (
    "The FTIP system computes strategy graph ensembles, per-strategy signals, and feature snapshots. "
    "Key endpoints: /prosperity/strategy_graph/run to populate ensembles, /prosperity/strategy_graph/latest/* for latest rows, "
    "/prosperity/latest/features and /prosperity/latest/signal for snapshots."
)


def _build_grounding(
    symbol: str,
    ensemble: Dict[str, Any],
    strategies: List[Dict[str, Any]],
    features: Optional[Dict[str, Any]],
    signal: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    strategy_ground = [
        {
            "strategy_id": s.get("strategy_id"),
            "as_of_date": s.get("as_of_date"),
            "confidence": s.get("confidence"),
            "normalized_score": s.get("normalized_score"),
            "signal": s.get("signal"),
        }
        for s in strategies
    ]
    return {
        "symbol": symbol,
        "ensemble_fields": {
            "as_of_date": ensemble.get("as_of_date"),
            "regime": ensemble.get("regime"),
            "final_signal": ensemble.get("final_signal"),
            "final_score": ensemble.get("final_score"),
            "final_confidence": ensemble.get("final_confidence"),
            "thresholds": ensemble.get("thresholds"),
            "strategies_used": ensemble.get("strategies_used"),
        },
        "strategies": strategy_ground,
        "features": list((features or {}).get("features", {}).keys()),
        "snapshot_signal": signal,
    }


def _top_strategies(
    strategies: List[Dict[str, Any]], ensemble: Dict[str, Any]
) -> List[Dict[str, Any]]:
    weights = {
        str(item.get("strategy_id")): abs(float(item.get("weight") or 0.0))
        for item in ensemble.get("strategies_used") or []
    }
    scored: List[tuple[float, Dict[str, Any]]] = []
    for strat in strategies:
        metric = weights.get(str(strat.get("strategy_id")))
        if metric is None:
            metric = abs(float(strat.get("confidence") or 0.0))
        rationale = strat.get("rationale")
        rationale_text = (
            "; ".join(rationale)
            if isinstance(rationale, list)
            else (str(rationale) if rationale else "")
        )
        scored.append(
            (
                metric,
                {
                    "strategy_id": strat.get("strategy_id"),
                    "signal": strat.get("signal"),
                    "confidence": strat.get("confidence"),
                    "normalized_score": strat.get("normalized_score"),
                    "rationale": rationale_text,
                },
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item[1] for item in scored[:3]]


def _clean_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _load_context(symbol: str, as_of_date: dt.date, lookback: int) -> Dict[str, Any]:
    clean_symbol = _clean_symbol(symbol)
    ensemble = ensemble_as_of(clean_symbol, lookback, as_of_date)
    if not ensemble:
        return {}
    strategies = strategies_as_of(clean_symbol, lookback, as_of_date)
    features = query.features_as_of(clean_symbol, lookback, as_of_date)
    try:
        signal = query.signal_as_of(clean_symbol, lookback, as_of_date)
    except db.DBError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    as_of = (
        dt.date.fromisoformat(ensemble["as_of_date"])
        if isinstance(ensemble.get("as_of_date"), str)
        else ensemble.get("as_of_date")
    )
    aligned_strategies = [
        s for s in strategies if s.get("as_of_date") == ensemble.get("as_of_date")
    ]
    return {
        "symbol": clean_symbol,
        "as_of_date": as_of,
        "lookback": lookback,
        "ensemble": ensemble,
        "strategies": aligned_strategies or strategies,
        "features": features,
        "signal": signal,
    }


def _build_explain_prompt(context: Dict[str, Any]) -> List[Dict[str, str]]:
    ensemble = context["ensemble"]
    strategies = context["strategies"]
    features = (context.get("features") or {}).get("features", {})
    snapshot_signal = context.get("signal")

    lines = [
        f"Symbol: {context['symbol']} (as_of_date={ensemble.get('as_of_date')}, lookback={context['lookback']})",
        f"Regime: {ensemble.get('regime')}",
        f"Final signal: {ensemble.get('final_signal')} score={ensemble.get('final_score')} confidence={ensemble.get('final_confidence')}",
        f"Thresholds: {ensemble.get('thresholds')}",
        f"Risk overlay applied: {ensemble.get('risk_overlay_applied')}",
    ]
    if snapshot_signal:
        lines.append(
            f"Latest snapshot signal: {snapshot_signal.get('signal')} score={snapshot_signal.get('score')} as_of={snapshot_signal.get('as_of')}"
        )
    if features:
        feat_snippet = ", ".join([f"{k}={v}" for k, v in list(features.items())[:8]])
        lines.append(f"Features: {feat_snippet}")
    for strat in _top_strategies(strategies, ensemble):
        lines.append(
            f"Strategy {strat.get('strategy_id')}: signal={strat.get('signal')} conf={strat.get('confidence')} score={strat.get('normalized_score')} rationale={strat.get('rationale')}"
        )

    user_prompt = "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_ask_prompt(
    question: str, context_items: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    blocks: List[str] = [SYSTEM_CONTEXT]
    for ctx in context_items:
        ensemble = ctx.get("ensemble") or {}
        block_lines = [
            f"Symbol: {ctx.get('symbol')}",
            f"As of {ensemble.get('as_of_date')} lookback={ctx.get('lookback')}",
            f"Final signal: {ensemble.get('final_signal')} score={ensemble.get('final_score')} conf={ensemble.get('final_confidence')} regime={ensemble.get('regime')}",
        ]
        block_lines.append(f"Thresholds: {ensemble.get('thresholds')}")
        strategies = ctx.get("strategies") or []
        for strat in _top_strategies(strategies, ensemble):
            block_lines.append(
                f"Strategy {strat.get('strategy_id')}: signal={strat.get('signal')} conf={strat.get('confidence')} score={strat.get('normalized_score')} rationale={strat.get('rationale')}"
            )
        blocks.append("\n".join(block_lines))
    content = (
        "\n\n".join(blocks)
        + f"\n\nQuestion: {question}\nAnswer using only the information above."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


@router.get("/health", response_model=HealthResponse)
async def narrator_health(request: Request) -> HealthResponse:
    trace_id = _trace_id(request)
    return HealthResponse(
        status="ok",
        llm_enabled=config.llm_enabled(),
        has_api_key=bool(config.openai_api_key()),
        model=config.llm_model(),
        trace_id=trace_id,
    )


@router.get(
    "/explain",
    response_model=NarratorExplainResponse,
    dependencies=[Depends(_rate_limit)],
)
async def narrator_explain(
    request: Request, symbol: str, as_of_date: dt.date, lookback: int = 252
) -> NarratorExplainResponse:
    trace_id = _trace_id(request)
    await _ensure_enabled(trace_id)

    context = _load_context(symbol, as_of_date, lookback)
    if not context:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"DB has no strategy graph for {symbol}",
                "trace_id": trace_id,
            },
        )

    messages = _build_explain_prompt(context)
    reply, model, _ = llm_client.complete_chat(
        messages,
        max_tokens=config.llm_max_tokens(),
        temperature=config.llm_temperature(),
        trace_id=trace_id,
    )

    ensemble = context["ensemble"]
    strategies = context["strategies"]
    features = context.get("features")
    signal = context.get("signal")
    grounding = _build_grounding(
        context["symbol"], ensemble, strategies, features, signal
    )

    metrics_tracker.record_narrator_call()

    return NarratorExplainResponse(
        symbol=context["symbol"],
        as_of_date=as_of_date,
        lookback=lookback,
        final_signal=ensemble.get("final_signal"),
        final_score=ensemble.get("final_score"),
        final_confidence=ensemble.get("final_confidence"),
        regime=ensemble.get("regime"),
        thresholds=ensemble.get("thresholds") or {},
        risk_overlay_applied=ensemble.get("risk_overlay_applied"),
        top_strategies=_top_strategies(strategies, ensemble),
        narration=reply,
        disclaimer=DISCLAIMER,
        grounding=grounding,
        trace_id=trace_id,
    )


@router.post(
    "/ask", response_model=NarratorAskResponse, dependencies=[Depends(_rate_limit)]
)
async def narrator_ask(
    payload: NarratorAskRequest, request: Request
) -> NarratorAskResponse:
    trace_id = _trace_id(request)
    await _ensure_enabled(trace_id)

    as_of_date = payload.as_of_date or dt.date.today()
    contexts: List[Dict[str, Any]] = []
    missing: List[str] = []
    if payload.symbols:
        for sym in payload.symbols:
            ctx = _load_context(sym, as_of_date, payload.lookback)
            if not ctx:
                missing.append(_clean_symbol(sym))
            else:
                contexts.append(ctx)
    if missing:
        msg = "; ".join(
            [
                f"DB has no strategy graph for {sym}; run /prosperity/strategy_graph/run first."
                for sym in missing
            ]
        )
        raise HTTPException(
            status_code=404, detail={"message": msg, "trace_id": trace_id}
        )

    if not contexts:
        contexts.append(
            {
                "symbol": "SYSTEM",
                "ensemble": {
                    "as_of_date": as_of_date.isoformat(),
                    "regime": None,
                    "final_signal": None,
                    "final_score": None,
                    "final_confidence": None,
                    "thresholds": {},
                },
                "strategies": [],
            }
        )

    messages = _build_ask_prompt(payload.question, contexts)
    reply, model, _ = llm_client.complete_chat(
        messages,
        max_tokens=config.llm_max_tokens(),
        temperature=config.llm_temperature(),
        trace_id=trace_id,
    )

    grounding = {
        "symbols": [ctx.get("symbol") for ctx in contexts],
        "trace_id": trace_id,
    }

    metrics_tracker.record_narrator_call()

    return NarratorAskResponse(
        answer=reply,
        trace_id=trace_id,
        model=model,
        disclaimer=DISCLAIMER,
        grounding=grounding,
    )
