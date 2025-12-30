from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from api import config, db, security
from api.ops import metrics_tracker
from api.prosperity import query
from api.prosperity import strategy_graph_db
from ftip.narrator import client as narrator_client
from ftip.narrator import prompts

logger = logging.getLogger("ftip.api.narrator")
router = APIRouter(tags=["narrator"])


class NarratorAskRequest(BaseModel):
    question: str = Field(..., description="User question")
    symbols: List[str]
    as_of_date: dt.date
    lookback: int = 252
    days: int = 365


class NarratorAskResponse(BaseModel):
    answer: str
    model: str
    trace_id: str
    context_used: Dict[str, Any]


class NarratorExplainRequest(BaseModel):
    symbol: str
    as_of_date: dt.date
    lookback: int = 252
    days: int = 365


class NarratorExplainResponse(BaseModel):
    symbol: str
    as_of: dt.date
    signal: Optional[str]
    explanation: str
    drivers: List[str]
    risks: List[str]
    what_to_watch: List[str]
    trace_id: str


class NarratorHealthResponse(BaseModel):
    status: str
    has_api_key: bool
    trace_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_id(request: Request) -> str:
    trace_id = getattr(request.state, "trace_id", None)
    if not trace_id:
        trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
        request.state.trace_id = trace_id
    return trace_id


def _ensure_openai_key(trace_id: str) -> None:
    if not config.openai_api_key():
        raise narrator_client.NarratorClient._missing_key_exc(trace_id)


def _ensure_db(trace_id: str) -> None:
    if not (db.db_enabled() and db.db_read_enabled()):
        raise HTTPException(status_code=503, detail={"message": "Prosperity DB read access required", "trace_id": trace_id})


def _clean_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _load_symbol_context(symbol: str, as_of_date: dt.date, lookback: int, days: int) -> Dict[str, Any]:
    sig = query.signal_as_of(symbol, lookback, as_of_date)
    feats = query.features_as_of(symbol, lookback, as_of_date)
    history = query.signal_history(symbol, lookback, as_of_date, days)

    return {
        "symbol": symbol,
        "as_of_date": as_of_date.isoformat(),
        "lookback": lookback,
        "signal": sig or {},
        "features": (feats or {}).get("features") or {},
        "history": history,
    }


def _load_strategy_graph(symbol: str, lookback: int, as_of_date: dt.date) -> Optional[Dict[str, Any]]:
    ensemble = strategy_graph_db.ensemble_as_of(symbol, lookback, as_of_date)
    strategies = strategy_graph_db.strategies_as_of(symbol, lookback, as_of_date)
    if not ensemble:
        return None
    return {"ensemble": ensemble, "strategies": strategies}


def _build_context(req: NarratorAskRequest) -> Dict[str, Any]:
    symbols_packet: List[Dict[str, Any]] = []
    for raw_sym in req.symbols:
        sym = _clean_symbol(raw_sym)
        symbols_packet.append(_load_symbol_context(sym, req.as_of_date, req.lookback, req.days))

    strategy_graph = None
    if symbols_packet:
        strategy_graph = _load_strategy_graph(symbols_packet[0]["symbol"], req.lookback, req.as_of_date)

    return prompts.build_context_packet(
        question=req.question,
        symbols=symbols_packet,
        strategy_graph=strategy_graph,
        meta={"as_of_date": req.as_of_date.isoformat(), "lookback": req.lookback, "days": req.days},
    )


def _build_explain_context(req: NarratorExplainRequest) -> Dict[str, Any]:
    sym = _clean_symbol(req.symbol)
    signal_ctx = _load_symbol_context(sym, req.as_of_date, req.lookback, req.days)
    strategy_graph = _load_strategy_graph(sym, req.lookback, req.as_of_date)

    return {
        "question": f"Explain the signal for {sym}",
        "symbol": sym,
        "as_of_date": req.as_of_date.isoformat(),
        "lookback": req.lookback,
        "signal": signal_ctx,
        "strategy_graph": strategy_graph,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=NarratorHealthResponse)
async def narrator_health(request: Request) -> NarratorHealthResponse:
    trace_id = _trace_id(request)
    logger.info("narrator.health", extra={"trace_id": trace_id})
    return NarratorHealthResponse(status="ok", has_api_key=bool(config.openai_api_key()), trace_id=trace_id)


@router.post("/ask", response_model=NarratorAskResponse)
async def narrator_ask(req: NarratorAskRequest, request: Request) -> NarratorAskResponse:
    trace_id = _trace_id(request)
    _ensure_db(trace_id)
    _ensure_openai_key(trace_id)

    logger.info(
        "narrator.ask.start",
        extra={"trace_id": trace_id, "symbols": req.symbols, "as_of_date": req.as_of_date.isoformat()},
    )

    context_packet = _build_context(req)
    messages = prompts.build_ask_prompt(req.question, context_packet, safe_mode=True)
    reply, model, _ = narrator_client.complete_chat(messages, trace_id=trace_id)

    metrics_tracker.record_narrator_call()
    return NarratorAskResponse(
        answer=f"{reply}\n\n{prompts.DISCLAIMER}",
        model=model,
        trace_id=trace_id,
        context_used=prompts.summarize_context_used(context_packet),
    )


@router.post("/explain-signal", response_model=NarratorExplainResponse)
async def narrator_explain_signal(req: NarratorExplainRequest, request: Request) -> NarratorExplainResponse:
    trace_id = _trace_id(request)
    _ensure_db(trace_id)
    _ensure_openai_key(trace_id)

    logger.info(
        "narrator.explain_signal.start",
        extra={"trace_id": trace_id, "symbol": req.symbol, "as_of_date": req.as_of_date.isoformat()},
    )

    context_packet = _build_explain_context(req)
    messages = prompts.build_explain_prompt(context_packet, safe_mode=True)
    reply, model, _ = narrator_client.complete_chat(messages, trace_id=trace_id)

    signal = context_packet.get("signal", {}).get("signal", {})
    narration = reply.strip()

    return NarratorExplainResponse(
        symbol=_clean_symbol(req.symbol),
        as_of=req.as_of_date,
        signal=signal.get("signal") if isinstance(signal, dict) else None,
        explanation=f"{narration}\n\n{prompts.DISCLAIMER}",
        drivers=["Model drivers referenced in context"],
        risks=["Model limitations and market regime shifts"],
        what_to_watch=["Threshold changes", "Feature stability"],
        trace_id=trace_id,
    )


__all__ = ["router"]
