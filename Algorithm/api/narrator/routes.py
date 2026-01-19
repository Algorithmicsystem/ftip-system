from __future__ import annotations

import datetime as dt
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from api import config, db, migrations, security
from api.ops import metrics_tracker
from api.prosperity import query
from api.prosperity import strategy_graph_db
from ftip.narrator import client as narrator_client
from ftip.narrator import prompts

logger = logging.getLogger("ftip.api.narrator")
router = APIRouter(tags=["narrator"])


class _ValidatedRequest(BaseModel):
    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def _clean_symbol(cls, value: str) -> str:
        cleaned = (value or "").strip().upper()
        if not cleaned:
            raise ValueError("symbol required")
        if not re.match(r"^[A-Z][A-Z0-9\\.\-]{0,9}$", cleaned):
            raise ValueError("symbol must be alphanumeric (.+-) up to 10 chars")
        return cleaned

class NarratorAskRequest(_ValidatedRequest):
    question: str = Field(..., description="User question")
    symbols: List[str]
    as_of_date: dt.date
    lookback: int = Field(252, ge=5, le=5000)
    days: int = Field(365, ge=1, le=3650)

    @field_validator("symbols")
    @classmethod
    def _clean_symbols(cls, value: List[str]) -> List[str]:
        cleaned = []
        for sym in value:
            sym_clean = (sym or "").strip().upper()
            if not sym_clean:
                continue
            if not re.match(r"^[A-Z][A-Z0-9\\.\-]{0,9}$", sym_clean):
                raise ValueError("symbol must be alphanumeric (.+-) up to 10 chars")
            cleaned.append(sym_clean)
        if not cleaned:
            raise ValueError("symbols required")
        return cleaned


class NarratorAskResponse(BaseModel):
    answer: str
    model: str
    trace_id: str
    context_used: Dict[str, Any]


class NarratorExplainRequest(_ValidatedRequest):
    symbol: str
    as_of_date: dt.date
    lookback: int = Field(252, ge=5, le=5000)
    days: int = Field(365, ge=1, le=3650)


class NarratorExplainResponse(BaseModel):
    symbol: str
    as_of: dt.date
    signal: Optional[str]
    explanation: str
    drivers: List[str]
    risks: List[str]
    what_to_watch: List[str]
    trace_id: str


class NarratorExplainStrategyRequest(_ValidatedRequest):
    symbol: str
    lookback: int = Field(252, ge=5, le=5000)
    days: int = Field(365, ge=1, le=3650)
    to_date: dt.date


class NarratorExplainStrategyResponse(BaseModel):
    symbol: str
    lookback: int
    window: Dict[str, Any]
    explanation: str
    graph: Dict[str, Any]
    trace_id: str


class NarratorHealthResponse(BaseModel):
    status: str
    has_api_key: bool
    trace_id: str


class NarratorDiagnoseResponse(BaseModel):
    status: str
    checks: List[Dict[str, Any]]
    trace_id: str


class NarratorExplainSignalRequest(_ValidatedRequest):
    symbol: str
    as_of_date: dt.date
    signal: Dict[str, Any]
    features: Dict[str, Any]
    quality: Dict[str, Any]
    bars: Dict[str, Any]
    sentiment: Dict[str, Any]


class NarratorExplainSignalResponse(BaseModel):
    symbol: str
    as_of_date: dt.date
    action: str
    confidence: Optional[float]
    explanation: str
    reasons: List[str]
    risks: List[str]
    invalidation: str
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
    days = min(max(days, 1), 3650)
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


def _build_transition_graph(symbol: str, lookback: int, to_date: dt.date, days: int) -> Optional[Dict[str, Any]]:
    window_days = min(max(days, 1), 3650)
    history = query.signal_history(symbol, lookback, to_date, window_days)
    if not history:
        return None

    try:
        ordered = sorted(history, key=lambda item: dt.date.fromisoformat(str(item.get("as_of"))))
    except Exception:  # pragma: no cover - defensive
        ordered = history[::-1]

    nodes: Dict[str, int] = {}
    edges: Dict[tuple[str, str], int] = {}
    series: List[Dict[str, Any]] = []

    for entry in ordered:
        sig = (entry.get("signal") or "UNKNOWN").upper()
        nodes[sig] = nodes.get(sig, 0) + 1
        series.append(
            {
                "as_of": entry.get("as_of"),
                "signal": sig,
                "score": entry.get("score"),
                "regime": entry.get("regime"),
                "confidence": entry.get("confidence"),
            }
        )

    for idx in range(len(ordered) - 1):
        from_sig = (ordered[idx].get("signal") or "UNKNOWN").upper()
        to_sig = (ordered[idx + 1].get("signal") or "UNKNOWN").upper()
        edges[(from_sig, to_sig)] = edges.get((from_sig, to_sig), 0) + 1

    window_from = to_date - dt.timedelta(days=window_days)
    nodes_list = [{"id": key, "count": count} for key, count in sorted(nodes.items())]
    edges_list = [
        {"from": pair[0], "to": pair[1], "count": count} for pair, count in sorted(edges.items())
    ]

    return {
        "symbol": symbol,
        "lookback": lookback,
        "window": {"days": window_days, "from": window_from.isoformat(), "to": to_date.isoformat()},
        "nodes": nodes_list,
        "edges": edges_list,
        "series_sample": series[-120:],
    }


def _trim_graph_for_prompt(graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **graph,
        "nodes": (graph.get("nodes") or [])[:50],
        "edges": (graph.get("edges") or [])[:100],
        "series_sample": (graph.get("series_sample") or [])[-50:],
    }


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


def _check(name: str, ok: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "fail", "details": details}


def _reason_mapping() -> Dict[str, str]:
    return {
        "TREND_UP": "Trend slope is positive over the long window.",
        "TREND_DOWN": "Trend slope is negative over the long window.",
        "MOMENTUM_STRONG": "Momentum is strong relative to volatility.",
        "MOMENTUM_WEAK": "Momentum is weak relative to volatility.",
        "SENTIMENT_POSITIVE": "News sentiment skews positive.",
        "SENTIMENT_NEGATIVE": "News sentiment skews negative.",
        "NEUTRAL_SCORE": "Signals are mixed and score is near neutral.",
    }


def _render_explain_signal(req: NarratorExplainSignalRequest, trace_id: str) -> NarratorExplainSignalResponse:
    signal = req.signal or {}
    action = (signal.get("action") or "HOLD").upper()
    confidence = signal.get("confidence")
    reason_codes = signal.get("reason_codes") or []
    reasons = [_reason_mapping().get(code, f"Signal reason: {code}.") for code in reason_codes]

    quality = req.quality or {}
    missing_notes = []
    if quality.get("sentiment_ok") is False or (req.sentiment or {}).get("headline_count") in (0, None):
        missing_notes.append("Sentiment coverage is thin or missing.")
    if quality.get("intraday_ok") is False:
        missing_notes.append("Intraday coverage is missing.")
    if quality.get("fundamentals_ok") is False:
        missing_notes.append("Fundamentals coverage is missing.")

    stop_loss = signal.get("stop_loss")
    invalidation = (
        f"Invalidated if price breaches stop-loss at {stop_loss:.2f}."
        if isinstance(stop_loss, (int, float))
        else "Invalidated if risk limits are hit."
    )
    risks = missing_notes or ["Normal market volatility applies. Use position sizing and risk limits."]
    explanation = (
        f"{action} signal with confidence {confidence:.2f}." if confidence is not None else f"{action} signal generated."
    )
    explanation += " This summary is informational only."

    return NarratorExplainSignalResponse(
        symbol=req.symbol,
        as_of_date=req.as_of_date,
        action=action,
        confidence=confidence if isinstance(confidence, (int, float)) else None,
        explanation=explanation,
        reasons=reasons,
        risks=risks,
        invalidation=invalidation,
        trace_id=trace_id,
    )


def _migrations_status() -> tuple[bool, Dict[str, Any]]:
    if not db.db_enabled():
        return False, {"message": "database disabled"}
    if not db.db_read_enabled():
        return False, {"message": "database read access disabled"}
    try:
        migrations.ensure_schema()
        rows = db.safe_fetchall("SELECT version, applied_at FROM schema_migrations ORDER BY version ASC")
        versions = [row[0] for row in rows]
        return True, {"versions": versions}
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"error": str(exc)}


def _latest_signal_exists(symbol: str, lookback: int) -> tuple[bool, Dict[str, Any]]:
    if not (db.db_enabled() and db.db_read_enabled()):
        return False, {"message": "database disabled"}
    try:
        exists = bool(query.latest_signal(symbol, lookback))
        return exists, {"found": exists, "symbol": symbol, "lookback": lookback}
    except Exception as exc:  # pragma: no cover - defensive
        return False, {"error": str(exc)}


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


@router.post("/explain-signal", response_model=NarratorExplainSignalResponse)
async def narrator_explain_signal(req: NarratorExplainSignalRequest, request: Request) -> NarratorExplainSignalResponse:
    trace_id = _trace_id(request)
    logger.info(
        "narrator.explain_signal.start",
        extra={"trace_id": trace_id, "symbol": req.symbol, "as_of_date": req.as_of_date.isoformat()},
    )
    return _render_explain_signal(req, trace_id)


@router.post("/explain-strategy-graph", response_model=NarratorExplainStrategyResponse)
async def narrator_explain_strategy_graph(
    req: NarratorExplainStrategyRequest, request: Request
) -> NarratorExplainStrategyResponse:
    trace_id = _trace_id(request)
    _ensure_db(trace_id)
    _ensure_openai_key(trace_id)

    logger.info(
        "narrator.explain_strategy_graph.start",
        extra={"trace_id": trace_id, "symbol": req.symbol, "to_date": req.to_date.isoformat()},
    )

    graph = _build_transition_graph(req.symbol, req.lookback, req.to_date, req.days)
    if not graph:
        raise HTTPException(status_code=404, detail={"message": "graph context not found", "trace_id": trace_id})

    trimmed_graph = _trim_graph_for_prompt(graph)
    messages = prompts.build_strategy_graph_prompt(trimmed_graph, safe_mode=True)
    max_tokens = min(config.llm_max_tokens(), 500)
    reply, model, _ = narrator_client.complete_chat(messages, trace_id=trace_id, max_tokens=max_tokens)

    metrics_tracker.record_narrator_call()
    return NarratorExplainStrategyResponse(
        symbol=graph["symbol"],
        lookback=graph["lookback"],
        window=graph["window"],
        explanation=f"{reply.strip()}\n\n{prompts.DISCLAIMER}",
        graph=graph,
        trace_id=trace_id,
    )


@router.post("/diagnose", response_model=NarratorDiagnoseResponse)
async def narrator_diagnose(request: Request) -> NarratorDiagnoseResponse:
    trace_id = _trace_id(request)
    checks: List[Dict[str, Any]] = []

    auth_env = [name for name in ("FTIP_API_KEY", "FTIP_API_KEYS", "FTIP_API_KEY_PRIMARY") if os.getenv(name)]
    openai_env = [name for name in ("OPENAI_API_KEY", "OpenAI_ftip-system") if os.getenv(name)]

    checks.append(
        _check(
            "auth",
            security.auth_enabled(),
            {"auth_enabled": security.auth_enabled(), "env_vars": auth_env},
        )
    )

    db_ok = db.db_enabled() and db.db_read_enabled()
    checks.append(
        _check(
            "database",
            db_ok,
            {"enabled": db.db_enabled(), "read_enabled": db.db_read_enabled(), "write_enabled": db.db_write_enabled()},
        )
    )

    mig_ok, mig_details = _migrations_status()
    checks.append(_check("migrations", mig_ok, mig_details))

    signal_ok, signal_details = _latest_signal_exists("AAPL", 252)
    checks.append(_check("latest_signal", signal_ok, signal_details))

    openai_ok = bool(config.openai_api_key())
    checks.append(_check("openai_api_key", openai_ok, {"env_vars": openai_env}))

    overall_status = "ok" if all(item.get("status") == "pass" for item in checks) else "degraded"
    return NarratorDiagnoseResponse(status=overall_status, checks=checks, trace_id=trace_id)


__all__ = ["router"]
