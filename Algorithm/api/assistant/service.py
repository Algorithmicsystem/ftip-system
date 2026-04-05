from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException
from openai import APIError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from api import config
from api.assistant import intelligence, orchestrator, prompts, reports, strategy
from api.assistant.storage import AssistantStorage, storage

logger = logging.getLogger(__name__)
_SYMBOL_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")


def _build_client() -> OpenAI:
    api_key = config.openai_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="LLM API key not configured")
    return OpenAI(
        api_key=api_key,
        timeout=config.llm_timeout_seconds(),
        max_retries=config.llm_max_retries(),
    )


def _require_api_key() -> None:
    if not config.openai_api_key():
        raise HTTPException(status_code=500, detail="LLM API key not configured")


def _safe_completion(
    messages: List[ChatCompletionMessageParam],
) -> Tuple[str, str, Dict[str, Optional[int]]]:
    try:
        client = _build_client()
        resp = client.chat.completions.create(
            model=config.llm_model(),
            messages=messages,
            max_tokens=400,
        )
        reply = resp.choices[0].message.content or ""
        usage = resp.usage
        usage_map = {
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        return reply, resp.model or config.llm_model(), usage_map
    except HTTPException:
        raise
    except APIStatusError as exc:
        logger.warning("LLM upstream error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM upstream error")
    except APIError as exc:
        logger.warning("LLM API error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM upstream error")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM unexpected error: %s", exc)
        raise HTTPException(status_code=502, detail="LLM upstream error")


def _ensure_llm_enabled() -> None:
    if not config.llm_enabled():
        raise HTTPException(
            status_code=503,
            detail="Assistant is disabled (set FTIP_LLM_ENABLED=1 to enable)",
        )


def _prepare_history(
    session_id: Optional[str], store: AssistantStorage
) -> Tuple[str, List[Dict[str, str]]]:
    if session_id is None:
        new_session = store.create_session()
        return new_session, []

    session = store.get_session(session_id)
    if session is None:
        new_session = store.create_session()
        return new_session, []

    messages = store.get_messages(session_id=session_id)
    history: List[Dict[str, str]] = []
    for msg in messages:
        history.append({"role": msg["role"], "content": msg["content"]})
    return session_id, history


def _normalize_active_analysis_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not context:
        return {}

    active = context.get("active_analysis")
    if isinstance(active, dict):
        return {k: v for k, v in active.items() if v not in (None, "")}

    keys = (
        "report_id",
        "symbol",
        "as_of_date",
        "horizon",
        "risk_mode",
        "session_id",
        "scenario",
        "analysis_depth",
        "refresh_mode",
        "market_regime",
        "signal",
        "freshness_status",
        "report_version",
    )
    return {key: context.get(key) for key in keys if context.get(key) not in (None, "")}


def _extract_symbol_from_message(message: str) -> Optional[str]:
    if not message:
        return None
    ignored = {
        "I",
        "A",
        "AN",
        "THE",
        "IT",
        "AND",
        "OR",
        "TO",
        "FOR",
        "OF",
        "BUY",
        "SELL",
        "HOLD",
        "BASED",
        "ON",
        "WHAT",
        "WHY",
        "HOW",
        "THIS",
        "THAT",
    }
    for match in _SYMBOL_PATTERN.findall(message):
        symbol = match.upper()
        if symbol not in ignored:
            return symbol
    return None


def _load_report_from_reference(
    reference: Dict[str, Any],
    *,
    session_id: Optional[str],
    store: AssistantStorage,
    require_exact_symbol: bool = False,
) -> Optional[Dict[str, Any]]:
    report_id = reference.get("report_id")
    if report_id:
        report = store.get_latest_analysis_report(
            report_id=report_id,
            session_id=reference.get("session_id") or session_id,
        )
        if report:
            return report

    symbol = reference.get("symbol")
    as_of_date = reference.get("as_of_date")
    horizon = reference.get("horizon")
    risk_mode = reference.get("risk_mode")

    if symbol or as_of_date or horizon or risk_mode:
        report = store.get_latest_analysis_report(
            session_id=session_id,
            symbol=symbol,
            as_of_date=as_of_date,
            horizon=horizon,
            risk_mode=risk_mode,
        )
        if report:
            return report

        report = store.get_latest_analysis_report(
            symbol=symbol,
            as_of_date=as_of_date,
            horizon=horizon,
            risk_mode=risk_mode,
        )
        if report:
            return report

        if symbol and not require_exact_symbol:
            report = store.get_latest_analysis_report(session_id=session_id, symbol=symbol)
            if report:
                return report
            return store.get_latest_analysis_report(symbol=symbol)

    return None


def _resolve_active_report(
    *,
    session_id: str,
    message: str,
    context: Optional[Dict[str, Any]],
    store: AssistantStorage,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    session = store.get_session(session_id)
    session_metadata = (session or {}).get("metadata") or {}
    context_ref = _normalize_active_analysis_context(context)
    session_ref = _normalize_active_analysis_context(
        session_metadata if isinstance(session_metadata, dict) else None
    )
    explicit_symbol = _extract_symbol_from_message(message)

    if explicit_symbol:
        explicit_ref = {
            "symbol": explicit_symbol,
            "horizon": context_ref.get("horizon") or session_ref.get("horizon"),
            "risk_mode": context_ref.get("risk_mode") or session_ref.get("risk_mode"),
        }
        report = _load_report_from_reference(
            explicit_ref,
            session_id=session_id,
            store=store,
            require_exact_symbol=True,
        )
        return report, explicit_ref

    for candidate in (context_ref, session_ref):
        if candidate:
            report = _load_report_from_reference(candidate, session_id=session_id, store=store)
            if report:
                return report, candidate

    if context_ref.get("symbol") or session_ref.get("symbol"):
        return None, context_ref or session_ref

    report = store.get_latest_analysis_report(session_id=session_id)
    if report:
        return report, reports.build_active_analysis_reference(report)

    report = store.get_latest_analysis_report()
    if report:
        return report, reports.build_active_analysis_reference(report)

    return None, context_ref or session_ref


def _no_analysis_reply(reference: Dict[str, Any]) -> str:
    symbol = reference.get("symbol")
    if symbol:
        return (
            f"No stored analysis report exists for {symbol} yet. "
            "Run Assistant Analyze first, then ask follow-up questions about that report."
        )
    return (
        "No analysis report has been generated yet. "
        "Run Assistant Analyze first, then I can explain the signal, drivers, risks, and strategy view from that report."
    )


async def generate_analysis_report(
    request: Dict[str, Any],
    store: AssistantStorage = storage,
) -> Dict[str, Any]:
    session_id = request.get("session_id")
    sid, _history = _prepare_history(session_id, store)

    symbol = orchestrator.normalize_symbol(request.get("symbol") or "")
    horizon = str(request.get("horizon") or "").strip()
    risk_mode = str(request.get("risk_mode") or "").strip()
    trace_id = str(request.get("trace_id") or uuid.uuid4())
    freshness = await orchestrator.ensure_freshness(symbol)
    as_of_date = freshness["as_of_date"]
    job_context = intelligence.build_analysis_job_context(
        {**request, "symbol": symbol, "horizon": horizon, "risk_mode": risk_mode},
        session_id=sid,
        trace_id=trace_id,
        as_of_date=as_of_date,
        freshness=freshness,
    )

    await orchestrator.run_features(symbol, as_of_date)
    await orchestrator.run_signals(symbol, as_of_date)

    signal = orchestrator.fetch_signal(symbol, as_of_date)
    if not signal:
        raise HTTPException(status_code=404, detail="signal not available after compute")

    key_features = orchestrator.fetch_key_features(symbol, as_of_date)
    quality = orchestrator.fetch_quality(symbol, as_of_date, freshness)
    evidence = {
        "reason_codes": signal.get("reason_codes") or [],
        "reason_details": signal.get("reason_details") or {},
        "sources": [
            "market_bars_daily",
            "market_bars_intraday",
            "fundamentals_quarterly",
            "news_raw",
            "sentiment_daily",
            "features_daily",
            "signals_daily",
            "quality_daily",
        ],
    }
    data_bundle = intelligence.build_normalized_data_bundle(
        job_context=job_context,
        freshness=freshness,
        signal=signal,
        key_features=key_features,
        quality=quality,
    )
    feature_factor_bundle = intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal=signal,
        key_features=key_features,
        quality=quality,
    )
    strategy_bundle = strategy.build_strategy_artifact(
        job_context=job_context,
        signal=signal,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
    )
    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date=as_of_date,
        horizon=horizon,
        risk_mode=risk_mode,
        signal=signal,
        key_features=key_features,
        quality=quality,
        evidence=evidence,
        job_context=job_context,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy=strategy_bundle,
    )
    job_context_id = store.save_artifact(sid, intelligence.ANALYSIS_JOB_KIND, job_context)
    data_bundle_id = store.save_artifact(sid, intelligence.DATA_BUNDLE_KIND, data_bundle)
    factor_bundle_id = store.save_artifact(
        sid, intelligence.FEATURE_FACTOR_BUNDLE_KIND, feature_factor_bundle
    )
    strategy_id = store.save_artifact(sid, strategy.STRATEGY_ARTIFACT_KIND, strategy_bundle)
    report_id = store.save_artifact(sid, reports.ANALYSIS_REPORT_KIND, report)
    active_analysis = reports.build_active_analysis_reference(
        report, session_id=sid, report_id=report_id
    )
    store.upsert_session_metadata(
        sid,
        {
            "active_analysis": active_analysis,
            "analysis_job_context": {
                "job_id": job_context.get("job_id"),
                "trace_id": job_context.get("trace_id"),
                "artifact_id": job_context_id,
            },
        },
    )
    store.add_message(
        sid,
        "system",
        prompts.summarize_analysis_report(report),
        extra={
            "artifact_id": report_id,
            "artifact_kind": reports.ANALYSIS_REPORT_KIND,
            "analysis_job_artifact_id": job_context_id,
            "data_bundle_artifact_id": data_bundle_id,
            "feature_factor_artifact_id": factor_bundle_id,
            "strategy_artifact_id": strategy_id,
            "active_analysis": active_analysis,
        },
    )
    return {
        **report,
        "session_id": sid,
        "report_id": report_id,
        "analysis_job_artifact_id": job_context_id,
        "data_bundle_artifact_id": data_bundle_id,
        "feature_factor_artifact_id": factor_bundle_id,
        "strategy_artifact_id": strategy_id,
        "active_analysis": active_analysis,
    }


def chat_with_assistant(
    request: Dict[str, Any], store: AssistantStorage = storage
) -> Dict[str, Any]:
    _ensure_llm_enabled()
    _require_api_key()

    session_id = request.get("session_id")
    message = request.get("message") or ""
    context = request.get("context")

    sid, history = _prepare_history(session_id, store)
    report, requested_reference = _resolve_active_report(
        session_id=sid,
        message=message,
        context=context,
        store=store,
    )

    active_analysis = (
        reports.build_active_analysis_reference(
            report,
            session_id=report.get("session_id") or sid,
            report_id=report.get("report_id"),
        )
        if report
        else {
            **requested_reference,
            "session_id": sid,
        }
        if requested_reference
        else None
    )

    if message:
        store.add_message(
            sid,
            "user",
            message,
            extra={"active_analysis": active_analysis},
        )

    if not report:
        reply = _no_analysis_reply(requested_reference)
        store.add_message(
            sid,
            "assistant",
            reply,
            extra={"active_analysis": active_analysis, "report_found": False},
        )
        return {
            "session_id": sid,
            "reply": reply,
            "citations": ["no_analysis_report"],
            "active_analysis": active_analysis,
            "report_found": False,
        }

    store.upsert_session_metadata(sid, {"active_analysis": active_analysis})
    messages: List[ChatCompletionMessageParam] = prompts.build_grounded_chat_messages(
        history, message, report, context
    )
    reply, model_used, usage = _safe_completion(messages)
    grounding_context = {
        "report_id": report.get("report_id"),
        "active_analysis": active_analysis,
        "question": message,
        "signal_summary": report.get("signal_summary"),
        "overall_analysis": report.get("overall_analysis"),
        "strategy_view": report.get("strategy_view"),
        "risks_weaknesses_invalidators": report.get("risks_weaknesses_invalidators"),
        "evidence_provenance": report.get("evidence_provenance"),
    }
    grounding_artifact_id = store.save_artifact(
        sid, strategy.CHAT_GROUNDING_CONTEXT_KIND, grounding_context
    )

    store.add_message(
        sid,
        "assistant",
        reply,
        model=model_used,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
        extra={
            "artifact_id": report.get("report_id"),
            "artifact_kind": reports.ANALYSIS_REPORT_KIND,
            "grounding_artifact_id": grounding_artifact_id,
            "active_analysis": active_analysis,
            "report_found": True,
        },
    )

    return {
        "session_id": sid,
        "reply": reply,
        "citations": [
            f"analysis_report:{report.get('report_id')}",
            "signal_summary",
            "macro_geopolitical_analysis",
            "overall_analysis",
            "strategy_view",
            "risks_weaknesses_invalidators",
        ],
        "active_analysis": active_analysis,
        "report_found": True,
    }


def explain_signal(
    payload: Dict[str, Any],
    *,
    signal_fetcher: Optional[Callable[[str, str, int], Any]] = None,
    store: AssistantStorage = storage,
) -> Dict[str, Any]:
    _ensure_llm_enabled()
    _require_api_key()

    if signal_fetcher is None:
        from api import main as api_main  # late import to avoid circular

        signal_fetcher = api_main.compute_signal_for_symbol

    symbol = payload.get("symbol")
    as_of = payload.get("as_of")
    lookback = int(payload.get("lookback") or 252)

    result = signal_fetcher(symbol, as_of, lookback)
    summary = (
        prompts.summarize_signal(result.model_dump())
        if hasattr(result, "model_dump")
        else prompts.summarize_signal(result)
    )

    history: List[Dict[str, str]] = []
    messages = prompts.build_chat_messages(
        history, f"Explain this signal: {summary}", None
    )
    reply, model_used, usage = _safe_completion(messages)

    session_id = payload.get("session_id") or store.create_session(
        metadata={"symbol": symbol, "as_of": as_of}
    )
    store.add_message(session_id, "system", summary)
    store.add_message(
        session_id,
        "assistant",
        reply,
        model=model_used,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )
    store.save_artifact(
        session_id,
        "signal_explanation",
        {"symbol": symbol, "as_of": as_of, "lookback": lookback, "summary": summary},
    )

    return {"session_id": session_id, "reply": reply, "citations": ["signal_summary"]}


def explain_backtest(
    payload: Dict[str, Any],
    *,
    backtest_runner: Optional[Callable[..., Any]] = None,
    store: AssistantStorage = storage,
) -> Dict[str, Any]:
    _ensure_llm_enabled()
    _require_api_key()

    backtest_request_model = None
    if backtest_runner is None:
        from api import main as api_main

        backtest_runner = api_main.portfolio_backtest
        backtest_request_model = api_main.PortfolioBacktestRequest

    req_obj = backtest_request_model(**payload) if backtest_request_model else payload
    result = backtest_runner(req_obj)  # type: ignore[arg-type]
    summary = (
        prompts.summarize_backtest(result.model_dump())
        if hasattr(result, "model_dump")
        else prompts.summarize_backtest(result)
    )

    messages = prompts.build_chat_messages(
        [], f"Explain this backtest: {summary}", None
    )
    reply, model_used, usage = _safe_completion(messages)

    session_id = payload.get("session_id") or store.create_session(
        metadata={"kind": "backtest"}
    )
    store.add_message(session_id, "system", summary)
    store.add_message(
        session_id,
        "assistant",
        reply,
        model=model_used,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )
    store.save_artifact(
        session_id, "backtest_explanation", {"summary": summary, "request": payload}
    )

    return {"session_id": session_id, "reply": reply, "citations": ["backtest_summary"]}


def title_session(
    session_id: str, hint: Optional[str] = None, store: AssistantStorage = storage
) -> Dict[str, Any]:
    _ensure_llm_enabled()

    history = store.get_messages(session_id=session_id)
    summary_text = hint or "Summarize this conversation with a short title."
    messages = prompts.build_chat_messages(history, summary_text, None)
    reply, model_used, usage = _safe_completion(messages)

    store.add_message(
        session_id,
        "assistant",
        reply,
        model=model_used,
        tokens_in=usage.get("prompt_tokens"),
        tokens_out=usage.get("completion_tokens"),
    )
    store.upsert_session_metadata(session_id, {"title": reply})
    return {"session_id": session_id, "title": reply}
