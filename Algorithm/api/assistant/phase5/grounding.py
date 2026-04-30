from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from api.assistant import reports
from api.assistant.storage import AssistantStorage


_SYMBOL_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
_IGNORED_SYMBOLS = {
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

_ACTIVE_ANALYSIS_KEYS = (
    "report_id",
    "session_id",
    "symbol",
    "as_of_date",
    "horizon",
    "risk_mode",
    "scenario",
    "analysis_depth",
    "refresh_mode",
    "market_regime",
    "signal",
    "conviction_tier",
    "strategy_posture",
    "actionability_score",
    "freshness_status",
    "report_version",
    "strategy_version",
)


def normalize_active_analysis_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not context:
        return {}
    active = context.get("active_analysis") if isinstance(context, dict) else None
    payload = active if isinstance(active, dict) else context
    return {key: payload.get(key) for key in _ACTIVE_ANALYSIS_KEYS if payload.get(key) not in (None, "")}


def extract_symbol_from_message(message: str) -> Optional[str]:
    if not message:
        return None
    for match in _SYMBOL_PATTERN.findall(message):
        symbol = match.upper()
        if symbol not in _IGNORED_SYMBOLS:
            return symbol
    return None


def load_report_from_reference(
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


def resolve_active_report(
    *,
    session_id: str,
    message: str,
    context: Optional[Dict[str, Any]],
    store: AssistantStorage,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    session = store.get_session(session_id)
    session_metadata = (session or {}).get("metadata") or {}
    context_ref = normalize_active_analysis_context(context)
    session_ref = normalize_active_analysis_context(
        session_metadata if isinstance(session_metadata, dict) else None
    )
    explicit_symbol = extract_symbol_from_message(message)

    if explicit_symbol:
        explicit_ref = {
            "symbol": explicit_symbol,
            "horizon": context_ref.get("horizon") or session_ref.get("horizon"),
            "risk_mode": context_ref.get("risk_mode") or session_ref.get("risk_mode"),
        }
        report = load_report_from_reference(
            explicit_ref,
            session_id=session_id,
            store=store,
            require_exact_symbol=True,
        )
        return report, explicit_ref

    for candidate in (context_ref, session_ref):
        if candidate:
            report = load_report_from_reference(candidate, session_id=session_id, store=store)
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


def no_active_analysis_reply(reference: Dict[str, Any], *, intent: str) -> str:
    symbol = reference.get("symbol")
    if symbol:
        return (
            f"No stored analysis report exists for {symbol} yet. "
            f"Run Assistant Analyze for {symbol} first, then I can answer {intent.replace('_', ' ')} follow-ups using the active report and strategy artifact."
        )
    return (
        "No analysis report is active yet. "
        "Run Assistant Analyze first, then I can explain the signal, strategy posture, risks, invalidators, and evidence from the active report."
    )
