from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant import reports

from . import grounding
from .context import build_narrator_context
from .prompts import build_grounded_narrator_messages
from .routing import route_question


def _citations_for_report(
    report: Dict[str, Any],
    *,
    relevant_sections: List[str],
) -> List[str]:
    citations = [f"analysis_report:{report.get('report_id')}"]
    citations.extend(section for section in relevant_sections if report.get(section))
    return citations


def prepare_narrator_exchange(
    *,
    session_id: str,
    history: List[Dict[str, str]],
    message: str,
    context: Optional[Dict[str, Any]],
    store: Any,
) -> Dict[str, Any]:
    route = route_question(message)
    report, requested_reference = grounding.resolve_active_report(
        session_id=session_id,
        message=message,
        context=context,
        store=store,
    )

    if not report:
        active_analysis = (
            {
                **requested_reference,
                "session_id": session_id,
            }
            if requested_reference
            else None
        )
        return {
            "report": None,
            "requested_reference": requested_reference,
            "active_analysis": active_analysis,
            "route": route,
            "reply": grounding.no_active_analysis_reply(
                requested_reference,
                intent=str(route.get("intent") or "analysis"),
            ),
            "citations": ["no_analysis_report"],
            "messages": None,
            "narrator_context": None,
            "grounding_payload": {
                "report_found": False,
                "question": message,
                "route": route,
                "requested_reference": requested_reference,
            },
        }

    active_analysis = reports.build_active_analysis_reference(
        report,
        session_id=report.get("session_id") or session_id,
        report_id=report.get("report_id"),
    )
    narrator_context = build_narrator_context(
        report,
        active_analysis=active_analysis,
        route=route,
        user_message=message,
        caller_context=context,
    )
    messages = build_grounded_narrator_messages(history, message, narrator_context)
    return {
        "report": report,
        "requested_reference": requested_reference,
        "active_analysis": active_analysis,
        "route": route,
        "reply": None,
        "citations": _citations_for_report(
            report,
            relevant_sections=route.get("relevant_sections") or [],
        ),
        "messages": messages,
        "narrator_context": narrator_context,
        "grounding_payload": {
            "report_found": True,
            "report_id": report.get("report_id"),
            "question": message,
            "route": route,
            "active_analysis": active_analysis,
            "selected_sections": narrator_context.get("selected_sections"),
            "signal_snapshot": narrator_context.get("signal_snapshot"),
            "evidence_summary": narrator_context.get("evidence_summary"),
            "invalidators": narrator_context.get("invalidators"),
            "freshness_and_coverage": narrator_context.get("freshness_and_coverage"),
            "uncertainty_notes": narrator_context.get("uncertainty_notes"),
        },
    }
