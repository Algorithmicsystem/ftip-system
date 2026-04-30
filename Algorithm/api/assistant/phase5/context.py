from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_list(values: Iterable[Any], *, limit: int = 4) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        items.append(str(value))
        if len(items) >= limit:
            break
    return items


def _top_score_snapshot(proprietary_scores: Dict[str, Any], *, limit: int = 5) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for name, payload in proprietary_scores.items():
        if not isinstance(payload, dict):
            continue
        score = _safe_float(payload.get("score"))
        if score is None:
            continue
        ranked.append(
            {
                "name": name,
                "score": round(score, 2),
                "coverage_status": payload.get("coverage_status") or "unknown",
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


def _scenario_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = (report.get("strategy") or {}).get("scenario_matrix") or report.get("scenario_matrix") or {}
    return {
        name: {
            "summary": (payload or {}).get("summary"),
            "supporting_conditions": _compact_list(
                (payload or {}).get("supporting_conditions") or []
            ),
            "risk_conditions": _compact_list((payload or {}).get("risk_conditions") or []),
            "expected_posture_shift": (payload or {}).get("expected_posture_shift"),
            "confidence_level": (payload or {}).get("confidence_level"),
            "fragility_notes": _compact_list((payload or {}).get("fragility_notes") or []),
        }
        for name, payload in scenarios.items()
    }


def _section_catalog(report: Dict[str, Any]) -> Dict[str, str]:
    return {
        "signal_summary": report.get("signal_summary") or "",
        "technical_analysis": report.get("technical_analysis") or "",
        "fundamental_analysis": report.get("fundamental_analysis") or "",
        "statistical_analysis": report.get("statistical_analysis") or "",
        "sentiment_analysis": report.get("sentiment_analysis") or "",
        "macro_geopolitical_analysis": report.get("macro_geopolitical_analysis") or "",
        "risk_quality_analysis": report.get("risk_quality_analysis") or "",
        "overall_analysis": report.get("overall_analysis") or "",
        "strategy_view": report.get("strategy_view") or "",
        "risks_weaknesses_invalidators": report.get("risks_weaknesses_invalidators") or "",
        "evidence_provenance": report.get("evidence_provenance") or "",
    }


def _evidence_snapshot(report: Dict[str, Any]) -> Dict[str, Any]:
    why_signal = report.get("why_this_signal") or {}
    domain_agreement = report.get("domain_agreement") or {}
    strategy = report.get("strategy") or {}
    return {
        "strong_evidence": _compact_list(
            [
                item.get("detail") or item.get("label")
                for item in (why_signal.get("top_positive_drivers") or [])
            ]
        ),
        "weak_evidence": _compact_list(
            [
                item.get("detail") or item.get("label")
                for item in (why_signal.get("top_negative_drivers") or [])
            ]
            + list(strategy.get("confidence_degraders") or [])
        ),
        "missing_evidence": _compact_list(
            list(why_signal.get("missing_data_warnings") or [])
            + list(report.get("uncertainty_notes") or [])
        ),
        "conflicting_evidence": _compact_list(
            list(domain_agreement.get("strongest_conflicting_domains") or [])
            + list(domain_agreement.get("agreement_flags") or [])
        ),
    }


def build_narrator_context(
    report: Dict[str, Any],
    *,
    active_analysis: Optional[Dict[str, Any]],
    route: Dict[str, Any],
    user_message: str,
    caller_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    active_analysis = active_analysis or {}
    strategy = report.get("strategy") or {}
    proprietary_scores = report.get("proprietary_scores") or {}
    regime = report.get("regime_intelligence") or {}
    fragility = report.get("fragility_intelligence") or {}
    agreement = report.get("domain_agreement") or {}
    invalidators = report.get("invalidators") or {}
    sections = _section_catalog(report)
    selected_sections = {
        name: sections.get(name)
        for name in route.get("relevant_sections") or []
        if sections.get(name)
    }

    freshness_summary = report.get("freshness_summary") or {}
    active_context = {
        "session_id": active_analysis.get("session_id"),
        "symbol": active_analysis.get("symbol") or report.get("symbol"),
        "as_of_date": active_analysis.get("as_of_date") or report.get("as_of_date"),
        "horizon": active_analysis.get("horizon") or report.get("horizon"),
        "risk_mode": active_analysis.get("risk_mode") or report.get("risk_mode"),
        "scenario": active_analysis.get("scenario") or report.get("scenario"),
        "signal": active_analysis.get("signal") or strategy.get("final_signal"),
        "strategy_posture": active_analysis.get("strategy_posture")
        or strategy.get("strategy_posture"),
        "conviction_tier": active_analysis.get("conviction_tier")
        or strategy.get("conviction_tier"),
        "freshness_status": active_analysis.get("freshness_status")
        or freshness_summary.get("overall_status"),
        "report_version": active_analysis.get("report_version") or report.get("report_version"),
        "strategy_version": active_analysis.get("strategy_version")
        or report.get("strategy_version")
        or strategy.get("strategy_version"),
    }

    return {
        "active_context": active_context,
        "question": user_message,
        "question_intent": route.get("intent"),
        "answer_mode": route.get("answer_mode"),
        "routing_confidence": route.get("routing_confidence"),
        "matched_keywords": route.get("matched_keywords") or [],
        "signal_snapshot": {
            "final_signal": strategy.get("final_signal")
            or (report.get("signal") or {}).get("final_action")
            or (report.get("signal") or {}).get("action"),
            "strategy_posture": strategy.get("strategy_posture") or report.get("strategy_posture"),
            "confidence_score": report.get("confidence_score"),
            "conviction_tier": strategy.get("conviction_tier"),
            "actionability_score": strategy.get("actionability_score"),
            "participant_fit": strategy.get("participant_fit") or [],
            "strategy_summary": report.get("strategy_summary"),
        },
        "top_proprietary_scores": _top_score_snapshot(proprietary_scores),
        "regime_intelligence": {
            "regime_label": regime.get("regime_label"),
            "regime_confidence": regime.get("regime_confidence"),
            "regime_instability": regime.get("regime_instability"),
            "transition_risk": regime.get("transition_risk"),
            "breakout_readiness": regime.get("breakout_readiness"),
        },
        "fragility_intelligence": {
            "instability_score": fragility.get("instability_score"),
            "fragility_index": proprietary_scores.get("Signal Fragility Index", {}).get("score")
            if isinstance(proprietary_scores.get("Signal Fragility Index"), dict)
            else None,
            "clean_setup_score": fragility.get("clean_setup_score"),
            "confidence_degradation_triggers": _compact_list(
                fragility.get("confidence_degradation_triggers") or []
            ),
        },
        "domain_agreement": {
            "domain_agreement_score": agreement.get("domain_agreement_score"),
            "domain_conflict_score": agreement.get("domain_conflict_score"),
            "strongest_confirming_domains": _compact_list(
                agreement.get("strongest_confirming_domains") or []
            ),
            "strongest_conflicting_domains": _compact_list(
                agreement.get("strongest_conflicting_domains") or []
            ),
        },
        "section_summaries": {
            "signal_summary": sections["signal_summary"],
            "technical_analysis": sections["technical_analysis"],
            "fundamental_analysis": sections["fundamental_analysis"],
            "sentiment_analysis": sections["sentiment_analysis"],
            "macro_geopolitical_analysis": sections["macro_geopolitical_analysis"],
            "risk_quality_analysis": sections["risk_quality_analysis"],
            "overall_analysis": sections["overall_analysis"],
            "strategy_view": sections["strategy_view"],
            "risks_weaknesses_invalidators": sections["risks_weaknesses_invalidators"],
            "evidence_provenance": sections["evidence_provenance"],
        },
        "selected_sections": selected_sections,
        "scenario_matrix": _scenario_snapshot(report),
        "invalidators": {
            "top_invalidators": _compact_list(invalidators.get("top_invalidators") or []),
            "regime_invalidators": _compact_list(invalidators.get("regime_invalidators") or []),
            "narrative_invalidators": _compact_list(
                invalidators.get("narrative_invalidators") or []
            ),
            "macro_invalidators": _compact_list(invalidators.get("macro_invalidators") or []),
            "quality_invalidators": _compact_list(
                invalidators.get("quality_freshness_invalidators") or []
            ),
            "confirmation_triggers": _compact_list(report.get("confirmation_triggers") or []),
            "deterioration_triggers": _compact_list(report.get("deterioration_triggers") or []),
        },
        "evidence_summary": _evidence_snapshot(report),
        "freshness_and_coverage": {
            "overall_status": freshness_summary.get("overall_status"),
            "domain_status": freshness_summary.get("domains") or {},
            "missing_data_warnings": _compact_list(report.get("missing_data_warnings") or []),
            "freshness_warnings": _compact_list(report.get("freshness_warnings") or []),
        },
        "uncertainty_notes": _compact_list(report.get("uncertainty_notes") or [], limit=6),
        "followup_questions": route.get("followup_questions") or [],
        "caller_context": caller_context or {},
    }
