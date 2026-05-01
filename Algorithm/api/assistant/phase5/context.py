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
        "evaluation_research_analysis": report.get("evaluation_research_analysis") or "",
        "deployment_readiness_summary": report.get("deployment_readiness_summary") or "",
        "deployment_permission_analysis": report.get("deployment_permission_analysis") or "",
        "risk_budget_exposure_analysis": report.get("risk_budget_exposure_analysis") or "",
        "rollout_stage_summary": report.get("rollout_stage_summary") or "",
        "portfolio_context_summary": report.get("portfolio_context_summary") or "",
        "portfolio_fit_analysis": report.get("portfolio_fit_analysis") or "",
        "execution_quality_analysis": report.get("execution_quality_analysis") or "",
        "portfolio_workflow_summary": report.get("portfolio_workflow_summary") or "",
        "learning_summary": report.get("learning_summary") or "",
        "regime_learning_summary": report.get("regime_learning_summary") or "",
        "adaptation_queue_summary": report.get("adaptation_queue_summary") or "",
        "experiment_registry_summary": report.get("experiment_registry_summary") or "",
        "archetype_motif_summary": report.get("archetype_motif_summary") or "",
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
    deployment_readiness = report.get("deployment_readiness") or {}
    deployment_mode = deployment_readiness.get("deployment_mode") or {}
    model_readiness = deployment_readiness.get("model_readiness") or {}
    deployment_permission = deployment_readiness.get("deployment_permission") or {}
    risk_budgeting = deployment_readiness.get("risk_budgeting") or {}
    rollout = deployment_readiness.get("rollout_workflow") or {}
    drift_monitor = deployment_readiness.get("drift_monitor") or {}
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
        "deployment_mode": active_analysis.get("deployment_mode") or report.get("deployment_mode"),
        "deployment_permission": active_analysis.get("deployment_permission")
        or report.get("deployment_permission"),
        "trust_tier": active_analysis.get("trust_tier") or report.get("trust_tier"),
        "live_readiness_status": active_analysis.get("live_readiness_status")
        or report.get("model_readiness_status"),
        "live_readiness_score": active_analysis.get("live_readiness_score")
        or report.get("live_readiness_score"),
        "rollout_stage": active_analysis.get("rollout_stage") or report.get("rollout_stage"),
        "candidate_classification": active_analysis.get("candidate_classification")
        or report.get("candidate_classification"),
        "ranked_opportunity_score": active_analysis.get("ranked_opportunity_score")
        or report.get("ranked_opportunity_score"),
        "portfolio_fit_quality": active_analysis.get("portfolio_fit_quality")
        or report.get("portfolio_fit_quality"),
        "size_band": active_analysis.get("size_band") or report.get("size_band"),
        "setup_archetype": active_analysis.get("setup_archetype")
        or ((report.get("setup_archetype") or {}).get("archetype_name")),
        "research_version": active_analysis.get("research_version")
        or report.get("research_version"),
        "learning_priority": active_analysis.get("learning_priority")
        or report.get("learning_priority"),
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
            "evaluation_research_analysis": sections["evaluation_research_analysis"],
            "deployment_readiness_summary": sections["deployment_readiness_summary"],
            "deployment_permission_analysis": sections["deployment_permission_analysis"],
            "risk_budget_exposure_analysis": sections["risk_budget_exposure_analysis"],
            "rollout_stage_summary": sections["rollout_stage_summary"],
            "portfolio_context_summary": sections["portfolio_context_summary"],
            "portfolio_fit_analysis": sections["portfolio_fit_analysis"],
            "execution_quality_analysis": sections["execution_quality_analysis"],
            "portfolio_workflow_summary": sections["portfolio_workflow_summary"],
            "learning_summary": sections["learning_summary"],
            "regime_learning_summary": sections["regime_learning_summary"],
            "adaptation_queue_summary": sections["adaptation_queue_summary"],
            "experiment_registry_summary": sections["experiment_registry_summary"],
            "archetype_motif_summary": sections["archetype_motif_summary"],
        },
        "evaluation_snapshot": report.get("evaluation") or {},
        "deployment_readiness_snapshot": {
            "deployment_mode": deployment_mode.get("active_mode") or report.get("deployment_mode"),
            "trust_tier": deployment_permission.get("trust_tier") or report.get("trust_tier"),
            "deployment_permission": deployment_permission.get("deployment_permission")
            or report.get("deployment_permission"),
            "paper_vs_live_classification": deployment_permission.get("paper_vs_live_classification"),
            "model_readiness_status": model_readiness.get("model_readiness_status")
            or report.get("model_readiness_status"),
            "live_readiness_score": model_readiness.get("live_readiness_score")
            or report.get("live_readiness_score"),
            "blockers": _compact_list(
                deployment_permission.get("deployment_blockers")
                or model_readiness.get("live_readiness_blockers")
                or report.get("deployment_blockers")
                or report.get("live_readiness_blockers")
                or []
            ),
            "minimum_required_review": deployment_permission.get("minimum_required_review")
            or report.get("minimum_required_review"),
            "human_review_required": deployment_permission.get("human_review_required")
            if deployment_permission
            else report.get("human_review_required"),
            "risk_budget_tier": risk_budgeting.get("risk_budget_tier")
            or report.get("risk_budget_tier"),
            "exposure_caution_level": risk_budgeting.get("exposure_caution_level")
            or report.get("exposure_caution_level"),
            "rollout_stage": rollout.get("rollout_stage") or report.get("rollout_stage"),
            "readiness_checkpoint": rollout.get("readiness_checkpoint")
            or report.get("readiness_checkpoint"),
            "pause_recommended": drift_monitor.get("pause_recommended")
            if drift_monitor
            else report.get("pause_recommended"),
            "degrade_to_paper_recommended": drift_monitor.get("degrade_to_paper_recommended")
            if drift_monitor
            else report.get("degrade_to_paper_recommended"),
            "drift_alerts": _compact_list(
                drift_monitor.get("drift_alerts")
                or report.get("drift_alerts")
                or []
            ),
        },
        "portfolio_snapshot": {
            "candidate_classification": report.get("candidate_classification"),
            "ranked_opportunity_score": report.get("ranked_opportunity_score"),
            "portfolio_candidate_score": report.get("portfolio_candidate_score"),
            "watchlist_priority_score": report.get("watchlist_priority_score"),
            "portfolio_fit_quality": report.get("portfolio_fit_quality"),
            "size_band": report.get("size_band"),
            "weight_band": report.get("weight_band"),
            "risk_budget_band": report.get("risk_budget_band"),
            "overlap_score": report.get("overlap_score"),
            "redundancy_score": report.get("redundancy_score"),
            "diversification_contribution_score": report.get("diversification_contribution_score"),
            "execution_quality_score": report.get("execution_quality_score"),
            "friction_penalty": report.get("friction_penalty"),
            "turnover_penalty": report.get("turnover_penalty"),
            "candidate_blockers": _compact_list(report.get("candidate_blockers") or []),
            "active_warnings": _compact_list(
                [
                    report.get("concentration_warning"),
                    report.get("cluster_concentration_warning"),
                    report.get("sector_crowding_warning"),
                    report.get("fragility_cluster_warning"),
                    report.get("macro_exposure_warning"),
                    report.get("theme_exposure_warning"),
                ]
            ),
            "cohort_ranking": [
                {
                    "symbol": item.get("symbol"),
                    "portfolio_rank": item.get("portfolio_rank"),
                    "candidate_classification": item.get("candidate_classification"),
                    "portfolio_candidate_score": item.get("portfolio_candidate_score"),
                    "portfolio_fit_quality": item.get("portfolio_fit_quality"),
                    "deployment_permission": item.get("deployment_permission"),
                }
                for item in (report.get("cohort_ranking") or [])[:6]
            ],
        },
        "learning_snapshot": {
            "research_version": report.get("research_version"),
            "setup_archetype": (report.get("setup_archetype") or {}).get("archetype_name"),
            "deployment_caution_level": (report.get("setup_archetype") or {}).get(
                "deployment_caution_level"
            ),
            "learning_priority": report.get("learning_priority"),
            "top_drift_alert": (report.get("learning_drift_alerts") or [{}])[0],
            "top_reweighting_candidate": (report.get("reweighting_candidates") or [{}])[0],
            "top_hypothesis": (report.get("research_hypotheses") or [{}])[0],
            "top_experiment": (
                ((report.get("experiment_registry") or {}).get("open_experiments") or [{}])[0]
            ),
            "active_motifs": report.get("active_motifs") or [],
            "regime_learning": (report.get("regime_conditioned_learnings") or [])[:4],
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
