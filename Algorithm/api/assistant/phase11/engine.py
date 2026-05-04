from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant import reports
from api.assistant.phase9.common import build_candidate_snapshot, clamp as clamp_score, safe_float
from api.assistant.phase9.execution import build_execution_quality
from api.assistant.phase9.ranking import build_candidate_ranking
from api.assistant.storage import AssistantStorage

from .common import (
    PORTFOLIO_RISK_MODEL_ARTIFACT_KIND,
    PORTFOLIO_RISK_MODEL_VERSION,
    compact_list,
    now_utc,
)
from .covariance import build_return_profile, pairwise_relationship
from .exposures import build_factor_exposure, exposure_similarity
from .overlap import pairwise_overlap, summarize_overlap
from .portfolio import build_portfolio_risk_overlay
from .utility import enrich_with_marginal_utility


def _load_cohort_reports(
    *,
    current_report: Dict[str, Any],
    session_id: str | None,
    store: AssistantStorage,
    limit: int = 14,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=reports.ANALYSIS_REPORT_KIND, limit=150)
    same_session: Dict[str, Dict[str, Any]] = {}
    same_profile: Dict[str, Dict[str, Any]] = {}
    current_symbol = str(current_report.get("symbol") or "")
    for artifact in artifacts:
        payload = artifact.get("payload") or {}
        symbol = str(payload.get("symbol") or "")
        if not symbol or symbol == current_symbol:
            continue
        if payload.get("horizon") != current_report.get("horizon"):
            continue
        if payload.get("risk_mode") != current_report.get("risk_mode"):
            continue
        candidate = {
            **payload,
            "report_id": artifact.get("id"),
            "session_id": artifact.get("session_id"),
        }
        if str(artifact.get("session_id")) == str(session_id):
            same_session.setdefault(symbol, candidate)
        else:
            same_profile.setdefault(symbol, candidate)
    peers = list(same_session.values())
    seen = {str(item.get("symbol") or "") for item in peers}
    for item in same_profile.values():
        if len(peers) >= limit - 1:
            break
        symbol = str(item.get("symbol") or "")
        if symbol and symbol not in seen:
            peers.append(item)
            seen.add(symbol)
    return peers[: max(0, limit - 1)]


def _base_candidate_row(report: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = build_candidate_snapshot(
        report,
        report_id=report.get("report_id"),
        session_id=report.get("session_id"),
    )
    execution = build_execution_quality(snapshot)
    ranking = build_candidate_ranking(snapshot, execution)
    candidate_classification = (
        report.get("candidate_classification")
        or snapshot.get("current_candidate_classification")
        or "watchlist_candidate"
    )
    size_band = report.get("size_band")
    if not size_band:
        permission = str(snapshot.get("deployment_permission") or "analysis_only")
        size_band = (
            "exploratory allocation band"
            if permission.endswith("eligible")
            else "paper / shadow band"
            if permission == "paper_shadow_only"
            else "watchlist only"
        )
    portfolio_fit_quality = safe_float(report.get("portfolio_fit_quality"))
    if portfolio_fit_quality is None:
        portfolio_fit_quality = clamp_score(
            (float(ranking.get("portfolio_candidate_score") or 0.0) * 0.62)
            + (float(execution.get("execution_quality_score") or 0.0) * 0.16)
            + (float(snapshot.get("live_readiness_score") or 0.0) * 0.10)
            - max(0.0, float(snapshot.get("signal_fragility_index") or 0.0) - 52.0) * 0.20,
            0.0,
            100.0,
        )

    return {
        **snapshot,
        **execution,
        **ranking,
        "candidate_classification": candidate_classification,
        "size_band": size_band,
        "weight_band": report.get("weight_band") or snapshot.get("current_weight_band"),
        "risk_budget_band": report.get("risk_budget_band"),
        "portfolio_fit_quality": round(float(portfolio_fit_quality), 2),
        "candidate_blockers": compact_list(report.get("candidate_blockers") or [], limit=8),
        "active_warnings": compact_list(
            [
                report.get("concentration_warning"),
                report.get("cluster_concentration_warning"),
                report.get("sector_crowding_warning"),
                report.get("fragility_cluster_warning"),
                report.get("macro_exposure_warning"),
                report.get("theme_exposure_warning"),
            ],
            limit=8,
        ),
        "report": report,
    }


def _portfolio_risk_model_summary(current: Dict[str, Any], portfolio_overlay: Dict[str, Any]) -> str:
    return (
        f"Portfolio-fit rank is {current.get('portfolio_fit_rank')} with marginal utility {current.get('marginal_portfolio_utility')} / 100 and hidden overlap {current.get('hidden_overlap_score')} / 100. "
        f"Portfolio stress is {portfolio_overlay.get('portfolio_stress_score')} / 100 and diversification health is {portfolio_overlay.get('diversification_health_score')} / 100."
    )


def _hidden_overlap_summary(current: Dict[str, Any]) -> str:
    return (
        f"Realized overlap is {current.get('overlap_score')} / 100, hidden overlap is {current.get('hidden_overlap_score')} / 100, redundancy is {current.get('redundancy_score')} / 100, and complementarity is {current.get('complementarity_score')} / 100. "
        f"Closest overlap is {current.get('most_redundant_symbol') or 'none'}. Drivers: {', '.join(current.get('overlap_drivers') or ['none'])}."
    )


def _factor_exposure_summary(current: Dict[str, Any]) -> str:
    vector = current.get("factor_exposure_vector") or {}
    macro_profile = current.get("macro_exposure_profile") or {}
    loadings = ", ".join(current.get("factor_loading_summary") or []) or "none"
    return (
        f"Exposure cluster is {current.get('exposure_cluster') or 'unknown'} with style affinity {current.get('style_affinity') or 'unknown'} and exposure confidence {current.get('exposure_confidence')} / 100. "
        f"Market beta proxy is {macro_profile.get('market_beta_proxy')}, sector dependence is {macro_profile.get('sector_dependence_score')} / 100, and top loadings are {loadings}. "
        f"Factor vector currently emphasizes market beta {vector.get('market_beta')}, momentum {vector.get('momentum_trend')}, rates {vector.get('rates_sensitivity')}, and fragility {vector.get('fragility_sensitivity')}."
    )


def _concentration_summary(current: Dict[str, Any], portfolio_overlay: Dict[str, Any]) -> str:
    warnings = "; ".join(portfolio_overlay.get("portfolio_risk_warnings") or []) or "no major hidden concentration warning"
    return (
        f"Hidden concentration is {portfolio_overlay.get('hidden_concentration_score')} / 100, cluster risk is {portfolio_overlay.get('cluster_risk_score')} / 100, fragility cluster risk is {portfolio_overlay.get('fragility_cluster_risk')} / 100, and macro cluster risk is {portfolio_overlay.get('macro_cluster_risk')} / 100. "
        f"Diversification health is {portfolio_overlay.get('diversification_health_score')} / 100. {warnings}."
    )


def _replacement_summary(current: Dict[str, Any]) -> str:
    if current.get("replacement_candidate"):
        return (
            f"{current.get('replacement_candidate')} is currently the cleaner substitution candidate with substitution score {current.get('substitution_score')} / 100. "
            f"Better alternative flag is {'yes' if current.get('better_alternative_flag') else 'no'}, diversification upgrade flag is {'yes' if current.get('diversification_upgrade_flag') else 'no'}, and overlap reduction flag is {'yes' if current.get('overlap_reduction_flag') else 'no'}."
        )
    return (
        f"No superior substitution candidate is currently flagged. Replacement value score is {current.get('replacement_value_score')} / 100 and marginal diversification bonus is {current.get('marginal_diversification_bonus')}."
    )


def _portfolio_stress_summary(current: Dict[str, Any], portfolio_overlay: Dict[str, Any]) -> str:
    return (
        f"Portfolio stress is {portfolio_overlay.get('portfolio_stress_score')} / 100, portfolio fragility is {portfolio_overlay.get('portfolio_fragility_score')} / 100, and correlation-breakdown risk is {portfolio_overlay.get('correlation_breakdown_risk')} / 100. "
        f"Unstable portfolio state flag is {'on' if portfolio_overlay.get('unstable_portfolio_state_flag') else 'off'}, and current candidate gap / liquidity fragility is {current.get('implementation_fragility_score')} / 100."
    )


def build_portfolio_risk_model_artifact(
    *,
    current_report: Dict[str, Any],
    session_id: str | None,
    store: AssistantStorage,
) -> Dict[str, Any]:
    cohort_reports = [
        {
            **current_report,
            "report_id": current_report.get("report_id"),
            "session_id": session_id or current_report.get("session_id"),
        },
        *_load_cohort_reports(
            current_report=current_report,
            session_id=session_id,
            store=store,
        ),
    ]
    candidate_rows = [_base_candidate_row(report) for report in cohort_reports]

    history_profiles = {
        str(row.get("symbol") or ""): build_return_profile(row.get("report") or {})
        for row in candidate_rows
    }
    exposure_profiles = {
        str(row.get("symbol") or ""): build_factor_exposure(
            row.get("report") or {},
            row,
            history_profiles[str(row.get("symbol") or "")],
        )
        for row in candidate_rows
    }

    pairwise_rows: List[Dict[str, Any]] = []
    for left in candidate_rows:
        left_symbol = str(left.get("symbol") or "")
        for right in candidate_rows:
            right_symbol = str(right.get("symbol") or "")
            if not left_symbol or not right_symbol or left_symbol == right_symbol:
                continue
            relationship = pairwise_relationship(
                history_profiles[left_symbol],
                history_profiles[right_symbol],
            )
            similarity = exposure_similarity(
                exposure_profiles[left_symbol],
                exposure_profiles[right_symbol],
            )
            pairwise_rows.append(
                {
                    "symbol": left_symbol,
                    **pairwise_overlap(left, right, relationship, similarity),
                }
            )

    rows_with_overlap: List[Dict[str, Any]] = []
    for row in candidate_rows:
        overlap_summary, pairwise = summarize_overlap(row, pairwise_rows)
        rows_with_overlap.append(
            {
                **row,
                **exposure_profiles[str(row.get("symbol") or "")],
                **overlap_summary,
                "pairwise_relationships": pairwise,
            }
        )

    initial_overlay = build_portfolio_risk_overlay(rows_with_overlap, pairwise_rows)
    utility_rows = enrich_with_marginal_utility(
        rows_with_overlap,
        portfolio_overlay=initial_overlay,
    )
    final_overlay = build_portfolio_risk_overlay(utility_rows, pairwise_rows)

    for row in utility_rows:
        row.update(
            {
                "hidden_concentration_score": final_overlay.get("hidden_concentration_score"),
                "cluster_risk_score": final_overlay.get("cluster_risk_score"),
                "fragility_cluster_risk": final_overlay.get("fragility_cluster_risk"),
                "event_cluster_risk": final_overlay.get("event_cluster_risk"),
                "macro_cluster_risk": final_overlay.get("macro_cluster_risk"),
                "narrative_cluster_risk": final_overlay.get("narrative_cluster_risk"),
                "diversification_health_score": final_overlay.get("diversification_health_score"),
                "portfolio_stress_score": final_overlay.get("portfolio_stress_score"),
                "portfolio_fragility_score": final_overlay.get("portfolio_fragility_score"),
                "correlation_breakdown_risk": final_overlay.get("correlation_breakdown_risk"),
            }
        )
        row["active_warnings"] = compact_list(
            [*(row.get("active_warnings") or []), *(final_overlay.get("portfolio_risk_warnings") or [])],
            limit=10,
        )

    utility_rows.sort(
        key=lambda item: (
            float(item.get("portfolio_contribution_score") or 0.0),
            float(item.get("marginal_portfolio_utility") or 0.0),
        ),
        reverse=True,
    )
    for index, row in enumerate(utility_rows, start=1):
        row["portfolio_fit_rank"] = index
    current_symbol = str(current_report.get("symbol") or "")
    current = next(item for item in utility_rows if str(item.get("symbol") or "") == current_symbol)
    summaries = {
        "portfolio_risk_model_summary": _portfolio_risk_model_summary(current, final_overlay),
        "hidden_overlap_redundancy_analysis": _hidden_overlap_summary(current),
        "factor_exposure_summary": _factor_exposure_summary(current),
        "concentration_cluster_risk_analysis": _concentration_summary(current, final_overlay),
        "replacement_diversification_analysis": _replacement_summary(current),
        "portfolio_stress_fragility_summary": _portfolio_stress_summary(current, final_overlay),
    }

    return {
        "portfolio_risk_model_kind": PORTFOLIO_RISK_MODEL_ARTIFACT_KIND,
        "portfolio_risk_model_version": PORTFOLIO_RISK_MODEL_VERSION,
        "generated_at": now_utc(),
        "current_candidate": current,
        "cohort_portfolio_risk_ranking": [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "report",
                    "pairwise_relationships",
                }
            }
            for row in utility_rows[:12]
        ],
        "top_pairwise_relationships": current.get("pairwise_relationships") or [],
        "portfolio_overlay": final_overlay,
        **summaries,
    }
