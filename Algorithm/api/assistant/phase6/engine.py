from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional, Sequence

from api.assistant.storage import AssistantStorage

from .attribution import build_factor_attribution_summary
from .common import (
    EVALUATION_ARTIFACT_KIND,
    EVALUATION_VERSION,
    PREDICTION_RECORD_KIND,
    compact_list,
    score_value,
    safe_float,
)
from .linkage import link_realized_outcome
from .scorecards import (
    build_calibration_summary,
    build_ranking_scorecard,
    build_regime_breakdown,
    build_signal_scorecard,
    build_strategy_scorecard,
)


def _primary_participant_fit(report: Dict[str, Any]) -> Optional[str]:
    strategy = report.get("strategy") or {}
    fits = strategy.get("participant_fit") or report.get("participant_fit") or []
    if strategy.get("primary_participant_fit"):
        return strategy.get("primary_participant_fit")
    return fits[0] if fits else None


def _derived_slices(report: Dict[str, Any]) -> Dict[str, Any]:
    proprietary_scores = report.get("proprietary_scores") or {}
    domain_agreement = report.get("domain_agreement") or {}
    domain_availability = report.get("domain_availability") or {}
    data_bundle = report.get("data_bundle") or {}
    event_domain = data_bundle.get("event_catalyst_risk") or {}
    liquidity_domain = data_bundle.get("liquidity_execution_fragility") or {}
    breadth_domain = data_bundle.get("market_breadth_internals") or {}
    cross_asset_domain = data_bundle.get("cross_asset_confirmation") or {}
    stress_domain = data_bundle.get("stress_spillover_conditions") or {}
    narrative_crowding = score_value(proprietary_scores.get("Narrative Crowding Index"))
    macro_alignment = score_value(proprietary_scores.get("Macro Alignment Score"))
    fragility = score_value(proprietary_scores.get("Signal Fragility Index"))
    agreement_score = safe_float(domain_agreement.get("domain_agreement_score"))
    conflict_score = safe_float(domain_agreement.get("domain_conflict_score"))
    return {
        "crowding_state": (
            "crowded"
            if narrative_crowding is not None and narrative_crowding >= 60
            else "uncrowded"
            if narrative_crowding is not None and narrative_crowding <= 40
            else "balanced"
        ),
        "macro_alignment_state": (
            "supportive"
            if macro_alignment is not None and macro_alignment >= 60
            else "conflicted"
            if macro_alignment is not None and macro_alignment <= 40
            else "mixed"
        ),
        "fragility_state": (
            "fragile"
            if fragility is not None and fragility >= 60
            else "clean"
            if fragility is not None and fragility <= 40
            else "mixed"
        ),
        "agreement_state": (
            "aligned"
            if agreement_score is not None and conflict_score is not None and agreement_score - conflict_score >= 20
            else "conflicted"
            if agreement_score is not None and conflict_score is not None and conflict_score - agreement_score >= 10
            else "mixed"
        ),
        "freshness_quality": (report.get("freshness_summary") or {}).get("overall_status") or "unknown",
        "fundamental_coverage_level": (
            (domain_availability.get("fundamentals") or {}).get("coverage_status")
            or "unknown"
        ),
        "event_risk_state": event_domain.get("event_risk_classification") or "unknown",
        "liquidity_state": liquidity_domain.get("tradability_state") or "unknown",
        "breadth_state": breadth_domain.get("breadth_state") or "unknown",
        "cross_asset_state": (
            "conflicted"
            if safe_float(cross_asset_domain.get("cross_asset_conflict_score")) is not None
            and safe_float(cross_asset_domain.get("cross_asset_conflict_score")) >= 60
            else "supportive"
            if safe_float(cross_asset_domain.get("cross_asset_conflict_score")) is not None
            and safe_float(cross_asset_domain.get("cross_asset_conflict_score")) <= 35
            else "mixed"
        ),
        "stress_state": (
            "unstable"
            if safe_float(stress_domain.get("market_stress_score")) is not None
            and safe_float(stress_domain.get("market_stress_score")) >= 60
            else "stable"
        ),
    }


def build_prediction_record(
    report: Dict[str, Any],
    *,
    report_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    strategy = report.get("strategy") or {}
    regime = report.get("regime_intelligence") or {}
    proprietary_scores = {
        name: score_value(payload)
        for name, payload in (report.get("proprietary_scores") or {}).items()
    }
    canonical_alpha_core = report.get("canonical_alpha_core") or {}
    canonical_feature_vector = canonical_alpha_core.get("feature_vector") or {}
    canonical_signal_payload = canonical_alpha_core.get("signal_payload") or {}
    component_scores = {
        name: {
            "score": safe_float((payload or {}).get("score")),
            "normalized_score": safe_float((payload or {}).get("normalized_score")),
            "weight": safe_float((payload or {}).get("weight")),
        }
        for name, payload in (strategy.get("component_scores") or {}).items()
    }
    return {
        "prediction_version": EVALUATION_VERSION,
        "generated_at": report.get("generated_at"),
        "report_id": report_id,
        "session_id": session_id or report.get("session_id"),
        "symbol": report.get("symbol"),
        "as_of_date": report.get("as_of_date"),
        "horizon": report.get("horizon"),
        "horizon_days": (report.get("signal") or {}).get("horizon_days"),
        "risk_mode": report.get("risk_mode"),
        "scenario": report.get("scenario"),
        "analysis_depth": report.get("analysis_depth"),
        "signal_action": (report.get("signal") or {}).get("action"),
        "final_signal": strategy.get("final_signal")
        or (report.get("signal") or {}).get("final_action")
        or (report.get("signal") or {}).get("action"),
        "raw_score": safe_float((report.get("signal") or {}).get("score")),
        "raw_confidence": safe_float((report.get("signal") or {}).get("confidence")),
        "confidence": safe_float(strategy.get("confidence") or (report.get("signal") or {}).get("strategy_confidence")),
        "confidence_score": safe_float(strategy.get("confidence_score") or report.get("confidence_score")),
        "conviction_tier": strategy.get("conviction_tier"),
        "confidence_quality": strategy.get("confidence_quality"),
        "strategy_posture": strategy.get("strategy_posture") or report.get("strategy_posture"),
        "actionability_score": safe_float(strategy.get("actionability_score") or report.get("actionability_score")),
        "participant_fit": strategy.get("participant_fit") or report.get("participant_fit") or [],
        "participant_fit_primary": _primary_participant_fit(report),
        "fragility_tier": strategy.get("fragility_tier"),
        "regime_label": regime.get("regime_label")
        or (report.get("key_features") or {}).get("regime_label"),
        "regime_stability_score": proprietary_scores.get("Regime Stability Score"),
        "domain_agreement_score": safe_float((report.get("domain_agreement") or {}).get("domain_agreement_score")),
        "domain_conflict_score": safe_float((report.get("domain_agreement") or {}).get("domain_conflict_score")),
        "freshness_status": (report.get("freshness_summary") or {}).get("overall_status"),
        "quality_score": safe_float((report.get("quality") or {}).get("quality_score")),
        "missingness": safe_float((report.get("quality") or {}).get("missingness")),
        "invalidator_count": len((report.get("invalidators") or {}).get("top_invalidators") or []),
        "confirmation_trigger_count": len(report.get("confirmation_triggers") or []),
        "deterioration_trigger_count": len(report.get("deterioration_triggers") or []),
        "fragility_veto_count": len(report.get("fragility_vetoes") or []),
        "deployment_mode": report.get("deployment_mode"),
        "deployment_permission": report.get("deployment_permission"),
        "trust_tier": report.get("trust_tier"),
        "live_readiness_score": safe_float(report.get("live_readiness_score")),
        "rollout_stage": report.get("rollout_stage"),
        "candidate_classification": report.get("candidate_classification"),
        "portfolio_candidate_score": safe_float(report.get("portfolio_candidate_score")),
        "ranked_opportunity_score": safe_float(report.get("ranked_opportunity_score")),
        "portfolio_fit_quality": safe_float(report.get("portfolio_fit_quality")),
        "watchlist_priority_score": safe_float(report.get("watchlist_priority_score")),
        "execution_quality_score": safe_float(report.get("execution_quality_score")),
        "friction_penalty": safe_float(report.get("friction_penalty")),
        "turnover_penalty": safe_float(report.get("turnover_penalty")),
        "size_band": report.get("size_band"),
        "weight_band": report.get("weight_band"),
        "risk_budget_band": report.get("risk_budget_band"),
        "suppression_flags": list(report.get("suppression_flags") or []),
        "adjusted_confidence_notes": list(report.get("adjusted_confidence_notes") or []),
        "report_version": report.get("report_version"),
        "strategy_version": report.get("strategy_version"),
        "proprietary_scores": proprietary_scores,
        "strategy_component_scores": component_scores,
        "feature_vector": canonical_feature_vector,
        "signal_payload": canonical_signal_payload,
        "slices": _derived_slices(report),
    }


def _filter_predictions(
    records: Sequence[Dict[str, Any]],
    *,
    symbol: Optional[str],
    horizon: Optional[str],
    risk_mode: Optional[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in records:
        if symbol and str(record.get("symbol")) != str(symbol):
            continue
        if horizon and str(record.get("horizon")) != str(horizon):
            continue
        if risk_mode and str(record.get("risk_mode")) != str(risk_mode):
            continue
        output.append(record)
    return output


def _link_predictions(
    records: Sequence[Dict[str, Any]],
    *,
    bar_fetcher: Any = None,
) -> List[Dict[str, Any]]:
    evaluation_date = dt.datetime.now(dt.timezone.utc).date()
    linked: List[Dict[str, Any]] = []
    for record in records:
        outcome = link_realized_outcome(
            record,
            bar_fetcher=bar_fetcher,
            evaluation_as_of_date=evaluation_date,
        )
        linked.append({**record, "outcome": outcome})
    return linked


def _prediction_source_metadata(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dates = sorted({record.get("as_of_date") for record in records if record.get("as_of_date")})
    return {
        "total_predictions": len(records),
        "unique_symbols": len({record.get("symbol") for record in records if record.get("symbol")}),
        "unique_as_of_dates": len(dates),
        "earliest_as_of_date": dates[0] if dates else None,
        "latest_as_of_date": dates[-1] if dates else None,
        "report_versions": sorted({str(record.get("report_version")) for record in records if record.get("report_version")}),
        "strategy_versions": sorted({str(record.get("strategy_version")) for record in records if record.get("strategy_version")}),
    }


def _linked_status_counts(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        status = str((record.get("outcome") or {}).get("outcome_status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _ranking_specs() -> List[Any]:
    return [
        ("Opportunity Quality Score", lambda record: safe_float((record.get("proprietary_scores") or {}).get("Opportunity Quality Score")), True),
        ("Cross-Domain Conviction Score", lambda record: safe_float((record.get("proprietary_scores") or {}).get("Cross-Domain Conviction Score")), True),
        ("Signal Fragility Index", lambda record: safe_float((record.get("proprietary_scores") or {}).get("Signal Fragility Index")), False),
        ("confidence_score", lambda record: safe_float(record.get("confidence_score")), True),
        ("actionability_score", lambda record: safe_float(record.get("actionability_score")), True),
    ]


def _evaluation_text(
    *,
    cohort_name: str,
    source_meta: Dict[str, Any],
    signal_scorecard: Dict[str, Any],
    strategy_scorecard: Dict[str, Any],
    calibration_summary: Dict[str, Any],
    regime_breakdown: Dict[str, Any],
) -> Dict[str, str]:
    def _condition_labels(items: Sequence[Dict[str, Any]]) -> List[str]:
        labels: List[str] = []
        for item in items:
            dimension = item.get("dimension")
            label = item.get("label")
            if dimension and label:
                labels.append(f"{dimension}={label}")
        return labels

    matured = signal_scorecard.get("final_signal_overall", {}).get("matured_count") or 0
    total = source_meta.get("total_predictions") or 0
    if matured < 4:
        summary = (
            f"The Phase 6 evaluation layer is live for the {cohort_name} cohort, but only {matured} of {total} stored predictions have matured. "
            "Use the current scorecards as provisional research context rather than hard proof."
        )
    else:
        summary = (
            f"Across {matured} matured predictions in the {cohort_name} cohort, final strategy hit rate is "
            f"{signal_scorecard.get('final_signal_overall', {}).get('hit_rate')} with average forward return "
            f"{signal_scorecard.get('final_signal_overall', {}).get('average_forward_return')}. "
            f"Actionable setups versus watchlist / wait setups are separated by "
            f"{strategy_scorecard.get('actionable_vs_watchlist_return_spread')} average forward-return spread."
        )

    confidence_reliability = calibration_summary.get("confidence_reliability_score")
    confidence_text = (
        f"Confidence reliability currently scores {confidence_reliability} / 100, with monotonicity reading "
        f"{calibration_summary.get('confidence_monotonicity')}. "
        f"Drift notes: {', '.join(calibration_summary.get('calibration_drift_notes') or ['none'])}."
        if confidence_reliability is not None
        else "Confidence calibration is still sample-limited, so conviction remains researchable but provisional."
    )

    strongest = regime_breakdown.get("strongest_conditions") or []
    weakest = regime_breakdown.get("weakest_conditions") or []
    regime_text = (
        f"Historically strongest conditions are {compact_list(_condition_labels(strongest))}; "
        f"weakest conditions are {compact_list(_condition_labels(weakest))}."
        if strongest or weakest
        else "Regime-segmented evaluation is live, but not enough matured observations exist yet to isolate strong or weak setup conditions."
    )
    return {
        "evaluation_summary": summary,
        "confidence_reliability_summary": confidence_text,
        "regime_usefulness_summary": regime_text,
    }


def build_evaluation_artifact(
    *,
    current_report: Optional[Dict[str, Any]] = None,
    store: Optional[AssistantStorage] = None,
    prediction_records: Optional[Sequence[Dict[str, Any]]] = None,
    bar_fetcher: Any = None,
    cohort_symbol: Optional[str] = None,
    cohort_horizon: Optional[str] = None,
    cohort_risk_mode: Optional[str] = None,
    min_sample_size: int = 4,
) -> Dict[str, Any]:
    if current_report is not None:
        cohort_symbol = cohort_symbol or current_report.get("symbol")
        cohort_horizon = cohort_horizon or current_report.get("horizon")
        cohort_risk_mode = cohort_risk_mode or current_report.get("risk_mode")

    records = list(prediction_records or [])
    if not records and store is not None:
        artifacts = store.list_artifacts(kind=PREDICTION_RECORD_KIND, limit=750)
        records = [artifact.get("payload") or {} for artifact in artifacts]

    primary_records = _filter_predictions(
        records,
        symbol=None,
        horizon=cohort_horizon,
        risk_mode=cohort_risk_mode,
    )
    symbol_slice = _filter_predictions(
        primary_records,
        symbol=cohort_symbol,
        horizon=cohort_horizon,
        risk_mode=cohort_risk_mode,
    )
    linked_primary = _link_predictions(primary_records, bar_fetcher=bar_fetcher)
    linked_symbol = _link_predictions(symbol_slice, bar_fetcher=bar_fetcher)

    source_meta = _prediction_source_metadata(primary_records)
    linked_status = _linked_status_counts(linked_primary)
    signal_scorecard = build_signal_scorecard(linked_primary)
    strategy_scorecard = build_strategy_scorecard(linked_primary)
    ranking_scorecard = build_ranking_scorecard(linked_primary, score_specs=_ranking_specs())
    calibration_summary = build_calibration_summary(linked_primary)
    regime_breakdown = build_regime_breakdown(linked_primary)
    factor_attribution = build_factor_attribution_summary(linked_primary)
    text_summary = _evaluation_text(
        cohort_name=f"{cohort_horizon or 'all-horizon'} / {cohort_risk_mode or 'all-risk'}",
        source_meta=source_meta,
        signal_scorecard=signal_scorecard,
        strategy_scorecard=strategy_scorecard,
        calibration_summary=calibration_summary,
        regime_breakdown=regime_breakdown,
    )
    matured_count = signal_scorecard.get("final_signal_overall", {}).get("matured_count") or 0
    status = "available" if matured_count >= min_sample_size else "limited"

    return {
        "evaluation_kind": EVALUATION_ARTIFACT_KIND,
        "evaluation_version": EVALUATION_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "cohort_definition": {
            "symbol_slice": cohort_symbol,
            "horizon": cohort_horizon,
            "risk_mode": cohort_risk_mode,
            "minimum_matured_sample": min_sample_size,
            "primary_scope": "same_horizon_and_risk_mode",
        },
        "evaluation_run": {
            "model_identifier": sorted({record.get("report_version") for record in primary_records if record.get("report_version")}),
            "strategy_identifier": sorted({record.get("strategy_version") for record in primary_records if record.get("strategy_version")}),
            "data_window_identifier": {
                "earliest_prediction_date": source_meta.get("earliest_as_of_date"),
                "latest_prediction_date": source_meta.get("latest_as_of_date"),
            },
        },
        "prediction_linkage_summary": {
            **source_meta,
            "linked_outcome_status": linked_status,
        },
        "evaluation_cohorts": {
            "primary": {
                **source_meta,
                "linked_status": linked_status,
            },
            "symbol_slice": {
                **_prediction_source_metadata(symbol_slice),
                "linked_status": _linked_status_counts(linked_symbol),
            },
        },
        "signal_scorecard": signal_scorecard,
        "strategy_scorecard": strategy_scorecard,
        "ranking_scorecard": ranking_scorecard,
        "calibration_summary": calibration_summary,
        "regime_breakdown": regime_breakdown.get("regime_breakdown"),
        "bucket_results": ranking_scorecard.get("bucket_results"),
        "factor_attribution_summary": factor_attribution,
        "failure_modes": regime_breakdown.get("weakest_conditions"),
        "strongest_conditions": regime_breakdown.get("strongest_conditions"),
        "weakest_conditions": regime_breakdown.get("weakest_conditions"),
        **text_summary,
    }
