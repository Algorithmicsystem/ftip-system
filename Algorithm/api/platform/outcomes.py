from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from api.assistant.reports import sanitize_payload
from api.platform.contracts import (
    HorizonOutcome,
    OutcomeAttribution,
    OutcomeEvidenceStatus,
    OutcomeSnapshot,
    RecommendationOutcomeAssessment,
    TrackingStatus,
)
from api.platform.tracking import TRACKING_HORIZON_DAYS, build_tracking_status
from api.research.backtest.outcomes import (
    default_ohlc_bar_fetcher,
    evaluate_prediction_outcome,
)


OutcomeBarFetcher = Callable[[str, dt.date, int], List[Dict[str, Any]]]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _as_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _supported_horizons(paper_trade: Dict[str, Any]) -> List[str]:
    tracked = list(paper_trade.get("tracked_horizons") or [])
    output = [label for label in tracked if label in TRACKING_HORIZON_DAYS]
    return output or ["5d", "21d", "63d"]


def _validation_record(
    *,
    track: Dict[str, Any],
    paper_trade: Dict[str, Any],
    horizon_label: str,
) -> Dict[str, Any]:
    source_context = dict(track.get("start_source_context") or {})
    feature_vector = dict(source_context.get("feature_vector") or {})
    fragility_score = (
        ((track.get("start_engine_scores") or {}).get("critical_fragility") or {}).get("score")
    )
    return {
        "symbol": track.get("symbol"),
        "as_of_date": paper_trade.get("entry_reference_date"),
        "horizon_days": TRACKING_HORIZON_DAYS[horizon_label],
        "signal_action": track.get("signal_action_at_start") or "BUY",
        "final_signal": track.get("signal_action_at_start") or "BUY",
        "feature_vector": feature_vector,
        "proprietary_scores": {
            "Signal Fragility Index": fragility_score,
        },
    }


def _horizon_outcome(
    *,
    track: Dict[str, Any],
    paper_trade: Dict[str, Any],
    horizon_label: str,
    evaluation_as_of_date: Optional[dt.date],
    bar_fetcher: OutcomeBarFetcher,
) -> HorizonOutcome:
    record = _validation_record(
        track=track,
        paper_trade=paper_trade,
        horizon_label=horizon_label,
    )
    effective_fetcher = bar_fetcher
    if evaluation_as_of_date is not None:
        def effective_fetcher(symbol: str, as_of_date: dt.date, limit: int) -> List[Dict[str, Any]]:
            rows = bar_fetcher(symbol, as_of_date, limit)
            return [
                row
                for row in rows
                if (_as_date(row.get("as_of_date")) or as_of_date) <= evaluation_as_of_date
            ]
    outcome = evaluate_prediction_outcome(
        record,
        bar_fetcher=effective_fetcher,
        evaluation_as_of_date=evaluation_as_of_date,
    )
    status = str(outcome.get("outcome_status") or "pending")
    return HorizonOutcome(
        horizon_label=horizon_label,
        horizon_days=TRACKING_HORIZON_DAYS[horizon_label],
        status=status,
        matured=bool(outcome.get("matured")),
        entry_date=outcome.get("entry_date") or paper_trade.get("entry_reference_date"),
        exit_date=outcome.get("exit_date"),
        entry_price=outcome.get("entry_price") or paper_trade.get("entry_price"),
        exit_price=outcome.get("exit_price"),
        forward_return=outcome.get("forward_return"),
        gross_edge_return=outcome.get("gross_edge_return"),
        net_edge_return=outcome.get("net_edge_return"),
        gross_trade_return=outcome.get("gross_trade_return"),
        net_trade_return=outcome.get("net_trade_return"),
        final_signal_correct=outcome.get("final_signal_correct"),
        raw_signal_correct=outcome.get("raw_signal_correct"),
        mae=outcome.get("mae"),
        mfe=outcome.get("mfe"),
        invalidation_triggered=outcome.get("invalidation_triggered"),
        benchmark_forward_return=outcome.get("forward_return"),
        excess_return_vs_hold=(
            None
            if outcome.get("net_edge_return") is None or outcome.get("forward_return") is None
            else float(outcome.get("net_edge_return")) - float(outcome.get("forward_return"))
        ),
        warnings=[] if status == "matured" else [status],
        metadata=sanitize_payload(
            {
                "evaluation_as_of_date": outcome.get("evaluation_as_of_date"),
                "threshold": outcome.get("threshold"),
                "signal_half_life_days": outcome.get("signal_half_life_days"),
                "continuation_decay_score": outcome.get("continuation_decay_score"),
                "friction_cost_summary": outcome.get("friction_cost_summary") or {},
            }
        ),
    )


def build_outcome_windows(
    *,
    track: Dict[str, Any],
    paper_trade: Dict[str, Any],
    evaluation_as_of_date: Optional[dt.date] = None,
    bar_fetcher: OutcomeBarFetcher = default_ohlc_bar_fetcher,
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for horizon_label in _supported_horizons(paper_trade):
        outcome = _horizon_outcome(
            track=track,
            paper_trade=paper_trade,
            horizon_label=horizon_label,
            evaluation_as_of_date=evaluation_as_of_date,
            bar_fetcher=bar_fetcher,
        )
        output[horizon_label] = outcome.model_dump(mode="python")
    return output


def _evidence_label(
    matured: List[Dict[str, Any]],
    supportive_horizons: List[str],
    contradicted_horizons: List[str],
) -> str:
    if not matured:
        return "insufficient_sample"
    if len(supportive_horizons) >= max(1, len(matured) - 1) and not contradicted_horizons:
        return "supportive"
    if contradicted_horizons and len(contradicted_horizons) >= len(matured) / 2:
        return "weak"
    return "mixed"


def build_outcome_evidence_status(
    *,
    windows: Mapping[str, Dict[str, Any]],
    supportive_horizons: Optional[List[str]] = None,
    contradicted_horizons: Optional[List[str]] = None,
) -> OutcomeEvidenceStatus:
    matured = [payload for payload in windows.values() if payload.get("matured") is True]
    supportive = list(supportive_horizons or [])
    contradicted = list(contradicted_horizons or [])
    label = _evidence_label(matured, supportive, contradicted)
    warnings: List[str] = []
    if not matured:
        warnings.append("tracked evidence has not matured yet")
    if label == "weak":
        warnings.append("recent paper-tracked evidence contradicts the original recommendation")
    elif label == "mixed":
        warnings.append("paper-tracked evidence is mixed across available horizons")
    return OutcomeEvidenceStatus(
        status=label,
        label=label,
        sample_size=len(matured),
        warnings=warnings,
        rationale=(
            "Evidence is still pending because no tracked horizon has matured yet."
            if label == "insufficient_sample"
            else "Most matured horizons were supportive of the original recommendation."
            if label == "supportive"
            else "Matured horizons are mixed and still require caution."
            if label == "mixed"
            else "A meaningful share of matured horizons contradicted the original recommendation."
        ),
    )


def build_recommendation_outcome_assessment(
    *,
    track: Dict[str, Any],
    windows: Mapping[str, Dict[str, Any]],
) -> RecommendationOutcomeAssessment:
    matured = [
        payload
        for payload in windows.values()
        if payload.get("matured") is True
    ]
    supportive_horizons: List[str] = []
    contradicted_horizons: List[str] = []
    strongest_horizon = None
    weakest_horizon = None
    strongest_value = None
    weakest_value = None
    for label, payload in windows.items():
        net_edge = payload.get("net_edge_return")
        if payload.get("matured") is not True or net_edge is None:
            continue
        if payload.get("final_signal_correct") is True and float(net_edge) > 0:
            supportive_horizons.append(label)
        if float(net_edge) < 0 or payload.get("invalidation_triggered") is True:
            contradicted_horizons.append(label)
        if strongest_value is None or float(net_edge) > strongest_value:
            strongest_value = float(net_edge)
            strongest_horizon = label
        if weakest_value is None or float(net_edge) < weakest_value:
            weakest_value = float(net_edge)
            weakest_horizon = label

    evidence = build_outcome_evidence_status(
        windows=windows,
        supportive_horizons=supportive_horizons,
        contradicted_horizons=contradicted_horizons,
    )
    aligned = None
    if matured:
        aligned = len(supportive_horizons) >= len(contradicted_horizons)
    tier = str(track.get("deployability_tier_at_start") or "monitor_only")
    avg_net_edge = _mean(payload.get("net_edge_return") for payload in matured)
    deployability_justified = None
    if avg_net_edge is not None:
        if tier == "live_candidate":
            deployability_justified = avg_net_edge >= 0.01 and len(contradicted_horizons) == 0
        elif tier == "paper_trade_only":
            deployability_justified = avg_net_edge >= -0.005
        elif tier == "monitor_only":
            deployability_justified = avg_net_edge >= -0.02
        else:
            deployability_justified = avg_net_edge <= 0.0 or len(contradicted_horizons) > 0

    summary = (
        "Tracked horizons have not matured yet."
        if not matured
        else f"{len(supportive_horizons)} supportive horizon(s) versus {len(contradicted_horizons)} contradictory horizon(s)."
    )
    return RecommendationOutcomeAssessment(
        assessment_status=evidence.status,
        hindsight_label=(
            "aligned"
            if aligned is True
            else "contradicted"
            if aligned is False
            else "pending"
        ),
        aligned_in_hindsight=aligned,
        deployability_justified=deployability_justified,
        supportive_horizons=supportive_horizons,
        contradicted_horizons=contradicted_horizons,
        strongest_horizon=strongest_horizon,
        weakest_horizon=weakest_horizon,
        supportive_window_count=len(supportive_horizons),
        contradictory_window_count=len(contradicted_horizons),
        summary=summary,
        rationale=(
            evidence.rationale
            or "Paper-tracked evidence remains incomplete."
        ),
        evidence_status=evidence.status,
    )


def build_outcome_snapshot(
    *,
    track: Dict[str, Any],
    paper_trade: Dict[str, Any],
    evaluation_as_of_date: Optional[dt.date] = None,
    bar_fetcher: OutcomeBarFetcher = default_ohlc_bar_fetcher,
    snapshot_id: Optional[str] = None,
) -> OutcomeSnapshot:
    windows = build_outcome_windows(
        track=track,
        paper_trade=paper_trade,
        evaluation_as_of_date=evaluation_as_of_date,
        bar_fetcher=bar_fetcher,
    )
    assessment = build_recommendation_outcome_assessment(
        track=track,
        windows=windows,
    )
    evidence = build_outcome_evidence_status(
        windows=windows,
        supportive_horizons=assessment.supportive_horizons,
        contradicted_horizons=assessment.contradicted_horizons,
    )
    matured_horizons = [label for label, payload in windows.items() if payload.get("matured") is True]
    pending_horizons = [label for label, payload in windows.items() if payload.get("matured") is not True]
    tracking_status = build_tracking_status(
        status="complete" if pending_horizons == [] else "active",
        evidence_status=evidence.status,
        matured_horizons=matured_horizons,
        pending_horizons=pending_horizons,
        warnings=evidence.warnings,
        metadata={"paper_trade_id": paper_trade.get("paper_trade_id")},
        last_refreshed_at=now_utc(),
    )
    avg_net_edge = _mean(payload.get("net_edge_return") for payload in windows.values() if payload.get("matured") is True)
    avg_forward = _mean(payload.get("forward_return") for payload in windows.values() if payload.get("matured") is True)
    benchmark_comparison = sanitize_payload(
        {
            "average_net_edge_return": avg_net_edge,
            "average_forward_return": avg_forward,
            "average_excess_vs_hold": None if avg_net_edge is None or avg_forward is None else avg_net_edge - avg_forward,
        }
    )
    created_at = now_utc()
    return OutcomeSnapshot(
        snapshot_id=snapshot_id or str(uuid.uuid4()),
        organization_id=track.get("organization_id"),
        workspace_id=track.get("workspace_id"),
        workflow_id=str(track.get("workflow_id") or ""),
        dossier_id=str(track.get("dossier_id") or ""),
        track_id=str(track.get("track_id") or ""),
        paper_trade_id=str(paper_trade.get("paper_trade_id") or ""),
        symbol=str(track.get("symbol") or "").upper(),
        snapshot_date=(evaluation_as_of_date or dt.date.today()).isoformat(),
        evidence_mode="paper_tracked",
        tracking_status=tracking_status,
        windows={key: HorizonOutcome.model_validate(value) for key, value in windows.items()},
        assessment=assessment,
        evidence_status=evidence,
        benchmark_comparison=benchmark_comparison,
        metadata=sanitize_payload(
            {
                "entry_reference_date": paper_trade.get("entry_reference_date"),
                "tracked_horizons": paper_trade.get("tracked_horizons") or [],
                "recommendation_state_at_start": track.get("recommendation_state_at_start"),
                "deployability_tier_at_start": track.get("deployability_tier_at_start"),
            }
        ),
        created_at=created_at,
        updated_at=created_at,
    )


def build_outcome_attribution(
    *,
    track: Dict[str, Any],
    paper_trade: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> OutcomeAttribution:
    parsed = OutcomeSnapshot.model_validate(snapshot)
    summary = (
        f"{parsed.assessment.summary} Evidence status is {parsed.evidence_status.status} "
        f"for recommendation state {track.get('recommendation_state_at_start') or 'draft'}."
    )
    return OutcomeAttribution(
        attribution_id=str(uuid.uuid4()),
        organization_id=track.get("organization_id"),
        workspace_id=track.get("workspace_id"),
        workflow_id=str(track.get("workflow_id") or ""),
        dossier_id=str(track.get("dossier_id") or ""),
        track_id=str(track.get("track_id") or ""),
        paper_trade_id=str(paper_trade.get("paper_trade_id") or ""),
        symbol=str(track.get("symbol") or "").upper(),
        recommendation_state_at_start=track.get("recommendation_state_at_start"),
        deployability_tier_at_start=track.get("deployability_tier_at_start"),
        size_band_at_start=track.get("size_band_at_start"),
        regime_label=track.get("regime_label"),
        trade_family=track.get("trade_family"),
        strongest_engine_at_start=track.get("strongest_engine_at_start"),
        weakest_engine_at_start=track.get("weakest_engine_at_start"),
        source_label="paper_tracked",
        snapshot=parsed,
        benchmark_comparison=sanitize_payload(parsed.benchmark_comparison),
        summary=summary,
        created_at=now_utc(),
    )
