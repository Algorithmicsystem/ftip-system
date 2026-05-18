from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Mapping, Optional

from api.axiom.common import rounded
from api.axiom.contracts import (
    AxiomArtifact,
    AxiomHistoricalOutcome,
    AxiomHistoryRecord,
    EngineScore,
)


AXIOM_PHASE3_VERSION = "axiom50_phase3_v1"
AXIOM_SCORE_HISTORY_ARTIFACT_KIND = "assistant_axiom_score_history_artifact"
AXIOM_REPLAY_ARTIFACT_KIND = "assistant_axiom_replay_artifact"
AXIOM_CALIBRATION_ARTIFACT_KIND = "assistant_axiom_calibration_artifact"
AXIOM_PORTFOLIO_GOVERNANCE_ARTIFACT_KIND = "assistant_axiom_portfolio_governance_artifact"

_FORWARD_HORIZONS: Mapping[str, int] = {
    "5d": 5,
    "21d": 21,
    "63d": 63,
}


def forward_horizons() -> Dict[str, int]:
    return dict(_FORWARD_HORIZONS)


def _signal_action(report: Dict[str, Any], axiom: AxiomArtifact) -> str:
    signal = report.get("signal") or {}
    return str(
        signal.get("final_action")
        or signal.get("action")
        or axiom.source_context.get("signal_action")
        or "HOLD"
    ).upper()


def _signal_score(report: Dict[str, Any], axiom: AxiomArtifact) -> Optional[float]:
    signal = report.get("signal") or {}
    return rounded(
        signal.get("score")
        if signal.get("score") is not None
        else axiom.source_context.get("signal_score"),
        digits=6,
    )


def _signal_confidence(report: Dict[str, Any], axiom: AxiomArtifact) -> Optional[float]:
    signal = report.get("signal") or {}
    confidence = signal.get("confidence")
    if confidence is None:
        confidence = axiom.source_context.get("signal_confidence")
    if confidence is None:
        return None
    confidence_value = float(confidence)
    if confidence_value <= 1.0:
        confidence_value *= 100.0
    return rounded(confidence_value, digits=2)


def _theme_tag(report: Dict[str, Any]) -> Optional[str]:
    source_governance = report.get("source_governance") or {}
    export = source_governance.get("source_inventory_export") or {}
    _ = export  # keep for future provenance expansion
    sentiment = ((report.get("data_bundle") or {}).get("sentiment_narrative_flow") or {})
    topic_clusters = sentiment.get("topic_clusters") or sentiment.get("top_narratives") or []
    if topic_clusters:
        first = topic_clusters[0]
        if isinstance(first, dict):
            return str(first.get("topic") or first.get("label") or "broad_theme")
        return str(first)
    return None


def _snapshot_versions(report: Dict[str, Any], axiom: AxiomArtifact) -> Dict[str, Optional[str]]:
    canonical_lineage = (((report.get("data_bundle") or {}).get("canonical_alpha_core") or {}).get("lineage") or {})
    return {
        "snapshot_id": str(
            canonical_lineage.get("snapshot_id")
            or axiom.source_context.get("snapshot_id")
            or ""
        )
        or None,
        "snapshot_version": str(
            canonical_lineage.get("snapshot_version")
            or axiom.source_context.get("snapshot_version")
            or ""
        )
        or None,
        "feature_version": str(
            canonical_lineage.get("feature_version")
            or axiom.source_context.get("feature_version")
            or ""
        )
        or None,
        "signal_version": str(
            canonical_lineage.get("signal_version")
            or axiom.source_context.get("signal_version")
            or ""
        )
        or None,
    }


def build_axiom_history_record(
    *,
    report: Dict[str, Any],
    axiom_artifact: Dict[str, Any],
    build_metadata: Optional[Dict[str, Any]] = None,
    forward_outcomes: Optional[Dict[str, Dict[str, Any]]] = None,
    evidence_backed_deployability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    axiom = AxiomArtifact.model_validate(axiom_artifact)
    versions = _snapshot_versions(report, axiom)
    symbol_meta = (report.get("data_bundle") or {}).get("symbol_meta") or {}
    relative = (report.get("data_bundle") or {}).get("relative_context") or {}
    market_domain = (report.get("data_bundle") or {}).get("market_price_volume") or {}
    deployment = axiom.deployability_decision
    explanation = axiom.explanation or {}
    normalized_outcomes: Dict[str, AxiomHistoricalOutcome] = {}
    for label, payload in (forward_outcomes or {}).items():
        horizon_days = _FORWARD_HORIZONS.get(label)
        normalized_outcomes[label] = AxiomHistoricalOutcome(
            horizon_label=label,
            horizon_days=int(horizon_days or payload.get("horizon_days") or 0),
            matured=bool(payload.get("matured")),
            outcome_status=str(payload.get("outcome_status") or "") or None,
            entry_date=payload.get("entry_date"),
            exit_date=payload.get("exit_date"),
            gross_edge_return=rounded(payload.get("gross_edge_return"), digits=6),
            net_edge_return=rounded(payload.get("net_edge_return"), digits=6),
            gross_trade_return=rounded(payload.get("gross_trade_return"), digits=6),
            net_trade_return=rounded(payload.get("net_trade_return"), digits=6),
            final_signal_correct=payload.get("final_signal_correct"),
            estimated_cost_bps=rounded(payload.get("estimated_cost_bps"), digits=4),
            mae=rounded(payload.get("mae"), digits=6),
            mfe=rounded(payload.get("mfe"), digits=6),
            invalidation_triggered=payload.get("invalidation_triggered"),
            signal_half_life_days=payload.get("signal_half_life_days"),
            continuation_decay_score=rounded(
                payload.get("continuation_decay_score"),
                digits=4,
            ),
        )

    record = AxiomHistoryRecord(
        history_version=AXIOM_PHASE3_VERSION,
        framework_version=axiom.framework_version,
        symbol=axiom.symbol,
        as_of_date=axiom.as_of,
        signal_action=_signal_action(report, axiom),
        signal_score=_signal_score(report, axiom),
        signal_confidence=_signal_confidence(report, axiom),
        strategy_posture=str(
            (report.get("strategy") or {}).get("strategy_posture")
            or report.get("strategy_posture")
            or ""
        )
        or None,
        deployment_permission=str(report.get("deployment_permission") or "") or None,
        trust_tier=str(report.get("trust_tier") or "") or None,
        snapshot_id=versions["snapshot_id"],
        snapshot_version=versions["snapshot_version"],
        feature_version=versions["feature_version"],
        signal_version=versions["signal_version"],
        regime_label=axiom.regime_label,
        trade_family=axiom.trade_family,
        deployability_tier=axiom.deployability_tier,
        size_band_recommendation=deployment.size_band_recommendation,
        gross_opportunity=axiom.gross_opportunity,
        friction_burden=axiom.friction_burden,
        validated_edge=axiom.validated_edge,
        deployable_alpha_utility=axiom.deployable_alpha_utility,
        overall_coverage=rounded(
            (axiom.coverage_summary or {}).get("overall_coverage"),
            digits=2,
        )
        or 0.0,
        overall_confidence=rounded(
            (axiom.coverage_summary or {}).get("overall_confidence"),
            digits=2,
        )
        or 0.0,
        sector=str(symbol_meta.get("sector") or relative.get("sector") or "") or None,
        benchmark_proxy=str(
            ((report.get("data_bundle") or {}).get("macro_cross_asset") or {}).get("benchmark_proxy")
            or relative.get("benchmark_proxy")
            or ""
        )
        or None,
        theme_tag=_theme_tag(report),
        engine_scores={
            name: EngineScore.model_validate(payload)
            for name, payload in (axiom.engine_scores or {}).items()
        },
        invalidation_flags=list(axiom.invalidation_flags or []),
        top_positive_drivers=list(explanation.get("top_positive_drivers") or []),
        top_negative_drivers=list(explanation.get("top_negative_drivers") or []),
        explanation_summary=str(explanation.get("summary") or "") or None,
        coverage_summary=dict(axiom.coverage_summary or {}),
        diagnostics=dict(axiom.diagnostics or {}),
        source_context={
            **dict(axiom.source_context or {}),
            "latest_close": market_domain.get("latest_close"),
            "benchmark_proxy": relative.get("benchmark_proxy"),
        },
        build_metadata={
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **(build_metadata or {}),
        },
        forward_outcomes=normalized_outcomes,
        evidence_backed_deployability=dict(evidence_backed_deployability or {}),
    )
    return record.model_dump(mode="python")
