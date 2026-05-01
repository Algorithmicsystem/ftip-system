from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant import reports
from api.assistant.storage import AssistantStorage

from .archetypes import classify_archetype, summarize_archetype_library
from .common import (
    CONTINUOUS_LEARNING_ARTIFACT_KIND,
    CONTINUOUS_LEARNING_VERSION,
    compact_list,
    now_utc,
    build_learning_snapshot,
)
from .drift import build_drift_alerts
from .experiments import build_experiment_registry
from .hypotheses import build_research_hypotheses
from .interactions import build_feature_interaction_candidates
from .motifs import build_motif_library
from .regime import build_regime_conditioned_learnings
from .reweighting import build_reweighting_candidates


def _load_cohort_reports(
    *,
    current_report: Dict[str, Any],
    session_id: Optional[str],
    store: AssistantStorage,
    limit: int = 18,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=reports.ANALYSIS_REPORT_KIND, limit=120)
    same_session: Dict[str, Dict[str, Any]] = {}
    broader: Dict[str, Dict[str, Any]] = {}
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
        report_payload = {
            **payload,
            "report_id": artifact.get("id"),
            "session_id": artifact.get("session_id"),
        }
        if str(artifact.get("session_id")) == str(session_id):
            same_session.setdefault(symbol, report_payload)
        else:
            broader.setdefault(symbol, report_payload)

    peers = list(same_session.values())
    for item in broader.values():
        if len(peers) >= max(0, limit - 1):
            break
        if item.get("symbol") not in {peer.get("symbol") for peer in peers}:
            peers.append(item)
    return peers[: max(0, limit - 1)]


def _load_prior_learning_artifacts(
    *,
    store: AssistantStorage,
    horizon: Optional[str],
    risk_mode: Optional[str],
    limit: int = 30,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=CONTINUOUS_LEARNING_ARTIFACT_KIND, limit=limit)
    output: List[Dict[str, Any]] = []
    for artifact in artifacts:
        payload = artifact.get("payload") or {}
        cohort = payload.get("cohort_summary") or {}
        if horizon and cohort.get("horizon") and cohort.get("horizon") != horizon:
            continue
        if risk_mode and cohort.get("risk_mode") and cohort.get("risk_mode") != risk_mode:
            continue
        output.append(payload)
    return output


def _learning_summaries(
    *,
    active_archetype: Dict[str, Any],
    regime_learning: Dict[str, Any],
    reweighting_candidates: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    drift_alerts: List[Dict[str, Any]],
    experiment_registry: Dict[str, Any],
    motif_library: Dict[str, Any],
) -> Dict[str, str]:
    top_reweight = reweighting_candidates[0] if reweighting_candidates else {}
    top_hypothesis = hypotheses[0] if hypotheses else {}
    top_drift = drift_alerts[0] if drift_alerts else {}
    top_motif = (motif_library.get("active_motifs") or motif_library.get("motif_library") or [{}])[0]

    learning_summary = (
        f"Active setup archetype is {active_archetype.get('archetype_name') or 'unknown'}, with deployment caution {active_archetype.get('deployment_caution_level') or 'unknown'}. "
        f"The top current learning priority is {top_reweight.get('target_family') or top_hypothesis.get('hypothesis_title') or 'maintain observation mode'}, and the leading drift issue is {top_drift.get('affected_component') or 'none active'}."
    )
    regime_learning_summary = regime_learning.get("regime_learning_summary") or (
        "Regime-conditioned learning will attach once enough cohort context exists."
    )
    adaptation_queue_summary = (
        f"Top adaptation candidate is {top_reweight.get('target_family') or 'none'}, supported by sample size {top_reweight.get('sample_size') or 0}. "
        f"Leading hypothesis is {top_hypothesis.get('hypothesis_title') or 'none'}."
    )
    experiment_registry_summary = (
        experiment_registry.get("experiment_registry_summary")
        or "No experiment-registry summary is available yet."
    )
    archetype_motif_summary = (
        f"Active archetype is {active_archetype.get('archetype_name') or 'unknown'}, and the highest-signal motif is {top_motif.get('motif_summary') or 'none identified'}."
    )
    return {
        "learning_summary": learning_summary,
        "regime_learning_summary": regime_learning_summary,
        "adaptation_queue_summary": adaptation_queue_summary,
        "experiment_registry_summary": experiment_registry_summary,
        "archetype_motif_summary": archetype_motif_summary,
    }


def build_continuous_learning_artifact(
    *,
    current_report: Dict[str, Any],
    current_report_id: Optional[str],
    session_id: Optional[str],
    store: AssistantStorage,
) -> Dict[str, Any]:
    current_snapshot = build_learning_snapshot(
        current_report,
        report_id=current_report_id,
        session_id=session_id,
    )
    peer_reports = _load_cohort_reports(
        current_report=current_report,
        session_id=session_id,
        store=store,
    )
    cohort_snapshots: List[Dict[str, Any]] = []
    for report in [{**current_report, "report_id": current_report_id, "session_id": session_id}, *peer_reports]:
        snapshot = build_learning_snapshot(
            report,
            report_id=report.get("report_id"),
            session_id=report.get("session_id"),
        )
        snapshot["setup_archetype"] = classify_archetype(snapshot)
        cohort_snapshots.append(snapshot)

    current_snapshot = next(
        item for item in cohort_snapshots if item.get("symbol") == current_report.get("symbol")
    )
    active_archetype = current_snapshot.get("setup_archetype") or classify_archetype(current_snapshot)
    archetype_library = summarize_archetype_library(cohort_snapshots)
    regime_learning = build_regime_conditioned_learnings(
        current_snapshot,
        cohort_snapshots,
        current_report.get("evaluation") or {},
    )
    motif_library = build_motif_library(current_snapshot, cohort_snapshots)
    interactions = build_feature_interaction_candidates(cohort_snapshots)

    current_report = {
        **current_report,
        "setup_archetype": active_archetype,
        "signal_family_library": archetype_library,
    }
    drift_alerts = build_drift_alerts(current_report, current_snapshot, regime_learning)
    reweighting_candidates = build_reweighting_candidates(
        current_report,
        regime_learning,
        drift_alerts,
    )
    hypotheses = build_research_hypotheses(
        reweighting_candidates,
        drift_alerts,
        interactions,
    )
    prior_learning_artifacts = _load_prior_learning_artifacts(
        store=store,
        horizon=current_report.get("horizon"),
        risk_mode=current_report.get("risk_mode"),
    )
    experiment_registry = build_experiment_registry(
        current_report,
        reweighting_candidates,
        hypotheses,
        drift_alerts,
        prior_learning_artifacts,
    )
    summaries = _learning_summaries(
        active_archetype=active_archetype,
        regime_learning=regime_learning,
        reweighting_candidates=reweighting_candidates,
        hypotheses=hypotheses,
        drift_alerts=drift_alerts,
        experiment_registry=experiment_registry,
        motif_library=motif_library,
    )
    strongest_archetypes = compact_list(
        (item.get("archetype_name") for item in (archetype_library.get("archetype_cohorts") or [])[:3]),
        limit=3,
    )
    return {
        "continuous_learning_kind": CONTINUOUS_LEARNING_ARTIFACT_KIND,
        "continuous_learning_version": CONTINUOUS_LEARNING_VERSION,
        "generated_at": now_utc(),
        "cohort_summary": {
            "tracked_reports": len(cohort_snapshots),
            "peer_reports": max(0, len(cohort_snapshots) - 1),
            "unique_symbols": len({item.get("symbol") for item in cohort_snapshots if item.get("symbol")}),
            "horizon": current_report.get("horizon"),
            "risk_mode": current_report.get("risk_mode"),
            "prior_learning_cycles": len(prior_learning_artifacts),
        },
        "active_setup_archetype": active_archetype,
        "signal_family_library": {
            "active_archetype": active_archetype,
            "archetype_cohorts": archetype_library.get("archetype_cohorts") or [],
            "strongest_signal_families": strongest_archetypes,
        },
        "regime_conditioned_learnings": regime_learning.get("regime_conditioned_learnings") or [],
        "feature_interaction_candidates": interactions,
        "reweighting_candidates": reweighting_candidates,
        "research_hypotheses": hypotheses,
        "drift_alerts": drift_alerts,
        "experiment_registry": experiment_registry,
        "motif_discovery": motif_library,
        "improvement_queue": experiment_registry.get("improvement_queue") or [],
        **summaries,
    }
