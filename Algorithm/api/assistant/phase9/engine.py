from __future__ import annotations

from typing import Any, Dict, List, Tuple

from api.assistant import reports
from api.assistant.storage import AssistantStorage

from .common import (
    PORTFOLIO_CONSTRUCTION_ARTIFACT_KIND,
    PORTFOLIO_CONSTRUCTION_VERSION,
    build_candidate_snapshot,
    compact_list,
    now_utc,
)
from .execution import build_execution_quality
from .exposure import build_exposure_framework
from .overlap import summarize_overlap
from .ranking import build_candidate_ranking
from .sizing import build_size_band_support
from .workflow import build_workflow_state, classify_candidate


def _load_cohort_reports(
    *,
    current_report: Dict[str, Any],
    current_report_id: str | None,
    session_id: str | None,
    store: AssistantStorage,
    limit: int = 12,
) -> List[Dict[str, Any]]:
    artifacts = store.list_artifacts(kind=reports.ANALYSIS_REPORT_KIND, limit=100)
    same_session: Dict[str, Dict[str, Any]] = {}
    same_regime: Dict[str, Dict[str, Any]] = {}
    for artifact in artifacts:
        payload = artifact.get("payload") or {}
        symbol = str(payload.get("symbol") or "")
        if not symbol or symbol == str(current_report.get("symbol") or ""):
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
            same_regime.setdefault(symbol, report_payload)

    peers = list(same_session.values())
    for item in same_regime.values():
        if len(peers) >= limit - 1:
            break
        if item.get("symbol") not in {peer.get("symbol") for peer in peers}:
            peers.append(item)
    return peers[: max(0, limit - 1)]


def _rank_cohort(
    current_report: Dict[str, Any],
    *,
    current_report_id: str | None,
    session_id: str | None,
    store: AssistantStorage,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cohort_reports = [
        {
            **current_report,
            "report_id": current_report_id,
            "session_id": session_id,
        },
        *_load_cohort_reports(
            current_report=current_report,
            current_report_id=current_report_id,
            session_id=session_id,
            store=store,
        ),
    ]

    cohort_entries: List[Dict[str, Any]] = []
    for report in cohort_reports:
        snapshot = build_candidate_snapshot(
            report,
            report_id=report.get("report_id"),
            session_id=report.get("session_id"),
        )
        execution = build_execution_quality(snapshot)
        ranking = build_candidate_ranking(snapshot, execution)
        cohort_entries.append(
            {
                "report": report,
                "snapshot": snapshot,
                "execution": execution,
                "ranking": ranking,
            }
        )

    snapshots_only = [entry["snapshot"] for entry in cohort_entries]
    enriched_rows: List[Dict[str, Any]] = []
    for entry in cohort_entries:
        snapshot = entry["snapshot"]
        overlap_summary, peer_overlaps = summarize_overlap(
            snapshot,
            [peer["snapshot"] for peer in cohort_entries if peer["snapshot"]["symbol"] != snapshot["symbol"]],
        )
        exposure = build_exposure_framework(snapshot, snapshots_only)
        size_support = build_size_band_support(
            snapshot,
            entry["ranking"],
            entry["execution"],
            overlap_summary,
            exposure,
        )
        classification = classify_candidate(
            snapshot,
            entry["ranking"],
            overlap_summary,
            size_support,
        )
        portfolio_fit_quality = round(
            max(
                0.0,
                min(
                    100.0,
                    (entry["ranking"]["portfolio_candidate_score"] * 0.54)
                    + (overlap_summary["diversification_contribution_score"] * 0.20)
                    + (entry["execution"]["execution_quality_score"] * 0.14)
                    + (entry["ranking"]["deployability_rank"] * 0.12)
                    - max(0.0, overlap_summary["redundancy_score"] - 70.0) * 0.22,
                ),
            ),
            2,
        )
        enriched_rows.append(
            {
                **snapshot,
                **entry["ranking"],
                **entry["execution"],
                **overlap_summary,
                **exposure,
                **size_support,
                **classification,
                "portfolio_fit_quality": portfolio_fit_quality,
                "peer_overlaps": peer_overlaps,
            }
        )

    enriched_rows.sort(
        key=lambda item: (
            float(item.get("portfolio_candidate_score") or 0.0)
            + float(item.get("portfolio_fit_quality") or 0.0) * 0.25
        ),
        reverse=True,
    )
    for index, item in enumerate(enriched_rows, start=1):
        item["portfolio_rank"] = index

    current = next(
        item for item in enriched_rows if item.get("symbol") == current_report.get("symbol")
    )
    workflow = build_workflow_state(current, current, current, current, enriched_rows)
    current.update(workflow)
    return current, enriched_rows


def _summary_text(current: Dict[str, Any], cohort: List[Dict[str, Any]]) -> Dict[str, str]:
    higher_ranked = [item for item in cohort if item.get("portfolio_rank", 99) < current.get("portfolio_rank", 99)]
    top_peers = ", ".join(item.get("symbol") or "n/a" for item in cohort[:3])
    portfolio_context_summary = (
        f"Portfolio rank is {current.get('portfolio_rank')} of {len(cohort)} tracked candidates, with ranked opportunity score {current.get('ranked_opportunity_score')} / 100, portfolio candidate score {current.get('portfolio_candidate_score')} / 100, and classification {current.get('candidate_classification')}. "
        f"Deployability is {current.get('deployment_permission')} and portfolio fit quality is {current.get('portfolio_fit_quality')} / 100."
    )
    portfolio_fit_analysis = (
        f"Overlap score is {current.get('overlap_score')} / 100, redundancy score is {current.get('redundancy_score')} / 100, and diversification contribution is {current.get('diversification_contribution_score')} / 100. "
        f"The most redundant tracked peer is {current.get('most_redundant_symbol') or 'none'}, while the leading cohort names are {top_peers or 'none'}."
    )
    execution_quality_analysis = (
        f"Execution quality is {current.get('execution_quality_score')} / 100 with friction penalty {current.get('friction_penalty')} and turnover penalty {current.get('turnover_penalty')}. "
        f"Preferred posture is {current.get('execution_preferred_posture')}, wait-for-better-entry is {current.get('wait_for_better_entry_flag')}, and size band is {current.get('size_band')}."
    )
    workflow_summary = (
        f"Prioritized watchlist currently includes {', '.join(current.get('prioritized_watchlist') or []) or 'none'}, active portfolio candidates are {', '.join(current.get('active_portfolio_candidates') or []) or 'none'}, and rotation pressure is {current.get('rotation_pressure_score')} / 100."
    )
    if higher_ranked:
        workflow_summary += (
            f" Higher-ranked alternatives currently include {', '.join(item.get('symbol') or 'n/a' for item in higher_ranked[:3])}."
        )
    return {
        "portfolio_context_summary": portfolio_context_summary,
        "portfolio_fit_analysis": portfolio_fit_analysis,
        "execution_quality_analysis": execution_quality_analysis,
        "portfolio_workflow_summary": workflow_summary,
    }


def build_portfolio_construction_artifact(
    *,
    current_report: Dict[str, Any],
    current_report_id: str | None,
    session_id: str | None,
    store: AssistantStorage,
) -> Dict[str, Any]:
    current, cohort = _rank_cohort(
        current_report,
        current_report_id=current_report_id,
        session_id=session_id,
        store=store,
    )
    summaries = _summary_text(current, cohort)
    return {
        "portfolio_construction_kind": PORTFOLIO_CONSTRUCTION_ARTIFACT_KIND,
        "portfolio_construction_version": PORTFOLIO_CONSTRUCTION_VERSION,
        "generated_at": now_utc(),
        "current_candidate": current,
        "cohort_ranking": [
            {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "peer_overlaps",
                    "prioritized_watchlist",
                    "active_portfolio_candidates",
                    "blocked_candidates",
                    "stale_review_needed",
                }
            }
            for item in cohort[:12]
        ],
        "top_peer_overlaps": current.get("peer_overlaps") or [],
        "workflow": {
            "candidate_watchlist": current.get("candidate_watchlist") or [],
            "prioritized_watchlist": current.get("prioritized_watchlist") or [],
            "active_portfolio_candidates": current.get("active_portfolio_candidates") or [],
            "blocked_candidates": current.get("blocked_candidates") or [],
            "stale_review_needed": current.get("stale_review_needed") or [],
            "priority_shift_flag": current.get("priority_shift_flag"),
            "rebalance_attention_flag": current.get("rebalance_attention_flag"),
            "candidate_upgrade_reason": current.get("candidate_upgrade_reason"),
            "candidate_downgrade_reason": current.get("candidate_downgrade_reason"),
            "replacement_candidate_notes": current.get("replacement_candidate_notes"),
            "rotation_pressure_score": current.get("rotation_pressure_score"),
        },
        **summaries,
    }
