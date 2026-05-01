from __future__ import annotations

from typing import Any, Dict, List

from .common import overfit_risk, sample_confidence, slugify


def build_research_hypotheses(
    reweighting_candidates: List[Dict[str, Any]],
    drift_alerts: List[Dict[str, Any]],
    interaction_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hypotheses: List[Dict[str, Any]] = []

    for candidate in reweighting_candidates[:4]:
        target = str(candidate.get("target_family") or "factor")
        candidate_id = str(candidate.get("reweighting_candidate") or slugify(target))
        hypotheses.append(
            {
                "hypothesis_id": f"hyp_{candidate_id}",
                "hypothesis_title": f"Recalibrate {target.replace('_', ' ')} influence",
                "observed_pattern": candidate.get("rationale"),
                "candidate_improvement": (
                    candidate.get("suggested_weight_changes") or []
                ),
                "supporting_evidence": [
                    candidate.get("expected_impact_area"),
                    f"sample size {candidate.get('sample_size')}",
                ],
                "sample_size": candidate.get("sample_size") or 0,
                "confidence": candidate.get("confidence_in_recommendation") or 0.0,
                "next_validation_step": "run a controlled sandbox experiment and compare ranking monotonicity, calibration, and deployment gates before approval",
                "risk_of_overfit": candidate.get("risk_of_overfit") or "moderate",
            }
        )

    for alert in drift_alerts[:3]:
        drift_id = str(alert.get("drift_id") or slugify(alert.get("affected_component") or "drift"))
        hypotheses.append(
            {
                "hypothesis_id": f"hyp_{drift_id}",
                "hypothesis_title": f"Drift response for {str(alert.get('affected_component') or 'component').replace('_', ' ')}",
                "observed_pattern": " / ".join(alert.get("evidence") or []),
                "candidate_improvement": alert.get("adaptation_candidate"),
                "supporting_evidence": alert.get("evidence") or [],
                "sample_size": len(alert.get("evidence") or []),
                "confidence": 0.58 if alert.get("severity") == "moderate" else 0.72,
                "next_validation_step": "check whether the drift persists across the next learning cycle before escalating the change proposal",
                "risk_of_overfit": "moderate",
            }
        )

    for interaction in interaction_candidates[:3]:
        interaction_id = str(interaction.get("interaction_candidate") or "interaction")
        hypotheses.append(
            {
                "hypothesis_id": f"hyp_{slugify(interaction_id)}",
                "hypothesis_title": f"Validate interaction: {interaction_id.replace('_', ' ')}",
                "observed_pattern": interaction.get("description"),
                "candidate_improvement": "promote this interaction into a tracked experimental score overlay if follow-up validation remains stable",
                "supporting_evidence": [
                    f"conditional usefulness: {interaction.get('conditional_usefulness')}",
                    f"motif strength: {interaction.get('motif_strength')}",
                ],
                "sample_size": interaction.get("sample_size") or 0,
                "confidence": 0.44
                if interaction.get("validation_status") == "exploratory"
                else 0.61
                if interaction.get("validation_status") == "under_review"
                else 0.76,
                "next_validation_step": "track the interaction across more matured outcomes and compare it against the current archetype library before operational use",
                "risk_of_overfit": overfit_risk(int(interaction.get("sample_size") or 0)),
            }
        )

    hypotheses.sort(
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            item.get("sample_size") or 0,
        ),
        reverse=True,
    )
    for item in hypotheses:
        item["sample_confidence"] = sample_confidence(int(item.get("sample_size") or 0))
    return hypotheses[:8]
