from __future__ import annotations

from typing import Any, Dict

from .common import mode_policy


def build_deployment_mode_state() -> Dict[str, Any]:
    policy = mode_policy()
    active_mode = policy["mode"]
    return {
        "active_mode": active_mode,
        "mode_description": policy["description"],
        "rollout_stage": policy["rollout_stage"],
        "paper_allowed": policy["paper_allowed"],
        "live_allowed": policy["live_allowed"],
        "review_floor": policy["review_floor"],
        "max_permission": policy["max_permission"],
        "max_risk_mode_allowed": policy["max_risk_mode_allowed"],
        "policy_thresholds": {
            "min_live_readiness_score": policy["min_live_readiness_score"],
            "min_confidence_score": policy["min_confidence_score"],
            "min_actionability_score": policy["min_actionability_score"],
            "max_fragility_score": policy["max_fragility_score"],
            "min_calibration_score": policy["min_calibration_score"],
            "min_matured_predictions": policy["min_matured_predictions"],
        },
    }
