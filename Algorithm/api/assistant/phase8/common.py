from __future__ import annotations

import datetime as dt
import math
from typing import Any, Dict, Iterable, List, Optional

from api import config


DEPLOYMENT_READINESS_ARTIFACT_KIND = "assistant_deployment_readiness_artifact"
DEPLOYMENT_AUDIT_RECORD_KIND = "assistant_deployment_audit_record"
DEPLOYMENT_READINESS_VERSION = "phase8_capital_readiness_v1"

DEPLOYMENT_MODES = (
    "research_only",
    "paper_shadow",
    "low_risk_live",
    "limited_live",
    "scaled_live",
    "paused",
)

MODE_POLICIES: Dict[str, Dict[str, Any]] = {
    "research_only": {
        "rollout_stage": "historical_validation",
        "paper_allowed": False,
        "live_allowed": False,
        "review_floor": "analyst_review",
        "max_permission": "analysis_only",
        "max_risk_mode_allowed": "research_only",
        "min_live_readiness_score": 101.0,
        "min_confidence_score": 101.0,
        "min_actionability_score": 101.0,
        "max_fragility_score": 0.0,
        "min_calibration_score": 101.0,
        "min_matured_predictions": 0,
        "description": "Analysis and experimentation are allowed, but no paper or live deployment posture is permitted.",
    },
    "paper_shadow": {
        "rollout_stage": "forward_shadow_validation",
        "paper_allowed": True,
        "live_allowed": False,
        "review_floor": "analyst_review",
        "max_permission": "paper_shadow_only",
        "max_risk_mode_allowed": "paper_shadow",
        "min_live_readiness_score": 101.0,
        "min_confidence_score": 101.0,
        "min_actionability_score": 101.0,
        "max_fragility_score": 0.0,
        "min_calibration_score": 101.0,
        "min_matured_predictions": 0,
        "description": "Forward paper / shadow tracking is allowed so the platform can compare expected versus realized behavior without capital use.",
    },
    "low_risk_live": {
        "rollout_stage": "low_risk_live_pilot",
        "paper_allowed": True,
        "live_allowed": True,
        "review_floor": "senior_analyst_and_risk_review",
        "max_permission": "low_risk_live_eligible",
        "max_risk_mode_allowed": "low_risk_live",
        "min_live_readiness_score": 74.0,
        "min_confidence_score": 58.0,
        "min_actionability_score": 56.0,
        "max_fragility_score": 46.0,
        "min_calibration_score": 58.0,
        "min_matured_predictions": 8,
        "description": "Only cleaner, strongly validated setups can be considered, and they remain subject to explicit human review and tight risk bands.",
    },
    "limited_live": {
        "rollout_stage": "limited_live_monitoring",
        "paper_allowed": True,
        "live_allowed": True,
        "review_floor": "portfolio_manager_and_risk_review",
        "max_permission": "limited_live_eligible",
        "max_risk_mode_allowed": "limited_live",
        "min_live_readiness_score": 79.0,
        "min_confidence_score": 61.0,
        "min_actionability_score": 60.0,
        "max_fragility_score": 42.0,
        "min_calibration_score": 63.0,
        "min_matured_predictions": 14,
        "description": "Controlled live eligibility is available, but only under active monitoring and with deployment discipline still clearly tighter than scaled use.",
    },
    "scaled_live": {
        "rollout_stage": "scaled_live_governed",
        "paper_allowed": True,
        "live_allowed": True,
        "review_floor": "investment_committee_review",
        "max_permission": "scaled_live_eligible",
        "max_risk_mode_allowed": "scaled_live",
        "min_live_readiness_score": 86.0,
        "min_confidence_score": 67.0,
        "min_actionability_score": 67.0,
        "max_fragility_score": 35.0,
        "min_calibration_score": 70.0,
        "min_matured_predictions": 24,
        "description": "Scaled live consideration is only available after the platform has proven stable under the tighter earlier stages and remains fully governed.",
    },
    "paused": {
        "rollout_stage": "deployment_paused",
        "paper_allowed": False,
        "live_allowed": False,
        "review_floor": "risk_committee_review",
        "max_permission": "blocked_paused",
        "max_risk_mode_allowed": "paused",
        "min_live_readiness_score": 101.0,
        "min_confidence_score": 101.0,
        "min_actionability_score": 101.0,
        "max_fragility_score": 0.0,
        "min_calibration_score": 101.0,
        "min_matured_predictions": 0,
        "description": "Live-use support is paused. The system can still analyze, but it should not be used to justify capital deployment.",
    },
}


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def compact_list(values: Iterable[Any], *, limit: int = 6) -> List[str]:
    items: List[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, dict):
            label = value.get("domain") or value.get("label") or value.get("name") or str(value)
            items.append(str(label))
        else:
            items.append(str(value))
        if len(items) >= limit:
            break
    return items


def score_value(payload: Any) -> Optional[float]:
    if isinstance(payload, dict):
        return safe_float(payload.get("score"))
    return safe_float(payload)


def bool_score(flag: Any, *, yes: float = 100.0, no: float = 0.0, default: float = 50.0) -> float:
    if flag is None:
        return default
    return yes if bool(flag) else no


def status_score(status: Any) -> float:
    normalized = str(status or "unknown").strip().lower()
    mapping = {
        "fresh": 92.0,
        "available": 82.0,
        "mixed": 72.0,
        "mixed_stale": 46.0,
        "stale_but_usable": 38.0,
        "stale": 24.0,
        "partial": 58.0,
        "unavailable": 18.0,
        "unknown": 42.0,
    }
    return mapping.get(normalized, 42.0)


def deployment_mode() -> str:
    raw = str(config.env("FTIP_DEPLOYMENT_MODE", "research_only") or "research_only").strip().lower()
    if raw not in MODE_POLICIES:
        return "research_only"
    return raw


def mode_policy(mode: Optional[str] = None) -> Dict[str, Any]:
    resolved = mode or deployment_mode()
    return {**MODE_POLICIES.get(resolved, MODE_POLICIES["research_only"]), "mode": resolved}


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()
