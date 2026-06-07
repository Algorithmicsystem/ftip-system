"""Signal serializer: strips internal/proprietary fields before HTTP responses."""
from __future__ import annotations

from typing import Any, Dict, Set

_INTERNAL_FIELDS: Set[str] = {
    "thresholds",
    "stacked_meta",
    "calibration_meta",
    "calibration_loaded",
    "component_support",
    "reason_details",
    "event_penalties",
    "liquidity_penalties",
    "breadth_penalties",
    "cross_asset_penalties",
    "stress_penalties",
    "environment_penalties",
    "adjusted_confidence_notes",
    "base_score",
    "stacked_score",
    "effective_lookback",
    "lookback",
    "signal_hash",
}

_INTERNAL_META_FIELDS: Set[str] = {
    "depth_adjustments",
    "snapshot_id",
    "snapshot_version",
    "feature_hash",
    "signal_schema_version",
    "signal_version",
}


def safe_signal_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of payload with all internal fields removed."""
    out = {k: v for k, v in payload.items() if k not in _INTERNAL_FIELDS}
    if "meta" in out and isinstance(out["meta"], dict):
        out["meta"] = {k: v for k, v in out["meta"].items() if k not in _INTERNAL_META_FIELDS}
    if "features" in out:
        del out["features"]
    return out


def validate_response_safety(payload: Dict[str, Any]) -> bool:
    """Return True if no internal fields are present in the payload."""
    for key in _INTERNAL_FIELDS:
        if key in payload:
            return False
    return True
