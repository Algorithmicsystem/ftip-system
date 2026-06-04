"""Phase 16.3: Counterfactual Analysis Engine."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from api.assistant.phase3.common import clamp


# ---------------------------------------------------------------------------
# Component DAU impact model (approximate linear weights)
# ---------------------------------------------------------------------------

# (component_path, effective_weight_on_dau, direction: +1 = higher is more BUY)
_COMPONENT_MODEL: List[Tuple[str, List[str], float, int]] = [
    # (name, path_in_engine_scores, dau_weight_per_point, direction)
    ("eis_component",      ["fundamental_reality", "components", "eis_component"],      0.066,  1),
    ("caps_component",     ["fundamental_reality", "components", "caps_component"],     0.077,  1),
    ("flow_score",         ["flow_transmission",   "score"],                            0.160,  1),
    ("behavioral_score",   ["behavioral_distortion", "score"],                          0.120,  1),
    ("fragility_score",    ["critical_fragility",  "score"],                            0.120, -1),
    ("scps_component",     ["critical_fragility",  "components", "scps_component"],     0.050, -1),
    ("sentiment_score",    ["behavioral_distortion", "components", "asymmetric_sent_score"], 0.060, 1),
]

# Signal boundaries
_BUY_THRESHOLD = 65.0
_SELL_THRESHOLD = 40.0


def _get_nested(d: dict, path: List[str]) -> Optional[float]:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return float(cur) if cur is not None else None


def _signal_label_from_dau(dau: float) -> str:
    if dau >= _BUY_THRESHOLD:
        return "BUY"
    if dau <= _SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def _target_dau(signal_label: str, target_signal: Optional[str]) -> float:
    effective_target = target_signal or (
        "SELL" if signal_label == "BUY" else
        "BUY" if signal_label == "SELL" else
        "BUY"
    )
    if effective_target == "BUY":
        return _BUY_THRESHOLD
    if effective_target == "SELL":
        return _SELL_THRESHOLD
    return 52.5  # middle of HOLD range


def _probability_label(delta: float) -> str:
    if delta <= 10:
        return "high"
    if delta <= 25:
        return "medium"
    return "low"


def _timeline_label(delta: float) -> str:
    if delta <= 10:
        return "weeks"
    if delta <= 20:
        return "quarters"
    return "not_near_term"


def compute_counterfactuals(
    axiom_payload: Dict[str, Any],
    signal_label: str,
    target_signal: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return list of counterfactuals showing what would need to change to flip the signal."""
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)
    engine_scores = axiom_payload.get("engine_scores") or {}
    target_dau = _target_dau(signal_label, target_signal)
    dau_delta_needed = target_dau - dau  # negative means need to decrease DAU

    candidates = []
    for (comp_name, path, weight, direction) in _COMPONENT_MODEL:
        current_val = _get_nested(engine_scores, path)
        if current_val is None:
            continue
        # Impact of +1 point on DAU = weight × direction
        dau_per_point = weight * direction
        if abs(dau_per_point) < 1e-9:
            continue
        # Points needed in this component to achieve dau_delta_needed
        component_delta = dau_delta_needed / dau_per_point
        counterfactual_val = round(clamp(current_val + component_delta, 0.0, 100.0), 1)
        delta_abs = round(abs(component_delta), 2)
        direction_str = "increase" if component_delta > 0 else "decrease"
        dau_impact = round(component_delta * dau_per_point, 2)

        plain_english = (
            f"{comp_name} would need to {direction_str} by {delta_abs:.1f} points "
            f"(from {current_val:.1f} to {counterfactual_val:.1f}) "
            f"to move signal toward {target_signal or ('SELL' if signal_label == 'BUY' else 'BUY')}"
        )

        candidates.append({
            "component": comp_name,
            "current_value": round(current_val, 2),
            "counterfactual_value": counterfactual_val,
            "delta_needed": delta_abs,
            "direction": direction_str,
            "dau_impact": round(abs(dau_impact), 2),
            "plain_english": plain_english,
            "probability": _probability_label(delta_abs),
            "timeline": _timeline_label(delta_abs),
        })

    # Prefer feasible counterfactuals (delta < 30); always return at least some
    feasible = [c for c in candidates if c["delta_needed"] <= 30]
    if not feasible:
        feasible = sorted(candidates, key=lambda x: x["delta_needed"])

    return sorted(feasible, key=lambda x: x["delta_needed"])[:5]


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def compute_signal_sensitivity(
    axiom_payload: Dict[str, Any],
    component_name: str,
    delta_range: Tuple[float, float] = (-30.0, 30.0),
) -> Dict[str, Any]:
    """Simulate how DAU changes as a component changes across a range."""
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)
    engine_scores = axiom_payload.get("engine_scores") or {}

    # Find component in model
    weight = 0.0
    direction = 1
    current_val = 50.0
    for (comp_name, path, w, d) in _COMPONENT_MODEL:
        if comp_name == component_name:
            weight = w
            direction = d
            val = _get_nested(engine_scores, path)
            if val is not None:
                current_val = float(val)
            break

    dau_per_point = weight * direction
    lo, hi = delta_range
    step = 5.0
    n_steps = int((hi - lo) / step) + 1

    curve = []
    for i in range(n_steps):
        delta = lo + i * step
        simulated_dau = round(clamp(dau + delta * dau_per_point, 0.0, 100.0), 2)
        curve.append({"delta": round(delta, 1), "dau": simulated_dau})

    dau_values = [pt["dau"] for pt in curve]
    dau_range = max(dau_values) - min(dau_values)
    total_delta_range = abs(hi - lo)
    sensitivity_score = round(clamp(dau_range / max(total_delta_range, 1.0), 0.0, 1.0), 4)

    # Breakeven: delta needed to cross signal boundary
    target_boundary = _BUY_THRESHOLD if dau < _BUY_THRESHOLD else _SELL_THRESHOLD
    if abs(dau_per_point) > 0:
        breakeven = (target_boundary - dau) / dau_per_point
    else:
        breakeven = float("inf")

    return {
        "component": component_name,
        "current_value": round(current_val, 2),
        "current_dau": round(dau, 2),
        "sensitivity_curve": curve,
        "breakeven_delta": round(breakeven, 2) if abs(breakeven) < 999 else None,
        "sensitivity_score": sensitivity_score,
    }
