"""Strip internal IP fields from AXIOM API responses."""
from __future__ import annotations
from typing import Any, Dict

_INTERNAL_FIELDS = frozenset({
    "scps_component", "mtrs_score", "bfs_component", "caps_component",
    "eis_score", "kle_score", "cardi_score",
    # factor model internal fields
    "regime_relevance",
})


def sanitize_engine_breakdown(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove proprietary sub-component fields from engine breakdowns for unauthenticated responses."""
    result = dict(payload)
    engine_scores = dict(result.get("engine_scores") or {})
    for engine_name, engine_data in engine_scores.items():
        if not isinstance(engine_data, dict):
            continue
        engine_copy = dict(engine_data)
        comps = dict(engine_copy.get("components") or {})
        for field in _INTERNAL_FIELDS:
            comps.pop(field, None)
        engine_copy["components"] = comps
        engine_scores[engine_name] = engine_copy
    result["engine_scores"] = engine_scores

    # Strip regime_adjusted_loadings from alpha_decomposition for free-tier responses
    # (this contains proprietary IP about how factors are combined)
    alpha_decomp = result.get("alpha_decomposition")
    if isinstance(alpha_decomp, dict) and alpha_decomp:
        alpha_copy = dict(alpha_decomp)
        alpha_copy.pop("regime_adjusted_loadings", None)
        result["alpha_decomposition"] = alpha_copy

    return result
