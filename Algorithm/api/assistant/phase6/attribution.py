from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .common import safe_float
from .scorecards import build_ranking_scorecard


def _favorable_high(name: str) -> bool:
    text = name.lower()
    if "fragility" in text or "crowding" in text:
        return False
    return True


def _getter_for_score(name: str) -> Callable[[Dict[str, Any]], Optional[float]]:
    def _inner(record: Dict[str, Any]) -> Optional[float]:
        return safe_float((record.get("proprietary_scores") or {}).get(name))

    return _inner


def _getter_for_component(name: str) -> Callable[[Dict[str, Any]], Optional[float]]:
    def _inner(record: Dict[str, Any]) -> Optional[float]:
        component = (record.get("strategy_component_scores") or {}).get(name) or {}
        return safe_float(component.get("score"))

    return _inner


def build_factor_attribution_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "proprietary_score_attribution": [],
            "strategy_component_attribution": [],
        }

    first = records[0]
    score_names = sorted((first.get("proprietary_scores") or {}).keys())
    component_names = sorted((first.get("strategy_component_scores") or {}).keys())

    score_specs: List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], bool]] = [
        (name, _getter_for_score(name), _favorable_high(name)) for name in score_names
    ]
    component_specs: List[Tuple[str, Callable[[Dict[str, Any]], Optional[float]], bool]] = [
        (name, _getter_for_component(name), True) for name in component_names
    ]

    scorecard = build_ranking_scorecard(records, score_specs=score_specs)
    component_scorecard = build_ranking_scorecard(records, score_specs=component_specs)

    def _leaders(bucket_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        available = [
            result
            for result in bucket_results
            if result.get("status") == "available"
        ]
        available.sort(
            key=lambda item: abs(safe_float(item.get("favorable_vs_unfavorable_return_spread")) or 0.0),
            reverse=True,
        )
        return available[:8]

    return {
        "proprietary_score_attribution": _leaders(scorecard.get("bucket_results") or []),
        "strategy_component_attribution": _leaders(
            component_scorecard.get("bucket_results") or []
        ),
    }
