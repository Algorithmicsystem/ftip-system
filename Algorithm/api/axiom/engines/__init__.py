from __future__ import annotations

from typing import Dict

from api.assistant.phase3.common import clamp
from api.axiom.contracts import AxiomEngineInput, EngineScore


def stub_engine_score(
    *,
    engine_name: str,
    coverage_hint: float,
    relevant_domains: Dict[str, float] | None = None,
) -> EngineScore:
    relevant_domains = relevant_domains or {}
    domain_signal = max(relevant_domains.values(), default=0.0)
    coverage = clamp(max(coverage_hint, domain_signal), 0.0, 100.0)
    status = "partial" if coverage > 0 else "unavailable"
    summary = (
        f"{engine_name.replace('_', ' ').title()} is not implemented in the Phase 1 AXIOM slice yet. "
        "The contract is live, but the engine is exposing only honest partial coverage metadata."
        if coverage > 0
        else f"{engine_name.replace('_', ' ').title()} is not implemented in the Phase 1 AXIOM slice yet."
    )
    flags = ["phase1_not_implemented"]
    if coverage <= 0:
        flags.append("no_engine_coverage")
    return EngineScore(
        score=None,
        confidence=0.0,
        coverage=round(coverage, 2),
        status=status,
        components={},
        flags=flags,
        summary=summary,
    )


from .behavior import score_behavioral_distortion
from .flow import score_flow_transmission
from .fundamental import score_fundamental_reality
from .liquidity_convexity import score_liquidity_convexity
from .research_integrity import score_research_integrity
from .state_pricing import score_state_pricing
from .fragility import score_critical_fragility

__all__ = [
    "score_behavioral_distortion",
    "score_flow_transmission",
    "score_fundamental_reality",
    "score_liquidity_convexity",
    "score_research_integrity",
    "score_state_pricing",
    "score_critical_fragility",
    "stub_engine_score",
]
