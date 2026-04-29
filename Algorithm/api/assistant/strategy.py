from __future__ import annotations

from typing import Any, Dict

from api.assistant.phase4 import build_strategy_artifact as build_phase4_strategy_artifact


STRATEGY_ARTIFACT_KIND = "strategy_artifact"
CHAT_GROUNDING_CONTEXT_KIND = "chat_grounding_context"


def build_strategy_artifact(
    *,
    job_context: Dict[str, Any],
    signal: Dict[str, Any],
    data_bundle: Dict[str, Any],
    feature_factor_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    return build_phase4_strategy_artifact(
        job_context=job_context,
        signal=signal,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
    )
