from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .common import compact_list


def build_exposure_framework(
    current: Dict[str, Any],
    cohort: List[Dict[str, Any]],
) -> Dict[str, Any]:
    sector_counts = Counter(item.get("sector") or "unknown" for item in cohort)
    benchmark_counts = Counter(item.get("benchmark_proxy") or "unknown" for item in cohort)
    regime_counts = Counter(item.get("regime_label") or "unknown" for item in cohort)
    macro_counts = Counter(item.get("macro_state") or "mixed" for item in cohort)
    theme_counts = Counter(item.get("theme_tag") or "broad_theme" for item in cohort)
    fragile_cluster_count = sum(
        1 for item in cohort if float(item.get("signal_fragility_index") or 0.0) >= 58.0
    )

    sector_count = sector_counts[current.get("sector") or "unknown"]
    benchmark_count = benchmark_counts[current.get("benchmark_proxy") or "unknown"]
    regime_count = regime_counts[current.get("regime_label") or "unknown"]
    macro_count = macro_counts[current.get("macro_state") or "mixed"]
    theme_count = theme_counts[current.get("theme_tag") or "broad_theme"]
    cohort_size = max(len(cohort), 1)

    concentration_warning = (
        f"Sector concentration is rising: {sector_count} of {cohort_size} tracked candidates share {current.get('sector')}."
        if sector_count >= 3 or (cohort_size >= 4 and sector_count >= 2)
        else None
    )
    cluster_concentration_warning = (
        f"Benchmark / regime concentration is elevated around {current.get('benchmark_proxy')} and {current.get('regime_label')}."
        if benchmark_count >= 3 or regime_count >= 3
        else None
    )
    sector_crowding_warning = (
        f"Theme and sector clustering are elevated in {current.get('sector')}."
        if sector_count >= 2 and theme_count >= 2
        else None
    )
    fragility_cluster_warning = (
        "Multiple tracked candidates are simultaneously fragile, so cluster fragility is elevated."
        if fragile_cluster_count >= 2
        else None
    )
    macro_exposure_warning = (
        f"Macro exposure is clustering in a {current.get('macro_state')} state across {macro_count} tracked candidates."
        if macro_count >= max(2, int(cohort_size * 0.6))
        else None
    )
    theme_exposure_warning = (
        f"Narrative/theme overlap is concentrated around the {current.get('theme_tag')} proxy."
        if theme_count >= 2
        else None
    )

    active_warnings = [
        concentration_warning,
        cluster_concentration_warning,
        sector_crowding_warning,
        fragility_cluster_warning,
        macro_exposure_warning,
        theme_exposure_warning,
    ]
    if len([item for item in active_warnings if item]) >= 4:
        diversification_status = "concentrated"
    elif len([item for item in active_warnings if item]) >= 2:
        diversification_status = "mixed"
    else:
        diversification_status = "balanced"

    return {
        "concentration_warning": concentration_warning,
        "cluster_concentration_warning": cluster_concentration_warning,
        "sector_crowding_warning": sector_crowding_warning,
        "fragility_cluster_warning": fragility_cluster_warning,
        "macro_exposure_warning": macro_exposure_warning,
        "theme_exposure_warning": theme_exposure_warning,
        "diversification_status": diversification_status,
        "cluster_counts": {
            "sector": dict(sector_counts),
            "benchmark": dict(benchmark_counts),
            "regime": dict(regime_counts),
            "macro_state": dict(macro_counts),
            "theme": dict(theme_counts),
        },
        "active_warnings": compact_list(active_warnings, limit=8),
    }
