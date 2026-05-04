from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .common import clamp, compact_list, correlation, cosine_similarity, safe_float
from .covariance import PriceHistoryLoader, build_return_profile, default_price_history_loader


_SECTOR_PROXY_MAP = {
    "technology": "XLK",
    "financial services": "XLF",
    "financial": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "industrials": "XLI",
    "consumer defensive": "XLP",
    "consumer staples": "XLP",
    "utilities": "XLU",
    "communication services": "XLC",
    "consumer cyclical": "XLY",
    "materials": "XLB",
    "real estate": "XLRE",
}

_MACRO_PROXIES = {
    "market": "SPY",
    "rates": "TLT",
    "gold": "GLD",
    "energy": "USO",
    "dollar": "UUP",
}


def _proxy_corr(
    corr: Optional[float],
    profile: Dict[str, Any],
    *,
    fallback_score: Optional[float] = None,
) -> float:
    if profile.get("history_source") == "realized_history" and corr is not None:
        return float(corr)
    if fallback_score is not None:
        return float(clamp((fallback_score - 50.0) / 55.0, -0.75, 0.75))
    if corr is None:
        return 0.0
    return float(corr) * 0.18


def _aligned_correlation(
    left_profile: Dict[str, Any],
    right_profile: Dict[str, Any],
) -> Optional[float]:
    left_map = {
        date: float(value)
        for date, value in zip(left_profile.get("dates") or [], left_profile.get("returns") or [])
    }
    right_map = {
        date: float(value)
        for date, value in zip(right_profile.get("dates") or [], right_profile.get("returns") or [])
    }
    overlap_dates = sorted(set(left_map.keys()) & set(right_map.keys()))
    if len(overlap_dates) < 8:
        return None
    left_returns = [left_map[date] for date in overlap_dates]
    right_returns = [right_map[date] for date in overlap_dates]
    return correlation(left_returns, right_returns)


def _proxy_profile(
    symbol: str,
    report: Dict[str, Any],
    *,
    history_loader: Optional[PriceHistoryLoader] = None,
    lookback_days: int = 126,
) -> Dict[str, Any]:
    synthetic_report = {
        "symbol": symbol,
        "as_of_date": report.get("as_of_date"),
        "data_bundle": {},
    }
    return build_return_profile(
        synthetic_report,
        history_loader=history_loader or default_price_history_loader,
        lookback_days=lookback_days,
    )


def build_factor_exposure(
    report: Dict[str, Any],
    snapshot: Dict[str, Any],
    history_profile: Dict[str, Any],
    *,
    history_loader: Optional[PriceHistoryLoader] = None,
    lookback_days: int = 126,
) -> Dict[str, Any]:
    sector = str(snapshot.get("sector") or "unknown").lower()
    benchmark = str(snapshot.get("benchmark_proxy") or _MACRO_PROXIES["market"])
    sector_proxy = _SECTOR_PROXY_MAP.get(sector, benchmark)

    benchmark_profile = _proxy_profile(
        benchmark,
        report,
        history_loader=history_loader,
        lookback_days=lookback_days,
    )
    sector_profile = _proxy_profile(
        sector_proxy,
        report,
        history_loader=history_loader,
        lookback_days=lookback_days,
    )
    rates_profile = _proxy_profile(
        _MACRO_PROXIES["rates"],
        report,
        history_loader=history_loader,
        lookback_days=lookback_days,
    )
    gold_profile = _proxy_profile(
        _MACRO_PROXIES["gold"],
        report,
        history_loader=history_loader,
        lookback_days=lookback_days,
    )
    energy_profile = _proxy_profile(
        _MACRO_PROXIES["energy"],
        report,
        history_loader=history_loader,
        lookback_days=lookback_days,
    )
    dollar_profile = _proxy_profile(
        _MACRO_PROXIES["dollar"],
        report,
        history_loader=history_loader,
        lookback_days=lookback_days,
    )

    benchmark_corr = _aligned_correlation(history_profile, benchmark_profile)
    sector_corr = _aligned_correlation(history_profile, sector_profile)
    rates_corr = _aligned_correlation(history_profile, rates_profile)
    gold_corr = _aligned_correlation(history_profile, gold_profile)
    energy_corr = _aligned_correlation(history_profile, energy_profile)
    dollar_corr = _aligned_correlation(history_profile, dollar_profile)

    opportunity = safe_float(snapshot.get("opportunity_quality_score")) or 0.0
    conviction = safe_float(snapshot.get("cross_domain_conviction_score")) or 0.0
    regime_stability = safe_float(snapshot.get("regime_stability_score")) or 0.0
    fragility = safe_float(snapshot.get("signal_fragility_index")) or 50.0
    implementation_fragility = safe_float(snapshot.get("implementation_fragility_score")) or fragility
    event_overhang = safe_float(snapshot.get("event_overhang_score")) or 0.0
    crowding = safe_float(snapshot.get("narrative_crowding_index")) or 0.0
    macro_alignment = safe_float(snapshot.get("macro_alignment_score")) or 50.0
    fundamentals = safe_float(snapshot.get("fundamental_durability_score")) or 0.0
    ret_21d = safe_float(snapshot.get("ret_21d")) or 0.0

    momentum_dependence = clamp(
        50.0 + (ret_21d * 320.0) + ((opportunity - 50.0) * 0.18) + ((regime_stability - 50.0) * 0.12),
        0.0,
        100.0,
    )
    mean_reversion_dependence = clamp(
        50.0 - (ret_21d * 260.0) + max(0.0, 55.0 - regime_stability) * 0.28,
        0.0,
        100.0,
    )
    quality_growth_affinity = clamp(
        (fundamentals * 0.55) + (opportunity * 0.25) + (macro_alignment * 0.20),
        0.0,
        100.0,
    )
    fragility_profile = clamp(
        (fragility * 0.55) + (implementation_fragility * 0.25) + ((safe_float(snapshot.get("market_stress_score")) or 0.0) * 0.20),
        0.0,
        100.0,
    )

    benchmark_corr = _proxy_corr(
        benchmark_corr,
        benchmark_profile,
        fallback_score=safe_float(snapshot.get("benchmark_confirmation_score")) or macro_alignment,
    )
    sector_corr = _proxy_corr(
        sector_corr,
        sector_profile,
        fallback_score=safe_float(snapshot.get("sector_confirmation_score"))
        or safe_float(snapshot.get("benchmark_confirmation_score"))
        or macro_alignment,
    )
    rates_corr = _proxy_corr(rates_corr, rates_profile)
    gold_corr = _proxy_corr(gold_corr, gold_profile)
    energy_corr = _proxy_corr(
        energy_corr,
        energy_profile,
        fallback_score=58.0 if sector == "energy" else 46.0 if sector in {"materials", "industrials"} else 50.0,
    ) * 0.55
    dollar_corr = _proxy_corr(dollar_corr, dollar_profile) * 0.35

    factor_vector = {
        "market_beta": round((benchmark_corr or 0.0) * 100.0, 2),
        "sector_dependence": round((sector_corr or 0.0) * 100.0, 2),
        "momentum_trend": round(momentum_dependence - 50.0, 2),
        "mean_reversion": round(mean_reversion_dependence - 50.0, 2),
        "rates_sensitivity": round((rates_corr or 0.0) * 100.0, 2),
        "gold_sensitivity": round((gold_corr or 0.0) * 100.0, 2),
        "energy_sensitivity": round((energy_corr or 0.0) * 100.0, 2),
        "dollar_sensitivity": round((dollar_corr or 0.0) * 100.0, 2),
        "fragility_sensitivity": round(fragility_profile - 50.0, 2),
        "event_sensitivity": round(event_overhang - 50.0, 2),
        "narrative_crowding": round(crowding - 50.0, 2),
        "quality_growth": round(quality_growth_affinity - 50.0, 2),
    }

    style_scores = {
        "quality_growth": quality_growth_affinity,
        "momentum": momentum_dependence,
        "mean_reversion": mean_reversion_dependence,
        "event_driven": clamp(event_overhang * 0.62 + crowding * 0.18 + fragility * 0.20, 0.0, 100.0),
        "fragile_beta": clamp(((benchmark_corr or 0.0) * 50.0) + fragility_profile * 0.55, 0.0, 100.0),
    }
    style_affinity = max(style_scores.items(), key=lambda item: item[1])[0]

    macro_profile = {
        "benchmark": benchmark,
        "sector_proxy": sector_proxy,
        "market_beta_proxy": round((benchmark_corr or 0.0), 4) if benchmark_corr is not None else None,
        "sector_dependence_score": round(abs(sector_corr or 0.0) * 100.0, 2),
        "rates_sensitivity_score": round(abs(rates_corr or 0.0) * 100.0, 2),
        "commodity_sensitivity_score": round(abs(energy_corr or 0.0) * 100.0, 2),
        "fx_sensitivity_score": round(abs(dollar_corr or 0.0) * 100.0, 2),
    }

    top_loadings = sorted(
        factor_vector.items(),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    factor_loading_summary = compact_list(
        [
            f"{name.replace('_', ' ')} {value:+.1f}"
            for name, value in top_loadings[:4]
        ],
        limit=4,
    )

    real_proxy_count = sum(
        1
        for profile in (
            benchmark_profile,
            sector_profile,
            rates_profile,
            gold_profile,
            energy_profile,
            dollar_profile,
        )
        if profile.get("history_source") == "realized_history"
    )
    exposure_confidence = round(
        clamp(
            min(float(history_profile.get("relationship_confidence") or 30.0), 90.0)
            + real_proxy_count * 6.0
            - (10.0 if history_profile.get("history_source") == "synthetic_proxy" else 0.0),
            20.0,
            92.0,
        ),
        2,
    )

    exposure_cluster = style_affinity
    if abs(factor_vector["market_beta"]) >= 65.0 and factor_vector["fragility_sensitivity"] >= 10.0:
        exposure_cluster = "fragile_beta_cluster"
    elif factor_vector["rates_sensitivity"] <= -25.0 or factor_vector["dollar_sensitivity"] <= -20.0:
        exposure_cluster = "macro_conflict_cluster"
    elif factor_vector["quality_growth"] >= 18.0 and factor_vector["momentum_trend"] >= 10.0:
        exposure_cluster = "quality_momentum_cluster"

    return {
        "factor_exposure_vector": factor_vector,
        "factor_loading_summary": factor_loading_summary,
        "style_affinity": style_affinity,
        "macro_exposure_profile": macro_profile,
        "fragility_exposure_profile": {
            "fragility_sensitivity_score": round(fragility_profile, 2),
            "event_sensitivity_score": round(event_overhang, 2),
            "crowding_sensitivity_score": round(crowding, 2),
        },
        "exposure_confidence": exposure_confidence,
        "exposure_cluster": exposure_cluster,
    }


def exposure_similarity(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, Any]:
    left_vector = left.get("factor_exposure_vector") or {}
    right_vector = right.get("factor_exposure_vector") or {}
    cosine = cosine_similarity(left_vector, right_vector)
    similarity_score = round(clamp(((cosine or 0.0) + 1.0) * 50.0, 0.0, 100.0), 2)
    shared_cluster = left.get("exposure_cluster") == right.get("exposure_cluster")
    macro_overlap = 84.0 if left.get("macro_exposure_profile", {}).get("benchmark") == right.get("macro_exposure_profile", {}).get("benchmark") else 42.0
    exposure_confidence = round(
        clamp(
            min(
                float(left.get("exposure_confidence") or 25.0),
                float(right.get("exposure_confidence") or 25.0),
            ),
            18.0,
            92.0,
        ),
        2,
    )
    return {
        "factor_similarity_score": similarity_score,
        "shared_exposure_cluster": shared_cluster,
        "macro_exposure_similarity": macro_overlap,
        "exposure_similarity_confidence": exposure_confidence,
    }
