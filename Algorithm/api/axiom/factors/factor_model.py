"""Proprietary 12-Factor Model for AXIOM alpha decomposition."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from api.assistant.phase3.common import clamp


@dataclass
class FactorLoading:
    factor_name: str
    loading: float           # -1.0 to +1.0
    t_stat: float            # approximate t-stat (|loading| * sqrt(n_observations))
    theoretical_source: str  # book/paper reference
    regime_relevance: Dict[str, float] = field(default_factory=dict)


def _safe(components: dict, key: str, default: float = 50.0) -> float:
    v = components.get(key)
    return float(v) if v is not None else default


def _loading(score_0_100: float) -> float:
    return round(clamp((score_0_100 - 50.0) / 50.0, -1.0, 1.0), 4)


def _inv_loading(score_0_100: float) -> float:
    return round(clamp((50.0 - score_0_100) / 50.0, -1.0, 1.0), 4)


def _tstat(loading: float, n: int = 30) -> float:
    return round(abs(loading) * (n ** 0.5), 4)


# ---------------------------------------------------------------------------
# 12 Factor compute functions
# ---------------------------------------------------------------------------

def compute_eif(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """EIF — Earnings Integrity Factor. Grounded in Penman, Schilit, O'Glove."""
    comps = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    eis = _safe(comps, "earnings_quality_component")
    l = _loading(eis)
    return FactorLoading(
        factor_name="EIF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Penman Financial Statement Analysis; Schilit Financial Shenanigans; O'Glove Quality of Earnings",
        regime_relevance={"TRENDING": 0.15, "CHOPPY": 0.25, "HIGH_VOL": 0.20, "RECOVERY": 0.30},
    )


def compute_cmf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """CMF — Competitive Moat Factor. Grounded in Rappaport Creating Shareholder Value."""
    comps = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    caps = _safe(comps, "caps_component")
    l = _loading(caps)
    return FactorLoading(
        factor_name="CMF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Rappaport Creating Shareholder Value",
        regime_relevance={"TRENDING": 0.10, "CHOPPY": 0.25, "HIGH_VOL": 0.15, "RECOVERY": 0.25},
    )


def compute_baf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """BAF — Behavioral Asymmetry Factor. Grounded in Kahneman-Tversky Prospect Theory."""
    # Use behavioral engine score as proxy for asymmetric sentiment quality
    beh = engine_scores.get("behavioral_distortion") or {}
    beh_score = beh.get("score") or 50.0
    l = _loading(float(beh_score))
    return FactorLoading(
        factor_name="BAF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Kahneman-Tversky Prospect Theory (1979); Thinking Fast and Slow",
        regime_relevance={"TRENDING": 0.20, "CHOPPY": 0.15, "HIGH_VOL": 0.25, "RECOVERY": 0.15},
    )


def compute_klf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """KLF — Kyle Liquidity Factor. Grounded in Kyle (1985) Continuous Auctions."""
    comps = (engine_scores.get("liquidity_convexity") or {}).get("components") or {}
    kle = _safe(comps, "kle_score")
    l = _loading(kle)
    return FactorLoading(
        factor_name="KLF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Kyle (1985) Continuous Auctions and Insider Trading",
        regime_relevance={"TRENDING": 0.15, "CHOPPY": 0.20, "HIGH_VOL": 0.30, "RECOVERY": 0.20},
    )


def compute_scaf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """SCAF — Sornette Crash Avoidance Factor. INVERTED: low SCPS = positive loading."""
    comps = (engine_scores.get("critical_fragility") or {}).get("components") or {}
    scps = _safe(comps, "scps_component")
    l = _inv_loading(scps)  # inverted: low bubble score = good
    return FactorLoading(
        factor_name="SCAF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Sornette Why Stock Markets Crash (2003); log-periodic power law",
        regime_relevance={"TRENDING": 0.20, "CHOPPY": 0.15, "HIGH_VOL": 0.25, "RECOVERY": 0.10},
    )


def compute_icf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """ICF — Ilmanen Carry Factor. Grounded in Ilmanen Expected Returns."""
    comps = (engine_scores.get("state_pricing") or {}).get("components") or {}
    cardi = _safe(comps, "cardi_score")
    l = _loading(cardi)
    return FactorLoading(
        factor_name="ICF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Ilmanen Expected Returns (2011); cross-asset carry framework",
        regime_relevance={"TRENDING": 0.15, "CHOPPY": 0.20, "HIGH_VOL": 0.10, "RECOVERY": 0.20},
    )


def compute_gbf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """GBF — Grinold Breadth Factor. Grounded in Grinold-Kahn Active Portfolio Management."""
    # amqs_score from source_context if available, else derive from research engine
    amqs = engine_inputs.get("amqs_score")
    if amqs is None:
        # Proxy: use research_integrity engine score as breadth quality proxy
        res = engine_scores.get("research_integrity") or {}
        amqs = res.get("score") or 50.0
    l = _loading(float(amqs))
    return FactorLoading(
        factor_name="GBF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Grinold-Kahn Active Portfolio Management; Fundamental Law of Active Management",
        regime_relevance={"TRENDING": 0.15, "CHOPPY": 0.15, "HIGH_VOL": 0.15, "RECOVERY": 0.15},
    )


def compute_mtrf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """MTRF — Mandelbrot Tail Risk Factor. INVERTED: low tail risk = positive loading."""
    comps = (engine_scores.get("critical_fragility") or {}).get("components") or {}
    mtrs = _safe(comps, "mtrs_score")
    l = _inv_loading(mtrs)  # inverted: low fat-tail risk = good
    return FactorLoading(
        factor_name="MTRF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Mandelbrot The Variation of Certain Speculative Prices (1963)",
        regime_relevance={"TRENDING": 0.10, "CHOPPY": 0.20, "HIGH_VOL": 0.35, "RECOVERY": 0.15},
    )


def compute_mqf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """MQF — Momentum Quality Factor. Jegadeesh-Titman filtered by earnings quality."""
    flow_comps = (engine_scores.get("flow_transmission") or {}).get("components") or {}
    tq = _safe(flow_comps, "trend_quality_component")
    ts = _safe(flow_comps, "transmission_strength_component")
    momentum_score = (tq + ts) / 2.0

    fund_comps = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    eis = _safe(fund_comps, "earnings_quality_component")
    if eis > 60:
        quality_gate = 1.0
    elif eis > 40:
        quality_gate = 0.5
    else:
        quality_gate = 0.0

    raw_l = clamp(momentum_score / 100.0 * 2.0 - 1.0, -1.0, 1.0)
    l = round(raw_l * quality_gate, 4)
    return FactorLoading(
        factor_name="MQF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Jegadeesh-Titman (1993) Returns to Buying Winners; quality momentum filter",
        regime_relevance={"TRENDING": 0.35, "CHOPPY": 0.10, "HIGH_VOL": 0.05, "RECOVERY": 0.20},
    )


def compute_vif(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """VIF — Value Integrity Factor. Graham + Penman + Rappaport triple screen."""
    fund_comps = (engine_scores.get("fundamental_reality") or {}).get("components") or {}
    val = _safe(fund_comps, "valuation_gap_component")
    eis = _safe(fund_comps, "earnings_quality_component")
    caps = _safe(fund_comps, "caps_component")

    if val > 60 and eis > 60 and caps > 55:
        l = round(clamp((val + eis + caps - 175.0) / 75.0, 0.0, 1.0), 4)
    else:
        l = 0.0
    return FactorLoading(
        factor_name="VIF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Graham Security Analysis; Penman + Rappaport triple quality-value screen",
        regime_relevance={"TRENDING": 0.10, "CHOPPY": 0.30, "HIGH_VOL": 0.15, "RECOVERY": 0.35},
    )


def compute_rtf(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """RTF — Regime Transition Factor. Kindleberger + Marks regime awareness."""
    regime_label = str(engine_inputs.get("regime_label") or "CHOPPY").upper()
    regime_strength = float(engine_inputs.get("regime_strength") or 0.5)

    if regime_label in ("TRENDING", "BULL_TRENDING", "RECOVERY"):
        direction_sign = 1.0
    elif regime_label in ("HIGH_VOL", "BEAR_STRESS", "LIQUIDITY_FRACTURE"):
        direction_sign = -1.0
    else:
        direction_sign = 0.0

    l = round(clamp((regime_strength - 0.5) / 0.5 * direction_sign, -1.0, 1.0), 4)
    return FactorLoading(
        factor_name="RTF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="Kindleberger Manias Panics Crashes; Marks The Most Important Thing (2011)",
        regime_relevance={"TRENDING": 0.20, "CHOPPY": 0.15, "HIGH_VOL": 0.20, "RECOVERY": 0.20},
    )


def compute_ntff(engine_scores: dict, engine_inputs: dict) -> FactorLoading:
    """NTFF — Noise Trader Fade Factor. DeLong-Shleifer-Summers-Waldmann."""
    beh_comps = (engine_scores.get("behavioral_distortion") or {}).get("components") or {}
    # Use crowding_component as crowding proxy; high crowding = fade signal
    crowding = _safe(beh_comps, "crowding_component")
    l = _inv_loading(crowding)  # inverted: high crowding = short = negative loading
    return FactorLoading(
        factor_name="NTFF",
        loading=l,
        t_stat=_tstat(l),
        theoretical_source="DeLong, Shleifer, Summers, Waldmann (1990) Noise Trader Risk",
        regime_relevance={"TRENDING": 0.15, "CHOPPY": 0.20, "HIGH_VOL": 0.15, "RECOVERY": 0.15},
    )


_FACTOR_FUNCS = [
    compute_eif, compute_cmf, compute_baf, compute_klf,
    compute_scaf, compute_icf, compute_gbf, compute_mtrf,
    compute_mqf, compute_vif, compute_rtf, compute_ntff,
]


def compute_all_factor_loadings(
    engine_scores: dict,
    engine_inputs: dict,
) -> List[FactorLoading]:
    """Compute all 12 factor loadings. Gracefully returns loading=0.0 on any error."""
    results = []
    for fn in _FACTOR_FUNCS:
        try:
            results.append(fn(engine_scores, engine_inputs))
        except Exception:
            # Fallback: zero loading, unknown factor
            name = fn.__name__.replace("compute_", "").upper()
            results.append(FactorLoading(
                factor_name=name, loading=0.0, t_stat=0.0,
                theoretical_source="unavailable", regime_relevance={},
            ))
    return results
