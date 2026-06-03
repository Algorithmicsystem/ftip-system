"""Phase 9.1: ML Feature Vector Builder.

Extracts the 46-feature ML input vector from assembled AXIOM outputs.
All features are point-in-time safe — knowable at signal generation time.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import numpy as np

from api.assistant.phase3.common import clamp


_IC_STATE_MAP: Dict[str, float] = {
    "STRONG": 1.0,
    "MODERATE": 0.75,
    "WEAK": 0.5,
    "DEGRADED": 0.25,
    "INSUFFICIENT": 0.0,
}

_TRENDING_LABELS = frozenset({"TRENDING", "BULL_TRENDING", "TREND_CONFIRMED"})
_HIGHVOL_LABELS = frozenset({"HIGH_VOL", "BEAR_STRESS", "LIQUIDITY_FRACTURE"})


@dataclass
class AxiomMLFeatureVector:
    # AXIOM engine scores (7) — normalized to [0, 1]
    fundamental_score: Optional[float] = None
    state_pricing_score: Optional[float] = None
    behavioral_score: Optional[float] = None
    flow_score: Optional[float] = None
    liquidity_score: Optional[float] = None
    fragility_score: Optional[float] = None
    research_integrity_score: Optional[float] = None

    # AXIOM scorecard outputs (4) — normalized to [0, 1]
    gross_opportunity: Optional[float] = None
    friction_burden: Optional[float] = None
    validated_edge: Optional[float] = None
    deployable_alpha_utility: Optional[float] = None

    # Knowledge vault IP calculations (10) — normalized to [0, 1]
    eis_score: Optional[float] = None
    caps_score: Optional[float] = None
    scps_score: Optional[float] = None
    mtrs_score: Optional[float] = None
    kle_score: Optional[float] = None
    bfs_score: Optional[float] = None
    cardi_score: Optional[float] = None
    amqs_score: Optional[float] = None
    asymmetric_sent: Optional[float] = None
    pess_score: Optional[float] = None

    # Alternative data scores (4) — normalized to [0, 1]
    osms_score: Optional[float] = None
    nss_score: Optional[float] = None
    nms_score: Optional[float] = None
    ias_score: Optional[float] = None

    # Factor model outputs (12) — kept as-is, range [-1, 1]
    factor_eif: Optional[float] = None
    factor_cmf: Optional[float] = None
    factor_baf: Optional[float] = None
    factor_klf: Optional[float] = None
    factor_scaf: Optional[float] = None
    factor_icf: Optional[float] = None
    factor_gbf: Optional[float] = None
    factor_mtrf: Optional[float] = None
    factor_mqf: Optional[float] = None
    factor_vif: Optional[float] = None
    factor_rtf: Optional[float] = None
    factor_ntff: Optional[float] = None

    # Factor composite and alpha decomposition (3)
    factor_composite_score: Optional[float] = None   # normalized to [0, 1]
    idiosyncratic_alpha: Optional[float] = None       # normalized by /100
    factor_concentration: Optional[float] = None      # already [0, 1]

    # Market regime context (3) — [0, 1]
    regime_is_trending: Optional[float] = None
    regime_is_high_vol: Optional[float] = None
    regime_strength: Optional[float] = None

    # IC quality context (3)
    ic_state_numeric: Optional[float] = None            # [0, 1] via lookup
    ic_mean_21d: Optional[float] = None                  # kept as-is [-1, 1]
    effective_breadth_normalized: Optional[float] = None # breadth/30, capped 1.0


def _div(v: Any, divisor: float) -> Optional[float]:
    """Return float(v)/divisor if v is not None, else None."""
    if v is None:
        return None
    try:
        return float(v) / divisor
    except (TypeError, ValueError):
        return None


def _efloat(v: Any) -> Optional[float]:
    """Return float(v) if v is not None, else None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_feature_vector(
    axiom_payload: Dict[str, Any],
    factor_loadings: list,
    ic_state: Optional[Dict[str, Any]] = None,
) -> AxiomMLFeatureVector:
    """Extract all 46 ML features from assembled AXIOM payload.

    Missing values stay None — the ML model handles them via imputation (fill_value=0.5).
    All 0-100 scores are divided by 100. Factor loadings [-1,1] are kept as-is.
    """
    es = axiom_payload.get("engine_scores") or {}
    sc = axiom_payload.get("source_context") or {}
    ic = ic_state or {}

    def _engine_score(name: str) -> Optional[float]:
        v = (es.get(name) or {}).get("score")
        return _div(v, 100.0)

    def _comp(engine: str, comp_key: str) -> Optional[float]:
        v = ((es.get(engine) or {}).get("components") or {}).get(comp_key)
        return _div(v, 100.0)

    # Build factor loading lookup by factor_name
    factor_map: Dict[str, float] = {}
    for fl in (factor_loadings or []):
        if hasattr(fl, "factor_name"):
            factor_map[str(fl.factor_name)] = float(fl.loading)
        elif isinstance(fl, dict):
            name = fl.get("factor_name") or ""
            loading = fl.get("loading", 0.0)
            if name:
                factor_map[str(name)] = float(loading)

    alpha_decomp = axiom_payload.get("alpha_decomposition") or {}
    regime = str(axiom_payload.get("regime_label") or "CHOPPY").upper()
    ic_state_str = str(ic.get("ic_state") or "INSUFFICIENT").upper()

    # regime_strength in source_context is already [0, 1]
    rs_raw = sc.get("regime_strength")
    regime_str_val = float(rs_raw) if rs_raw is not None else 0.5

    # effective_breadth_normalized: breadth / 30, capped at 1.0
    eb = ic.get("effective_breadth") or 0
    try:
        eb_norm = min(float(eb) / 30.0, 1.0)
    except (TypeError, ValueError):
        eb_norm = None

    # eis_score may be stored as earnings_quality_component in fundamental
    eis = _comp("fundamental_reality", "eis_score")
    if eis is None:
        eis = _comp("fundamental_reality", "earnings_quality_component")

    # nms_score may be in behavioral components or source_context
    nms = _comp("behavioral_distortion", "nms_score")
    if nms is None:
        nms = _div(sc.get("nms"), 100.0)

    return AxiomMLFeatureVector(
        # Engine scores (7)
        fundamental_score=_engine_score("fundamental_reality"),
        state_pricing_score=_engine_score("state_pricing"),
        behavioral_score=_engine_score("behavioral_distortion"),
        flow_score=_engine_score("flow_transmission"),
        liquidity_score=_engine_score("liquidity_convexity"),
        fragility_score=_engine_score("critical_fragility"),
        research_integrity_score=_engine_score("research_integrity"),
        # Scorecard outputs (4)
        gross_opportunity=_div(axiom_payload.get("gross_opportunity"), 100.0),
        friction_burden=_div(axiom_payload.get("friction_burden"), 100.0),
        validated_edge=_div(axiom_payload.get("validated_edge"), 100.0),
        deployable_alpha_utility=_div(axiom_payload.get("deployable_alpha_utility"), 100.0),
        # Knowledge vault IP (10)
        eis_score=eis,
        caps_score=_comp("fundamental_reality", "caps_component"),
        scps_score=_comp("critical_fragility", "scps_component"),
        mtrs_score=_comp("critical_fragility", "mtrs_score"),
        kle_score=_comp("liquidity_convexity", "kle_score"),
        bfs_score=_comp("critical_fragility", "bfs_component"),
        cardi_score=_comp("state_pricing", "cardi_score"),
        amqs_score=_div(ic.get("amqs_score"), 100.0),
        asymmetric_sent=_div(sc.get("asymmetric_sent_score"), 100.0),
        pess_score=_comp("fundamental_reality", "pess_component"),
        # Alternative data (4)
        osms_score=_div(sc.get("osms"), 100.0) if sc.get("osms") is not None else _div(sc.get("osms_score"), 100.0),
        nss_score=_div(sc.get("nss"), 100.0) if sc.get("nss") is not None else _div(sc.get("nss_score"), 100.0),
        nms_score=nms,
        ias_score=_div(sc.get("ias"), 100.0) if sc.get("ias") is not None else _div(sc.get("ias_score"), 100.0),
        # Factor model (12) — loadings are [-1, 1], kept as-is
        factor_eif=factor_map.get("EIF"),
        factor_cmf=factor_map.get("CMF"),
        factor_baf=factor_map.get("BAF"),
        factor_klf=factor_map.get("KLF"),
        factor_scaf=factor_map.get("SCAF"),
        factor_icf=factor_map.get("ICF"),
        factor_gbf=factor_map.get("GBF"),
        factor_mtrf=factor_map.get("MTRF"),
        factor_mqf=factor_map.get("MQF"),
        factor_vif=factor_map.get("VIF"),
        factor_rtf=factor_map.get("RTF"),
        factor_ntff=factor_map.get("NTFF"),
        # Factor composite + alpha decomp (3)
        factor_composite_score=_div(axiom_payload.get("factor_composite_score"), 100.0),
        idiosyncratic_alpha=_div(alpha_decomp.get("idiosyncratic_alpha"), 100.0),
        factor_concentration=_efloat(alpha_decomp.get("factor_concentration")),
        # Regime context (3)
        regime_is_trending=1.0 if regime in _TRENDING_LABELS else 0.0,
        regime_is_high_vol=1.0 if regime in _HIGHVOL_LABELS else 0.0,
        regime_strength=regime_str_val,
        # IC quality (3)
        ic_state_numeric=_IC_STATE_MAP.get(ic_state_str, 0.0),
        ic_mean_21d=_efloat(ic.get("mean_ic")),
        effective_breadth_normalized=eb_norm,
    )


def feature_vector_to_array(
    fv: AxiomMLFeatureVector,
    fill_value: float = 0.5,
) -> np.ndarray:
    """Convert dataclass to numpy array of length 46.

    None values are replaced with fill_value (default 0.5 = neutral midpoint).
    Order is deterministic — always matches dataclass field declaration order.
    """
    arr = []
    for f in fields(fv):
        v = getattr(fv, f.name)
        arr.append(fill_value if v is None else float(v))
    return np.array(arr, dtype=np.float64)


def get_feature_names() -> List[str]:
    """Return ordered list of 46 feature names matching feature_vector_to_array order."""
    return [f.name for f in fields(AxiomMLFeatureVector)]
