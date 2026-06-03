"""Phase 8: Proprietary 12-Factor Model and Alpha Decomposition Engine tests."""
from __future__ import annotations
import pytest
from api.axiom.factors.factor_model import (
    compute_all_factor_loadings,
    compute_eif,
    compute_cmf,
    compute_baf,
    compute_klf,
    compute_scaf,
    compute_icf,
    compute_gbf,
    compute_mtrf,
    compute_mqf,
    compute_vif,
    compute_rtf,
    compute_ntff,
)
from api.axiom.factors.alpha_decomposition import (
    decompose_alpha,
    AlphaDecomposition,
    _herfindahl,
)
from api.axiom.factors.regime_factor_matrix import (
    FACTOR_REGIME_MATRIX,
    get_regime_factor_weights,
    compute_factor_composite_score,
)
from api.axiom.factors.factor_model import FactorLoading
from api.jobs.ic import SCORE_FIELDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_engine_scores(overrides=None):
    base = {
        "fundamental_reality": {"score": 60.0, "components": {
            "earnings_quality_component": 60.0,
            "caps_component": 60.0,
            "valuation_gap_component": 65.0,
        }},
        "state_pricing": {"score": 55.0, "components": {"cardi_score": 60.0}},
        "behavioral_distortion": {"score": 55.0, "components": {"crowding_component": 40.0}},
        "flow_transmission": {"score": 60.0, "components": {
            "trend_quality_component": 65.0,
            "transmission_strength_component": 60.0,
        }},
        "liquidity_convexity": {"score": 65.0, "components": {"kle_score": 70.0}},
        "critical_fragility": {"score": 50.0, "components": {
            "scps_component": 30.0,
            "mtrs_score": 35.0,
        }},
        "research_integrity": {"score": 60.0, "components": {}},
    }
    if overrides:
        for k, v in overrides.items():
            base[k] = v
    return base


def _make_engine_inputs(regime="TRENDING", strength=0.7):
    return {"regime_label": regime, "regime_strength": strength, "amqs_score": 65.0}


def _make_factor_loadings(loading_value: float = 0.5) -> list:
    """Create 12 FactorLoading objects all with the same loading value."""
    names = ["EIF", "CMF", "BAF", "KLF", "SCAF", "ICF", "GBF", "MTRF", "MQF", "VIF", "RTF", "NTFF"]
    return [
        FactorLoading(
            factor_name=n,
            loading=loading_value,
            t_stat=abs(loading_value) * (30 ** 0.5),
            theoretical_source="test",
            regime_relevance={"TRENDING": 0.15, "CHOPPY": 0.15, "HIGH_VOL": 0.15, "RECOVERY": 0.15},
        )
        for n in names
    ]


# ---------------------------------------------------------------------------
# Individual factor tests
# ---------------------------------------------------------------------------

def test_eif_loading_positive_for_high_eis():
    """EIF loading > 0 when earnings_quality_component is 80."""
    es = _make_engine_scores({
        "fundamental_reality": {"score": 80.0, "components": {
            "earnings_quality_component": 80.0,
            "caps_component": 60.0,
            "valuation_gap_component": 65.0,
        }}
    })
    fl = compute_eif(es, _make_engine_inputs())
    assert fl.loading > 0


def test_eif_loading_negative_for_low_eis():
    """EIF loading < 0 when earnings_quality_component is 20."""
    es = _make_engine_scores({
        "fundamental_reality": {"score": 20.0, "components": {
            "earnings_quality_component": 20.0,
            "caps_component": 60.0,
            "valuation_gap_component": 65.0,
        }}
    })
    fl = compute_eif(es, _make_engine_inputs())
    assert fl.loading < 0


def test_scaf_inverted():
    """SCAF loading < 0 when scps_component is 80 (high bubble risk = bad)."""
    es = _make_engine_scores({
        "critical_fragility": {"score": 80.0, "components": {
            "scps_component": 80.0,
            "mtrs_score": 35.0,
        }}
    })
    fl = compute_scaf(es, _make_engine_inputs())
    assert fl.loading < 0


def test_mtrf_inverted():
    """MTRF loading < 0 when mtrs_score is 80 (high tail risk = bad)."""
    es = _make_engine_scores({
        "critical_fragility": {"score": 80.0, "components": {
            "scps_component": 30.0,
            "mtrs_score": 80.0,
        }}
    })
    fl = compute_mtrf(es, _make_engine_inputs())
    assert fl.loading < 0


def test_vif_requires_all_three_fail_val():
    """VIF loading = 0.0 when valuation_gap_component is below threshold."""
    es = _make_engine_scores({
        "fundamental_reality": {"score": 60.0, "components": {
            "earnings_quality_component": 70.0,
            "caps_component": 60.0,
            "valuation_gap_component": 40.0,  # below 60
        }}
    })
    fl = compute_vif(es, _make_engine_inputs())
    assert fl.loading == 0.0


def test_vif_requires_all_three_fail_eis():
    """VIF loading = 0.0 when earnings_quality_component is below threshold."""
    es = _make_engine_scores({
        "fundamental_reality": {"score": 60.0, "components": {
            "earnings_quality_component": 40.0,  # below 60
            "caps_component": 60.0,
            "valuation_gap_component": 70.0,
        }}
    })
    fl = compute_vif(es, _make_engine_inputs())
    assert fl.loading == 0.0


def test_vif_positive_when_all_three_met():
    """VIF loading > 0 when all three criteria are met."""
    es = _make_engine_scores({
        "fundamental_reality": {"score": 75.0, "components": {
            "earnings_quality_component": 70.0,
            "caps_component": 60.0,
            "valuation_gap_component": 70.0,
        }}
    })
    fl = compute_vif(es, _make_engine_inputs())
    assert fl.loading > 0


def test_mqf_quality_gate_high_eis():
    """MQF loading equals full raw_l when eis > 60 (quality gate = 1.0)."""
    es = _make_engine_scores({
        "flow_transmission": {"score": 80.0, "components": {
            "trend_quality_component": 80.0,
            "transmission_strength_component": 80.0,
        }},
        "fundamental_reality": {"score": 70.0, "components": {
            "earnings_quality_component": 70.0,
            "caps_component": 60.0,
            "valuation_gap_component": 65.0,
        }},
    })
    fl = compute_mqf(es, _make_engine_inputs())
    # With eis=70 (>60), quality_gate=1.0, momentum_score=80, raw_l=0.6
    assert fl.loading > 0
    # Verify it's the full value (gate = 1.0)
    assert abs(fl.loading) > 0.4


def test_mqf_quality_gate_low_eis():
    """MQF loading = 0.0 when eis <= 40 (quality gate = 0.0)."""
    es = _make_engine_scores({
        "flow_transmission": {"score": 80.0, "components": {
            "trend_quality_component": 80.0,
            "transmission_strength_component": 80.0,
        }},
        "fundamental_reality": {"score": 30.0, "components": {
            "earnings_quality_component": 30.0,  # <= 40
            "caps_component": 60.0,
            "valuation_gap_component": 65.0,
        }},
    })
    fl = compute_mqf(es, _make_engine_inputs())
    assert fl.loading == 0.0


def test_rtf_positive_in_trending_regime():
    """RTF loading > 0 when regime is TRENDING with high strength."""
    ei = {"regime_label": "TRENDING", "regime_strength": 0.8}
    fl = compute_rtf({}, ei)
    assert fl.loading > 0


def test_rtf_negative_in_high_vol_regime():
    """RTF loading < 0 when regime is HIGH_VOL with high strength."""
    ei = {"regime_label": "HIGH_VOL", "regime_strength": 0.8}
    fl = compute_rtf({}, ei)
    assert fl.loading < 0


# ---------------------------------------------------------------------------
# Full 12-factor tests
# ---------------------------------------------------------------------------

def test_all_12_factors_compute():
    """compute_all_factor_loadings returns list of length 12."""
    es = _make_engine_scores()
    ei = _make_engine_inputs()
    result = compute_all_factor_loadings(es, ei)
    assert len(result) == 12


def test_all_factor_names_unique():
    """No duplicate factor names in result."""
    es = _make_engine_scores()
    ei = _make_engine_inputs()
    result = compute_all_factor_loadings(es, ei)
    names = [fl.factor_name for fl in result]
    assert len(names) == len(set(names))


def test_loadings_bounded():
    """All loadings are in [-1.0, 1.0]."""
    es = _make_engine_scores()
    ei = _make_engine_inputs()
    result = compute_all_factor_loadings(es, ei)
    for fl in result:
        assert -1.0 <= fl.loading <= 1.0, f"{fl.factor_name} loading {fl.loading} out of bounds"


def test_graceful_degradation_missing_scores():
    """Empty engine_scores dict does not raise exceptions."""
    result = compute_all_factor_loadings({}, {"regime_label": "CHOPPY", "regime_strength": 0.5})
    assert len(result) == 12
    for fl in result:
        assert -1.0 <= fl.loading <= 1.0


# ---------------------------------------------------------------------------
# Alpha decomposition tests
# ---------------------------------------------------------------------------

def test_decomposition_idiosyncratic_computed():
    """systematic + idiosyncratic ≈ total_dau (within 0.01 tolerance)."""
    es = _make_engine_scores()
    ei = _make_engine_inputs()
    loadings = compute_all_factor_loadings(es, ei)
    payload = {"deployable_alpha_utility": 65.0}
    decomp = decompose_alpha(payload, loadings, "TRENDING", symbol="AAPL", as_of_date="2026-01-01")
    assert abs(decomp.systematic_contribution + decomp.idiosyncratic_alpha - decomp.total_dau) < 0.01


def test_primary_driver_identified():
    """primary_driver is the factor name with highest |contribution|."""
    # Create loadings where MQF has a very high loading
    loadings = _make_factor_loadings(0.0)
    # Override MQF to have a large loading
    for i, fl in enumerate(loadings):
        if fl.factor_name == "MQF":
            loadings[i] = FactorLoading(
                factor_name="MQF",
                loading=0.9,
                t_stat=4.93,
                theoretical_source="test",
                regime_relevance={"TRENDING": 0.35, "CHOPPY": 0.10, "HIGH_VOL": 0.05, "RECOVERY": 0.20},
            )
            break
    payload = {"deployable_alpha_utility": 65.0}
    decomp = decompose_alpha(payload, loadings, "TRENDING", symbol="AAPL", as_of_date="2026-01-01")
    assert decomp.primary_driver == "MQF"


def test_herfindahl_bounded():
    """`_herfindahl([1,1,1,1])` is in [0,1]."""
    h = _herfindahl([1, 1, 1, 1])
    assert 0.0 <= h <= 1.0


def test_high_concentration_single_driver():
    """`_herfindahl([10,0,0,0])` = 1.0."""
    h = _herfindahl([10, 0, 0, 0])
    assert h == 1.0


def test_low_concentration_diversified():
    """`_herfindahl([1]*12)` is close to 1/12."""
    h = _herfindahl([1] * 12)
    assert abs(h - 1.0 / 12.0) < 0.001


# ---------------------------------------------------------------------------
# Regime factor matrix tests
# ---------------------------------------------------------------------------

def test_all_regimes_present():
    """TRENDING, CHOPPY, HIGH_VOL, RECOVERY are all in FACTOR_REGIME_MATRIX."""
    for regime in ("TRENDING", "CHOPPY", "HIGH_VOL", "RECOVERY"):
        assert regime in FACTOR_REGIME_MATRIX


def test_weights_sum_to_one_trending():
    """TRENDING regime weights sum to 1.0 (within 0.001)."""
    total = sum(FACTOR_REGIME_MATRIX["TRENDING"].values())
    assert abs(total - 1.0) < 0.001, f"TRENDING sum = {total}"


def test_weights_sum_to_one_choppy():
    """CHOPPY regime weights sum to 1.0 (within 0.001)."""
    total = sum(FACTOR_REGIME_MATRIX["CHOPPY"].values())
    assert abs(total - 1.0) < 0.001, f"CHOPPY sum = {total}"


def test_weights_sum_to_one_high_vol():
    """HIGH_VOL regime weights sum to 1.0 (within 0.001)."""
    total = sum(FACTOR_REGIME_MATRIX["HIGH_VOL"].values())
    assert abs(total - 1.0) < 0.001, f"HIGH_VOL sum = {total}"


def test_weights_sum_to_one_recovery():
    """RECOVERY regime weights sum to 1.0 (within 0.001)."""
    total = sum(FACTOR_REGIME_MATRIX["RECOVERY"].values())
    assert abs(total - 1.0) < 0.001, f"RECOVERY sum = {total}"


def test_factor_composite_score_bounded():
    """FCS is in [0, 100] with varied loadings."""
    # All positive loadings
    loadings_pos = _make_factor_loadings(0.6)
    fcs_pos = compute_factor_composite_score(loadings_pos, "TRENDING")
    assert 0.0 <= fcs_pos <= 100.0

    # All negative loadings
    loadings_neg = _make_factor_loadings(-0.6)
    fcs_neg = compute_factor_composite_score(loadings_neg, "TRENDING")
    assert 0.0 <= fcs_neg <= 100.0

    # Mixed
    es = _make_engine_scores()
    ei = _make_engine_inputs()
    loadings_mixed = compute_all_factor_loadings(es, ei)
    fcs_mixed = compute_factor_composite_score(loadings_mixed, "CHOPPY")
    assert 0.0 <= fcs_mixed <= 100.0


def test_trending_regime_favors_momentum():
    """TRENDING regime gives MQF a higher weight than VIF."""
    assert FACTOR_REGIME_MATRIX["TRENDING"]["MQF"] > FACTOR_REGIME_MATRIX["TRENDING"]["VIF"]


def test_high_vol_regime_favors_liquidity():
    """HIGH_VOL regime: KLF has the highest weight."""
    weights = FACTOR_REGIME_MATRIX["HIGH_VOL"]
    max_factor = max(weights, key=weights.get)
    assert max_factor == "KLF"


def test_recovery_regime_favors_value():
    """RECOVERY regime: VIF has the highest weight."""
    weights = FACTOR_REGIME_MATRIX["RECOVERY"]
    max_factor = max(weights, key=weights.get)
    assert max_factor == "VIF"


# ---------------------------------------------------------------------------
# Factor routes tests
# ---------------------------------------------------------------------------

def test_factor_routes_db_disabled(monkeypatch):
    """get_factor_loadings returns structured empty response when DB is disabled."""
    import api.db as db_module
    monkeypatch.setattr(db_module, "db_read_enabled", lambda: False)
    from api.axiom.factor_routes import get_factor_loadings
    result = get_factor_loadings("AAPL", as_of_date=None)
    assert result["symbol"] == "AAPL"
    assert result["factor_loadings"] == []
    assert result["factor_composite_score"] == 50.0
    assert "regime_sensitivity" in result


def test_dominant_factors_count(monkeypatch):
    """get_factor_loadings with mock payload returns at most 3 dominant factors."""
    import api.db as db_module

    mock_payload = {
        "factor_loadings_summary": [
            {"factor_name": "MQF", "loading": 0.8, "t_stat": 4.38},
            {"factor_name": "EIF", "loading": 0.6, "t_stat": 3.29},
            {"factor_name": "KLF", "loading": 0.5, "t_stat": 2.74},
            {"factor_name": "VIF", "loading": 0.3, "t_stat": 1.64},
        ],
        "alpha_decomposition": {},
        "factor_composite_score": 72.5,
        "regime_label": "TRENDING",
    }

    monkeypatch.setattr(db_module, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db_module, "safe_fetchone", lambda sql, params: (mock_payload,))

    from api.axiom.factor_routes import get_factor_loadings
    result = get_factor_loadings("AAPL", as_of_date=None)
    assert len(result["dominant_factors"]) <= 3
    assert len(result["dominant_factors"]) > 0


def test_regime_sensitivity_score_bounded(monkeypatch):
    """regime_sensitivity_score is in [0, 100]."""
    import api.db as db_module

    mock_payload = {
        "factor_loadings_summary": [
            {"factor_name": n, "loading": 0.3, "t_stat": 1.64}
            for n in ["EIF", "CMF", "BAF", "KLF", "SCAF", "ICF", "GBF", "MTRF", "MQF", "VIF", "RTF", "NTFF"]
        ],
        "alpha_decomposition": {},
        "factor_composite_score": 65.0,
        "regime_label": "TRENDING",
    }

    monkeypatch.setattr(db_module, "db_read_enabled", lambda: True)
    monkeypatch.setattr(db_module, "safe_fetchone", lambda sql, params: (mock_payload,))

    from api.axiom.factor_routes import get_factor_loadings
    result = get_factor_loadings("AAPL", as_of_date=None)
    score = result["regime_sensitivity"]["regime_sensitivity_score"]
    assert 0.0 <= score <= 100.0


def test_factor_routes_not_expose_regime_relevance(monkeypatch):
    """Factor endpoint response does NOT contain 'regime_relevance' in factor_loadings items."""
    import api.db as db_module
    monkeypatch.setattr(db_module, "db_read_enabled", lambda: False)
    from api.axiom.factor_routes import get_factor_loadings
    result = get_factor_loadings("AAPL", as_of_date=None)
    # Check top-level
    assert "regime_relevance" not in result
    # Check each factor loading item (should be empty list when DB disabled)
    for fl in result.get("factor_loadings", []):
        assert "regime_relevance" not in fl


# ---------------------------------------------------------------------------
# FCS directional tests
# ---------------------------------------------------------------------------

def test_fcs_high_for_all_positive_loadings():
    """FCS > 75 when all loadings are 0.8."""
    loadings = _make_factor_loadings(0.8)
    fcs = compute_factor_composite_score(loadings, "TRENDING")
    assert fcs > 75, f"Expected FCS > 75, got {fcs}"


def test_fcs_low_for_all_negative_loadings():
    """FCS < 25 when all loadings are -0.8."""
    loadings = _make_factor_loadings(-0.8)
    fcs = compute_factor_composite_score(loadings, "TRENDING")
    assert fcs < 25, f"Expected FCS < 25, got {fcs}"


# ---------------------------------------------------------------------------
# IC SCORE_FIELDS test
# ---------------------------------------------------------------------------

def test_factor_composite_in_ic_score_fields():
    """'factor_composite' is in SCORE_FIELDS from ic.py."""
    assert "factor_composite" in SCORE_FIELDS


# ---------------------------------------------------------------------------
# NTFF inversion test
# ---------------------------------------------------------------------------

def test_ntff_inverted_high_crowding():
    """NTFF loading < 0 when crowding_component is high (>50)."""
    es = _make_engine_scores({
        "behavioral_distortion": {"score": 80.0, "components": {"crowding_component": 80.0}}
    })
    fl = compute_ntff(es, _make_engine_inputs())
    assert fl.loading < 0, f"Expected NTFF < 0 for high crowding, got {fl.loading}"


def test_ntff_positive_low_crowding():
    """NTFF loading > 0 when crowding_component is low (<50)."""
    es = _make_engine_scores({
        "behavioral_distortion": {"score": 20.0, "components": {"crowding_component": 20.0}}
    })
    fl = compute_ntff(es, _make_engine_inputs())
    assert fl.loading > 0, f"Expected NTFF > 0 for low crowding, got {fl.loading}"


# ---------------------------------------------------------------------------
# Unknown regime fallback
# ---------------------------------------------------------------------------

def test_unknown_regime_uses_equal_weights():
    """Unknown regime label falls back to equal weights (1/12 each)."""
    weights = get_regime_factor_weights("UNKNOWN_REGIME_XYZ")
    expected = 1.0 / 12.0
    for factor, w in weights.items():
        assert abs(w - expected) < 0.001, f"{factor} weight {w} != {expected}"


def test_fcs_neutral_for_zero_loadings():
    """FCS ≈ 50 when all loadings are 0.0."""
    loadings = _make_factor_loadings(0.0)
    fcs = compute_factor_composite_score(loadings, "TRENDING")
    assert abs(fcs - 50.0) < 0.01, f"Expected FCS ≈ 50, got {fcs}"
