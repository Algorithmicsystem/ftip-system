from __future__ import annotations
from api.assistant.phase3.common import clamp

def compute_osms(options_data: dict) -> float:
    """Options Smart Money Score (0-100). Higher = more bullish smart money flow."""
    call_volume = options_data.get("call_volume")
    avg_30d_call_volume = options_data.get("avg_30d_call_volume")
    call_iv_atm = options_data.get("call_iv_atm")
    put_iv_atm = options_data.get("put_iv_atm")
    atm_iv = options_data.get("atm_iv")
    large_block_pct = options_data.get("large_block_pct")  # pct of premium in blocks > $100K
    pcr = options_data.get("put_call_ratio")

    components = {}

    # Component 1: unusual_call_volume_z — (call_vol/avg - 1) bounded [0,5], normalized 0-100
    if call_volume is not None and avg_30d_call_volume and float(avg_30d_call_volume) > 0:
        z = (float(call_volume) / float(avg_30d_call_volume)) - 1.0
        z_bounded = clamp(z, 0.0, 5.0)
        components["unusual_call_volume_z"] = (z_bounded / 5.0) * 100.0

    # Component 2: iv_skew_direction — (call_iv_atm - put_iv_atm) / atm_iv, bounded [-0.30, 0.30]
    # Positive skew (calls more expensive) = bullish = higher score
    if call_iv_atm is not None and put_iv_atm is not None and atm_iv and float(atm_iv) > 0:
        raw_skew = (float(call_iv_atm) - float(put_iv_atm)) / float(atm_iv)
        skew_bounded = clamp(raw_skew, -0.30, 0.30)
        components["iv_skew_direction"] = ((skew_bounded + 0.30) / 0.60) * 100.0

    # Component 3: large_block_premium — pct of premium in blocks > $100K, bounded [0, 0.80]
    if large_block_pct is not None:
        lb_bounded = clamp(float(large_block_pct), 0.0, 0.80)
        components["large_block_premium"] = (lb_bounded / 0.80) * 100.0

    # Component 4: put_call_ratio_score — inverted PCR, PCR<0.5=bullish=high score, PCR>1.5=bearish=low score
    if pcr is not None:
        pcr_bounded = clamp(float(pcr), 0.30, 2.0)
        # invert: low PCR → high score
        components["put_call_ratio_score"] = ((2.0 - pcr_bounded) / (2.0 - 0.30)) * 100.0

    if not components:
        return 50.0

    # Canonical weights — renormalize if some components missing
    canonical_weights = {
        "unusual_call_volume_z": 0.30,
        "iv_skew_direction": 0.25,
        "large_block_premium": 0.25,
        "put_call_ratio_score": 0.20,
    }
    total_w = sum(canonical_weights[k] for k in components)
    if total_w <= 0:
        return 50.0
    osms = sum(components[k] * canonical_weights[k] for k in components) / total_w
    return round(clamp(osms, 0.0, 100.0), 2)
