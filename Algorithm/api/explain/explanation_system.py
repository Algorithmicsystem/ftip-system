"""Phase 16.2: Evidence-Based Explanation System."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from api.assistant.phase3.common import clamp


@dataclass
class EvidenceItem:
    direction: str      # "supporting" | "contradicting" | "neutral"
    category: str       # "fundamental" | "technical" | "behavioral" | "macro" | "risk" | "alternative"
    description: str
    strength: float     # 0-1
    data_point: str
    source: str


def _strength(score: float, threshold: float = 50.0, scale: float = 50.0) -> float:
    return round(clamp(abs(score - threshold) / scale, 0.0, 1.0), 4)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def extract_evidence_items(
    axiom_payload: Dict[str, Any],
    signal_label: str,
) -> Dict[str, List[EvidenceItem]]:
    supporting: List[EvidenceItem] = []
    contradicting: List[EvidenceItem] = []
    neutral: List[EvidenceItem] = []

    engine_scores = axiom_payload.get("engine_scores") or {}
    is_buy = signal_label == "BUY"
    is_sell = signal_label == "SELL"

    # --- Fundamental ---
    fundamental = engine_scores.get("fundamental_reality") or {}
    comps_f = fundamental.get("components") or {}
    eis = comps_f.get("eis_component")
    caps = comps_f.get("caps_component")

    if eis is not None:
        eis = float(eis)
        if (is_buy and eis > 60) or (is_sell and eis < 40):
            supporting.append(EvidenceItem(
                direction="supporting",
                category="fundamental",
                description=f"Earnings quality (EIS) {'supports' if is_buy else 'confirms'} the {signal_label} thesis",
                strength=_strength(eis, 50.0, 50.0),
                data_point=f"EIS={eis:.1f}",
                source="fundamental_reality",
            ))
        elif (is_buy and eis < 40) or (is_sell and eis > 60):
            contradicting.append(EvidenceItem(
                direction="contradicting",
                category="fundamental",
                description=f"Earnings quality (EIS) contradicts the {signal_label} thesis",
                strength=_strength(eis, 50.0, 50.0),
                data_point=f"EIS={eis:.1f}",
                source="fundamental_reality",
            ))
        else:
            neutral.append(EvidenceItem(
                direction="neutral",
                category="fundamental",
                description=f"Earnings quality (EIS) is neutral",
                strength=0.1,
                data_point=f"EIS={eis:.1f}",
                source="fundamental_reality",
            ))

    if caps is not None:
        caps = float(caps)
        if (is_buy and caps > 65) or (is_sell and caps < 40):
            supporting.append(EvidenceItem(
                direction="supporting",
                category="fundamental",
                description=f"CAPS (Rappaport shareholder value) {'supports' if is_buy else 'confirms'} {signal_label}",
                strength=_strength(caps, 50.0, 50.0),
                data_point=f"CAPS={caps:.1f}",
                source="fundamental_reality",
            ))
        elif (is_buy and caps < 40) or (is_sell and caps > 65):
            contradicting.append(EvidenceItem(
                direction="contradicting",
                category="fundamental",
                description=f"CAPS contradicts the {signal_label} thesis",
                strength=_strength(caps, 50.0, 50.0),
                data_point=f"CAPS={caps:.1f}",
                source="fundamental_reality",
            ))

    # --- Flow (technical) ---
    flow = engine_scores.get("flow_transmission") or {}
    flow_score = float(flow.get("score") or 50.0)
    flow_comps = flow.get("components") or {}
    trend_quality = flow_comps.get("trend_quality")

    if (is_buy and flow_score > 60) or (is_sell and flow_score < 40):
        supporting.append(EvidenceItem(
            direction="supporting",
            category="technical",
            description=f"Flow transmission confirms {signal_label} direction",
            strength=_strength(flow_score, 50.0, 50.0),
            data_point=f"Flow={flow_score:.1f}",
            source="flow_transmission",
        ))
    elif (is_buy and flow_score < 40) or (is_sell and flow_score > 60):
        contradicting.append(EvidenceItem(
            direction="contradicting",
            category="technical",
            description=f"Flow transmission opposes the {signal_label} direction",
            strength=_strength(flow_score, 50.0, 50.0),
            data_point=f"Flow={flow_score:.1f}",
            source="flow_transmission",
        ))

    # --- Behavioral ---
    behavioral = engine_scores.get("behavioral_distortion") or {}
    beh_comps = behavioral.get("components") or {}
    sent = beh_comps.get("asymmetric_sent_score")
    crowding = beh_comps.get("crowding_score")

    if sent is not None:
        sent = float(sent)
        if (is_buy and sent > 60) or (is_sell and sent < 40):
            supporting.append(EvidenceItem(
                direction="supporting",
                category="behavioral",
                description="Asymmetric sentiment (Kahneman-Tversky) supports signal",
                strength=_strength(sent, 50.0, 50.0),
                data_point=f"Sentiment={sent:.1f}",
                source="behavioral_distortion",
            ))
        elif (is_buy and sent < 40) or (is_sell and sent > 60):
            contradicting.append(EvidenceItem(
                direction="contradicting",
                category="behavioral",
                description="Asymmetric sentiment contradicts signal direction",
                strength=_strength(sent, 50.0, 50.0),
                data_point=f"Sentiment={sent:.1f}",
                source="behavioral_distortion",
            ))

    if crowding is not None:
        crowding = float(crowding)
        if is_buy and crowding > 65:
            contradicting.append(EvidenceItem(
                direction="contradicting",
                category="behavioral",
                description="High crowding — trade is crowded, expected return is lower",
                strength=_strength(crowding, 50.0, 50.0),
                data_point=f"Crowding={crowding:.1f}",
                source="behavioral_distortion",
            ))

    # --- Fragility / Risk ---
    fragility_engine = engine_scores.get("critical_fragility") or {}
    frag_score = float(fragility_engine.get("score") or 50.0)
    frag_comps = fragility_engine.get("components") or {}
    scps = frag_comps.get("scps_component")

    if (is_buy and frag_score < 40) or (is_sell and frag_score > 60):
        supporting.append(EvidenceItem(
            direction="supporting",
            category="risk",
            description=f"Fragility is {'low' if frag_score < 40 else 'elevated'} — "
                        f"{'supports' if is_buy else 'supports'} {signal_label}",
            strength=_strength(frag_score, 50.0, 50.0),
            data_point=f"Fragility={frag_score:.1f}",
            source="critical_fragility",
        ))
    elif (is_buy and frag_score > 60) or (is_sell and frag_score < 40):
        contradicting.append(EvidenceItem(
            direction="contradicting",
            category="risk",
            description=f"Fragility level contradicts the {signal_label} thesis",
            strength=_strength(frag_score, 50.0, 50.0),
            data_point=f"Fragility={frag_score:.1f}",
            source="critical_fragility",
        ))

    if scps is not None:
        scps = float(scps)
        if is_buy and scps > 70:
            contradicting.append(EvidenceItem(
                direction="contradicting",
                category="risk",
                description="Sornette SCPS > 70 warns of bubble/crash conditions — strong BUY counter-signal",
                strength=_strength(scps, 50.0, 50.0),
                data_point=f"SCPS={scps:.1f}",
                source="critical_fragility",
            ))
        elif is_sell and scps > 70:
            supporting.append(EvidenceItem(
                direction="supporting",
                category="risk",
                description="High SCPS supports SELL — bubble/crash risk elevated",
                strength=_strength(scps, 50.0, 50.0),
                data_point=f"SCPS={scps:.1f}",
                source="critical_fragility",
            ))

    # --- Alternative data ---
    liquidity = engine_scores.get("liquidity_convexity") or {}
    liq_comps = liquidity.get("components") or {}
    osms = liq_comps.get("osms_component")
    ias = liq_comps.get("ias_component")

    if osms is not None:
        osms = float(osms)
        if (is_buy and osms > 65) or (is_sell and osms < 35):
            supporting.append(EvidenceItem(
                direction="supporting",
                category="alternative",
                description=f"Smart money flow (OSMS) {'confirms buying' if is_buy else 'confirms selling'} pressure",
                strength=_strength(osms, 50.0, 50.0),
                data_point=f"OSMS={osms:.1f}",
                source="liquidity_convexity",
            ))

    if ias is not None:
        ias = float(ias)
        if (is_buy and ias > 65) or (is_sell and ias < 35):
            supporting.append(EvidenceItem(
                direction="supporting",
                category="alternative",
                description=f"Institutional accumulation (IAS) supports {signal_label}",
                strength=_strength(ias, 50.0, 50.0),
                data_point=f"IAS={ias:.1f}",
                source="liquidity_convexity",
            ))

    # Sort and cap
    supporting.sort(key=lambda x: x.strength, reverse=True)
    contradicting.sort(key=lambda x: x.strength, reverse=True)

    return {
        "supporting": supporting[:5],
        "contradicting": contradicting[:3],
        "neutral": neutral,
    }


# ---------------------------------------------------------------------------
# Evidence balance
# ---------------------------------------------------------------------------

def compute_evidence_balance(evidence: Dict[str, List[EvidenceItem]]) -> Dict[str, Any]:
    supporting = evidence.get("supporting") or []
    contradicting = evidence.get("contradicting") or []

    sup_total = sum(e.strength for e in supporting)
    con_total = sum(e.strength for e in contradicting)
    net = round(sup_total - con_total, 4)

    sup_n = len(supporting)
    con_n = len(contradicting)

    if sup_n >= 3 and net > 1.0:
        quality = "strong"
    elif net > 0.5:
        quality = "moderate"
    elif abs(net) <= 0.3:
        quality = "mixed"
    else:
        quality = "weak"

    # Dominant theme: most common category in supporting evidence
    if supporting:
        from collections import Counter
        cat_counts = Counter(e.category for e in supporting)
        dominant_theme = cat_counts.most_common(1)[0][0]
    else:
        dominant_theme = "none"

    return {
        "supporting_count": sup_n,
        "contradicting_count": con_n,
        "net_evidence_score": net,
        "evidence_quality": quality,
        "dominant_theme": dominant_theme,
    }
