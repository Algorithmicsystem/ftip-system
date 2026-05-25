"""Session 15: Structured IC memo with lineage hash.

Produces tamper-evident investment-committee memos that embed a SHA-256
lineage hash over all deterministic inputs. The hash lets downstream
consumers verify the memo was not altered after generation.

SCHEMA_VERSION is included in the hash: format changes are detectable.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

_CONVICTION_TIERS = [(75.0, "HIGH"), (50.0, "MODERATE"), (25.0, "LOW")]


def conviction_tier(score: float) -> str:
    for threshold, label in _CONVICTION_TIERS:
        if score >= threshold:
            return label
    return "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Canonical inputs + lineage hash
# ---------------------------------------------------------------------------

def build_canonical_inputs(
    *,
    symbol: str,
    as_of_date: str,
    dau: float,
    fragility_score: float,
    liquidity_score: float,
    research_score: float,
    overall_confidence: float,
    deployability_tier: str,
    ic_state: str,
    hit_rate: Optional[float],
    fractional_kelly: float,
    max_weight: float,
    signal_label: str,
    regime_label: str,
    breadth_state: str,
    conviction_score: float,
    suggested_weight: float,
) -> Dict[str, Any]:
    """Return the ordered dict that is SHA-256 hashed.

    Only deterministic values — no timestamps or UUIDs.
    Floats are rounded to avoid IEEE-754 divergence between runs.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "dau": round(float(dau), 4),
        "fragility_score": round(float(fragility_score), 4),
        "liquidity_score": round(float(liquidity_score), 4),
        "research_score": round(float(research_score), 4),
        "overall_confidence": round(float(overall_confidence), 4),
        "deployability_tier": deployability_tier,
        "ic_state": ic_state,
        "hit_rate": round(float(hit_rate), 6) if hit_rate is not None else None,
        "fractional_kelly": round(float(fractional_kelly), 4),
        "max_weight": round(float(max_weight), 4),
        "signal_label": signal_label,
        "regime_label": regime_label,
        "breadth_state": breadth_state,
        "conviction_score": round(float(conviction_score), 4),
        "suggested_weight": round(float(suggested_weight), 6),
    }


def compute_lineage_hash(canonical_inputs: Dict[str, Any]) -> str:
    """SHA-256 over a deterministic JSON serialization of canonical inputs."""
    serialized = json.dumps(canonical_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Memo dataclass
# ---------------------------------------------------------------------------

@dataclass
class AxiomMemo:
    memo_id: str
    symbol: str
    as_of_date: str
    lineage_hash: str
    schema_version: str
    signal_label: str
    dau: float
    conviction_score: float
    conviction_tier_label: str
    suggested_weight: float
    regime_label: str
    ic_state: str
    memo_body: Dict[str, Any]
    canonical_inputs: Dict[str, Any]
    generated_at: str


# ---------------------------------------------------------------------------
# Memo builder
# ---------------------------------------------------------------------------

def build_memo(
    *,
    symbol: str,
    as_of_date: str,
    dau: float,
    fragility_score: float,
    liquidity_score: float,
    research_score: float,
    overall_confidence: float,
    deployability_tier: str,
    ic_state: str,
    hit_rate: Optional[float],
    fractional_kelly: float,
    max_weight: float,
    signal_label: str,
    regime_label: str,
    breadth_state: str,
    conviction_score: float,
    suggested_weight: float,
    kelly_gross_weight: float,
    fractional_kelly_applied: float,
    ic_kelly_multiplier: float,
    fragility_penalty_applied: float,
    active_constraint: str,
    size_band: str,
    downside_flags: List[str],
    rationale: str,
    data_source: str,
    engine_scores: Optional[Dict[str, Any]] = None,
) -> AxiomMemo:
    now = dt.datetime.utcnow().isoformat() + "Z"
    memo_id = str(uuid.uuid4())
    tier = conviction_tier(conviction_score)

    canonical = build_canonical_inputs(
        symbol=symbol,
        as_of_date=as_of_date,
        dau=dau,
        fragility_score=fragility_score,
        liquidity_score=liquidity_score,
        research_score=research_score,
        overall_confidence=overall_confidence,
        deployability_tier=deployability_tier,
        ic_state=ic_state,
        hit_rate=hit_rate,
        fractional_kelly=fractional_kelly,
        max_weight=max_weight,
        signal_label=signal_label,
        regime_label=regime_label,
        breadth_state=breadth_state,
        conviction_score=conviction_score,
        suggested_weight=suggested_weight,
    )
    lineage_hash = compute_lineage_hash(canonical)

    risk_flags: List[str] = list(downside_flags)
    if conviction_score < 25.0:
        risk_flags.append("insufficient_conviction")
    if not regime_label or regime_label in ("unknown", ""):
        risk_flags.append("regime_unclassified")
    if breadth_state in ("NEUTRAL", "STRESSED", None, ""):
        risk_flags.append(f"breadth_{(breadth_state or 'unknown').lower()}")

    weight_pct = f"{suggested_weight * 100:.2f}%"
    headline = (
        f"{symbol.upper()}: {signal_label} — {tier} conviction, {weight_pct} suggested weight"
    )

    body: Dict[str, Any] = {
        "memo_id": memo_id,
        "schema_version": SCHEMA_VERSION,
        "lineage_hash": lineage_hash,
        "generated_at": now,
        "symbol": symbol.upper(),
        "as_of_date": as_of_date,
        "executive_summary": {
            "headline": headline,
            "signal_direction": signal_label,
            "conviction_tier": tier,
            "conviction_score": round(conviction_score, 2),
            "deployability_tier": deployability_tier,
            "suggested_weight_pct": weight_pct,
        },
        "signal_context": {
            "signal_label": signal_label,
            "regime_label": regime_label,
            "breadth_state": breadth_state,
            "as_of_date": as_of_date,
        },
        "axiom_scorecard": {
            "dau": round(dau, 2),
            "deployability_tier": deployability_tier,
            "fragility_score": round(fragility_score, 2),
            "liquidity_score": round(liquidity_score, 2),
            "research_score": round(research_score, 2),
            "overall_confidence": round(overall_confidence, 2),
            **({"engine_scores": engine_scores} if engine_scores else {}),
        },
        "conviction_analysis": {
            "conviction_score": round(conviction_score, 2),
            "conviction_tier": tier,
            "dau_gate_passed": dau >= 65.0,
            "regime_favorable": regime_label not in (
                "euphoria_critical", "liquidity_fracture", "unknown", "", None
            ),
            "breadth_aligned": breadth_state in ("EXPANDING", "CONTRACTING", "STRESSED"),
            "ic_gate_passed": ic_state != "DEGRADED",
        },
        "position_sizing": {
            "suggested_weight": round(suggested_weight, 6),
            "suggested_weight_pct": weight_pct,
            "kelly_gross_weight": round(kelly_gross_weight, 6),
            "fractional_kelly_applied": round(fractional_kelly_applied, 4),
            "ic_kelly_multiplier": round(ic_kelly_multiplier, 4),
            "fragility_penalty_applied": round(fragility_penalty_applied, 4),
            "active_constraint": active_constraint,
            "size_band": size_band,
            "downside_flags": sorted(set(downside_flags)),
            "rationale": rationale,
        },
        "ic_calibration": {
            "ic_state": ic_state,
            "ic_kelly_multiplier": round(ic_kelly_multiplier, 4),
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "horizon": "21d",
        },
        "risk_flags": sorted(set(risk_flags)),
        "data_source": data_source,
        "compliance_attestation": (
            "This memo is system-generated by FTIP AXIOM v1. "
            "All inputs are recorded and verifiable via the lineage_hash. "
            "Not investment advice."
        ),
    }

    return AxiomMemo(
        memo_id=memo_id,
        symbol=symbol.upper(),
        as_of_date=as_of_date,
        lineage_hash=lineage_hash,
        schema_version=SCHEMA_VERSION,
        signal_label=signal_label,
        dau=round(dau, 2),
        conviction_score=round(conviction_score, 2),
        conviction_tier_label=tier,
        suggested_weight=round(suggested_weight, 6),
        regime_label=regime_label,
        ic_state=ic_state,
        memo_body=body,
        canonical_inputs=canonical,
        generated_at=now,
    )


# ---------------------------------------------------------------------------
# DB store / load
# ---------------------------------------------------------------------------

def store_memo(memo: AxiomMemo) -> bool:
    """INSERT memo; ON CONFLICT (lineage_hash) DO NOTHING — idempotent."""
    try:
        db.safe_execute(
            """
            INSERT INTO axiom_memos (
                memo_id, symbol, as_of_date, lineage_hash, schema_version,
                signal_label, dau, conviction_score, suggested_weight,
                regime_label, ic_state, memo_body, canonical_inputs
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
            ON CONFLICT (lineage_hash) DO NOTHING
            """,
            (
                memo.memo_id,
                memo.symbol,
                dt.date.fromisoformat(memo.as_of_date),
                memo.lineage_hash,
                memo.schema_version,
                memo.signal_label,
                memo.dau,
                memo.conviction_score,
                memo.suggested_weight,
                memo.regime_label,
                memo.ic_state,
                json.dumps(memo.memo_body),
                json.dumps(memo.canonical_inputs),
            ),
        )
        return True
    except Exception:
        logger.warning("memo.store_failed", extra={"memo_id": memo.memo_id})
        return False


def load_memo_by_id(memo_id: str) -> Optional[Dict[str, Any]]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT memo_id, symbol, as_of_date, lineage_hash, schema_version,
                   signal_label, dau, conviction_score, suggested_weight,
                   regime_label, ic_state, memo_body, canonical_inputs, created_at
            FROM axiom_memos WHERE memo_id = %s
            """,
            (memo_id,),
        )
        return _row_to_dict(row) if row else None
    except Exception:
        return None


def load_memo_by_hash(lineage_hash: str) -> Optional[Dict[str, Any]]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            """
            SELECT memo_id, symbol, as_of_date, lineage_hash, schema_version,
                   signal_label, dau, conviction_score, suggested_weight,
                   regime_label, ic_state, memo_body, canonical_inputs, created_at
            FROM axiom_memos WHERE lineage_hash = %s
            """,
            (lineage_hash,),
        )
        return _row_to_dict(row) if row else None
    except Exception:
        return None


def _row_to_dict(row: tuple) -> Dict[str, Any]:
    keys = (
        "memo_id", "symbol", "as_of_date", "lineage_hash", "schema_version",
        "signal_label", "dau", "conviction_score", "suggested_weight",
        "regime_label", "ic_state", "memo_body", "canonical_inputs", "created_at",
    )
    d = dict(zip(keys, row))
    for date_key in ("as_of_date", "created_at"):
        val = d.get(date_key)
        if hasattr(val, "isoformat"):
            d[date_key] = val.isoformat()
    for json_key in ("memo_body", "canonical_inputs"):
        val = d.get(json_key)
        if isinstance(val, str):
            try:
                d[json_key] = json.loads(val)
            except Exception:
                pass
    return d
