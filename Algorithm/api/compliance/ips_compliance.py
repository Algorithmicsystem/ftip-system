"""Phase 20.2: Investment Policy Statement compliance engine."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)


@dataclass
class IPSConstraints:
    portfolio_id: str
    tenant_id: str

    # Position limits
    max_single_position_weight: float = 0.10
    max_sector_concentration: float = 0.30
    max_equity_weight: float = 0.80
    max_alternatives_weight: float = 0.20
    min_cash_weight: float = 0.05

    # Quality constraints
    min_credit_rating: Optional[str] = None
    min_dau_for_equity: float = 50.0
    max_fragility_score: float = 70.0

    # Risk constraints
    max_portfolio_var_1d: float = 0.02
    max_tracking_error: float = 0.05

    # Exclusions
    prohibited_symbols: List[str] = field(default_factory=list)
    prohibited_sectors: List[str] = field(default_factory=list)
    esg_required: bool = False

    # Rebalancing
    rebalancing_threshold: float = 0.05
    rebalancing_frequency: str = "quarterly"


def check_ips_compliance(
    allocation: Dict[str, Any],
    ips: IPSConstraints,
    portfolio_value_usd: float = 1_000_000,
) -> Dict[str, Any]:
    """Check an allocation dict against IPS constraints.

    allocation format: list of {symbol, weight, sector, axiom_dau, fragility_score}
    or dict with "allocations" key holding that list.
    """
    positions: List[Dict[str, Any]] = []
    if isinstance(allocation, dict):
        positions = allocation.get("allocations", [])
    elif isinstance(allocation, list):
        positions = allocation

    violations: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    remediation_actions: List[str] = []

    def _violation(constraint: str, current: float, limit: float, severity: str, note: str = ""):
        violations.append({
            "constraint": constraint,
            "current_value": round(current, 4),
            "limit": round(limit, 4),
            "severity": severity,
            "note": note,
        })

    def _warning(constraint: str, current: float, limit: float):
        warnings.append({
            "constraint": constraint,
            "current_value": round(current, 4),
            "limit": round(limit, 4),
            "note": "Within 20% of limit",
        })

    prohibited_upper = [s.upper() for s in ips.prohibited_symbols]

    for pos in positions:
        sym = str(pos.get("symbol", "")).upper()
        weight = float(pos.get("weight", 0.0))
        dau = pos.get("axiom_dau")
        fragility = pos.get("fragility_score")

        # Prohibited symbol
        if sym in prohibited_upper:
            _violation("prohibited_symbol", weight, 0.0, "critical",
                       f"{sym} is in the prohibited securities list")
            remediation_actions.append(f"Remove {sym} from allocation (prohibited)")

        # Single position cap
        if weight > ips.max_single_position_weight:
            _violation("single_position_weight", weight, ips.max_single_position_weight, "major",
                       f"{sym} weight {weight:.1%} exceeds max {ips.max_single_position_weight:.1%}")
            remediation_actions.append(
                f"Reduce {sym} from {weight:.1%} to {ips.max_single_position_weight:.1%}"
            )
        elif weight > ips.max_single_position_weight * 0.8:
            _warning("single_position_weight", weight, ips.max_single_position_weight)

        # DAU quality constraint
        if dau is not None and float(dau) < ips.min_dau_for_equity:
            _violation("min_dau_for_equity", float(dau), ips.min_dau_for_equity, "minor",
                       f"{sym} DAU={dau:.1f} below minimum {ips.min_dau_for_equity:.1f}")
            remediation_actions.append(f"Remove {sym}: DAU {dau:.1f} below minimum threshold")

        # Fragility constraint
        if fragility is not None and float(fragility) > ips.max_fragility_score:
            _violation("max_fragility_score", float(fragility), ips.max_fragility_score, "minor",
                       f"{sym} fragility={fragility:.1f} above max {ips.max_fragility_score:.1f}")
            remediation_actions.append(f"Remove or reduce {sym}: fragility {fragility:.1f} too high")

    # Sector concentration
    sector_weights: Dict[str, float] = {}
    for pos in positions:
        sector = str(pos.get("sector", "Unknown"))
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(pos.get("weight", 0.0))

    for sector, sw in sector_weights.items():
        if sw > ips.max_sector_concentration:
            _violation("sector_concentration", sw, ips.max_sector_concentration, "major",
                       f"{sector} sector weight {sw:.1%} exceeds max {ips.max_sector_concentration:.1%}")
            remediation_actions.append(
                f"Reduce {sector} sector exposure from {sw:.1%} to {ips.max_sector_concentration:.1%}"
            )
        elif sw > ips.max_sector_concentration * 0.8:
            _warning("sector_concentration", sw, ips.max_sector_concentration)

    # Total equity weight
    total_weight = sum(float(p.get("weight", 0.0)) for p in positions)
    if total_weight > ips.max_equity_weight:
        _violation("max_equity_weight", total_weight, ips.max_equity_weight, "critical",
                   f"Total equity weight {total_weight:.1%} exceeds max {ips.max_equity_weight:.1%}")
        remediation_actions.append(
            f"Reduce total equity allocation from {total_weight:.1%} to {ips.max_equity_weight:.1%}"
        )

    # Prohibited sectors
    prohibited_sector_upper = [s.upper() for s in ips.prohibited_sectors]
    for sector in sector_weights:
        if sector.upper() in prohibited_sector_upper:
            _violation("prohibited_sector", sector_weights[sector], 0.0, "critical",
                       f"Sector '{sector}' is prohibited")
            remediation_actions.append(f"Remove all positions in prohibited sector: {sector}")

    # Compute compliance score
    critical_count = sum(1 for v in violations if v["severity"] == "critical")
    major_count = sum(1 for v in violations if v["severity"] == "major")
    minor_count = sum(1 for v in violations if v["severity"] == "minor")
    score = clamp(
        100.0 - (critical_count * 25 + major_count * 10 + minor_count * 3),
        0.0, 100.0,
    )

    compliant = len(violations) == 0
    return {
        "compliant": compliant,
        "violations": violations,
        "warnings": warnings,
        "compliance_score": round(score, 1),
        "remediation_actions": list(dict.fromkeys(remediation_actions)),
    }


def generate_ips_compliant_allocation(
    raw_allocation: Dict[str, Any],
    ips: IPSConstraints,
) -> Dict[str, Any]:
    """Adjust allocation to satisfy IPS constraints and return renormalized result."""
    positions: List[Dict[str, Any]] = list(raw_allocation.get("allocations", []))
    prohibited_upper = {s.upper() for s in ips.prohibited_symbols}
    prohibited_sectors_upper = {s.upper() for s in ips.prohibited_sectors}

    # Step 1: Remove prohibited symbols and prohibited sectors
    positions = [
        p for p in positions
        if str(p.get("symbol", "")).upper() not in prohibited_upper
        and str(p.get("sector", "")).upper() not in prohibited_sectors_upper
    ]

    # Step 2: Remove positions below min_dau_for_equity
    positions = [
        p for p in positions
        if p.get("axiom_dau") is None or float(p.get("axiom_dau", 0)) >= ips.min_dau_for_equity
    ]

    # Step 3: Remove positions above max_fragility_score
    positions = [
        p for p in positions
        if p.get("fragility_score") is None
        or float(p.get("fragility_score", 0)) <= ips.max_fragility_score
    ]

    # Step 4: Cap single positions
    for p in positions:
        if float(p.get("weight", 0.0)) > ips.max_single_position_weight:
            p["weight"] = ips.max_single_position_weight

    # Step 5: Reduce sector overweights proportionally
    sector_groups: Dict[str, List[Dict]] = {}
    for p in positions:
        s = str(p.get("sector", "Unknown"))
        sector_groups.setdefault(s, []).append(p)

    for sector, members in sector_groups.items():
        sw = sum(float(m.get("weight", 0.0)) for m in members)
        if sw > ips.max_sector_concentration and sw > 0:
            scale = ips.max_sector_concentration / sw
            for m in members:
                m["weight"] = float(m.get("weight", 0.0)) * scale

    # Step 6: Renormalize iteratively — re-cap after each normalization pass
    for _ in range(10):  # max 10 iterations to converge
        total = sum(float(p.get("weight", 0.0)) for p in positions)
        if total <= 0:
            break
        for p in positions:
            p["weight"] = float(p.get("weight", 0.0)) / total
        # Re-apply position cap after normalization
        capped = False
        for p in positions:
            if float(p.get("weight", 0.0)) > ips.max_single_position_weight:
                p["weight"] = ips.max_single_position_weight
                capped = True
        if not capped:
            break
    # Final rounding
    for p in positions:
        p["weight"] = round(float(p.get("weight", 0.0)), 6)

    result = dict(raw_allocation)
    result["allocations"] = positions
    result["ips_adjusted"] = True
    result["position_count"] = len(positions)
    result["portfolio_weight_total"] = round(
        sum(float(p.get("weight", 0.0)) for p in positions), 6
    )
    return result


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_ips_constraints(ips: IPSConstraints) -> bool:
    if not db.db_write_enabled():
        return False
    import dataclasses
    try:
        payload = json.dumps(dataclasses.asdict(ips))
        db.safe_execute(
            """
            INSERT INTO ips_constraints (portfolio_id, tenant_id, constraint_json, created_at, updated_at)
            VALUES (%s, %s, %s::jsonb, now(), now())
            ON CONFLICT (portfolio_id) DO UPDATE
               SET constraint_json = EXCLUDED.constraint_json, updated_at = now()
            """,
            (ips.portfolio_id, ips.tenant_id, payload),
        )
        return True
    except Exception as exc:
        logger.warning("ips.save_failed portfolio=%s err=%s", ips.portfolio_id, exc)
        return False


def load_ips_constraints(portfolio_id: str) -> Optional[IPSConstraints]:
    if not db.db_read_enabled():
        return None
    try:
        row = db.safe_fetchone(
            "SELECT constraint_json FROM ips_constraints WHERE portfolio_id = %s",
            (portfolio_id,),
        )
        if not row or not row[0]:
            return None
        data = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return IPSConstraints(**{k: v for k, v in data.items()
                                 if k in IPSConstraints.__dataclass_fields__})
    except Exception as exc:
        logger.warning("ips.load_failed portfolio=%s err=%s", portfolio_id, exc)
        return None
