"""Phase 12.3: Company Intelligence Dossier.

Maintains a living intelligence record for each symbol that grows over time.
After 12 months this dossier contains more intelligence about a company than
anything a new competitor could build from scratch.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event type registry
# ---------------------------------------------------------------------------

EVENT_TYPES = {
    "eis_deterioration",
    "eis_recovery",
    "caps_upgrade",
    "caps_downgrade",
    "regime_transition",
    "earnings_stress",
    "earnings_beat",
    "earnings_miss",
    "signal_success",
    "signal_failure",
    "scps_spike",
    "scps_normalization",
    "insider_activity",
    "sector_rotation",
    "management_quality_change",
}

# Event type weights for IQ scoring (higher = more informative)
DOSSIER_EVENT_WEIGHTS: Dict[str, float] = {
    "earnings_beat": 2.0,
    "earnings_miss": 2.0,
    "eis_deterioration": 1.8,
    "eis_recovery": 1.8,
    "scps_spike": 1.6,
    "caps_downgrade": 1.5,
    "caps_upgrade": 1.5,
    "signal_success": 1.2,
    "signal_failure": 1.2,
    "earnings_stress": 1.1,
    "regime_transition": 1.0,
    "scps_normalization": 0.8,
    "insider_activity": 0.8,
    "sector_rotation": 0.6,
    "management_quality_change": 0.6,
}

_RECENCY_HALF_LIFE_DAYS = 90.0


@dataclass
class DossierEvent:
    symbol: str
    event_date: dt.date
    event_type: str
    event_summary: str
    event_payload: Dict[str, Any]
    impact_score: float
    axiom_score_before: Optional[float] = None
    axiom_score_after: Optional[float] = None

    def __post_init__(self) -> None:
        self.impact_score = clamp(self.impact_score, 0.0, 100.0)


def _score_from_payload(payload: Dict, *path, default: Optional[float] = None) -> Optional[float]:
    try:
        v: Any = payload
        for k in path:
            v = v[k]
        return float(v) if v is not None else default
    except (KeyError, TypeError, ValueError):
        return default


def _detect_events_from_payloads(
    symbol: str,
    as_of_date: dt.date,
    current_payload: Dict,
    prior_payload: Dict,
) -> List[DossierEvent]:
    """Pure-computation event detection from AXIOM payload comparison."""
    events: List[DossierEvent] = []
    dau_before = _score_from_payload(prior_payload, "deployable_alpha_utility") or 0.0

    # --- EIS: Earnings Integrity Score ---
    eis_now = _score_from_payload(
        current_payload, "engine_scores", "fundamental_reality", "components", "earnings_quality_component"
    )
    eis_prior = _score_from_payload(
        prior_payload, "engine_scores", "fundamental_reality", "components", "earnings_quality_component"
    )
    if eis_now is not None and eis_prior is not None:
        delta = eis_now - eis_prior
        if delta < -10.0:
            events.append(DossierEvent(
                symbol=symbol, event_date=as_of_date,
                event_type="eis_deterioration",
                event_summary=f"EIS dropped {abs(delta):.1f} pts ({eis_prior:.1f}→{eis_now:.1f})",
                event_payload={"eis_before": eis_prior, "eis_after": eis_now, "delta": delta},
                impact_score=clamp(abs(delta) * 3.0, 0.0, 100.0),
                axiom_score_before=dau_before,
            ))
        elif delta > 10.0:
            events.append(DossierEvent(
                symbol=symbol, event_date=as_of_date,
                event_type="eis_recovery",
                event_summary=f"EIS recovered {delta:.1f} pts ({eis_prior:.1f}→{eis_now:.1f})",
                event_payload={"eis_before": eis_prior, "eis_after": eis_now, "delta": delta},
                impact_score=clamp(delta * 3.0, 0.0, 100.0),
                axiom_score_before=dau_before,
            ))

    # --- CAPS: Competitive Advantage Period Score ---
    caps_now = _score_from_payload(
        current_payload, "engine_scores", "fundamental_reality", "components", "caps_component"
    )
    caps_prior = _score_from_payload(
        prior_payload, "engine_scores", "fundamental_reality", "components", "caps_component"
    )
    if caps_now is not None and caps_prior is not None:
        delta = caps_now - caps_prior
        if delta < -8.0:
            events.append(DossierEvent(
                symbol=symbol, event_date=as_of_date,
                event_type="caps_downgrade",
                event_summary=f"CAPS declined {abs(delta):.1f} pts",
                event_payload={"caps_before": caps_prior, "caps_after": caps_now, "delta": delta},
                impact_score=clamp(abs(delta) * 5.0, 0.0, 100.0),
                axiom_score_before=dau_before,
            ))
        elif delta > 8.0:
            events.append(DossierEvent(
                symbol=symbol, event_date=as_of_date,
                event_type="caps_upgrade",
                event_summary=f"CAPS improved {delta:.1f} pts",
                event_payload={"caps_before": caps_prior, "caps_after": caps_now, "delta": delta},
                impact_score=clamp(delta * 5.0, 0.0, 100.0),
                axiom_score_before=dau_before,
            ))

    # --- SCPS: Sornette Critical Point Score ---
    scps_now = _score_from_payload(
        current_payload, "engine_scores", "critical_fragility", "components", "scps_component"
    )
    scps_prior = _score_from_payload(
        prior_payload, "engine_scores", "critical_fragility", "components", "scps_component"
    )
    if scps_now is not None and scps_prior is not None:
        if scps_now >= 70.0 and scps_prior < 70.0:
            events.append(DossierEvent(
                symbol=symbol, event_date=as_of_date,
                event_type="scps_spike",
                event_summary=f"Sornette crossed bubble threshold: {scps_prior:.1f}→{scps_now:.1f}",
                event_payload={"scps_before": scps_prior, "scps_after": scps_now},
                impact_score=clamp(scps_now, 0.0, 100.0),
                axiom_score_before=dau_before,
            ))
        elif scps_now < 50.0 and scps_prior >= 70.0:
            events.append(DossierEvent(
                symbol=symbol, event_date=as_of_date,
                event_type="scps_normalization",
                event_summary=f"Sornette score normalized: {scps_prior:.1f}→{scps_now:.1f}",
                event_payload={"scps_before": scps_prior, "scps_after": scps_now},
                impact_score=50.0,
                axiom_score_before=dau_before,
            ))

    # --- Regime transition ---
    regime_now = current_payload.get("regime_label")
    regime_prior = prior_payload.get("regime_label")
    if regime_now and regime_prior and regime_now != regime_prior:
        events.append(DossierEvent(
            symbol=symbol, event_date=as_of_date,
            event_type="regime_transition",
            event_summary=f"Regime: {regime_prior} → {regime_now}",
            event_payload={"from_regime": regime_prior, "to_regime": regime_now},
            impact_score=40.0,
            axiom_score_before=dau_before,
        ))

    # --- Earnings stress (PESS) ---
    pess = _score_from_payload(
        current_payload, "engine_scores", "critical_fragility", "components", "pess_component"
    )
    days_to_earnings = _score_from_payload(
        current_payload, "engine_inputs", "fundamental", "days_to_next_earnings"
    )
    if pess is not None and pess > 65.0 and days_to_earnings is not None and days_to_earnings < 60:
        events.append(DossierEvent(
            symbol=symbol, event_date=as_of_date,
            event_type="earnings_stress",
            event_summary=f"Pre-earnings stress: PESS={pess:.1f}, {int(days_to_earnings)}d to earnings",
            event_payload={"pess_score": pess, "days_to_earnings": days_to_earnings},
            impact_score=clamp(pess, 0.0, 100.0),
            axiom_score_before=dau_before,
        ))

    return events


def _compute_iq_score(
    days_of_data: int,
    event_count: int,
    net_signal_accuracy: float,
    events: Optional[List[Dict]] = None,
) -> float:
    """Intelligence Quality Score (0–100) with recency-weighted event richness."""
    import math
    base = min(days_of_data / 252.0, 1.0) * 50.0

    # Recency-weighted event richness: each event decays by half-life of 90 days
    if events:
        today = dt.date.today()
        weighted_sum = 0.0
        for ev in events:
            ev_type = str(ev.get("event_type") or "")
            weight = DOSSIER_EVENT_WEIGHTS.get(ev_type, 1.0)
            try:
                ev_date = dt.date.fromisoformat(str(ev.get("event_date") or today))
            except ValueError:
                ev_date = today
            age_days = max(0, (today - ev_date).days)
            decay = math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)
            weighted_sum += weight * decay
        event_richness = min(weighted_sum / 25.0, 1.0) * 30.0
    else:
        event_richness = min(event_count / 50.0, 1.0) * 30.0

    accuracy_bonus = net_signal_accuracy * 20.0
    return round(clamp(base + event_richness + accuracy_bonus, 0.0, 100.0), 2)


def detect_and_record_dossier_events(symbol: str, as_of_date: dt.date) -> List[DossierEvent]:
    """Compare today's AXIOM payload vs 21 days ago and detect events."""
    if not db.db_read_enabled():
        return []
    try:
        prior_date = as_of_date - dt.timedelta(days=21)

        def _load(date: dt.date) -> Dict:
            row = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol = %s AND as_of_date = %s",
                (symbol, date),
            )
            if not row or not row[0]:
                return {}
            return row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}")

        current = _load(as_of_date)
        prior = _load(prior_date)
        if not current:
            return []
        return _detect_events_from_payloads(symbol, as_of_date, current, prior)
    except Exception as exc:
        logger.warning("dossier_detect_failed sym=%s err=%s", symbol, exc)
        return []


def get_company_dossier(symbol: str, lookback_days: int = 365) -> Dict:
    """Return complete dossier for a symbol."""
    empty_summary = {
        "total_eis_events": 0,
        "total_signal_events": 0,
        "net_signal_accuracy": 0.0,
        "regime_transitions": 0,
        "bubble_warnings": 0,
        "earnings_stress_events": 0,
    }
    if not db.db_read_enabled():
        return {
            "symbol": symbol,
            "dossier_start_date": None,
            "event_count": 0,
            "events": [],
            "summary": empty_summary,
            "intelligence_quality_score": 0.0,
        }

    try:
        since = dt.date.today() - dt.timedelta(days=lookback_days)
        rows = db.safe_fetchall(
            """
            SELECT event_id, event_date, event_type, event_summary,
                   event_payload, impact_score, axiom_score_before, axiom_score_after
              FROM company_intelligence_archive
             WHERE symbol = %s AND event_date >= %s
             ORDER BY event_date DESC
            """,
            (symbol, since),
        ) or []

        events = [
            {
                "event_id": str(r[0]),
                "event_date": str(r[1]),
                "event_type": str(r[2]),
                "event_summary": str(r[3] or ""),
                "event_payload": r[4] if isinstance(r[4], dict) else {},
                "impact_score": float(r[5] or 0),
                "axiom_score_before": float(r[6]) if r[6] is not None else None,
                "axiom_score_after": float(r[7]) if r[7] is not None else None,
            }
            for r in rows
        ]

        eis_count = sum(1 for e in events if "eis" in e["event_type"])
        signal_success = sum(1 for e in events if e["event_type"] == "signal_success")
        signal_failure = sum(1 for e in events if e["event_type"] == "signal_failure")
        signal_total = signal_success + signal_failure
        net_accuracy = signal_success / signal_total if signal_total > 0 else 0.0

        summary = {
            "total_eis_events": eis_count,
            "total_signal_events": signal_total,
            "net_signal_accuracy": round(net_accuracy, 4),
            "regime_transitions": sum(1 for e in events if e["event_type"] == "regime_transition"),
            "bubble_warnings": sum(1 for e in events if e["event_type"] == "scps_spike"),
            "earnings_stress_events": sum(1 for e in events if e["event_type"] == "earnings_stress"),
        }

        dossier_start = str(rows[-1][1]) if rows else None
        days_of_data = (dt.date.today() - since).days if rows else 0
        iq = _compute_iq_score(days_of_data, len(events), net_accuracy, events)

        return {
            "symbol": symbol,
            "dossier_start_date": dossier_start,
            "event_count": len(events),
            "events": events,
            "summary": summary,
            "intelligence_quality_score": iq,
        }
    except Exception as exc:
        logger.warning("get_company_dossier_failed sym=%s err=%s", symbol, exc)
        return {
            "symbol": symbol,
            "dossier_start_date": None,
            "event_count": 0,
            "events": [],
            "summary": empty_summary,
            "intelligence_quality_score": 0.0,
        }


def run_dossier_update_job(as_of_date: Optional[dt.date] = None) -> Dict:
    """Process all symbols and upsert detected events."""
    aod = as_of_date or dt.date.today()
    if not db.db_read_enabled():
        return {"symbols_processed": 0, "events_created": 0, "as_of_date": aod.isoformat()}

    try:
        sym_rows = db.safe_fetchall(
            "SELECT DISTINCT symbol FROM axiom_scores_daily WHERE as_of_date = %s LIMIT 50",
            (aod,),
        ) or []
        symbols = [str(r[0]) for r in sym_rows]

        total_events = 0
        for sym in symbols:
            events = detect_and_record_dossier_events(sym, aod)
            for ev in events:
                try:
                    db.safe_execute(
                        """
                        INSERT INTO company_intelligence_archive
                            (event_id, symbol, event_date, event_type, event_summary,
                             event_payload, impact_score, axiom_score_before, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, now())
                        ON CONFLICT (event_id) DO NOTHING
                        """,
                        (str(uuid.uuid4()), sym, ev.event_date, ev.event_type,
                         ev.event_summary, json.dumps(ev.event_payload),
                         ev.impact_score, ev.axiom_score_before),
                    )
                    total_events += 1
                except Exception:
                    pass

        return {
            "symbols_processed": len(symbols),
            "events_created": total_events,
            "as_of_date": aod.isoformat(),
        }
    except Exception as exc:
        logger.warning("dossier_update_job_failed err=%s", exc)
        return {"symbols_processed": 0, "events_created": 0, "as_of_date": aod.isoformat()}
