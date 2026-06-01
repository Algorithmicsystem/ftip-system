"""Session 13: Regime-gated alert engine.

Fires an alert for a symbol only when all three conviction layers agree:
  1. AXIOM DAU >= min_dau threshold (signal strength gate)
  2. Regime label is in the favorable set (not euphoric or fractured)
  3. Market breadth state aligns with the signal direction

Alerts are stored in signal_alert_events and optionally delivered to a
webhook URL. One event per rule per day (idempotent).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api import db, security

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    dependencies=[Depends(security.require_prosperity_api_key)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime catalogue
# ---------------------------------------------------------------------------

_FAVORABLE_REGIMES_BUY = frozenset({
    "fundamental_convergence",
    "compensation_capture",
    "behavioral_continuation",
    "convexity_opportunity",
    "recovery_reset",
})
_FAVORABLE_REGIMES_SELL = frozenset({
    "liquidity_fracture",
    "euphoria_critical",
})
_VETO_REGIMES_BUY = frozenset({"euphoria_critical", "liquidity_fracture"})
_VETO_REGIMES_SELL = frozenset({"fundamental_convergence", "compensation_capture"})

_BREADTH_ALIGNED_BUY  = frozenset({"EXPANDING"})
_BREADTH_ALIGNED_SELL = frozenset({"CONTRACTING", "STRESSED"})

_IC_MULTIPLIER = {
    "STRONG":       1.00,
    "MODERATE":     0.90,
    "WEAK":         0.75,
    "INSUFFICIENT": 0.70,
    "DEGRADED":     0.00,
}

_WEBHOOK_TIMEOUT_S = 10

_SIGNAL_EMOJI = {"BUY": ":large_green_circle:", "SELL": ":red_circle:", "HOLD": ":large_yellow_circle:"}


def format_slack_blocks(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return a Slack Incoming Webhook payload with blocks layout."""
    symbol   = event["symbol"]
    signal   = event.get("signal", "UNKNOWN")
    dau      = event.get("dau", 0.0)
    regime   = event.get("regime", "—")
    breadth  = event.get("breadth_state", "—")
    ic       = event.get("ic_state", "—")
    conv     = event.get("conviction_score", 0.0)
    date_str = event.get("as_of_date", "")
    emoji    = _SIGNAL_EMOJI.get(signal.upper(), ":white_circle:")

    return {
        "text": f"{emoji} {symbol} {signal} alert  (conviction {conv:.1f})",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {symbol} — {signal} Signal Alert",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol*\n{symbol}"},
                    {"type": "mrkdwn", "text": f"*Signal*\n{signal}"},
                    {"type": "mrkdwn", "text": f"*DAU Score*\n{dau:.1f}"},
                    {"type": "mrkdwn", "text": f"*Conviction*\n{conv:.1f}"},
                    {"type": "mrkdwn", "text": f"*Regime*\n{regime}"},
                    {"type": "mrkdwn", "text": f"*Breadth*\n{breadth}"},
                    {"type": "mrkdwn", "text": f"*IC State*\n{ic}"},
                    {"type": "mrkdwn", "text": f"*Date*\n{date_str}"},
                ],
            },
            {"type": "divider"},
        ],
    }


# ---------------------------------------------------------------------------
# Conviction scoring
# ---------------------------------------------------------------------------

def compute_conviction_score(
    *,
    dau: float,
    signal_label: str,
    regime_label: str,
    breadth_state: str,
    ic_state: str,
    min_dau: float = 65.0,
    favorable_regimes: Optional[frozenset] = None,
    require_breadth_alignment: bool = True,
) -> float:
    """
    Returns a 0–100 conviction score. Score > 0 means the alert should fire.
    Score = 0 means a hard veto (regime or IC blocks it).

    Breakdown:
      - DAU contribution:    0–50 points (scaled by how far above min_dau)
      - Regime contribution: 0, 15, or 30 points
      - Breadth contribution: 0 or 20 points
      × IC multiplier (0.0 if DEGRADED)
    """
    signal = (signal_label or "").upper()

    # Gate: DAU below threshold → no signal
    if dau < min_dau:
        return 0.0

    # Gate: IC degraded → no signal
    ic_mult = _IC_MULTIPLIER.get(ic_state, 0.70)
    if ic_mult == 0.0:
        return 0.0

    # DAU contribution (0–50)
    dau_range = max(1.0, 100.0 - min_dau)
    dau_contribution = ((dau - min_dau) / dau_range) * 50.0

    # Regime contribution (0–30)
    if favorable_regimes is None:
        if signal == "BUY":
            favorable_regimes = _FAVORABLE_REGIMES_BUY
        elif signal == "SELL":
            favorable_regimes = _FAVORABLE_REGIMES_SELL
        else:
            favorable_regimes = frozenset()   # HOLD: neutral, no special favored set
    # HOLD signals are not vetoed by regime (they fire on strength, not direction)
    if signal in ("BUY", "SELL"):
        veto = _VETO_REGIMES_BUY if signal == "BUY" else _VETO_REGIMES_SELL
    else:
        veto = frozenset()
    if regime_label in veto:
        return 0.0  # hard veto
    if regime_label in favorable_regimes:
        regime_contribution = 30.0
    elif regime_label in ("indeterminate", "", None):
        regime_contribution = 5.0
    else:
        regime_contribution = 12.0   # neutral regime

    # Breadth contribution (0–20)
    if signal == "BUY":
        aligned = breadth_state in _BREADTH_ALIGNED_BUY
    elif signal == "SELL":
        aligned = breadth_state in _BREADTH_ALIGNED_SELL
    else:
        aligned = True   # HOLD ignores breadth direction
    if require_breadth_alignment and not aligned:
        return 0.0   # hard gate if breadth misaligned
    breadth_contribution = 20.0 if aligned else 0.0

    raw = dau_contribution + regime_contribution + breadth_contribution
    return round(min(100.0, raw * ic_mult), 2)


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------

def deliver_webhook(url: str, payload: Dict[str, Any]) -> int:
    """POST payload to url. Returns HTTP status code (or 0 on connection error)."""
    try:
        resp = httpx.post(url, json=payload, timeout=_WEBHOOK_TIMEOUT_S)
        return resp.status_code
    except Exception as exc:
        logger.warning("alert.webhook.error url=%s err=%s", url, exc)
        return 0


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _load_active_rules() -> List[Dict[str, Any]]:
    if not db.db_read_enabled():
        return []
    rows = db.safe_fetchall(
        """
        SELECT rule_id, symbol, min_dau, signal_filter, favorable_regimes,
               require_breadth_alignment, min_conviction_score, webhook_url, meta,
               COALESCE(channel_type, 'generic') AS channel_type
        FROM signal_alert_rules
        WHERE is_active = true
        ORDER BY symbol, rule_id
        """,
        (),
    )
    return [
        {
            "rule_id": r[0], "symbol": r[1], "min_dau": float(r[2] or 65),
            "signal_filter": list(r[3] or []),
            "favorable_regimes": frozenset(r[4]) if r[4] else None,
            "require_breadth_alignment": bool(r[5]),
            "min_conviction_score": float(r[6] or 35),
            "webhook_url": r[7],
            "meta": r[8] or {},
            "channel_type": str(r[9] or "generic"),
        }
        for r in rows if r
    ]


def _load_axiom_scores_batch(
    symbols: List[str], as_of_date: dt.date
) -> Dict[str, Dict[str, Any]]:
    """Return {symbol: {dau, regime_label, deployability_tier, signal_label}}."""
    if not db.db_read_enabled() or not symbols:
        return {}
    rows = db.safe_fetchall(
        """
        SELECT
            a.symbol,
            (a.payload->>'deployable_alpha_utility')::numeric  AS dau,
            a.payload->>'regime_label'                          AS regime_label,
            a.payload->>'deployability_tier'                    AS deployability_tier,
            COALESCE(p.signal, 'UNKNOWN')                       AS signal_label,
            (a.payload->>'overall_confidence')::numeric         AS confidence
        FROM axiom_scores_daily a
        LEFT JOIN prosperity_signals_daily p
            ON p.symbol = a.symbol AND p.as_of = a.as_of_date
        WHERE a.symbol = ANY(%s)
          AND a.as_of_date = %s
        """,
        (symbols, as_of_date),
    )
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sym = row[0]
        result[sym] = {
            "dau":               float(row[1] or 0),
            "regime_label":      str(row[2] or "indeterminate"),
            "deployability_tier": str(row[3] or "monitor_only"),
            "signal_label":      str(row[4] or "UNKNOWN"),
            "confidence":        float(row[5] or 50),
        }
    return result


def _load_breadth_state(as_of_date: dt.date) -> str:
    if not db.db_read_enabled():
        return "UNKNOWN"
    row = db.safe_fetchone(
        """
        SELECT breadth_state
        FROM market_breadth_daily
        WHERE as_of_date = %s
        """,
        (as_of_date,),
    )
    return str(row[0]) if row and row[0] else "UNKNOWN"


def _load_ic_state(as_of_date: dt.date) -> str:
    if not db.db_read_enabled():
        return "INSUFFICIENT"
    row = db.safe_fetchone(
        """
        SELECT ic_state
        FROM signal_ic_daily
        WHERE score_field = 'composite' AND horizon_label = '21d'
          AND as_of_date <= %s
        ORDER BY as_of_date DESC
        LIMIT 1
        """,
        (as_of_date,),
    )
    return str(row[0]) if row and row[0] else "INSUFFICIENT"


def _already_fired(rule_id: str, as_of_date: dt.date) -> bool:
    if not db.db_read_enabled():
        return False
    row = db.safe_fetchone(
        "SELECT 1 FROM signal_alert_events WHERE rule_id=%s AND as_of_date=%s",
        (rule_id, as_of_date),
    )
    return row is not None


def _store_event(event: Dict[str, Any]) -> None:
    if not db.db_write_enabled():
        return
    db.safe_execute(
        """
        INSERT INTO signal_alert_events (
            event_id, rule_id, symbol, as_of_date, signal_label, dau,
            regime_label, breadth_state, ic_state, conviction_score,
            webhook_delivered, webhook_status_code, payload
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT (rule_id, as_of_date) DO UPDATE SET
            conviction_score      = EXCLUDED.conviction_score,
            webhook_delivered     = EXCLUDED.webhook_delivered,
            webhook_status_code   = EXCLUDED.webhook_status_code,
            payload               = EXCLUDED.payload
        """,
        (
            event["event_id"], event["rule_id"], event["symbol"],
            event["as_of_date"], event["signal_label"], event["dau"],
            event["regime_label"], event["breadth_state"], event["ic_state"],
            event["conviction_score"], event["webhook_delivered"],
            event["webhook_status_code"], json.dumps(event["payload"]),
        ),
    )


# ---------------------------------------------------------------------------
# Core scan
# ---------------------------------------------------------------------------

@dataclass
class AlertScanSummary:
    as_of_date: str
    rules_evaluated: int = 0
    fired: int = 0
    suppressed: int = 0
    already_fired_today: int = 0
    webhook_delivered: int = 0
    webhook_failed: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)


def run_alert_scan(as_of_date: dt.date) -> AlertScanSummary:
    summary = AlertScanSummary(as_of_date=as_of_date.isoformat())

    rules = _load_active_rules()
    if not rules:
        return summary

    symbols = list({r["symbol"] for r in rules})
    scores_batch  = _load_axiom_scores_batch(symbols, as_of_date)
    breadth_state = _load_breadth_state(as_of_date)
    ic_state      = _load_ic_state(as_of_date)

    for rule in rules:
        summary.rules_evaluated += 1
        rule_id = rule["rule_id"]
        symbol  = rule["symbol"]

        # Skip if already fired today
        if _already_fired(rule_id, as_of_date):
            summary.already_fired_today += 1
            continue

        scores = scores_batch.get(symbol)
        if not scores:
            summary.suppressed += 1
            continue

        dau            = scores["dau"]
        regime_label   = scores["regime_label"]
        signal_label   = scores["signal_label"]
        deploy_tier    = scores["deployability_tier"]

        # Optional signal filter
        if rule["signal_filter"] and signal_label not in rule["signal_filter"]:
            summary.suppressed += 1
            continue

        # Tier gate
        if deploy_tier in ("not_actionable",):
            summary.suppressed += 1
            continue

        conviction = compute_conviction_score(
            dau=dau,
            signal_label=signal_label,
            regime_label=regime_label,
            breadth_state=breadth_state,
            ic_state=ic_state,
            min_dau=rule["min_dau"],
            favorable_regimes=rule["favorable_regimes"],
            require_breadth_alignment=rule["require_breadth_alignment"],
        )

        if conviction < rule["min_conviction_score"]:
            summary.suppressed += 1
            continue

        # Build and store event
        event_id = str(uuid.uuid4())
        webhook_delivered = False
        webhook_status_code = None

        event_payload = {
            "symbol": symbol, "signal": signal_label, "dau": round(dau, 2),
            "regime": regime_label, "breadth_state": breadth_state,
            "ic_state": ic_state, "conviction_score": conviction,
            "as_of_date": as_of_date.isoformat(), "rule_id": rule_id,
        }

        # Deliver webhook — use Slack blocks format when channel_type is 'slack'
        if rule.get("webhook_url"):
            if rule.get("channel_type") == "slack":
                outbound = format_slack_blocks(event_payload)
            else:
                outbound = event_payload
            code = deliver_webhook(rule["webhook_url"], outbound)
            webhook_status_code = code
            webhook_delivered = 200 <= code < 300
            if webhook_delivered:
                summary.webhook_delivered += 1
            else:
                summary.webhook_failed += 1

        event = {
            "event_id": event_id, "rule_id": rule_id, "symbol": symbol,
            "as_of_date": as_of_date, "signal_label": signal_label,
            "dau": dau, "regime_label": regime_label,
            "breadth_state": breadth_state, "ic_state": ic_state,
            "conviction_score": conviction,
            "webhook_delivered": webhook_delivered,
            "webhook_status_code": webhook_status_code,
            "payload": event_payload,
        }
        _store_event(event)
        summary.fired += 1
        summary.events.append(event_payload)
        logger.info(
            "alert.fired symbol=%s signal=%s conviction=%.1f regime=%s breadth=%s ic=%s",
            symbol, signal_label, conviction, regime_label, breadth_state, ic_state,
        )

    return summary


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class AlertScanRequest(BaseModel):
    as_of_date: Optional[str] = None


class AlertRuleRequest(BaseModel):
    rule_id: Optional[str] = None         # omit to auto-generate
    symbol: str
    min_dau: float = Field(default=65.0, ge=0.0, le=100.0)
    signal_filter: Optional[List[str]] = None
    favorable_regimes: Optional[List[str]] = None
    require_breadth_alignment: bool = True
    min_conviction_score: float = Field(default=35.0, ge=0.0, le=100.0)
    webhook_url: Optional[str] = None
    channel_type: str = Field(default="generic", pattern="^(generic|slack)$")


@router.post("/alerts/daily-scan")
def alerts_daily_scan(req: AlertScanRequest) -> Dict[str, Any]:
    as_of_date = (
        dt.date.fromisoformat(req.as_of_date)
        if req.as_of_date else dt.date.today()
    )
    result = run_alert_scan(as_of_date)
    return {
        "status": "ok",
        "as_of_date": result.as_of_date,
        "rules_evaluated": result.rules_evaluated,
        "fired": result.fired,
        "suppressed": result.suppressed,
        "already_fired_today": result.already_fired_today,
        "webhook_delivered": result.webhook_delivered,
        "webhook_failed": result.webhook_failed,
        "events": result.events,
    }


@router.post("/alerts/rules")
def upsert_alert_rule(req: AlertRuleRequest) -> Dict[str, Any]:
    if not db.db_write_enabled():
        return {"status": "db_write_disabled"}

    rule_id = req.rule_id or str(uuid.uuid4())
    db.safe_execute(
        """
        INSERT INTO signal_alert_rules (
            rule_id, symbol, min_dau, signal_filter, favorable_regimes,
            require_breadth_alignment, min_conviction_score, webhook_url, channel_type, is_active
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,true)
        ON CONFLICT (rule_id) DO UPDATE SET
            symbol                    = EXCLUDED.symbol,
            min_dau                   = EXCLUDED.min_dau,
            signal_filter             = EXCLUDED.signal_filter,
            favorable_regimes         = EXCLUDED.favorable_regimes,
            require_breadth_alignment = EXCLUDED.require_breadth_alignment,
            min_conviction_score      = EXCLUDED.min_conviction_score,
            webhook_url               = EXCLUDED.webhook_url,
            channel_type              = EXCLUDED.channel_type,
            updated_at                = now()
        """,
        (
            rule_id, req.symbol, req.min_dau,
            req.signal_filter or None,
            req.favorable_regimes or None,
            req.require_breadth_alignment,
            req.min_conviction_score,
            req.webhook_url,
            req.channel_type,
        ),
    )
    return {"status": "ok", "rule_id": rule_id, "symbol": req.symbol}


@router.get("/alerts/recent")
def recent_alerts(
    symbol: Optional[str] = Query(default=None),
    days: int = Query(default=7, ge=1, le=90),
) -> Dict[str, Any]:
    if not db.db_read_enabled():
        return {"status": "db_read_disabled", "events": []}

    cutoff = dt.date.today() - dt.timedelta(days=days)
    if symbol:
        rows = db.safe_fetchall(
            """
            SELECT event_id, rule_id, symbol, as_of_date, signal_label,
                   dau, regime_label, breadth_state, ic_state, conviction_score,
                   webhook_delivered, webhook_status_code
            FROM signal_alert_events
            WHERE symbol = %s AND as_of_date >= %s
            ORDER BY as_of_date DESC
            LIMIT 200
            """,
            (symbol, cutoff),
        )
    else:
        rows = db.safe_fetchall(
            """
            SELECT event_id, rule_id, symbol, as_of_date, signal_label,
                   dau, regime_label, breadth_state, ic_state, conviction_score,
                   webhook_delivered, webhook_status_code
            FROM signal_alert_events
            WHERE as_of_date >= %s
            ORDER BY as_of_date DESC
            LIMIT 200
            """,
            (cutoff,),
        )
    events = [
        {
            "event_id": r[0], "rule_id": r[1], "symbol": r[2],
            "as_of_date": str(r[3]), "signal_label": r[4],
            "dau": float(r[5]) if r[5] is not None else None,
            "regime_label": r[6], "breadth_state": r[7], "ic_state": r[8],
            "conviction_score": float(r[9]) if r[9] is not None else None,
            "webhook_delivered": r[10], "webhook_status_code": r[11],
        }
        for r in rows if r
    ]
    return {"status": "ok", "events": events, "count": len(events)}
