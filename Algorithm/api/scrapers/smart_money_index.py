"""Smart Money Index (SMI) aggregator.

Combines four alpha signals into a single 0-100 composite score:

  SMI = ICS × 0.35 + CIS × 0.25 + DPS × 0.25 + MCS × 0.15

  ICS  Insider Confidence Score      (SEC EDGAR Form 4)
  CIS  Congress Influence Score      (housestockwatcher.com)
  DPS  Dark Pool Score               (FINRA OTC)
  MCS  Management Confidence Score   (SEC 8-K earnings release)

Divergence detection:
  SMI > 60 AND DAU < 40 → "Institutional Accumulation Divergence"
  SMI < 40 AND DAU > 60 → "Smart Money Distribution Signal"
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ICS_WEIGHT = 0.35
_CIS_WEIGHT = 0.25
_DPS_WEIGHT = 0.25
_MCS_WEIGHT = 0.15

_SMI_BUY_THRESHOLD = 60.0
_SMI_SELL_THRESHOLD = 40.0
_DAU_BUY_THRESHOLD = 60.0
_DAU_SELL_THRESHOLD = 40.0


def compute_smart_money_index(
    symbol: str,
    dau: Optional[float] = None,
    days_back: int = 90,
    prefetched_congress_trades: list = None,
    prefetched_dark_pool: Optional[Dict[str, Any]] = None,
    prefetched_insider_transactions: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute the Smart Money Index for *symbol*.

    *dau* (Deployable Alpha Utility) is optional but required for divergence
    detection.  If omitted the divergence fields will always be False.

    Returns:
      symbol, smi_score, smi_signal, ics_score, cis_score, dps_score,
      mcs_score, divergence_detected, divergence_type, key_insight,
      component_details
    """
    # ---- ICS: Insider Confidence Score ---------------------------------
    ics_score = 50.0
    ics_detail: Dict[str, Any] = {}
    try:
        from api.scrapers.sec_insider import compute_insider_score, fetch_insider_transactions
        txns = (
            prefetched_insider_transactions
            if prefetched_insider_transactions is not None
            else fetch_insider_transactions(symbol, days_back=days_back)
        )
        ics_data = compute_insider_score(symbol, txns)
        ics_score = float(ics_data.get("ics_score") or 50.0)
        ics_detail = {
            "net_signal": ics_data.get("net_signal"),
            "buy_count": ics_data.get("buy_count"),
            "sell_count": ics_data.get("sell_count"),
            "officer_buy_count": ics_data.get("officer_buy_count"),
        }
    except Exception as exc:
        logger.debug("smi.ics_failed symbol=%s error=%s", symbol, exc)

    # ---- CIS: Congress Influence Score ---------------------------------
    cis_score = 50.0
    cis_detail: Dict[str, Any] = {}
    try:
        from api.scrapers.congress_trading import compute_congress_score, fetch_recent_congress_trades
        trades = prefetched_congress_trades
        if trades is None:
            trades = fetch_recent_congress_trades(days_back=days_back)
        cis_data = compute_congress_score(symbol, trades)
        cis_score = float(cis_data.get("congress_score") or 50.0)
        cis_detail = {
            "net_signal": cis_data.get("net_signal"),
            "buy_count": cis_data.get("buy_count"),
            "sell_count": cis_data.get("sell_count"),
        }
    except Exception as exc:
        logger.debug("smi.cis_failed symbol=%s error=%s", symbol, exc)

    # ---- DPS: Dark Pool Score ------------------------------------------
    dps_score = 50.0
    dps_detail: Dict[str, Any] = {}
    try:
        from api.scrapers.finra_dark_pool import fetch_finra_otc_data
        dps_data = (
            prefetched_dark_pool
            if prefetched_dark_pool is not None
            else fetch_finra_otc_data(symbol)
        )
        dps_score = float(dps_data.get("dark_pool_score") or 50.0)
        dps_detail = {
            "dark_pool_volume_pct": dps_data.get("dark_pool_volume_pct"),
            "signal": dps_data.get("signal"),
        }
    except Exception as exc:
        logger.debug("smi.dps_failed symbol=%s error=%s", symbol, exc)

    # ---- MCS: Management Confidence Score ------------------------------
    mcs_score = 50.0
    mcs_detail: Dict[str, Any] = {}
    try:
        from api.scrapers.earnings_transcript import fetch_transcript_sentiment
        mcs_data = fetch_transcript_sentiment(symbol, days_back=days_back)
        mcs_score = float(mcs_data.get("mcs_score") or 50.0)
        mcs_detail = {
            "filing_date": mcs_data.get("filing_date"),
            "positive_hits": mcs_data.get("positive_hits"),
            "negative_hits": mcs_data.get("negative_hits"),
        }
    except Exception as exc:
        logger.debug("smi.mcs_failed symbol=%s error=%s", symbol, exc)

    # ---- Composite SMI -------------------------------------------------
    smi_raw = (
        ics_score * _ICS_WEIGHT
        + cis_score * _CIS_WEIGHT
        + dps_score * _DPS_WEIGHT
        + mcs_score * _MCS_WEIGHT
    )
    smi_score = round(min(max(smi_raw, 0.0), 100.0), 2)

    if smi_score >= _SMI_BUY_THRESHOLD:
        smi_signal = "ACCUMULATION"
    elif smi_score <= _SMI_SELL_THRESHOLD:
        smi_signal = "DISTRIBUTION"
    else:
        smi_signal = "NEUTRAL"

    # ---- Divergence detection ------------------------------------------
    divergence_detected = False
    divergence_type = ""
    key_insight = _build_key_insight(smi_score, smi_signal, ics_score, cis_score, dps_score, mcs_score)

    if dau is not None:
        if smi_score > _SMI_BUY_THRESHOLD and dau < _DAU_SELL_THRESHOLD:
            divergence_detected = True
            divergence_type = "Institutional Accumulation Divergence"
            key_insight = (
                f"Smart money is accumulating {symbol} (SMI {smi_score:.0f}) "
                f"while public signals show weakness (DAU {dau:.0f}). "
                "Insiders and institutions may be positioning ahead of a catalyst."
            )
        elif smi_score < _SMI_SELL_THRESHOLD and dau > _DAU_BUY_THRESHOLD:
            divergence_detected = True
            divergence_type = "Smart Money Distribution Signal"
            key_insight = (
                f"Smart money is distributing {symbol} (SMI {smi_score:.0f}) "
                f"despite strong public signals (DAU {dau:.0f}). "
                "Institutional selling into strength may precede a reversal."
            )

    return {
        "symbol": symbol,
        "smi_score": smi_score,
        "smi_signal": smi_signal,
        "ics_score": round(ics_score, 2),
        "cis_score": round(cis_score, 2),
        "dps_score": round(dps_score, 2),
        "mcs_score": round(mcs_score, 2),
        "divergence_detected": divergence_detected,
        "divergence_type": divergence_type,
        "key_insight": key_insight,
        "component_details": {
            "insider": ics_detail,
            "congress": cis_detail,
            "dark_pool": dps_detail,
            "management": mcs_detail,
        },
    }


def _build_key_insight(
    smi: float, signal: str,
    ics: float, cis: float, dps: float, mcs: float,
) -> str:
    strongest = max(
        [("insider buying", ics), ("congressional trading", cis),
         ("dark pool activity", dps), ("management tone", mcs)],
        key=lambda x: x[1],
    )
    weakest = min(
        [("insider buying", ics), ("congressional trading", cis),
         ("dark pool activity", dps), ("management tone", mcs)],
        key=lambda x: x[1],
    )

    if signal == "ACCUMULATION":
        return (
            f"Smart money signals are bullish (SMI {smi:.0f}). "
            f"Strongest signal: {strongest[0]} ({strongest[1]:.0f}). "
            f"Weakest: {weakest[0]} ({weakest[1]:.0f})."
        )
    if signal == "DISTRIBUTION":
        return (
            f"Smart money signals are bearish (SMI {smi:.0f}). "
            f"Most negative: {weakest[0]} ({weakest[1]:.0f}). "
            f"Best remaining: {strongest[0]} ({strongest[1]:.0f})."
        )
    return (
        f"Mixed smart money signals (SMI {smi:.0f}). "
        f"Strongest: {strongest[0]} ({strongest[1]:.0f}), "
        f"weakest: {weakest[0]} ({weakest[1]:.0f})."
    )
