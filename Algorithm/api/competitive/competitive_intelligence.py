"""Phase 18.1: Competitive Intelligence Engine."""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from api import db
from api.assistant.phase3.common import clamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static sector groups — guaranteed competitor identification without DB
# ---------------------------------------------------------------------------

AXIOM_SECTOR_GROUPS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AVGO", "ACN"],
    "Consumer Discretionary": ["TSLA", "AMZN", "HD", "MCD"],
    "Consumer Staples": ["WMT", "COST", "PG", "PEP", "KO"],
    "Healthcare": ["JNJ", "UNH", "MRK", "ABBV", "LLY", "TMO", "DHR"],
    "Financials": ["JPM", "V", "MA"],
    "Energy": ["CVX", "XOM"],
    "Utilities": ["NEE"],
}

# Reverse mapping: symbol -> sector
_SYMBOL_TO_SECTOR: Dict[str, str] = {
    sym: sector
    for sector, symbols in AXIOM_SECTOR_GROUPS.items()
    for sym in symbols
}


def get_sector_for_symbol(symbol: str) -> str:
    """Return sector name from static groups; 'unknown' if not found."""
    return _SYMBOL_TO_SECTOR.get(symbol.upper(), "unknown")


def get_static_competitors(symbol: str, max_competitors: int = 10) -> List[str]:
    """Return sector peers from static groups (no DB required)."""
    sector = _SYMBOL_TO_SECTOR.get(symbol.upper())
    if not sector:
        return []
    return [s for s in AXIOM_SECTOR_GROUPS[sector] if s != symbol.upper()][:max_competitors]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompetitorProfile:
    symbol: str
    competitor_symbol: str
    relationship_type: str
    competitive_distance: float

    dau_advantage: float
    eis_advantage: float
    caps_advantage: float
    fragility_advantage: float
    momentum_advantage: float

    competitive_position_score: float
    competitive_trend: str
    key_advantage: str
    key_vulnerability: str


@dataclass
class CompetitiveIntelligenceReport:
    symbol: str
    as_of_date: dt.date
    sector: str
    competitor_count: int
    competitors: List[CompetitorProfile]

    sector_dau_rank: int
    sector_eis_rank: int
    sector_caps_rank: int
    sector_size: int

    competitive_position: str
    market_share_momentum: str
    best_competitor: str
    most_dangerous_competitor: str

    competitive_intelligence_score: float


# ---------------------------------------------------------------------------
# Pure helpers (exposed for testing)
# ---------------------------------------------------------------------------

def _classify_position(rank: int, sector_size: int) -> str:
    if sector_size <= 0:
        return "unknown"
    if rank <= 2:
        return "leader"
    pct = rank / sector_size
    if pct <= 0.25:
        return "strong"
    if pct <= 0.75:
        return "middle"
    if pct <= 0.90:
        return "lagging"
    return "tail"


def _competitive_score(rank: int, sector_size: int) -> float:
    if sector_size <= 0:
        return 0.0
    return round(100.0 * (1.0 - rank / sector_size), 2)


def _extract_axiom_scores(payload: Dict[str, Any]) -> Dict[str, float]:
    engines = payload.get("engine_scores") or {}
    fundamental = engines.get("fundamental_reality") or {}
    fund_comps = fundamental.get("components") or {}
    fragility = engines.get("critical_fragility") or {}
    flow = engines.get("flow_transmission") or {}
    return {
        "dau": float(payload.get("deployable_alpha_utility") or 50.0),
        "eis": float(fund_comps.get("eis_component") or 50.0),
        "caps": float(fund_comps.get("caps_component") or 50.0),
        "fragility": float(fragility.get("score") or 50.0),
        "flow": float(flow.get("score") or 50.0),
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_competitor_profile(
    symbol: str,
    competitor: str,
    symbol_payload: Dict[str, Any],
    competitor_payload: Dict[str, Any],
    relationship_type: str = "sector_peer",
) -> CompetitorProfile:
    sym = _extract_axiom_scores(symbol_payload)
    comp = _extract_axiom_scores(competitor_payload)

    dau_adv = sym["dau"] - comp["dau"]
    eis_adv = sym["eis"] - comp["eis"]
    caps_adv = sym["caps"] - comp["caps"]
    frag_adv = comp["fragility"] - sym["fragility"]   # higher = target is safer
    mom_adv = sym["flow"] - comp["flow"]

    pos_score = clamp(
        50.0
        + dau_adv * 0.35
        + eis_adv * 0.20
        + caps_adv * 0.20
        + frag_adv * 0.15
        + mom_adv * 0.10,
        0.0, 100.0,
    )

    advantages = {
        "dau": dau_adv,
        "eis": eis_adv,
        "caps": caps_adv,
        "fragility": frag_adv,
        "momentum": mom_adv,
    }
    positive = {k: v for k, v in advantages.items() if v > 0}
    negative = {k: v for k, v in advantages.items() if v < 0}

    key_adv = max(positive, key=lambda k: abs(positive[k])) if positive else "none"
    key_vuln = min(negative, key=negative.__getitem__) if negative else "none"

    dist = clamp(abs(dau_adv) / 100.0, 0.0, 1.0)

    return CompetitorProfile(
        symbol=symbol,
        competitor_symbol=competitor,
        relationship_type=relationship_type,
        competitive_distance=round(dist, 3),
        dau_advantage=round(dau_adv, 2),
        eis_advantage=round(eis_adv, 2),
        caps_advantage=round(caps_adv, 2),
        fragility_advantage=round(frag_adv, 2),
        momentum_advantage=round(mom_adv, 2),
        competitive_position_score=round(pos_score, 2),
        competitive_trend="stable",
        key_advantage=key_adv,
        key_vulnerability=key_vuln,
    )


def compute_market_share_momentum(
    symbol: str,
    competitors: List[str],
    axiom_payloads: Dict[str, Any],
) -> str:
    sym_payload = axiom_payloads.get(symbol, {})
    sym_growth = float(sym_payload.get("revenue_growth_ttm", 0.0))

    comp_growths = [
        float(axiom_payloads.get(c, {}).get("revenue_growth_ttm", 0.0))
        for c in competitors
    ]
    if not comp_growths:
        return "stable"

    comp_avg = sum(comp_growths) / len(comp_growths)

    if sym_growth > comp_avg + 0.05:
        return "gaining"
    if sym_growth < comp_avg - 0.05:
        return "losing"
    return "stable"


def identify_competitors(
    symbol: str,
    sector: str,
    as_of_date: dt.date,
    max_competitors: int = 10,
) -> List[str]:
    competitors: List[str] = []

    if db.db_enabled():
        # Method 1: symbol_linkage table
        try:
            rows = db.safe_fetchall(
                """
                SELECT CASE WHEN symbol_a = %s THEN symbol_b ELSE symbol_a END AS peer
                  FROM symbol_linkage
                 WHERE (symbol_a = %s OR symbol_b = %s)
                   AND link_type IN ('sector_peer', 'competitor')
                 LIMIT %s
                """,
                (symbol, symbol, symbol, max_competitors),
            ) or []
            competitors = [str(r[0]) for r in rows if r[0] != symbol]
        except Exception:
            pass

        # Method 2: DB sector fallback
        if len(competitors) < 3:
            try:
                rows = db.safe_fetchall(
                    """
                    SELECT DISTINCT symbol
                      FROM axiom_scores_daily
                     WHERE payload->>'sector' = %s
                       AND symbol != %s
                       AND as_of_date >= %s - interval '7 days'
                     ORDER BY symbol LIMIT %s
                    """,
                    (sector, symbol, as_of_date, max_competitors),
                ) or []
                for r in rows:
                    sym = str(r[0])
                    if sym not in competitors:
                        competitors.append(sym)
            except Exception:
                pass

    # Method 3: static sector groups — always works without DB
    if len(competitors) < 2:
        static = get_static_competitors(symbol, max_competitors)
        for s in static:
            if s not in competitors:
                competitors.append(s)

    return competitors[:max_competitors]


def generate_competitive_intelligence_report(
    symbol: str,
    as_of_date: Optional[dt.date] = None,
) -> CompetitiveIntelligenceReport:
    as_of_date = as_of_date or dt.date.today()

    default = CompetitiveIntelligenceReport(
        symbol=symbol,
        as_of_date=as_of_date,
        sector="unknown",
        competitor_count=0,
        competitors=[],
        sector_dau_rank=1,
        sector_eis_rank=1,
        sector_caps_rank=1,
        sector_size=1,
        competitive_position="leader",
        market_share_momentum="stable",
        best_competitor="unknown",
        most_dangerous_competitor="unknown",
        competitive_intelligence_score=50.0,
    )

    if not db.db_enabled():
        static_sector = get_sector_for_symbol(symbol)
        static_peers = get_static_competitors(symbol)
        profiles = [
            compute_competitor_profile(symbol, peer, {}, {})
            for peer in static_peers[:5]
        ]
        return CompetitiveIntelligenceReport(
            symbol=symbol,
            as_of_date=as_of_date,
            sector=static_sector,
            competitor_count=len(profiles),
            competitors=profiles,
            sector_dau_rank=1,
            sector_eis_rank=1,
            sector_caps_rank=1,
            sector_size=len(AXIOM_SECTOR_GROUPS.get(static_sector, [symbol])),
            competitive_position="leader",
            market_share_momentum="stable",
            best_competitor=static_peers[0] if static_peers else "unknown",
            most_dangerous_competitor=static_peers[0] if static_peers else "unknown",
            competitive_intelligence_score=50.0,
        )

    try:
        row = db.safe_fetchone(
            """
            SELECT payload, payload->>'sector' AS sector
              FROM axiom_scores_daily
             WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1
            """,
            (symbol,),
        )
        if not row or not row[0]:
            return default

        symbol_payload = row[0] if isinstance(row[0], dict) else {}
        sector = str(row[1] or "unknown")

        competitors = identify_competitors(symbol, sector, as_of_date)
        if not competitors:
            return default

        # Fetch all sector payloads for ranking
        sector_rows = db.safe_fetchall(
            """
            SELECT symbol, payload
              FROM axiom_scores_daily
             WHERE payload->>'sector' = %s
               AND as_of_date >= %s - interval '7 days'
            """,
            (sector, as_of_date),
        ) or []

        all_payloads: Dict[str, Dict] = {symbol: symbol_payload}
        for sr in sector_rows:
            sym_key = str(sr[0])
            p = sr[1] if isinstance(sr[1], dict) else {}
            all_payloads[sym_key] = p

        all_symbols_dau = [
            (s, float(p.get("deployable_alpha_utility") or 0.0))
            for s, p in all_payloads.items()
        ]
        all_symbols_dau.sort(key=lambda x: x[1], reverse=True)
        sector_size = len(all_symbols_dau)

        symbol_dau = float(symbol_payload.get("deployable_alpha_utility") or 0.0)
        dau_rank = next(
            (i + 1 for i, (s, _) in enumerate(all_symbols_dau) if s == symbol),
            sector_size,
        )
        eis_rank = 1
        caps_rank = 1

        profiles = [
            compute_competitor_profile(symbol, c, symbol_payload, all_payloads.get(c, {}))
            for c in competitors
            if c in all_payloads
        ]

        competitive_pos = _classify_position(dau_rank, sector_size)
        comp_score = _competitive_score(dau_rank, sector_size)

        all_payloads_for_momentum = {
            s: {"revenue_growth_ttm": float(p.get("revenue_growth_ttm", 0.0))}
            for s, p in all_payloads.items()
        }
        momentum = compute_market_share_momentum(symbol, competitors, all_payloads_for_momentum)

        best_comp = max(competitors, key=lambda c: float(all_payloads.get(c, {}).get("deployable_alpha_utility") or 0.0)) if competitors else "unknown"
        danger_comp = best_comp

        return CompetitiveIntelligenceReport(
            symbol=symbol,
            as_of_date=as_of_date,
            sector=sector,
            competitor_count=len(profiles),
            competitors=profiles,
            sector_dau_rank=dau_rank,
            sector_eis_rank=eis_rank,
            sector_caps_rank=caps_rank,
            sector_size=sector_size,
            competitive_position=competitive_pos,
            market_share_momentum=momentum,
            best_competitor=best_comp,
            most_dangerous_competitor=danger_comp,
            competitive_intelligence_score=comp_score,
        )

    except Exception as exc:
        logger.warning("competitive_report_failed symbol=%s err=%s", symbol, exc)
        return default
