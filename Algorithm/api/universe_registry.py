"""AXIOM Universe Registry — manages 10,000 symbol universe.

Source of truth for which symbols AXIOM covers and at what scoring depth.
Replaces the hardcoded AXIOM_UNIVERSE list with a DB-driven registry.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional

from api import db

logger = logging.getLogger(__name__)

# ── Tier thresholds ─────────────────────────────────────────────────────────
TIER_1_MIN_MARKET_CAP = 10_000_000_000   # $10B+
TIER_1_MIN_VOLUME     = 1_000_000        # 1M+ avg daily volume
TIER_2_MIN_MARKET_CAP = 2_000_000_000    # $2B+
TIER_2_MIN_VOLUME     = 500_000          # 500K+ avg daily volume

# ── Bootstrap list (used when DB not available or not yet seeded) ────────────
# Matches the current production AXIOM_UNIVERSE exactly so the pipeline
# continues to work before the registry is seeded.
BOOTSTRAP_TIER1: List[str] = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "JPM", "JNJ", "UNH",
    "V", "PG", "HD", "CVX", "MRK", "ABBV", "LLY", "PEP", "KO", "MCD",
    "WMT", "COST", "ACN", "TMO", "DHR", "NEE", "XOM", "AVGO", "MA", "GOOGL",
    # S&P 500 additions for bootstrap
    "BRK-B", "RTX", "QCOM", "AMD", "GS", "HON", "IBM", "CAT", "AMGN", "LOW",
    "T", "SPGI", "INTU", "BLK", "ISRG", "GILD", "SYK", "ADI", "ELV", "DE",
    "REGN", "VRTX", "PLD", "ZTS", "CB", "MDLZ", "CI", "MMC", "SO", "DUK",
    "NKE", "CRM", "TXN", "NFLX", "LIN", "PM", "HUM", "CL", "TGT", "BMY",
    "PNC", "AON", "WM", "EMR", "NOC", "GM", "F", "USB", "GE", "TJX",
    "ETN", "NSC", "FCX", "APD", "ECL", "D", "EW", "FIS", "PSA", "AFL",
    "SHW", "CARR", "AIG", "HCA", "BA", "LMT", "BAC", "WFC", "MS",
]


def _assign_tier(market_cap_usd: Optional[int], avg_daily_volume: Optional[int]) -> int:
    """Assign tier based on market cap and daily volume."""
    cap = market_cap_usd or 0
    vol = avg_daily_volume or 0
    if cap >= TIER_1_MIN_MARKET_CAP and vol >= TIER_1_MIN_VOLUME:
        return 1
    if cap >= TIER_2_MIN_MARKET_CAP or vol >= TIER_2_MIN_VOLUME:
        return 2
    return 3


def get_symbols_by_tier(tier: int) -> List[str]:
    """Get active symbols for a specific tier from the DB registry."""
    if not db.db_enabled():
        return BOOTSTRAP_TIER1[:30] if tier == 1 else []
    try:
        rows = db.safe_fetchall(
            "SELECT symbol FROM axiom_universe_registry "
            "WHERE tier = %s AND active = TRUE "
            "ORDER BY market_cap_usd DESC NULLS LAST",
            (tier,),
        )
        return [r[0] for r in rows] if rows else (BOOTSTRAP_TIER1[:30] if tier == 1 else [])
    except Exception as exc:
        logger.warning("universe_registry.get_tier_failed tier=%d err=%s", tier, exc)
        return BOOTSTRAP_TIER1[:30] if tier == 1 else []


def get_all_active_symbols() -> List[str]:
    """Get all active symbols across all tiers, ordered by tier then market cap."""
    if not db.db_enabled():
        return list(BOOTSTRAP_TIER1)
    try:
        rows = db.safe_fetchall(
            "SELECT symbol FROM axiom_universe_registry "
            "WHERE active = TRUE "
            "ORDER BY tier ASC, market_cap_usd DESC NULLS LAST",
        )
        return [r[0] for r in rows] if rows else list(BOOTSTRAP_TIER1)
    except Exception as exc:
        logger.warning("universe_registry.get_all_failed err=%s", exc)
        return list(BOOTSTRAP_TIER1)


def get_tier_for_symbol(symbol: str) -> int:
    """Get the tier for a specific symbol. Returns 3 if not found."""
    if not db.db_enabled():
        return 1 if symbol in BOOTSTRAP_TIER1 else 3
    try:
        row = db.safe_fetchone(
            "SELECT tier FROM axiom_universe_registry "
            "WHERE symbol = %s AND active = TRUE",
            (symbol,),
        )
        return int(row[0]) if row else 3
    except Exception:
        return 3


def upsert_symbol(
    symbol: str,
    company_name: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    country: str = "US",
    exchange: Optional[str] = None,
    market_cap_usd: Optional[int] = None,
    avg_daily_volume: Optional[int] = None,
) -> int:
    """Insert or update a symbol in the registry. Returns the assigned tier."""
    tier = _assign_tier(market_cap_usd, avg_daily_volume)
    if not db.db_enabled():
        return tier
    try:
        db.safe_execute(
            """
            INSERT INTO axiom_universe_registry
                (symbol, company_name, sector, industry, country, exchange,
                 market_cap_usd, avg_daily_volume, tier, last_validated, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (symbol) DO UPDATE SET
                company_name     = COALESCE(EXCLUDED.company_name,     axiom_universe_registry.company_name),
                sector           = COALESCE(EXCLUDED.sector,           axiom_universe_registry.sector),
                industry         = COALESCE(EXCLUDED.industry,         axiom_universe_registry.industry),
                market_cap_usd   = COALESCE(EXCLUDED.market_cap_usd,   axiom_universe_registry.market_cap_usd),
                avg_daily_volume = COALESCE(EXCLUDED.avg_daily_volume, axiom_universe_registry.avg_daily_volume),
                tier             = EXCLUDED.tier,
                last_validated   = EXCLUDED.last_validated,
                updated_at       = now()
            """,
            (symbol, company_name, sector, industry, country, exchange,
             market_cap_usd, avg_daily_volume, tier, dt.date.today()),
        )
    except Exception as exc:
        logger.warning("universe_registry.upsert_failed symbol=%s err=%s", symbol, exc)
    return tier


def sync_symbol_metadata_from_yfinance(symbol: str) -> bool:
    """Fetch metadata for a symbol from yfinance and upsert into registry.

    Returns True if symbol is valid and was upserted.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        # Validate symbol has price data
        has_price = (
            info.get("regularMarketPrice") is not None
            or info.get("currentPrice") is not None
            or info.get("navPrice") is not None
        )
        if not has_price and not info.get("longName"):
            return False

        market_cap = info.get("marketCap")
        volume = info.get("averageVolume") or info.get("averageDailyVolume10Day")

        upsert_symbol(
            symbol=symbol,
            company_name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            country=info.get("country", "US"),
            exchange=info.get("exchange"),
            market_cap_usd=int(market_cap) if market_cap else None,
            avg_daily_volume=int(volume) if volume else None,
        )
        return True
    except Exception as exc:
        logger.debug("sync_metadata_failed symbol=%s err=%s", symbol, exc)
        return False


def get_registry_stats() -> Dict[str, Any]:
    """Return counts and top symbols for the universe stats endpoint."""
    if not db.db_enabled():
        return {
            "total_symbols": len(BOOTSTRAP_TIER1),
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "last_seeded": None,
            "top_10_by_market_cap": [],
            "source": "bootstrap_fallback",
        }
    try:
        counts = db.safe_fetchall(
            "SELECT tier, COUNT(*) FROM axiom_universe_registry "
            "WHERE active = TRUE GROUP BY tier ORDER BY tier",
        )
        tier_map = {int(r[0]): int(r[1]) for r in (counts or [])}

        total_row = db.safe_fetchone(
            "SELECT COUNT(*), MAX(last_validated) FROM axiom_universe_registry WHERE active = TRUE",
        )
        total = int(total_row[0]) if total_row else 0
        last_validated = total_row[1].isoformat() if (total_row and total_row[1]) else None

        top10 = db.safe_fetchall(
            "SELECT symbol, company_name, market_cap_usd, tier "
            "FROM axiom_universe_registry "
            "WHERE active = TRUE AND market_cap_usd IS NOT NULL "
            "ORDER BY market_cap_usd DESC LIMIT 10",
        )
        top10_list = [
            {"symbol": r[0], "company_name": r[1], "market_cap_usd": r[2], "tier": r[3]}
            for r in (top10 or [])
        ]
        return {
            "total_symbols": total,
            "tier1_count": tier_map.get(1, 0),
            "tier2_count": tier_map.get(2, 0),
            "tier3_count": tier_map.get(3, 0),
            "last_seeded": last_validated,
            "top_10_by_market_cap": top10_list,
        }
    except Exception as exc:
        logger.warning("universe_registry.stats_failed err=%s", exc)
        return {
            "total_symbols": 0,
            "tier1_count": 0,
            "tier2_count": 0,
            "tier3_count": 0,
            "last_seeded": None,
            "top_10_by_market_cap": [],
            "error": str(exc),
        }
