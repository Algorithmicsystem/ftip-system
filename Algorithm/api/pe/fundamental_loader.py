"""Phase 25: Real fundamental data loader for PE/SMB analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def load_company_fundamentals(symbol: str) -> Dict[str, Any]:
    """Load real fundamental data for PE analysis. Combines yfinance quarterly data with AXIOM scores."""
    result: Dict[str, Any] = {
        "symbol": symbol,
        "market_cap": None,
        "sector": "Unknown",
        "employees": None,
        "revenue_ttm": None,
        "gross_margin": None,
        "op_margin": None,
        "fcf_margin": None,
        "revenue_growth_yoy": None,
        "debt_to_equity": None,
        "return_on_assets": None,
        "axiom_dau": None,
        "axiom_signal": None,
        "axiom_eis": None,
        "axiom_caps": None,
    }

    # --- yfinance ---
    try:
        import importlib.util
        if importlib.util.find_spec("yfinance"):
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            result["market_cap"] = info.get("marketCap")
            result["sector"] = info.get("sector") or info.get("industryDisp") or "Unknown"
            result["employees"] = info.get("fullTimeEmployees")
            result["revenue_ttm"] = info.get("totalRevenue")
            result["gross_margin"] = info.get("grossMargins")
            result["op_margin"] = info.get("operatingMargins")
            result["fcf_margin"] = info.get("freeCashflow") and info.get("totalRevenue") and (
                info["freeCashflow"] / info["totalRevenue"]
            )
            result["debt_to_equity"] = info.get("debtToEquity")
            result["return_on_assets"] = info.get("returnOnAssets")

            # Revenue growth: use yfinance earnings quarterly trend
            try:
                qf = ticker.quarterly_financials
                if qf is not None and not qf.empty and len(qf.columns) >= 5:
                    rev_row = qf.loc["Total Revenue"] if "Total Revenue" in qf.index else None
                    if rev_row is not None:
                        recent_ttm = float(rev_row.iloc[:4].sum())
                        prior_ttm = float(rev_row.iloc[1:5].sum())
                        if prior_ttm > 0:
                            result["revenue_growth_yoy"] = (recent_ttm - prior_ttm) / prior_ttm
            except Exception:
                pass

            # FCF margin fallback from cash flow
            if result["fcf_margin"] is None:
                try:
                    cf = ticker.quarterly_cashflow
                    if cf is not None and not cf.empty:
                        fcf_row = cf.loc["Free Cash Flow"] if "Free Cash Flow" in cf.index else None
                        if fcf_row is not None and result["revenue_ttm"] and result["revenue_ttm"] > 0:
                            fcf_ttm = float(fcf_row.iloc[:4].sum())
                            result["fcf_margin"] = fcf_ttm / result["revenue_ttm"]
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("fundamental_loader.yfinance symbol=%s err=%s", symbol, exc)

    # --- AXIOM DB scores ---
    try:
        from api import db
        if db.db_read_enabled():
            row = db.safe_fetchone(
                "SELECT payload FROM axiom_scores_daily WHERE symbol = %s ORDER BY as_of_date DESC LIMIT 1",
                (symbol,),
            )
            if row and row[0]:
                import json as _json
                p = row[0] if isinstance(row[0], dict) else _json.loads(row[0])
                result["axiom_dau"] = p.get("deployable_alpha_utility")
                sig_val = result["axiom_dau"]
                if sig_val is not None:
                    result["axiom_signal"] = "BUY" if sig_val >= 65 else ("SELL" if sig_val <= 40 else "HOLD")
                eng = p.get("engine_scores") or {}
                fr = (eng.get("fundamental_reality") or {}).get("components") or {}
                result["axiom_eis"] = fr.get("eis_component")
                result["axiom_caps"] = fr.get("caps_component")
                # Fill gaps from AXIOM meta
                meta = p.get("symbol_meta") or {}
                if result["sector"] == "Unknown" and meta.get("sector"):
                    result["sector"] = meta["sector"]
    except Exception as exc:
        logger.debug("fundamental_loader.axiom symbol=%s err=%s", symbol, exc)

    return result
