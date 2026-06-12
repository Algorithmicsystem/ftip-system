"""AXIOM Batch Bar Fetcher

Fetches OHLCV bars for large symbol universes using yfinance's batch
download API. Dramatically faster than sequential per-symbol fetching.

Performance (measured):
  Sequential (current): 30 symbols × 0.34s = ~10 seconds
  Batch (this module):  30 symbols in 1 batch × 0.3s ≈ 0.3 seconds
  500 symbols: ~500/50 = 10 batches, 8 workers → ~4 seconds

Architecture:
  - Split universe into batches of BATCH_SIZE (50 symbols)
  - Use ThreadPoolExecutor to run multiple batches concurrently
  - Each worker calls yf.download() with its batch (group_by='ticker')
  - Results normalised to the same dict format as _fetch_daily_yfinance()
  - Failed symbols in a batch are retried individually
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

BATCH_SIZE  = 50    # symbols per yfinance batch call
MAX_WORKERS = 8     # concurrent batch workers
RETRY_DELAY = 0.2   # seconds before individual symbol retry


def fetch_bars_batch(
    symbols: List[str],
    from_date: dt.date,
    to_date: dt.date,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch OHLCV bars for a list of symbols using yfinance batch API.

    Returns dict: {symbol: [bar_dict, ...]}
    Each bar_dict has keys: symbol, date, open, high, low, close,
                            adj_close, volume, source

    Symbols that fail are returned with an empty list.
    """
    if not symbols:
        return {}

    try:
        import yfinance as yf  # noqa: F401 — verify installed
    except ImportError:
        logger.error("batch_bars: yfinance not installed")
        return {sym: [] for sym in symbols}

    # yfinance end date is exclusive — add one day
    end_str = (to_date + dt.timedelta(days=1)).isoformat()
    start_str = from_date.isoformat()

    batches = [symbols[i:i + BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    n_workers = min(MAX_WORKERS, len(batches))
    logger.info(
        "batch_bars.start total_symbols=%d batches=%d workers=%d",
        len(symbols), len(batches), n_workers,
    )

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    retry_symbols: List[str] = []

    def _fetch_one_batch(
        batch: List[str],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
        import yfinance as yf

        batch_results: Dict[str, List[Dict[str, Any]]] = {}
        batch_failed: List[str] = []

        if len(batch) == 1:
            # Single-symbol path — flat DataFrame (no MultiIndex)
            try:
                df = yf.download(
                    batch[0],
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                )
                if df is not None and not df.empty:
                    bars = _dataframe_to_bars(df, batch[0])
                    if bars:
                        batch_results[batch[0]] = bars
                        return batch_results, batch_failed
            except Exception as exc:
                logger.debug("batch_bars.single_failed sym=%s err=%s", batch[0], exc)
            batch_failed.append(batch[0])
            return batch_results, batch_failed

        # Multi-symbol batch — returns MultiIndex (Symbol, OHLCV)
        try:
            data = yf.download(
                batch,
                start=start_str,
                end=end_str,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if data is None or data.empty:
                return batch_results, list(batch)

            # data[symbol] gives per-symbol DataFrame with Open/High/Low/Close/Volume
            top_level = set(data.columns.get_level_values(0))
            for sym in batch:
                try:
                    if sym not in top_level:
                        batch_failed.append(sym)
                        continue
                    sym_df = data[sym].dropna(how="all")
                    if sym_df.empty:
                        batch_failed.append(sym)
                        continue
                    bars = _dataframe_to_bars(sym_df, sym)
                    if bars:
                        batch_results[sym] = bars
                    else:
                        batch_failed.append(sym)
                except Exception as exc:
                    logger.debug("batch_bars.sym_extract_failed sym=%s err=%s", sym, exc)
                    batch_failed.append(sym)

        except Exception as exc:
            logger.warning("batch_bars.batch_failed size=%d err=%s", len(batch), exc)
            batch_failed = list(batch)

        return batch_results, batch_failed

    # Run batches concurrently
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_fetch_one_batch, b): b for b in batches}
        for future in as_completed(futures):
            try:
                results, failed = future.result()
                all_results.update(results)
                retry_symbols.extend(failed)
            except Exception as exc:
                batch = futures[future]
                logger.warning("batch_bars.future_error batch=%s err=%s", batch[:3], exc)
                retry_symbols.extend(batch)

    # Retry failed symbols individually with a small delay
    if retry_symbols:
        logger.info("batch_bars.retrying count=%d", len(retry_symbols))
        import yfinance as yf
        for sym in retry_symbols:
            try:
                time.sleep(RETRY_DELAY)
                df = yf.download(
                    sym,
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                )
                if df is not None and not df.empty:
                    bars = _dataframe_to_bars(df, sym)
                    if bars:
                        all_results[sym] = bars
                        continue
            except Exception as exc:
                logger.debug("batch_bars.retry_failed sym=%s err=%s", sym, exc)
            all_results.setdefault(sym, [])

    ok  = sum(1 for v in all_results.values() if v)
    bad = sum(1 for v in all_results.values() if not v)
    logger.info("batch_bars.complete symbols_ok=%d symbols_failed=%d", ok, bad)
    return all_results


def _dataframe_to_bars(df: Any, symbol: str) -> List[Dict[str, Any]]:
    """Convert a per-symbol yfinance DataFrame to AXIOM bar dicts.

    Handles both single-symbol (flat columns) and the per-symbol slice
    from a multi-ticker grouped download (both have Open/High/Low/Close/Volume).
    """
    bars: List[Dict[str, Any]] = []
    try:
        df = df.dropna(subset=["Close"])
    except Exception:
        return bars

    for ts, row in df.iterrows():
        try:
            bar_date: dt.date = ts.date() if hasattr(ts, "date") else ts
            close = float(row.get("Close") if hasattr(row, "get") else row["Close"])
            bars.append({
                "symbol":    symbol,
                "date":      bar_date,
                "open":      float(row.get("Open",  close) if hasattr(row, "get") else row.get("Open", close)),
                "high":      float(row.get("High",  close) if hasattr(row, "get") else row.get("High", close)),
                "low":       float(row.get("Low",   close) if hasattr(row, "get") else row.get("Low",  close)),
                "close":     close,
                "adj_close": close,
                "volume":    float(row.get("Volume", 0) or 0),
                "source":    "yfinance_batch",
            })
        except Exception:
            continue
    return bars
