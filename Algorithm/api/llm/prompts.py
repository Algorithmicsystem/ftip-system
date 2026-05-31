from __future__ import annotations

from typing import Dict, List

DISCLAIMER = "For research/education only. Not financial advice."

FEATURE_HINTS = {
    # --- Price returns ---
    "ret_1d": "1-day price return",
    "ret_3d": "3-day price return",
    "ret_5d": "5-day price return",
    "ret_10d": "10-day price return",
    "ret_21d": "21-day (1-month) price return",
    "ret_63d": "63-day (quarterly) price return",
    "ret_126d": "126-day (6-month) price return",
    "ret_252d": "252-day (1-year) price return",
    # --- Momentum ---
    "mom_5": "very short-term momentum over 1 week",
    "mom_21": "1-month momentum",
    "mom_63": "quarterly momentum",
    "mom_126": "6-month momentum",
    "mom_252": "12-month (annual) momentum",
    "mom_vol_adj_21d": "21-day momentum adjusted for realized volatility (Sharpe-like)",
    # --- Volatility ---
    "volatility_ann": "annualized realized volatility (63-day window)",
    "vol_21d": "21-day realized annualized volatility",
    "vol_63d": "63-day realized annualized volatility",
    "vol_126d": "126-day realized annualized volatility",
    # --- ATR ---
    "atr_14": "14-day Average True Range (raw points)",
    "atr_pct": "14-day ATR as fraction of last close (normalized volatility proxy)",
    # --- Trend ---
    "trend_sma20_50": "trend strength: 20-day SMA relative to 50-day SMA",
    "trend_slope_21d": "log-price OLS slope over 21 days (short trend direction)",
    "trend_r2_21d": "R² of 21-day log-price OLS fit (trend quality)",
    "trend_slope_63d": "log-price OLS slope over 63 days (medium trend direction)",
    "trend_r2_63d": "R² of 63-day log-price OLS fit (trend quality)",
    # --- Oscillators and volume ---
    "rsi14": "RSI (14-day) momentum oscillator; >70 overbought, <30 oversold",
    "volume_z20": "volume z-score vs 20-day history; high positive = unusual buying interest",
    "dollar_vol_21d": "21-day average daily dollar volume (liquidity proxy)",
    # --- Drawdown ---
    "maxdd_63d": "maximum drawdown over 63 days (near-term downside risk)",
    "maxdd_252d": "maximum drawdown over 252 days (1-year peak-to-trough)",
    # --- Regime ---
    "regime_label": "current price-action regime: trend / choppy / high_vol",
    "regime_strength": "regime confidence score (0–1)",
    "signal_regime": "canonical regime label used by the signal engine",
    # --- Sentiment ---
    "sentiment_score": "latest news/social sentiment score (negative to positive)",
    "sentiment_surprise": "sentiment score minus its recent mean (surprise component)",
    # --- Base price ---
    "last_close": "most recent close price",
    # --- Event / earnings depth scores ---
    "event_overhang_score": "composite event risk overhang (0–100; high = near major catalyst)",
    "event_uncertainty_score": "uncertainty surrounding upcoming events (0–100)",
    "catalyst_burst_score": "recent burst of catalytic news activity (0–100)",
    "days_to_next_event": "calendar days to nearest scheduled event (earnings, etc.)",
    "earnings_window_flag": "true if currently within the earnings announcement window",
    "post_event_instability_flag": "true if price action unstable following a recent event",
    # --- Liquidity / implementation depth scores ---
    "implementation_fragility_score": "composite implementation risk (0–100; high = hard to execute cleanly)",
    "liquidity_quality_score": "market liquidity quality score (0–100; high = liquid)",
    "friction_proxy_score": "estimated trading friction (0–100; high = costly to trade)",
    "execution_cleanliness_score": "historical execution cleanliness proxy (0–100)",
    # --- Breadth depth scores ---
    "breadth_confirmation_score": "market-breadth confirmation of signal direction (0–100)",
    "internal_market_divergence_score": "divergence within market internals (0–100; high = fragile rally)",
    "leadership_concentration_score": "concentration of market leadership in few stocks (0–100)",
    "benchmark_confirmation_score": "benchmark-level confirmation of signal (0–100)",
    "sector_confirmation_score": "sector-level confirmation of signal direction (0–100)",
    # --- Cross-asset depth scores ---
    "macro_asset_alignment_score": "macro asset-class alignment with signal (0–100)",
    "cross_asset_conflict_score": "cross-asset contradictions to signal thesis (0–100; high = conflict)",
    "cross_asset_divergence_score": "divergence between related assets (0–100)",
    # --- Stress depth scores ---
    "market_stress_score": "composite market stress indicator (0–100; high = stressed)",
    "spillover_risk_score": "risk of stress spillover from other asset classes (0–100)",
    "correlation_breakdown_proxy": "proxy for correlation regime breakdown (0–100)",
    "volatility_shock_score": "probability-weighted volatility shock score (0–100)",
    "stress_transition_score": "likelihood of transitioning to a stressed regime (0–100)",
    "unstable_environment_flag": "true if environment is classified as unstable",
    "defensive_regime_flag": "true if model is in a defensive operating regime",
}


def _style_instruction(style: str) -> str:
    if style == "memo":
        return "Use a professional research memo tone with short paragraphs."
    if style == "detailed":
        return "Use 2-3 crisp paragraphs and a few bullets."
    return "Be concise and direct."


def feature_driver_lines(features: Dict[str, float], top_n: int = 3) -> List[str]:
    items = sorted(
        features.items(), key=lambda kv: abs(float(kv[1] or 0.0)), reverse=True
    )
    drivers: List[str] = []
    for name, val in items[:top_n]:
        desc = FEATURE_HINTS.get(name, name)
        drivers.append(f"{name} ({desc}): {round(float(val), 4)}")
    return drivers


def make_disclaimer() -> str:
    return DISCLAIMER


def system_prompt() -> str:
    return (
        "You are the FTIP Narrator. Explain model outputs using only provided context. "
        "Avoid promises or guarantees, do not offer investment advice, and always remind users that outputs are uncertain."
    )


def build_signal_prompt(
    signal: Dict[str, object],
    drivers: List[str],
    history: List[Dict[str, object]],
    style: str,
) -> List[Dict[str, str]]:
    history_lines = []
    for row in history:
        history_lines.append(
            f"{row.get('as_of')}: signal={row.get('signal')} score={row.get('score')} regime={row.get('regime')}"
        )

    prompt = (
        f"Summarize the signal for {signal.get('symbol')} as of {signal.get('as_of')} with lookback {signal.get('lookback')}. "
        f"Signal={signal.get('signal')} (score_mode={signal.get('score_mode')}, score={signal.get('score')}, "
        f"thresholds={signal.get('thresholds')}). Regime={signal.get('regime')} and confidence={signal.get('confidence')}."
    )
    if drivers:
        prompt += " Top drivers: " + "; ".join(drivers) + "."
    if history_lines:
        prompt += " Recent stored history: " + " | ".join(history_lines) + "."

    prompt += " Focus on explanations, risks (volatility/drawdown), and what the model observed. "
    prompt += _style_instruction(style)
    prompt += " Close with a one-line disclaimer."

    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]


def build_portfolio_prompt(
    summary: Dict[str, object], style: str
) -> List[Dict[str, str]]:
    perf = summary.get("performance") or {}
    contributors = summary.get("contributors") or []
    exposures = summary.get("exposures") or []

    sections: List[str] = []
    sections.append(
        "Portfolio backtest summary: return={return_} sharpe={sharpe} max_drawdown={mdd} turnover={turnover}.".format(
            return_=perf.get("return"),
            sharpe=perf.get("sharpe"),
            mdd=perf.get("max_drawdown"),
            turnover=perf.get("turnover"),
        )
    )
    if contributors:
        sections.append("Top contributors: " + "; ".join(contributors) + ".")
    if exposures:
        sections.append("Risk exposures or notes: " + "; ".join(exposures) + ".")

    prompt = " ".join(sections)
    prompt += " Provide a narrative and bullet takeaways. " + _style_instruction(style)
    prompt += (
        " Close with a disclaimer and note uncertainty; do not promise performance."
    )

    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]


def build_ask_prompt(question: str, context: Dict[str, object]) -> List[Dict[str, str]]:
    symbols_ctx = context.get("symbols") or {}
    parts: List[str] = []
    for sym, payload in symbols_ctx.items():
        sig = payload.get("signal") or {}
        stats = payload.get("market_stats") or {}
        parts.append(
            f"{sym}: signal={sig.get('signal')} score={sig.get('score')} regime={sig.get('regime')} "
            f"thresholds={sig.get('thresholds')} features={sig.get('features')} market_stats={stats}"
        )

    prompt = (
        "Use only the provided context to answer. If unsure, say the data is insufficient. "
        "Do not speculate about prices. Respond with a short answer and key citations (fields you relied on)."
    )
    prompt += " Context packets: " + " | ".join(parts)
    prompt += f" Question: {question}"
    prompt += " Always add a disclaimer about uncertainty and educational use."

    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": prompt},
    ]


def extract_bullets(text: str) -> List[str]:
    bullets: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "•", "*")):
            bullets.append(stripped.lstrip("-•* "))
    if not bullets and text:
        bullets.append(text.strip())
    return bullets


__all__ = [
    "DISCLAIMER",
    "build_signal_prompt",
    "build_portfolio_prompt",
    "build_ask_prompt",
    "feature_driver_lines",
    "make_disclaimer",
    "extract_bullets",
]
