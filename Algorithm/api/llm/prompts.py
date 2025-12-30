from __future__ import annotations

from typing import Dict, List

DISCLAIMER = "For research/education only. Not financial advice."

FEATURE_HINTS = {
    "mom_5": "very short-term momentum over 1 week",
    "mom_21": "1-month momentum",
    "mom_63": "quarterly momentum",
    "trend_sma20_50": "trend strength (20 vs 50 day moving averages)",
    "volatility_ann": "annualized volatility",
    "rsi14": "RSI (14-day) momentum oscillator",
    "volume_z20": "volume z-score vs 20-day history",
    "last_close": "most recent close price",
}


def _style_instruction(style: str) -> str:
    if style == "memo":
        return "Use a professional research memo tone with short paragraphs."
    if style == "detailed":
        return "Use 2-3 crisp paragraphs and a few bullets."
    return "Be concise and direct."


def feature_driver_lines(features: Dict[str, float], top_n: int = 3) -> List[str]:
    items = sorted(features.items(), key=lambda kv: abs(float(kv[1] or 0.0)), reverse=True)
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


def build_signal_prompt(signal: Dict[str, object], drivers: List[str], history: List[Dict[str, object]], style: str) -> List[Dict[str, str]]:
    history_lines = []
    for row in history:
        history_lines.append(
            f"{row.get('as_of')}: signal={row.get('signal')} score={row.get('score')} regime={row.get('regime')}"
        )

    prompt = (
        f"Summarize the signal for {signal.get('symbol')} as of {signal.get('as_of')} with lookback {signal.get('lookback')}. "
        f"Signal={signal.get('signal')} (score_mode={signal.get('score_mode')}, score={signal.get('score')}, "
        f"thresholds={signal.get('thresholds')}). Regime={signal.get('regime')} and confidence={signal.get('confidence')}.")
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


def build_portfolio_prompt(summary: Dict[str, object], style: str) -> List[Dict[str, str]]:
    perf = summary.get("performance") or {}
    contributors = summary.get("contributors") or []
    exposures = summary.get("exposures") or []

    sections: List[str] = []
    sections.append(
        "Portfolio backtest summary: return={return_} sharpe={sharpe} max_drawdown={mdd} turnover={turnover}.".format(
            return_=perf.get("return"), sharpe=perf.get("sharpe"), mdd=perf.get("max_drawdown"), turnover=perf.get("turnover")
        )
    )
    if contributors:
        sections.append("Top contributors: " + "; ".join(contributors) + ".")
    if exposures:
        sections.append("Risk exposures or notes: " + "; ".join(exposures) + ".")

    prompt = " ".join(sections)
    prompt += " Provide a narrative and bullet takeaways. " + _style_instruction(style)
    prompt += " Close with a disclaimer and note uncertainty; do not promise performance."

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
