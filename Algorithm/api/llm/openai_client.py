"""
AXIOM OpenAI Client — lightweight LLM synthesis layer.

Uses OPENAI_API_KEY from Railway environment via api.config.
Falls back gracefully to None on missing key or any call failure.
Never blocks the main response — LLM is always additive, never required.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from api import config

logger = logging.getLogger(__name__)

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

ANALYST_SYSTEM = (
    "You are an institutional investment research analyst at a top hedge fund. "
    "Write precisely and concisely. Use specific numbers from the data provided. "
    "Never speculate or add information not in the data. No hedging language. "
    "Write 2-3 sentences maximum unless specified otherwise."
)


def is_available() -> bool:
    return bool(config.openai_api_key())


def get_openai_client():
    """Return a configured OpenAI SDK client instance."""
    import openai
    api_key = config.openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    return openai.OpenAI(api_key=api_key)


def call_openai(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Call OpenAI with timeout and error handling.
    Returns None on any failure — caller handles fallback.
    """
    api_key = config.openai_api_key()
    if not api_key:
        return None

    try:
        import httpx
        response = httpx.post(
            OPENAI_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config.llm_model() or OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=8.0,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("openai_call_failed err=%s", exc)
        return None


def synthesize_signal_explanation(data: Dict[str, Any]) -> Optional[str]:
    return call_openai(
        system_prompt=ANALYST_SYSTEM,
        user_prompt=(
            f"Write a 3-sentence investment note explaining this signal:\n"
            f"Symbol: {data['symbol']}\n"
            f"Signal: {data['signal_label']} (DAU: {data['dau']:.1f}/100)\n"
            f"Earnings Quality (EIS): {data['eis_score']:.0f}/100\n"
            f"Competitive Advantage (CAPS): {data['caps_score']:.0f}/100\n"
            f"Factor Composite: {data['factor_composite']:.0f}/100\n"
            f"Primary Driver: {data['primary_driver']}\n"
            f"Regime: {data['regime_label']}\n"
            f"Key Risk: {data['top_risk']}\n\n"
            f"Be specific about the numbers. Explain WHY this is {data['signal_label']}."
        ),
        max_tokens=200,
    )


def synthesize_morning_briefing(data: Dict[str, Any]) -> Optional[str]:
    return call_openai(
        system_prompt=(
            "You are writing the daily morning intelligence briefing for a hedge fund. "
            "Be precise, professional, and specific about the numbers. "
            "2 paragraphs maximum. No hedging language."
        ),
        user_prompt=(
            f"Write today's morning briefing based on this data:\n"
            f"Date: {data['date']}\n"
            f"Regime: {data['regime']}\n"
            f"SRI (Systemic Risk): {data['sri']:.1f}/100\n"
            f"Universe: {data['n_buy']} BUY / {data['n_hold']} HOLD / {data['n_sell']} SELL signals\n"
            f"Top Signal: {data['top_symbol']} (DAU {data['top_dau']:.1f}, driver: {data['top_driver']})\n"
            f"Avg Universe DAU: {data['avg_dau']:.1f}\n"
            f"IC State: {data['ic_state']}\n"
            f"Pipeline: {data['symbols_ok']}/30 symbols processed"
        ),
        max_tokens=300,
    )


def synthesize_pe_analysis(data: Dict[str, Any]) -> Optional[str]:
    return call_openai(
        system_prompt=(
            "You are a private equity analyst writing a deal assessment. "
            "Be direct and quantitative. Focus on what the numbers mean for PE value creation."
        ),
        user_prompt=(
            f"Assess this PE target based on AXIOM intelligence:\n"
            f"Company: {data['symbol']}\n"
            f"AXIOM Signal: {data['signal_label']} (DAU {data['dau']:.1f})\n"
            f"Deal Attractiveness Score: {data['das_score']:.0f}/100 (Grade: {data['das_grade']})\n"
            f"Schilit Risk: {data['schilit_risk']}\n"
            f"Investment Thesis: {data['investment_thesis']}\n\n"
            f"Write a 2-sentence PE deal assessment. What is the opportunity and what is the risk?"
        ),
        max_tokens=150,
    )


def synthesize_smb_recommendation(data: Dict[str, Any]) -> Optional[str]:
    return call_openai(
        system_prompt=(
            "You are a business advisor writing an executive summary for a small business owner. "
            "Be direct and action-focused. Use dollar amounts."
        ),
        user_prompt=(
            f"Write a 2-sentence executive summary for this business:\n"
            f"Pricing Power Score: {data['pricing_score']}/100\n"
            f"Credit Rating: {data['credit_rating']} (DSCR: {data['dscr']:.1f}x)\n"
            f"Working Capital Opportunity: ${data['cash_opportunity']:,}\n"
            f"Top Action: {data['top_action']}\n\n"
            f"What should the owner do first and why?"
        ),
        max_tokens=150,
    )
