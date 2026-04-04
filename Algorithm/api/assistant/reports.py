from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from decimal import Decimal
from numbers import Real
from typing import Any, Dict, Iterable, Optional


ANALYSIS_REPORT_KIND = "analysis_report"


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: sanitize_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_payload(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Real):
        try:
            number = float(value)
            return number if math.isfinite(number) else None
        except (TypeError, ValueError, OverflowError):
            return value
    return value


def _iso_date(value: Any) -> str:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value or "")


def _fmt_num(value: Any, *, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}f}"


def _fmt_pct(value: Any, *, digits: int = 1, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return str(value)
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.{digits}f}%"


def _fmt_list(values: Iterable[Any]) -> str:
    cleaned = [str(value) for value in values if value not in (None, "", [], {})]
    return ", ".join(cleaned) if cleaned else "none"


def _signal_bias(action: str) -> str:
    action_upper = (action or "HOLD").upper()
    if action_upper == "BUY":
        return "bullish"
    if action_upper == "SELL":
        return "bearish"
    return "neutral"


def _entry_text(signal: Dict[str, Any]) -> str:
    entry_low = signal.get("entry_low")
    entry_high = signal.get("entry_high")
    if entry_low is None and entry_high is None:
        return "No explicit entry band is available in the stored signal."
    if entry_low is not None and entry_high is not None:
        return f"Entry band is {_fmt_num(entry_low, digits=2)} to {_fmt_num(entry_high, digits=2)}."
    if entry_low is not None:
        return f"Entry reference is {_fmt_num(entry_low, digits=2)}."
    return f"Entry reference is {_fmt_num(entry_high, digits=2)}."


def build_active_analysis_reference(
    report: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    return sanitize_payload(
        {
            "report_id": report_id,
            "session_id": session_id or report.get("session_id"),
            "symbol": report.get("symbol"),
            "as_of_date": report.get("as_of_date"),
            "horizon": report.get("horizon"),
            "risk_mode": report.get("risk_mode"),
        }
    )


def build_analysis_report(
    *,
    symbol: str,
    as_of_date: Any,
    horizon: str,
    risk_mode: str,
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    as_of_text = _iso_date(as_of_date)
    action = (signal.get("action") or "HOLD").upper()
    score = signal.get("score")
    confidence = signal.get("confidence")
    regime = key_features.get("regime_label") or "unknown"
    quality_score = quality.get("quality_score")
    warnings = quality.get("warnings") or []
    reason_codes = signal.get("reason_codes") or []
    signal_view = {
        **signal,
        "horizon": horizon,
        "risk_mode": risk_mode,
    }

    signal_summary = " ".join(
        [
            f"As of {as_of_text}, FTIP's stored signal for {symbol} is {action}.",
            f"The model bias is {_signal_bias(action)} with score {_fmt_num(score)} and confidence {_fmt_num(confidence)}.",
            f"This report is framed on the {horizon} horizon with {risk_mode} risk mode.",
            f"Primary reason codes: {_fmt_list(reason_codes)}.",
        ]
    )

    technical_analysis = " ".join(
        [
            f"Short-term returns are {_fmt_pct(key_features.get('ret_1d'))} over 1 day, {_fmt_pct(key_features.get('ret_5d'))} over 5 days, and {_fmt_pct(key_features.get('ret_21d'))} over 21 days.",
            f"Trend slopes are {_fmt_num(key_features.get('trend_slope_21d'), signed=True)} on 21 days and {_fmt_num(key_features.get('trend_slope_63d'), signed=True)} on 63 days.",
            f"Volatility reads {_fmt_pct(key_features.get('vol_21d'))} on 21 days and {_fmt_pct(key_features.get('vol_63d'))} on 63 days, while ATR percent is {_fmt_pct(key_features.get('atr_pct'))}.",
            f"The regime label is {regime} with strength {_fmt_num(key_features.get('regime_strength'))}.",
        ]
    )

    fundamentals_ok = quality.get("fundamentals_ok")
    if fundamentals_ok is True:
        fundamental_analysis = (
            "Dedicated fundamentals were marked available in quality data, but this assistant report does not yet carry a richer PIT fundamentals block. "
            "Treat the current view as primarily driven by prices, features, signal construction, and sentiment freshness."
        )
    elif fundamentals_ok is False:
        fundamental_analysis = (
            "Fundamental coverage is flagged as missing in quality data, so this report should be read as mostly technical/statistical rather than fundamentally anchored."
        )
    else:
        fundamental_analysis = (
            "No dedicated fundamental block is attached to this assistant report. The current output is grounded mainly in price features, signal scores, and news/sentiment availability."
        )

    statistical_analysis = " ".join(
        [
            f"From a statistical perspective, the signal score is {_fmt_num(score)} and confidence is {_fmt_num(confidence)}.",
            f"Risk-adjusted momentum on 21 days is {_fmt_num(key_features.get('mom_vol_adj_21d'), signed=True)} and the stored horizon_days field is {signal.get('horizon_days') or 'n/a'}.",
            f"Quality score is {_fmt_num(quality_score, digits=0)} and missingness is {_fmt_num(quality.get('missingness'))}.",
        ]
    )

    sentiment_analysis = " ".join(
        [
            f"Sentiment score is {_fmt_num(key_features.get('sentiment_score'), signed=True)} with surprise {_fmt_num(key_features.get('sentiment_surprise'), signed=True)}.",
            f"Freshness flags show news_ok={quality.get('news_ok')} and sentiment_ok={quality.get('sentiment_ok')}.",
            f"Evidence sources currently attached are {_fmt_list(evidence.get('sources') or [])}.",
        ]
    )

    risk_quality_analysis = " ".join(
        [
            f"Data quality flags are bars_ok={quality.get('bars_ok')}, news_ok={quality.get('news_ok')}, sentiment_ok={quality.get('sentiment_ok')}, and intraday_ok={quality.get('intraday_ok')}.",
            _entry_text(signal),
            f"Stop loss is {_fmt_num(signal.get('stop_loss'), digits=2)}, take-profit levels are {_fmt_num(signal.get('take_profit_1'), digits=2)} and {_fmt_num(signal.get('take_profit_2'), digits=2)}.",
            f"Warnings: {_fmt_list(warnings)}. Anomaly flags: {_fmt_list(quality.get('anomaly_flags') or [])}.",
        ]
    )

    if action == "BUY":
        overall_analysis = (
            f"The current system view on {symbol} is constructive, but the edge is only as strong as the reported score/confidence pair. "
            f"The stored analysis leans bullish because the computed signal is BUY, with regime {regime} and reason codes {_fmt_list(reason_codes)}. "
            "This should be read as a model state description, not personal trading advice."
        )
        strategy_view = (
            f"The implied strategy stance is to favor long exposure only if the signal remains intact on the {horizon} horizon, data quality stays acceptable, "
            "and execution respects the stored stop/take-profit structure and any quality warnings."
        )
    elif action == "SELL":
        overall_analysis = (
            f"The current system view on {symbol} is defensive or bearish. "
            f"The computed signal is SELL, so the analysis points to downside or de-risking conditions rather than upside conviction. "
            "This is still a system narrative, not individualized advice."
        )
        strategy_view = (
            f"The implied strategy stance is to avoid bullish positioning or to express a bearish/hedged view only within the {horizon} framework and with strict risk controls."
        )
    else:
        overall_analysis = (
            f"The current system view on {symbol} is neutral. "
            f"The computed signal is HOLD, which means the report does not show a strong directional edge after weighing signal score, confidence, and quality context. "
            "This is a wait-for-better-setup posture rather than a recommendation."
        )
        strategy_view = (
            f"The implied strategy stance is to stay patient, monitor whether score/confidence improve, and only act when a clearer directional setup appears for the {horizon} horizon."
        )

    report = {
        "report_kind": ANALYSIS_REPORT_KIND,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": symbol,
        "as_of_date": as_of_text,
        "horizon": horizon,
        "risk_mode": risk_mode,
        "signal": signal_view,
        "key_features": key_features,
        "quality": quality,
        "evidence": evidence,
        "signal_summary": signal_summary,
        "technical_analysis": technical_analysis,
        "fundamental_analysis": fundamental_analysis,
        "statistical_analysis": statistical_analysis,
        "sentiment_analysis": sentiment_analysis,
        "risk_quality_analysis": risk_quality_analysis,
        "overall_analysis": overall_analysis,
        "strategy_view": strategy_view,
    }
    return sanitize_payload(report)
