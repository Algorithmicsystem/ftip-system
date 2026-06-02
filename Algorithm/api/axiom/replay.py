from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from api.alpha import build_canonical_features, build_canonical_signal
from api import db
from api.assistant import intelligence, reports, strategy
from api.axiom.calibration import build_axiom_calibration_artifact
from api.axiom.engine import AXIOM_FRAMEWORK_VERSION, build_axiom_artifact
from api.axiom.history import (
    AXIOM_CALIBRATION_ARTIFACT_KIND,
    AXIOM_PHASE3_VERSION,
    AXIOM_REPLAY_ARTIFACT_KIND,
    build_axiom_history_record,
    forward_horizons,
)
from api.axiom.persistence import (
    load_axiom_history_records,
    persist_axiom_calibration_snapshot,
    persist_axiom_replay_run,
    persist_axiom_score_record,
)
from api.axiom.portfolio import build_evidence_backed_deployability
from api.research import build_research_snapshot
from api.research.backtest.outcomes import default_ohlc_bar_fetcher, evaluate_prediction_outcome


AXIOM_REPLAY_VERSION = "axiom50_phase3_replay_v1"


def _snapshot_builder(
    symbol: str,
    as_of_date: dt.date,
    lookback: int,
    *,
    include_reference_context: bool = True,
) -> Dict[str, Any]:
    return build_research_snapshot(
        symbol,
        as_of_date,
        lookback,
        lookback_days=max(420, lookback * 2),
        include_reference_context=include_reference_context,
    )


def _coerce_quarterly_fundamentals(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if any("revenue" in row or "report_date" in row for row in rows):
        return [dict(row) for row in rows]
    grouped: Dict[str, Dict[str, Any]] = {}
    metric_map = {
        "revenue": "revenue",
        "eps": "eps",
        "gross_margin": "gross_margin",
        "op_margin": "op_margin",
        "operating_margin": "op_margin",
        "fcf": "fcf",
        "free_cash_flow": "fcf",
    }
    for row in rows:
        period_end = str(row.get("period_end") or row.get("fiscal_period_end") or "")
        if not period_end:
            continue
        payload = grouped.setdefault(
            period_end,
            {
                "fiscal_period_end": period_end,
                "report_date": period_end,
                "source": row.get("source") or "fundamentals_pit",
                "ingested_at": row.get("as_of_ts") or row.get("published_ts"),
            },
        )
        metric_key = str(row.get("metric_key") or "").strip().lower()
        mapped_key = metric_map.get(metric_key)
        if mapped_key:
            payload[mapped_key] = row.get("metric_value")
    normalized = list(grouped.values())
    normalized.sort(key=lambda item: str(item.get("fiscal_period_end") or ""), reverse=True)
    return normalized


def _coerce_sentiment_history(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def _coerce_recent_news(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        published = payload.get("published_at") or payload.get("ingested_at")
        payload.setdefault("ingested_at", published)
        normalized.append(payload)
    normalized.sort(key=lambda item: str(item.get("ingested_at") or ""), reverse=True)
    return normalized


def _synthetic_freshness(snapshot: Dict[str, Any], as_of_date: dt.date) -> Dict[str, Any]:
    bars = list(snapshot.get("price_bars") or [])
    intraday = list(snapshot.get("intraday_bars") or [])
    news = list(snapshot.get("news") or [])
    sentiment = list(snapshot.get("sentiment_history") or [])
    fundamentals = list(snapshot.get("fundamentals") or [])
    return {
        "as_of_date": as_of_date.isoformat(),
        "bars_updated_at": (bars[-1] if bars else {}).get("ingested_at"),
        "intraday_updated_at": (intraday[-1] if intraday else {}).get("ingested_at"),
        "news_updated_at": (news[0] if news else {}).get("ingested_at"),
        "sentiment_updated_at": (sentiment[-1] if sentiment else {}).get("computed_at"),
        "fundamentals_updated_at": (fundamentals[0] if fundamentals else {}).get("ingested_at"),
    }


def _quality_payload(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    quality = dict(snapshot.get("quality") or {})
    if not quality:
        quality = {
            "quality_score": 0,
            "warnings": ["historical quality record unavailable"],
        }
    quality.setdefault("bars_ok", bool(snapshot.get("price_bars")))
    quality.setdefault("fundamentals_ok", bool(snapshot.get("fundamentals")))
    quality.setdefault("sentiment_ok", bool(snapshot.get("sentiment_history")))
    quality.setdefault("news_ok", bool(snapshot.get("news")))
    quality.setdefault("warnings", [])
    return quality


def _key_features(feature_vector: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "ret_1d",
        "ret_5d",
        "ret_21d",
        "vol_21d",
        "vol_63d",
        "atr_pct",
        "mom_vol_adj_21d",
        "trend_slope_21d",
        "trend_slope_63d",
        "regime_label",
        "regime_strength",
        "sentiment_score",
        "sentiment_surprise",
    )
    return {key: feature_vector.get(key) for key in keys if feature_vector.get(key) is not None}


def _signal_payload(signal_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(signal_result.get("payload") or {})
    action = str(signal_result.get("action") or payload.get("signal") or "HOLD").upper()
    score = signal_result.get("score")
    confidence = signal_result.get("confidence")
    return {
        "action": action,
        "score": score,
        "confidence": confidence,
        "reason_codes": list(payload.get("reason_codes") or []),
        "reason_details": dict(payload.get("reason_details") or {}),
        "horizon_days": 21,
    }


def _feature_factor_bundle(
    data_bundle: Dict[str, Any],
    signal: Dict[str, Any],
    key_features: Dict[str, Any],
    quality: Dict[str, Any],
) -> Dict[str, Any]:
    return intelligence.build_feature_factor_bundle(
        data_bundle=data_bundle,
        signal=signal,
        key_features=key_features,
        quality=quality,
    )


def _build_historical_data_bundle(
    *,
    symbol: str,
    as_of_date: dt.date,
    snapshot: Dict[str, Any],
    feature_vector: Dict[str, Any],
    feature_meta: Dict[str, Any],
    signal_result: Dict[str, Any],
) -> Dict[str, Any]:
    quality = _quality_payload(snapshot)
    key_features = _key_features(feature_vector)
    freshness = _synthetic_freshness(snapshot, as_of_date)
    symbol_meta = dict(snapshot.get("symbol_meta") or intelligence._load_symbol_meta(symbol))
    daily_bars = list(snapshot.get("price_bars") or [])
    intraday_bars = list(snapshot.get("intraday_bars") or [])
    market_domain = {
        "latest_close": feature_vector.get("latest_close") or (daily_bars[-1] if daily_bars else {}).get("close"),
        "previous_close": (daily_bars[-2] if len(daily_bars) >= 2 else {}).get("close"),
        "latest_open": (daily_bars[-1] if daily_bars else {}).get("open"),
        "day_return": feature_vector.get("ret_1d"),
        "ret_1d": feature_vector.get("ret_1d"),
        "ret_3d": feature_vector.get("ret_3d"),
        "ret_5d": feature_vector.get("ret_5d"),
        "ret_10d": feature_vector.get("ret_10d"),
        "ret_21d": feature_vector.get("ret_21d"),
        "ret_63d": feature_vector.get("ret_63d"),
        "ret_126d": feature_vector.get("ret_126d"),
        "ret_252d": feature_vector.get("ret_252d"),
        "realized_vol_5d": feature_vector.get("vol_5d"),
        "realized_vol_10d": feature_vector.get("vol_10d"),
        "realized_vol_21d": feature_vector.get("vol_21d"),
        "realized_vol_63d": feature_vector.get("vol_63d"),
        "realized_vol_126d": feature_vector.get("vol_126d"),
        "realized_vol_252d": feature_vector.get("vol_252d"),
        "atr_pct": feature_vector.get("atr_pct"),
        "gap_pct": feature_vector.get("gap_pct"),
        "volume_anomaly": feature_vector.get("volume_anomaly"),
        "positive_day_ratio_10d": feature_vector.get("positive_day_ratio_10d"),
        "positive_day_ratio_21d": feature_vector.get("positive_day_ratio_21d"),
        "positive_day_ratio_63d": feature_vector.get("positive_day_ratio_63d"),
        "return_dispersion_10d": feature_vector.get("return_dispersion_10d"),
        "return_dispersion_21d": feature_vector.get("return_dispersion_21d"),
        "return_dispersion_63d": feature_vector.get("return_dispersion_63d"),
        "downside_asymmetry_21d": feature_vector.get("downside_asymmetry_21d"),
        "downside_asymmetry_63d": feature_vector.get("downside_asymmetry_63d"),
        "maxdd_21d": feature_vector.get("maxdd_21d"),
        "maxdd_63d": feature_vector.get("maxdd_63d"),
        "maxdd_126d": feature_vector.get("maxdd_126d"),
        "gap_instability_10d": feature_vector.get("gap_instability_10d"),
        "abs_gap_mean_10d": feature_vector.get("abs_gap_mean_10d"),
        "up_down_volume_ratio_21d": feature_vector.get("up_down_volume_ratio_21d"),
        "vol_of_vol_proxy": feature_vector.get("vol_of_vol_proxy"),
        "compression_ratio": feature_vector.get("compression_ratio"),
        "range_position_21d": feature_vector.get("range_position_21d"),
        "range_expansion_ratio": feature_vector.get("range_expansion_ratio"),
        "breakout_distance_63d": feature_vector.get("breakout_distance_63d"),
        "support_21d": feature_vector.get("support_21d"),
        "resistance_21d": feature_vector.get("resistance_21d"),
        "recent_bars": daily_bars[-5:],
        "recent_intraday_bars": intraday_bars[-12:],
        "price_series": [b["close"] for b in daily_bars if b.get("close") is not None],
        "return_series": intelligence._to_returns([b.get("close") for b in daily_bars]),
        "volume_series": [b["volume"] for b in daily_bars if b.get("volume") is not None],
        "avg_volume_21d": intelligence._mean([b.get("volume") for b in daily_bars[-21:] if b.get("volume") is not None]) if daily_bars else None,
        "meta": {
            "coverage_score": min(len(daily_bars) / 252.0, 1.0),
            "sources": [snapshot.get("provenance", {}).get("market_bars_source") or "historical_snapshot"],
            "latest_ingested_at": (daily_bars[-1] if daily_bars else {}).get("ingested_at"),
            "intraday_latest_ingested_at": (intraday_bars[-1] if intraday_bars else {}).get("ingested_at"),
            "coverage_status": "available" if daily_bars else "limited",
            "data_quality_note": "Historical AXIOM replay is using point-in-time bars available at the replay date.",
        },
    }
    technical_domain = intelligence._technical_domain(daily_bars, key_features)
    fundamentals = intelligence._load_fundamentals(symbol, as_of_date)
    if not fundamentals:
        fundamentals = _coerce_quarterly_fundamentals(snapshot.get("fundamentals") or [])
    fundamental_domain = intelligence._fundamental_domain(fundamentals, as_of_date, quality)
    sentiment_history = intelligence._load_sentiment_history(symbol, as_of_date)
    if not sentiment_history:
        sentiment_history = _coerce_sentiment_history(snapshot.get("sentiment_history") or [])
    recent_news = intelligence._load_recent_news(symbol, as_of_date)
    if not recent_news:
        recent_news = _coerce_recent_news(snapshot.get("news") or [])
    sentiment_domain = intelligence._sentiment_domain(
        symbol,
        as_of_date,
        sentiment_history,
        recent_news,
        key_features,
    )
    job_context = {
        "symbol": symbol,
        "as_of_date": as_of_date.isoformat(),
        "horizon": "swing",
        "risk_mode": "balanced",
        "scenario": "historical_replay",
        "analysis_depth": "historical_replay",
        "refresh_mode": "historical_replay",
        "market_regime": "historical_replay",
        "canonical_lineage": {
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_version": snapshot.get("snapshot_version"),
            "feature_version": feature_meta.get("feature_version"),
            "signal_version": (signal_result.get("payload") or {}).get("signal_version")
            or signal_result.get("signal_version"),
        },
        "canonical_feature_vector": feature_vector,
        "canonical_feature_meta": feature_meta,
        "canonical_signal_payload": signal_result.get("payload") or {},
        "canonical_signal_meta": {
            "historical_replay": True,
            "snapshot_generated_at": snapshot.get("generated_at"),
        },
    }
    macro_domain = intelligence._macro_cross_asset_domain(symbol_meta, as_of_date, job_context, market_domain)
    geopolitical_domain = intelligence._geopolitical_domain(recent_news)
    relative_domain = intelligence._relative_context_domain(
        symbol,
        as_of_date,
        symbol_meta,
        market_domain,
        macro_domain,
        key_features,
    )
    quality_domain = intelligence._quality_provenance_domain(
        freshness,
        quality,
        {
            "macro_cross_asset": macro_domain,
            "relative_context": relative_domain,
        },
        as_of_date,
    )
    depth_domains = intelligence._canonical_depth_domains(job_context)
    bundle = {
        "symbol_meta": symbol_meta,
        "canonical_alpha_core": {
            "lineage": job_context.get("canonical_lineage") or {},
            "feature_vector": feature_vector,
            "feature_meta": feature_meta,
            "signal_payload": signal_result.get("payload") or {},
            "signal_meta": job_context.get("canonical_signal_meta") or {},
        },
        "market_price_volume": market_domain,
        "technical_market_structure": technical_domain,
        "fundamental_filing": fundamental_domain,
        "sentiment_narrative_flow": sentiment_domain,
        "macro_cross_asset": macro_domain,
        "geopolitical_policy": geopolitical_domain,
        "relative_context": relative_domain,
        "event_catalyst_risk": depth_domains.get("event_catalyst_risk") or {},
        "liquidity_execution_fragility": depth_domains.get("liquidity_execution_fragility") or {},
        "market_breadth_internals": depth_domains.get("market_breadth_internals") or {},
        "cross_asset_confirmation": depth_domains.get("cross_asset_confirmation") or {},
        "stress_spillover_conditions": depth_domains.get("stress_spillover_conditions") or {},
        "quality_provenance": quality_domain,
        "raw_supporting_fields": {
            "signal": _signal_payload(signal_result),
            "key_features": key_features,
            "quality": quality,
            "canonical_lineage": job_context.get("canonical_lineage") or {},
            "canonical_feature_meta": feature_meta,
            "canonical_signal_meta": job_context.get("canonical_signal_meta") or {},
            "recent_news_headlines": [row.get("title") for row in recent_news[:8]],
            "sentiment_history": sentiment_history[-5:],
            "recent_daily_bars": daily_bars[-10:],
            "recent_intraday_bars": intraday_bars[-12:],
            "fundamental_quarters": fundamentals[:4],
        },
    }
    domain_availability = intelligence._build_domain_availability_map(bundle)
    bundle["domain_availability"] = domain_availability
    bundle["quality_provenance"] = {
        **(bundle.get("quality_provenance") or {}),
        "domain_availability": domain_availability,
    }
    bundle["normalized_domains"] = {
        "market": bundle.get("market_price_volume") or {},
        "technical": bundle.get("technical_market_structure") or {},
        "fundamentals": bundle.get("fundamental_filing") or {},
        "news_sentiment_narrative": bundle.get("sentiment_narrative_flow") or {},
        "macro": bundle.get("macro_cross_asset") or {},
        "geopolitical": bundle.get("geopolitical_policy") or {},
        "cross_asset": bundle.get("relative_context") or {},
        "quality_provenance": bundle.get("quality_provenance") or {},
    }
    return bundle


def _available_dates_from_history(
    symbol: str,
    *,
    start: dt.date,
    end: dt.date,
    bar_history: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> List[dt.date]:
    if bar_history and symbol in bar_history:
        dates: List[dt.date] = []
        for row in bar_history[symbol]:
            value = row.get("as_of_date") or row.get("date") or row.get("timestamp")
            if not value:
                continue
            try:
                row_date = dt.date.fromisoformat(str(value)[:10])
            except ValueError:
                continue
            if start <= row_date <= end:
                dates.append(row_date)
        return sorted(set(dates))

    rows = db.safe_fetchall(
        """
        SELECT as_of_date
        FROM market_bars_daily
        WHERE symbol = %s
          AND as_of_date BETWEEN %s AND %s
        ORDER BY as_of_date ASC
        """,
        (symbol, start, end),
    )
    if rows:
        return [row[0] for row in rows]
    prosperity_rows = db.safe_fetchall(
        """
        SELECT date
        FROM prosperity_daily_bars
        WHERE symbol = %s
          AND date BETWEEN %s AND %s
        ORDER BY date ASC
        """,
        (symbol, start, end),
    )
    return [row[0] for row in prosperity_rows]


def _outcome_bar_fetcher(
    *,
    symbol: str,
    bar_history: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> Callable[[str, dt.date, int], List[Dict[str, Any]]]:
    if not bar_history or symbol not in bar_history:
        return default_ohlc_bar_fetcher

    rows = list(bar_history[symbol])

    def _fetch(_symbol: str, as_of_date: dt.date, limit: int) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            value = row.get("as_of_date") or row.get("date") or row.get("timestamp")
            if not value:
                continue
            try:
                row_date = dt.date.fromisoformat(str(value)[:10])
            except ValueError:
                continue
            if row_date < as_of_date:
                continue
            filtered.append(
                {
                    "as_of_date": row_date.isoformat(),
                    "open": row.get("open") or row.get("close"),
                    "high": row.get("high") or row.get("close"),
                    "low": row.get("low") or row.get("close"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                }
            )
        filtered.sort(key=lambda item: item["as_of_date"])
        return filtered[:limit]

    return _fetch


def _forward_outcomes(
    history_record: Dict[str, Any],
    *,
    bar_fetcher: Callable[[str, dt.date, int], List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    outcomes: Dict[str, Dict[str, Any]] = {}
    for label, days in forward_horizons().items():
        validation_record = {
            "symbol": history_record.get("symbol"),
            "as_of_date": history_record.get("as_of_date"),
            "horizon": "swing",
            "horizon_days": days,
            "signal_action": history_record.get("signal_action"),
            "final_signal": history_record.get("signal_action"),
            "score": history_record.get("signal_score"),
            "confidence_score": history_record.get("signal_confidence"),
            "feature_vector": history_record.get("source_context", {}).get("canonical_feature_vector")
            or history_record.get("source_context", {}).get("feature_vector")
            or {},
            "proprietary_scores": {
                "Signal Fragility Index": (
                    ((history_record.get("engine_scores") or {}).get("critical_fragility") or {}).get("score")
                ),
            },
        }
        outcomes[label] = evaluate_prediction_outcome(
            validation_record,
            bar_fetcher=bar_fetcher,
            cost_model={},
        )
    return outcomes


def build_axiom_replay_record(
    *,
    symbol: str,
    as_of_date: dt.date,
    lookback: int = 252,
    snapshot_builder: Callable[[str, dt.date, int], Dict[str, Any]] = _snapshot_builder,
    bar_history: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    snapshot = snapshot_builder(symbol, as_of_date, lookback)
    if len(snapshot.get("price_bars") or []) < 30:
        return None
    quality = _quality_payload(snapshot)
    quality_score = int(quality.get("quality_score") or 0)
    feature_payload = build_canonical_features(snapshot)
    signal_result = build_canonical_signal(snapshot, feature_payload, quality_score=quality_score)
    signal_wrapper = {
        "action": signal_result.get("signal"),
        "score": signal_result.get("score"),
        "confidence": signal_result.get("confidence"),
        "payload": signal_result,
        "feature_payload": feature_payload,
        "feature_vector": feature_payload.get("features") or {},
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_version": snapshot.get("snapshot_version"),
        "feature_version": feature_payload.get("feature_version"),
        "signal_version": signal_result.get("signal_version"),
    }
    data_bundle = _build_historical_data_bundle(
        symbol=symbol,
        as_of_date=as_of_date,
        snapshot=snapshot,
        feature_vector=signal_wrapper["feature_vector"],
        feature_meta=feature_payload.get("feature_meta") or {},
        signal_result=signal_wrapper,
    )
    key_features = _key_features(signal_wrapper["feature_vector"])
    signal_payload = _signal_payload(signal_wrapper)
    feature_factor_bundle = _feature_factor_bundle(
        data_bundle,
        signal_payload,
        key_features,
        quality,
    )
    strategy_bundle = strategy.build_strategy_artifact(
        job_context={
            "symbol": symbol,
            "as_of_date": as_of_date.isoformat(),
            "horizon": "swing",
            "risk_mode": "balanced",
            "scenario": "historical_replay",
            "analysis_depth": "historical_replay",
            "refresh_mode": "historical_replay",
            "market_regime": "historical_replay",
        },
        signal=signal_payload,
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
    )
    axiom_artifact = build_axiom_artifact(
        normalized_bundle=data_bundle,
        job_context={
            "symbol": symbol,
            "as_of_date": as_of_date.isoformat(),
            "horizon": "swing",
            "risk_mode": "balanced",
            "scenario": "historical_replay",
            "analysis_depth": "historical_replay",
            "refresh_mode": "historical_replay",
            "market_regime": "historical_replay",
            "canonical_lineage": {
                "snapshot_id": snapshot.get("snapshot_id"),
                "snapshot_version": snapshot.get("snapshot_version"),
                "feature_version": feature_payload.get("feature_version"),
                "signal_version": signal_result.get("signal_version"),
            },
            "canonical_feature_vector": signal_wrapper["feature_vector"],
            "canonical_feature_meta": feature_payload.get("feature_meta") or {},
            "canonical_signal_payload": signal_result,
            "canonical_signal_meta": {"historical_replay": True},
        },
        feature_factor_bundle=feature_factor_bundle,
        strategy_bundle=strategy_bundle,
        report_context={},
    )
    report = reports.build_analysis_report(
        symbol=symbol,
        as_of_date=as_of_date.isoformat(),
        horizon="swing",
        risk_mode="balanced",
        signal=signal_payload,
        key_features=key_features,
        quality=quality,
        evidence={
            "reason_codes": list(signal_result.get("reason_codes") or []),
            "reason_details": dict(signal_result.get("reason_details") or {}),
            "sources": ["historical_replay"],
        },
        job_context={
            "scenario": "historical_replay",
            "analysis_depth": "historical_replay",
            "refresh_mode": "historical_replay",
            "market_regime": "historical_replay",
        },
        data_bundle=data_bundle,
        feature_factor_bundle=feature_factor_bundle,
        strategy=strategy_bundle,
    )
    bar_fetcher = _outcome_bar_fetcher(symbol=symbol, bar_history=bar_history)
    history_record = build_axiom_history_record(
        report=report,
        axiom_artifact=axiom_artifact,
        build_metadata={
            "replay_version": AXIOM_REPLAY_VERSION,
            "phase3_version": AXIOM_PHASE3_VERSION,
            "lookback": lookback,
        },
        forward_outcomes={},
        evidence_backed_deployability={},
    )
    history_record["source_context"]["canonical_feature_vector"] = signal_wrapper["feature_vector"]
    history_record["source_context"]["feature_vector"] = signal_wrapper["feature_vector"]
    history_record["source_context"]["signal_payload"] = signal_result
    history_record["forward_outcomes"] = _forward_outcomes(history_record, bar_fetcher=bar_fetcher)
    return history_record


def run_axiom_replay(
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    lookback: int = 252,
    persist: bool = True,
    session_id: Optional[str] = None,
    store: Optional[Any] = None,
    bar_history: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
    snapshot_builder: Callable[[str, dt.date, int], Dict[str, Any]] = _snapshot_builder,
) -> Dict[str, Any]:
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    records: List[Dict[str, Any]] = []
    for symbol in symbols:
        dates = _available_dates_from_history(
            symbol,
            start=start,
            end=end,
            bar_history=bar_history,
        )
        for as_of_date in dates:
            record = build_axiom_replay_record(
                symbol=symbol,
                as_of_date=as_of_date,
                lookback=lookback,
                snapshot_builder=snapshot_builder,
                bar_history=bar_history,
            )
            if record:
                records.append(record)

    records.sort(key=lambda item: (str(item.get("as_of_date") or ""), str(item.get("symbol") or "")))
    for record in records:
        cutoff = dt.date.fromisoformat(str(record.get("as_of_date")))
        calibration = build_axiom_calibration_artifact(
            records=records,
            as_of_date=cutoff,
            horizon_label="21d",
        )
        record["evidence_backed_deployability"] = build_evidence_backed_deployability(
            current_axiom=record,
            calibration_summary=calibration,
        )
        if persist:
            persist_axiom_score_record(record)

    calibration = build_axiom_calibration_artifact(
        records=records,
        as_of_date=end,
        horizon_label="21d",
    )
    if persist:
        persist_axiom_calibration_snapshot(calibration)

    run_payload = {
        "run_id": str(uuid.uuid4()),
        "artifact_kind": AXIOM_REPLAY_ARTIFACT_KIND,
        "replay_version": AXIOM_REPLAY_VERSION,
        "framework_version": AXIOM_FRAMEWORK_VERSION,
        "symbols": list(symbols),
        "date_start": start_date,
        "date_end": end_date,
        "lookback": lookback,
        "record_count": len(records),
        "calibration_status": calibration.get("status"),
        "records": records,
        "calibration": calibration,
    }
    if persist:
        persist_axiom_replay_run(run_payload)
    if store is not None and session_id:
        store.save_artifact(session_id, AXIOM_REPLAY_ARTIFACT_KIND, run_payload)
        store.save_artifact(session_id, AXIOM_CALIBRATION_ARTIFACT_KIND, calibration)
    return run_payload


def load_or_build_axiom_calibration(
    *,
    symbols: Sequence[str],
    as_of_date: str,
    store: Optional[Any] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    records = load_axiom_history_records(
        symbols=symbols,
        end_date=as_of_date,
        store=store,
        session_id=session_id,
    )
    return build_axiom_calibration_artifact(
        records=records,
        as_of_date=dt.date.fromisoformat(as_of_date),
        horizon_label="21d",
    )
