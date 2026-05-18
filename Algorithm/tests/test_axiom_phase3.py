from __future__ import annotations

import copy
import datetime as dt
from typing import Any, Dict, List, Mapping, Sequence

from api.axiom import (
    build_axiom_calibration_artifact,
    build_axiom_portfolio_governance,
    rank_axiom_history_records,
    run_axiom_replay,
)
from api.axiom.persistence import (
    load_axiom_history_records,
    persist_axiom_artifacts_to_store,
)
from api.assistant.storage import AssistantStorage


def _bar_history(
    symbol: str,
    *,
    start: dt.date,
    count: int,
    shock_index: int | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        day = start + dt.timedelta(days=index)
        close = 100.0 + (index * 0.55)
        if shock_index is not None and index >= shock_index:
            close += 35.0
        rows.append(
            {
                "as_of_date": day.isoformat(),
                "open": close - 0.6,
                "high": close + 1.1,
                "low": close - 1.2,
                "close": round(close, 4),
                "volume": 1_250_000 + (index * 12_500),
                "source": "synthetic_test",
                "ingested_at": f"{day.isoformat()}T00:00:00Z",
            }
        )
    return rows


def _snapshot_builder_from_history(
    bar_history: Mapping[str, Sequence[Dict[str, Any]]],
) -> Any:
    def _builder(symbol: str, as_of_date: dt.date, lookback: int) -> Dict[str, Any]:
        full_history = list(bar_history[symbol])
        price_bars = [
            dict(row)
            for row in full_history
            if str(row.get("as_of_date") or "") <= as_of_date.isoformat()
        ][-lookback:]
        sentiment_history = [
            {
                "as_of_date": row["as_of_date"],
                "sentiment_score": 0.18 if idx % 3 else 0.05,
                "computed_at": f"{row['as_of_date']}T00:00:00Z",
            }
            for idx, row in enumerate(price_bars[-24:])
        ]
        news = [
            {
                "title": f"{symbol} replay headline {idx}",
                "published_at": f"{row['as_of_date']}T08:00:00Z",
                "ingested_at": f"{row['as_of_date']}T08:05:00Z",
                "source": "synthetic_test",
            }
            for idx, row in enumerate(price_bars[-8:])
        ]
        fundamentals = [
            {
                "fiscal_period_end": "2023-09-30",
                "report_date": "2023-11-02",
                "source": "synthetic_test",
                "ingested_at": "2023-11-02T00:00:00Z",
                "revenue": 12_500_000_000.0,
                "eps": 1.92,
                "gross_margin": 0.61,
                "op_margin": 0.24,
                "fcf": 1_800_000_000.0,
            },
            {
                "fiscal_period_end": "2023-12-31",
                "report_date": "2024-02-01",
                "source": "synthetic_test",
                "ingested_at": "2024-02-01T00:00:00Z",
                "revenue": 13_300_000_000.0,
                "eps": 2.14,
                "gross_margin": 0.63,
                "op_margin": 0.27,
                "fcf": 2_050_000_000.0,
            },
        ]
        return {
            "snapshot_id": f"{symbol}-{as_of_date.isoformat()}",
            "snapshot_version": "phase8_canonical_snapshot_v1",
            "generated_at": f"{as_of_date.isoformat()}T00:00:00Z",
            "symbol_meta": {
                "symbol": symbol,
                "sector": "Technology",
                "industry": "Semiconductors",
            },
            "price_bars": price_bars,
            "intraday_bars": [],
            "fundamentals": fundamentals,
            "news": news,
            "sentiment_history": sentiment_history,
            "quality": {
                "quality_score": 87,
                "warnings": [],
                "bars_ok": True,
                "fundamentals_ok": True,
                "sentiment_ok": True,
                "news_ok": True,
            },
            "provenance": {"market_bars_source": "synthetic_test"},
        }

    return _builder


def _sample_replay_run() -> Dict[str, Any]:
    start = dt.date(2024, 1, 2)
    history = {
        "NVDA": _bar_history("NVDA", start=start, count=150, shock_index=85),
        "AAPL": _bar_history("AAPL", start=start, count=150, shock_index=95),
    }
    snapshot_builder = _snapshot_builder_from_history(history)
    return run_axiom_replay(
        symbols=["NVDA", "AAPL"],
        start_date="2024-02-01",
        end_date="2024-03-10",
        lookback=90,
        persist=False,
        bar_history=history,
        snapshot_builder=snapshot_builder,
    )


def test_axiom_replay_builds_history_without_lookahead():
    start = dt.date(2024, 1, 2)
    history = {"NVDA": _bar_history("NVDA", start=start, count=140, shock_index=90)}
    snapshot_builder = _snapshot_builder_from_history(history)

    replay = run_axiom_replay(
        symbols=["NVDA"],
        start_date="2024-02-05",
        end_date="2024-02-08",
        lookback=80,
        persist=False,
        bar_history=history,
        snapshot_builder=snapshot_builder,
    )

    assert replay["record_count"] == 4
    first_record = replay["records"][0]
    latest_close = (first_record.get("source_context") or {}).get("latest_close")
    assert latest_close is not None
    expected_close = next(
        row["close"]
        for row in history["NVDA"]
        if row["as_of_date"] == first_record["as_of_date"]
    )
    assert latest_close == expected_close
    assert latest_close < history["NVDA"][-1]["close"]
    assert set(first_record["forward_outcomes"].keys()) == {"5d", "21d", "63d"}
    assert first_record["build_metadata"]["replay_version"] == "axiom50_phase3_replay_v1"


def test_axiom_history_persistence_round_trip_in_memory_store():
    replay = _sample_replay_run()
    record = next(item for item in replay["records"] if item["symbol"] == "NVDA")
    calibration = replay["calibration"]
    store = AssistantStorage(use_memory=True)
    session_id = store.create_session()

    artifact_ids = persist_axiom_artifacts_to_store(
        store=store,
        session_id=session_id,
        history_record=record,
        calibration_artifact=calibration,
    )
    loaded = load_axiom_history_records(
        symbols=["NVDA"],
        store=store,
        session_id=session_id,
    )

    assert artifact_ids["history_artifact_id"] is not None
    assert artifact_ids["calibration_artifact_id"] is not None
    assert len(loaded) == 1
    assert loaded[0]["symbol"] == record["symbol"]
    assert loaded[0]["framework_version"] == record["framework_version"]


def test_axiom_calibration_artifact_builds_bucket_and_group_summaries():
    replay = _sample_replay_run()
    calibration = replay["calibration"]

    assert calibration["status"] in {"available", "limited"}
    assert calibration["matured_count"] > 0
    assert calibration["dau_bucket_summary"]["status"] == "available"
    assert calibration["dau_bucket_summary"]["bucket_count"] >= 2
    assert calibration["regime_outcome_summary"]
    assert calibration["trade_family_outcome_summary"]
    assert calibration["deployability_tier_outcome_summary"]
    assert calibration["engine_conditioned_outcome_summary"]


def test_axiom_portfolio_governance_and_ranking_penalize_overlap():
    calibration = build_axiom_calibration_artifact(
        records=[],
        horizon_label="21d",
        min_sample_size=1,
    )
    calibration.update(
        {
            "status": "available",
            "matured_count": 18,
            "evidence_supportive_for_live": True,
            "evidence_supportive_for_paper": True,
        }
    )
    current_report = {
        "symbol": "NVDA",
        "as_of_date": "2024-03-01",
        "hidden_overlap_score": 82.0,
        "portfolio_fit_quality": 44.0,
        "portfolio_construction": {
            "current_candidate": {
                "symbol": "NVDA",
                "overlap_score": 82.0,
                "portfolio_fit_quality": 44.0,
            }
        },
    }
    axiom_artifact = {
        "deployability_tier": "live_candidate",
        "trade_family": "convergence",
        "deployable_alpha_utility": 74.0,
        "invalidation_flags": ["macro_alignment_break"],
        "deployability_decision": {"monitoring_triggers": ["breadth weakens"]},
        "engine_scores": {
            "critical_fragility": {"score": 32.0},
            "liquidity_convexity": {"score": 61.0},
            "research_integrity": {"score": 73.0},
        },
    }
    governance = build_axiom_portfolio_governance(
        current_report=current_report,
        axiom_artifact=axiom_artifact,
        calibration_summary=calibration,
    )
    assert governance["overlap_penalty"] > 0
    assert governance["portfolio_rank_score"] < axiom_artifact["deployable_alpha_utility"]
    assert governance["final_size_band"] in {"small", "medium", "large", "none"}

    record_a = {
        "symbol": "NVDA",
        "sector": "Technology",
        "regime_label": "fundamental_convergence",
        "trade_family": "convergence",
        "theme_tag": "ai",
        "deployable_alpha_utility": 76.0,
        "deployability_tier": "live_candidate",
        "size_band_recommendation": "large",
        "engine_scores": copy.deepcopy(axiom_artifact["engine_scores"]),
        "evidence_backed_deployability": {"size_band": "large", "deployability_tier": "live_candidate"},
    }
    record_b = {
        **record_a,
        "symbol": "AMD",
        "deployable_alpha_utility": 72.0,
    }
    ranked = rank_axiom_history_records([record_a, record_b], current_holdings=["NVDA"])
    nvda_ranked = next(item for item in ranked if item["symbol"] == "NVDA")
    assert nvda_ranked["overlap_penalty"] >= 8.0
    assert nvda_ranked["portfolio_fit_label"] in {
        "core_candidate",
        "tactical_candidate",
        "watchlist_only",
        "avoid",
    }
