from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.assistant.phase6.scorecards import build_calibration_summary, build_signal_scorecard

from .common import as_date, mean, safe_float


def build_walkforward_validation(
    records: List[Dict[str, Any]],
    *,
    mode: str = "anchored",
    train_window: int = 252,
    validation_window: int = 63,
    step_window: int = 63,
    min_window_records: int = 4,
) -> Dict[str, Any]:
    matured = [
        record
        for record in records
        if (record.get("outcome") or {}).get("matured") is True and as_date(record.get("as_of_date")) is not None
    ]
    matured.sort(key=lambda record: as_date(record.get("as_of_date")) or as_date("1970-01-01"))
    unique_dates = sorted({as_date(record.get("as_of_date")) for record in matured if as_date(record.get("as_of_date")) is not None})
    if len(unique_dates) < train_window + 2:
        return {
            "walkforward_version": "phase10_walkforward_v1",
            "status": "insufficient_sample",
            "window_count": 0,
            "windows": [],
            "calibration_drift_summary": "Not enough matured point-in-time predictions exist yet for walk-forward validation.",
            "feature_signal_stability_notes": [],
        }

    mode = str(mode or "anchored").lower()
    windows: List[Dict[str, Any]] = []
    start_idx = train_window
    while start_idx + 1 < len(unique_dates):
        train_dates = unique_dates[:start_idx] if mode == "anchored" else unique_dates[max(0, start_idx - train_window) : start_idx]
        validation_dates = unique_dates[start_idx : start_idx + validation_window]
        if len(validation_dates) < max(2, validation_window // 3):
            break
        train_records = [record for record in matured if as_date(record.get("as_of_date")) in set(train_dates)]
        validation_records = [record for record in matured if as_date(record.get("as_of_date")) in set(validation_dates)]
        if len(train_records) < min_window_records or len(validation_records) < min_window_records:
            start_idx += step_window
            continue
        train_signal = build_signal_scorecard(train_records).get("final_signal_overall") or {}
        validation_signal = build_signal_scorecard(validation_records).get("final_signal_overall") or {}
        validation_calibration = build_calibration_summary(validation_records)
        windows.append(
            {
                "train_window": {
                    "start": train_dates[0].isoformat(),
                    "end": train_dates[-1].isoformat(),
                    "sample_count": len(train_records),
                    "matured_count": train_signal.get("matured_count"),
                    "average_edge_return": train_signal.get("average_edge_return"),
                    "hit_rate": train_signal.get("hit_rate"),
                },
                "validation_window": {
                    "start": validation_dates[0].isoformat(),
                    "end": validation_dates[-1].isoformat(),
                    "sample_count": len(validation_records),
                    "matured_count": validation_signal.get("matured_count"),
                    "average_edge_return": validation_signal.get("average_edge_return"),
                    "hit_rate": validation_signal.get("hit_rate"),
                },
                "validation_reliability": validation_calibration.get("confidence_reliability_score"),
                "validation_monotonicity": validation_calibration.get("confidence_monotonicity"),
                "edge_delta_vs_train": round(
                    (safe_float(validation_signal.get("average_edge_return")) or 0.0)
                    - (safe_float(train_signal.get("average_edge_return")) or 0.0),
                    6,
                ),
                "hit_rate_delta_vs_train": round(
                    (safe_float(validation_signal.get("hit_rate")) or 0.0)
                    - (safe_float(train_signal.get("hit_rate")) or 0.0),
                    6,
                ),
            }
        )
        start_idx += step_window

    if not windows:
        return {
            "walkforward_version": "phase10_walkforward_v1",
            "status": "insufficient_sample",
            "window_count": 0,
            "windows": [],
            "calibration_drift_summary": "Walk-forward windows were configured, but too few matured records landed inside each validation slice.",
            "feature_signal_stability_notes": [],
        }

    validation_edges = [safe_float(window.get("validation_window", {}).get("average_edge_return")) for window in windows]
    validation_hits = [safe_float(window.get("validation_window", {}).get("hit_rate")) for window in windows]
    reliability = [safe_float(window.get("validation_reliability")) for window in windows]
    drift_notes: List[str] = []
    avg_edge = mean(validation_edges) or 0.0
    avg_hit = mean(validation_hits) or 0.0
    if avg_edge < 0:
        drift_notes.append("Net edge is negative across validation windows, so the canonical engine is not holding up cleanly out of sample.")
    if avg_hit < 0.5:
        drift_notes.append("Hit rate falls below 50% across walk-forward slices, so directionality remains unstable.")
    if reliability and (mean(reliability) or 0.0) < 50:
        drift_notes.append("Confidence reliability softens out of sample, which suggests calibration drift.")
    if not drift_notes:
        drift_notes.append("Walk-forward slices remain directionally constructive, with no major out-of-sample drift note active.")

    return {
        "walkforward_version": "phase10_walkforward_v1",
        "status": "available",
        "window_count": len(windows),
        "mode": mode,
        "train_window_days": train_window,
        "validation_window_days": validation_window,
        "step_window_days": step_window,
        "windows": windows,
        "average_validation_edge_return": round(avg_edge, 6),
        "average_validation_hit_rate": round(avg_hit, 4),
        "average_validation_reliability": round(mean(reliability) or 0.0, 4) if reliability else None,
        "calibration_drift_summary": " ".join(drift_notes),
        "feature_signal_stability_notes": drift_notes,
    }

