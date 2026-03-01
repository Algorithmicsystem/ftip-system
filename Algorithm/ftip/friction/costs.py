from __future__ import annotations


def bps_to_cash(notional: float, bps: float) -> float:
    return max(0.0, notional) * max(0.0, bps) / 10000.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator
