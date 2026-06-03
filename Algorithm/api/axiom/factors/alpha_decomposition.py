"""Alpha Decomposition Engine — decomposes DAU into factor contributions."""
from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from api.assistant.phase3.common import clamp
from api.axiom.factors.factor_model import FactorLoading


@dataclass
class AlphaDecomposition:
    symbol: str
    as_of_date: str          # ISO date string
    total_dau: float
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    systematic_contribution: float = 0.0
    idiosyncratic_alpha: float = 0.0
    primary_driver: str = ""
    factor_concentration: float = 0.0
    regime_adjusted_loadings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "as_of_date": self.as_of_date,
            "total_dau": self.total_dau,
            "factor_contributions": self.factor_contributions,
            "systematic_contribution": round(self.systematic_contribution, 4),
            "idiosyncratic_alpha": round(self.idiosyncratic_alpha, 4),
            "primary_driver": self.primary_driver,
            "factor_concentration": round(self.factor_concentration, 4),
            "regime_adjusted_loadings": self.regime_adjusted_loadings,
        }


def _herfindahl(values: list) -> float:
    total = sum(abs(v) for v in values)
    if total == 0:
        return 0.0
    shares = [abs(v) / total for v in values]
    return round(sum(s * s for s in shares), 4)


def decompose_alpha(
    axiom_payload: dict,
    factor_loadings: List[FactorLoading],
    regime_label: str,
    symbol: str = "",
    as_of_date: str = "",
) -> AlphaDecomposition:
    dau = float(axiom_payload.get("deployable_alpha_utility") or 50.0)

    factor_contributions: Dict[str, float] = {}
    regime_adjusted: Dict[str, float] = {}

    for fl in factor_loadings:
        rw = fl.regime_relevance.get(regime_label, 0.15)
        contribution = round(fl.loading * rw * 50.0, 4)
        factor_contributions[fl.factor_name] = contribution
        regime_adjusted[fl.factor_name] = round(fl.loading * rw, 4)

    systematic = sum(factor_contributions.values())
    idiosyncratic = round(dau - systematic, 4)

    primary = max(factor_contributions, key=lambda k: abs(factor_contributions[k]), default="")
    concentration = _herfindahl(list(factor_contributions.values()))

    return AlphaDecomposition(
        symbol=symbol,
        as_of_date=as_of_date,
        total_dau=round(dau, 4),
        factor_contributions=factor_contributions,
        systematic_contribution=round(systematic, 4),
        idiosyncratic_alpha=idiosyncratic,
        primary_driver=primary,
        factor_concentration=concentration,
        regime_adjusted_loadings=regime_adjusted,
    )
