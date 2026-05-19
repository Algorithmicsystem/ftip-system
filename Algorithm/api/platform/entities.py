from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, Optional

from api.platform.contracts import CoverageEntity


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_coverage_entity(
    *,
    symbol: Optional[str] = None,
    display_name: Optional[str] = None,
    entity_type: str = "public_equity",
    sector: Optional[str] = None,
    strategy: Optional[str] = None,
    theme: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    entity_id: Optional[str] = None,
) -> CoverageEntity:
    normalized_symbol = str(symbol or "").strip().upper() or None
    resolved_name = str(display_name or normalized_symbol or "Coverage Entity")
    return CoverageEntity(
        entity_id=str(entity_id or uuid.uuid4()),
        symbol=normalized_symbol,
        entity_type=entity_type,
        display_name=resolved_name,
        sector=sector,
        strategy=strategy,
        theme=theme,
        metadata=dict(metadata or {}),
        created_at=now_utc(),
        updated_at=now_utc(),
    )

