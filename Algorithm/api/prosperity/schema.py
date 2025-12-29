"""SQL helpers for Prosperity DB (placeholder for future extensions)."""

from __future__ import annotations

import json
from typing import Any, Dict


def to_jsonb(payload: Dict[str, Any]) -> str:
    return json.dumps(payload)
