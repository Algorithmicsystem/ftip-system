from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


__all__ = ["hash_payload"]
