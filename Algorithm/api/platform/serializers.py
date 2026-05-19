from __future__ import annotations

import hashlib
import json
from typing import Any

from api.assistant.reports import sanitize_payload


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(sanitize_payload(payload), sort_keys=True, separators=(",", ":"))


def content_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()
