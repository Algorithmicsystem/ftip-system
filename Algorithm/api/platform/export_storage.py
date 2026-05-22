from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Tuple

from api import config
from api.assistant.reports import sanitize_payload
from api.platform.contracts import ExportStorageRef


_IN_MEMORY_STORE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def default_storage_backend_name(*, use_memory: bool) -> str:
    configured = str(
        config.env("FTIP_PLATFORM_EXPORT_STORAGE_BACKEND", "") or ""
    ).strip()
    if configured:
        return configured
    return "in_memory_store" if use_memory else "local_file_store"


def default_storage_root() -> Path:
    configured = str(
        config.env("FTIP_PLATFORM_EXPORT_STORAGE_ROOT", ".platform_export_storage")
        or ".platform_export_storage"
    )
    return Path(configured).expanduser()


def clear_in_memory_export_storage() -> None:
    _IN_MEMORY_STORE.clear()


def store_export_content(
    *,
    storage_backend: str,
    storage_key: str,
    rendered_content: str,
    content_type: str,
    metadata: Dict[str, Any] | None = None,
) -> ExportStorageRef:
    normalized = str(storage_backend or "in_memory_store").strip().lower()
    payload = {
        "rendered_content": rendered_content,
        "content_type": content_type,
        "metadata": sanitize_payload(metadata or {}),
        "stored_at": _now_utc(),
    }
    if normalized == "in_memory_store":
        _IN_MEMORY_STORE[(normalized, storage_key)] = payload
        return ExportStorageRef(
            storage_backend=normalized,
            storage_key=storage_key,
            retrieval_hint=f"in_memory://{storage_key}",
            content_type=content_type,
            size_bytes=len(rendered_content.encode("utf-8")),
        )
    if normalized != "local_file_store":
        raise ValueError(f"Unsupported export storage backend: {normalized}")
    root = default_storage_root()
    path = root / storage_key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered_content, encoding="utf-8")
    return ExportStorageRef(
        storage_backend=normalized,
        storage_key=storage_key,
        retrieval_hint=str(path),
        local_path=str(path),
        content_type=content_type,
        size_bytes=path.stat().st_size,
    )


def retrieve_export_content(
    *,
    storage_backend: str,
    storage_key: str,
) -> Dict[str, Any]:
    normalized = str(storage_backend or "in_memory_store").strip().lower()
    if normalized == "in_memory_store":
        item = _IN_MEMORY_STORE.get((normalized, storage_key))
        if item is None:
            raise FileNotFoundError(storage_key)
        return sanitize_payload(item)
    if normalized != "local_file_store":
        raise ValueError(f"Unsupported export storage backend: {normalized}")
    path = default_storage_root() / storage_key
    if not path.exists():
        raise FileNotFoundError(str(path))
    return {
        "rendered_content": path.read_text(encoding="utf-8"),
        "content_type": "application/octet-stream",
        "metadata": {},
        "stored_at": _now_utc(),
        "local_path": str(path),
        "size_bytes": path.stat().st_size,
    }


__all__ = [
    "clear_in_memory_export_storage",
    "default_storage_backend_name",
    "default_storage_root",
    "retrieve_export_content",
    "store_export_content",
]
