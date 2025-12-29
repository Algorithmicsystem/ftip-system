from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json

from api import config, db


class AssistantStorage:
    def __init__(self, use_memory: Optional[bool] = None):
        self.use_memory = bool(use_memory) if use_memory is not None else not config.db_enabled()
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._messages: List[Dict[str, Any]] = []
        self._artifacts: List[Dict[str, Any]] = []

    def _now(self) -> float:
        return time.time()

    def create_session(self, *, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        sid = str(uuid.uuid4())
        if self.use_memory:
            self._sessions[sid] = {
                "id": sid,
                "title": title,
                "metadata": metadata or {},
                "created_at": self._now(),
                "updated_at": self._now(),
            }
            return sid

        db.exec1(
            """
            INSERT INTO assistant_sessions (id, title, metadata)
            VALUES (%s, %s, %s::jsonb)
            """,
            (sid, title, Json(metadata) if metadata is not None else None),
        )
        return sid

    def upsert_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> None:
        if self.use_memory:
            session = self._sessions.get(session_id)
            if session:
                merged = {**(session.get("metadata") or {}), **metadata}
                session["metadata"] = merged
                session["updated_at"] = self._now()
            return

        db.exec1(
            """
            UPDATE assistant_sessions
            SET metadata=COALESCE(metadata, '{}'::jsonb) || %s::jsonb, updated_at=now()
            WHERE id=%s
            """,
            (Json(metadata), session_id),
        )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            return self._sessions.get(session_id)

        row = db.fetch1(
            """
            SELECT id, title, metadata, created_at, updated_at
            FROM assistant_sessions WHERE id=%s
            """,
            (session_id,),
        )
        if not row:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "metadata": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        model: Optional[str] = None,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        mid = str(uuid.uuid4())
        if self.use_memory:
            self._messages.append(
                {
                    "id": mid,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "created_at": self._now(),
                    "model": model,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "extra": extra or {},
                }
            )
            return mid

        db.exec1(
            """
            INSERT INTO assistant_messages
              (id, session_id, role, content, model, tokens_in, tokens_out, extra)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (mid, session_id, role, content, model, tokens_in, tokens_out, Json(extra) if extra else None),
        )
        db.exec1("UPDATE assistant_sessions SET updated_at=now() WHERE id=%s", (session_id,))
        return mid

    def get_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        if self.use_memory:
            items = [m for m in self._messages if m.get("session_id") == session_id]
            items.sort(key=lambda m: m.get("created_at", 0))
            return items[-limit:]

        rows = db.fetchall(
            """
            SELECT id, role, content, created_at, model, tokens_in, tokens_out, extra
            FROM assistant_messages
            WHERE session_id=%s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (session_id, limit),
        )
        messages: List[Dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "created_at": row[3],
                    "model": row[4],
                    "tokens_in": row[5],
                    "tokens_out": row[6],
                    "extra": row[7],
                }
            )
        messages.reverse()
        return messages

    def save_artifact(self, session_id: str, kind: str, payload: Dict[str, Any]) -> str:
        aid = str(uuid.uuid4())
        if self.use_memory:
            self._artifacts.append(
                {
                    "id": aid,
                    "session_id": session_id,
                    "kind": kind,
                    "payload": payload,
                    "created_at": self._now(),
                }
            )
            return aid

        db.exec1(
            """
            INSERT INTO assistant_artifacts (id, session_id, kind, payload)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (aid, session_id, kind, Json(payload)),
        )
        return aid


storage = AssistantStorage()
