from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from psycopg.types.json import Json

from api import config, db
from api.assistant.reports import ANALYSIS_REPORT_KIND, sanitize_payload


class AssistantStorage:
    def __init__(self, use_memory: Optional[bool] = None):
        self.use_memory = (
            bool(use_memory) if use_memory is not None else not config.db_enabled()
        )
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._messages: List[Dict[str, Any]] = []
        self._artifacts: List[Dict[str, Any]] = []

    def _now(self) -> float:
        return time.time()

    def create_session(
        self, *, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
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
            (sid, title, Json(sanitize_payload(metadata)) if metadata is not None else None),
        )
        return sid

    def upsert_session_metadata(
        self, session_id: str, metadata: Dict[str, Any]
    ) -> None:
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
            (Json(sanitize_payload(metadata)), session_id),
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
            "metadata": sanitize_payload(row[2]) if row[2] is not None else row[2],
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
            (
                mid,
                session_id,
                role,
                content,
                model,
                tokens_in,
                tokens_out,
                Json(sanitize_payload(extra)) if extra else None,
            ),
        )
        db.exec1(
            "UPDATE assistant_sessions SET updated_at=now() WHERE id=%s", (session_id,)
        )
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
                    "extra": sanitize_payload(row[7]) if row[7] is not None else row[7],
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
            (aid, session_id, kind, Json(sanitize_payload(payload))),
        )
        return aid

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        if self.use_memory:
            for artifact in self._artifacts:
                if artifact.get("id") == artifact_id:
                    return artifact
            return None

        row = db.fetch1(
            """
            SELECT id, session_id, kind, payload, created_at
            FROM assistant_artifacts
            WHERE id=%s
            """,
            (artifact_id,),
        )
        if not row:
            return None
        return {
            "id": str(row[0]),
            "session_id": str(row[1]),
            "kind": row[2],
            "payload": sanitize_payload(row[3]) if row[3] is not None else row[3],
            "created_at": row[4],
        }

    def get_latest_analysis_report(
        self,
        *,
        report_id: Optional[str] = None,
        session_id: Optional[str] = None,
        symbol: Optional[str] = None,
        as_of_date: Optional[str] = None,
        horizon: Optional[str] = None,
        risk_mode: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        filters = {
            "symbol": symbol,
            "as_of_date": as_of_date,
            "horizon": horizon,
            "risk_mode": risk_mode,
        }

        if report_id:
            artifact = self.get_artifact(report_id)
            if not artifact or artifact.get("kind") != ANALYSIS_REPORT_KIND:
                return None
            if session_id and str(artifact.get("session_id")) != str(session_id):
                return None
            payload = artifact.get("payload") or {}
            for key, value in filters.items():
                if value is not None and str(payload.get(key)) != str(value):
                    return None
            return {
                **payload,
                "report_id": artifact.get("id"),
                "session_id": str(artifact.get("session_id")),
            }

        if self.use_memory:
            candidates: List[Dict[str, Any]] = []
            for artifact in self._artifacts:
                if artifact.get("kind") != ANALYSIS_REPORT_KIND:
                    continue
                if session_id and str(artifact.get("session_id")) != str(session_id):
                    continue
                payload = artifact.get("payload") or {}
                matched = True
                for key, value in filters.items():
                    if value is not None and str(payload.get(key)) != str(value):
                        matched = False
                        break
                if matched:
                    candidates.append(artifact)
            if not candidates:
                return None
            candidates.sort(key=lambda item: item.get("created_at", 0), reverse=True)
            payload = candidates[0].get("payload") or {}
            return {
                **payload,
                "report_id": candidates[0].get("id"),
                "session_id": str(candidates[0].get("session_id")),
            }

        where_clauses = ["kind=%s"]
        params: List[Any] = [ANALYSIS_REPORT_KIND]
        if session_id:
            where_clauses.append("session_id=%s")
            params.append(session_id)

        payload_filters = {
            "symbol": symbol,
            "as_of_date": as_of_date,
            "horizon": horizon,
            "risk_mode": risk_mode,
        }
        for key, value in payload_filters.items():
            if value is None:
                continue
            where_clauses.append(f"payload->>'{key}' = %s")
            params.append(str(value))

        row = db.fetch1(
            f"""
            SELECT id, session_id, payload, created_at
            FROM assistant_artifacts
            WHERE {' AND '.join(where_clauses)}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            tuple(params),
        )
        if not row:
            return None
        payload = row[2] or {}
        return {
            **payload,
            "report_id": str(row[0]),
            "session_id": str(row[1]),
        }


storage = AssistantStorage()
