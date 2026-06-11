"""AI Document Extraction API endpoints."""
from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Query, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/extract",
    tags=["extract"],
)


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert ExtractionResult dataclass to JSON-serializable dict."""
    d = asdict(result)
    # Convert ExtractedField list to plain dicts (already done by asdict)
    return d


# ---------------------------------------------------------------------------
# POST /extract/preview — extract without saving
# ---------------------------------------------------------------------------

@router.post("/preview")
async def preview_extraction(
    file: UploadFile = File(...),
    entity_hint: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Extract financial data from an uploaded document without saving to DB."""
    from api.ai_extraction.extractor import extract_from_file
    file_bytes = await file.read()
    result = extract_from_file(file_bytes, file.filename or "document", entity_hint)
    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# POST /extract/pe/portco/{entity_id}
# ---------------------------------------------------------------------------

@router.post("/pe/portco/{entity_id}")
async def extract_pe_portco(
    entity_id: str,
    file: UploadFile = File(...),
    period_override: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Upload portco financial statement → extract + save + run AXIOM intelligence."""
    from api.ai_extraction.extractor import extract_from_file
    from api import db

    file_bytes = await file.read()
    result = extract_from_file(file_bytes, file.filename or "document", entity_id)

    saved = False
    intelligence: Dict[str, Any] = {}

    if result.overall_confidence >= 0.6:
        # Save to DB
        if db.db_write_enabled():
            try:
                period_end = (
                    dt.date.fromisoformat(period_override)
                    if period_override
                    else (
                        dt.date.fromisoformat(result.period_end_date)
                        if result.period_end_date
                        else dt.date.today()
                    )
                )
                financials = {
                    "revenue": result.revenue,
                    "ebitda": result.ebitda,
                    "net_income": result.net_income,
                    "total_debt": result.total_debt,
                    "cash": result.cash_and_equivalents,
                    "capex": result.capex,
                    "free_cash_flow": result.free_cash_flow,
                }
                from api.jobs.pe_intelligence import store_entity_financials
                store_entity_financials(entity_id, period_end, financials)
                saved = True
            except Exception as exc:
                logger.debug("extract.pe.save_failed entity=%s err=%s", entity_id, exc)

        # Run forensic + DAS intelligence
        try:
            from api.pe.schilit_analyzer import SchilitForensicEngine
            from api.pe.das_engine import compute_das_score
            fund_inputs = {
                "gross_margin": result.gross_margin,
                "op_margin": result.op_margin,
                "fcf_margin": result.fcf_margin,
                "revenue_growth_yoy": None,
                "debt_to_equity": (
                    result.total_debt / (result.total_assets - result.total_debt)
                    if result.total_assets and result.total_debt and result.total_assets > result.total_debt
                    else None
                ),
            }
            forensic = SchilitForensicEngine().analyze(entity_id, {
                "revenue_growth_yoy": 0.0,
                "gross_margin": result.gross_margin or 0.0,
                "op_margin": result.op_margin or 0.0,
            })
            das = compute_das_score(entity_id, fund_inputs)
            intelligence = {
                "forensic_risk": forensic.overall_risk,
                "forensic_summary": forensic.forensic_summary,
                "das_score": das.das_score,
                "das_grade": das.das_grade,
                "investment_thesis": das.investment_thesis,
            }
        except Exception as exc:
            logger.debug("extract.pe.intelligence_failed entity=%s err=%s", entity_id, exc)

    # Store audit record
    _save_audit_record(entity_id, "pe_portco", result)

    return {
        "entity_id": entity_id,
        "extraction": _result_to_dict(result),
        "intelligence": intelligence,
        "saved": saved,
    }


# ---------------------------------------------------------------------------
# POST /extract/smb/entity/{entity_id}
# ---------------------------------------------------------------------------

@router.post("/smb/entity/{entity_id}")
async def extract_smb_entity(
    entity_id: str,
    file: UploadFile = File(...),
    month_override: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Upload SMB financial statement → extract + save + run AXIOM CFO intelligence."""
    from api.ai_extraction.extractor import extract_from_file
    from api import db

    file_bytes = await file.read()
    result = extract_from_file(file_bytes, file.filename or "document", entity_id)

    saved = False
    intelligence: Dict[str, Any] = {}

    if result.overall_confidence >= 0.6:
        if db.db_write_enabled():
            try:
                month_end = (
                    dt.date.fromisoformat(month_override)
                    if month_override
                    else (
                        dt.date.fromisoformat(result.period_end_date)
                        if result.period_end_date
                        else dt.date.today().replace(day=1) - dt.timedelta(days=1)
                    )
                )
                financials: Dict[str, Any] = {}
                if result.revenue is not None:
                    financials["revenue"] = result.revenue
                if result.gross_profit is not None and result.revenue:
                    financials["cogs"] = result.revenue - result.gross_profit
                if result.operating_income is not None:
                    financials["operating_expenses"] = (
                        (result.gross_profit or 0) - result.operating_income
                        if result.gross_profit
                        else None
                    )
                if result.net_income is not None:
                    financials["net_income"] = result.net_income
                if result.cash_and_equivalents is not None:
                    financials["cash_balance"] = result.cash_and_equivalents
                if result.accounts_receivable is not None:
                    financials["accounts_receivable"] = result.accounts_receivable
                if result.accounts_payable is not None:
                    financials["accounts_payable"] = result.accounts_payable
                from api.jobs.smb_intelligence import store_smb_financials
                store_smb_financials(entity_id, month_end, financials)
                saved = True
            except Exception as exc:
                logger.debug("extract.smb.save_failed entity=%s err=%s", entity_id, exc)

        # Run SMB intelligence
        try:
            from api.smb.public_intelligence import compute_smb_intelligence
            intelligence = compute_smb_intelligence(entity_id)
        except Exception as exc:
            logger.debug("extract.smb.intelligence_failed entity=%s err=%s", entity_id, exc)

    _save_audit_record(entity_id, "smb_entity", result)

    return {
        "entity_id": entity_id,
        "extraction": _result_to_dict(result),
        "intelligence": intelligence,
        "saved": saved,
    }


# ---------------------------------------------------------------------------
# POST /extract/bulk — paste financial text directly
# ---------------------------------------------------------------------------

class BulkExtractIn(BaseModel):
    entity_id: str
    entity_type: str = "unknown"
    text: str


@router.post("/bulk")
async def extract_bulk(body: BulkExtractIn) -> Dict[str, Any]:
    """Extract financials from pasted text (QuickBooks export, copied table, etc.)."""
    from api.ai_extraction.extractor import extract_financials_from_text
    result = extract_financials_from_text(body.text, "pasted_text", body.entity_id)
    _save_audit_record(body.entity_id, body.entity_type, result)
    return _result_to_dict(result)


# ---------------------------------------------------------------------------
# GET /extract/review-queue
# ---------------------------------------------------------------------------

@router.get("/review-queue")
def get_review_queue() -> Dict[str, Any]:
    """Return all pending extractions needing human review."""
    from api import db
    if not db.db_read_enabled():
        return {"status": "db_disabled", "items": []}
    try:
        rows = db.safe_fetchall(
            """
            SELECT id, entity_id, entity_type, filename, document_type,
                   period, overall_confidence, fields_needing_review, created_at
              FROM extraction_audit
             WHERE status = 'pending'
             ORDER BY created_at DESC
             LIMIT 50
            """
        ) or []
        items = []
        for row in rows:
            items.append({
                "id": str(row[0]),
                "entity_id": row[1],
                "entity_type": row[2],
                "filename": row[3],
                "document_type": row[4],
                "period": row[5],
                "overall_confidence": float(row[6]) if row[6] is not None else None,
                "fields_needing_review": list(row[7]) if row[7] else [],
                "created_at": row[8].isoformat() if row[8] else None,
            })
        return {"status": "ok", "count": len(items), "items": items}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "items": []}


# ---------------------------------------------------------------------------
# PATCH /extract/review/{extraction_id}
# ---------------------------------------------------------------------------

class ReviewIn(BaseModel):
    field_name: Optional[str] = None
    corrected_value: Optional[float] = None
    approved: bool = False


@router.patch("/review/{extraction_id}")
def review_extraction(extraction_id: str, body: ReviewIn) -> Dict[str, Any]:
    """Apply human correction or approval to a pending extraction."""
    from api import db
    if not db.db_write_enabled():
        return {"status": "db_disabled"}
    try:
        corrections: Dict[str, Any] = {}
        if body.field_name and body.corrected_value is not None:
            corrections[body.field_name] = body.corrected_value
        new_status = "approved" if body.approved else "pending"
        db.safe_execute(
            """
            UPDATE extraction_audit
               SET status = %s,
                   corrections = COALESCE(corrections, '{}'::jsonb) || %s::jsonb,
                   reviewed_at = now()
             WHERE id = %s
            """,
            (new_status, json.dumps(corrections), extraction_id),
        )
        return {"status": "updated", "extraction_id": extraction_id, "new_status": new_status}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_audit_record(entity_id: str, entity_type: str, result: Any) -> None:
    """Persist extraction result to extraction_audit for review queue."""
    from api import db
    if not db.db_write_enabled():
        return
    try:
        extracted_json = _result_to_dict(result)
        # Make sure fields list is JSON-serializable (already is from asdict)
        period_end = None
        if result.period_end_date:
            try:
                period_end = dt.date.fromisoformat(result.period_end_date)
            except Exception:
                pass
        db.safe_execute(
            """
            INSERT INTO extraction_audit
                (entity_id, entity_type, filename, document_type, period,
                 period_end_date, overall_confidence, extracted_json,
                 fields_needing_review, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            """,
            (
                entity_id,
                entity_type,
                result.document_name,
                result.document_type,
                result.period,
                period_end,
                result.overall_confidence,
                json.dumps(extracted_json),
                result.fields_needing_review or [],
                "pending" if result.fields_needing_review else "approved",
            ),
        )
    except Exception as exc:
        logger.debug("extract.audit_save_failed entity=%s err=%s", entity_id, exc)
