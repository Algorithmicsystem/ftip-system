"""
AXIOM AI Financial Document Extractor

Reads any financial document (PDF, Excel, CSV, images of statements)
and extracts structured financial data into AXIOM's schema.

The AI job: understand the document the way a smart analyst would.
Find revenue even if it's called "Net Sales" or "Total Revenue" or
"Turnover". Understand that EBITDA might be presented as
Operating Income + D&A. Flag anything ambiguous. Never guess blindly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedField:
    """A single extracted financial field with confidence."""
    field_name: str
    value: Optional[float]
    raw_text: str
    confidence: float
    notes: str
    needs_review: bool


@dataclass
class ExtractionResult:
    """Complete extraction result from one document."""
    document_name: str
    document_type: str      # 'income_statement'|'balance_sheet'|'cash_flow'|'full_financials'|'unknown'
    period: Optional[str]
    period_end_date: Optional[str]
    entity_name: Optional[str]

    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    ebitda: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    accounts_receivable: Optional[float] = None
    accounts_payable: Optional[float] = None

    gross_margin: Optional[float] = None
    ebitda_margin: Optional[float] = None
    op_margin: Optional[float] = None
    net_margin: Optional[float] = None
    fcf_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None

    fields: List[ExtractedField] = field(default_factory=list)
    overall_confidence: float = 0.0
    fields_needing_review: List[str] = field(default_factory=list)
    extraction_notes: str = ""
    currency: str = "USD"
    unit: str = "actual"
    raw_text_sample: str = ""


AXIOM_EXTRACTION_SCHEMA = """
You are an expert financial analyst. Extract financial data from the
provided document and return it as structured JSON.

IMPORTANT RULES:
1. Map synonyms correctly:
   - "Net Sales" / "Total Revenue" / "Turnover" / "Sales" → revenue
   - "Gross Profit" / "Gross Income" → gross_profit
   - "EBITDA" / "Operating Income before D&A" → ebitda
   - "Operating Income" / "EBIT" / "Operating Profit" → operating_income
   - "Net Income" / "Net Profit" / "Net Earnings" / "Profit after Tax" → net_income
   - "Free Cash Flow" / "FCF" → free_cash_flow
   - "Cash from Operations" / "Operating Cash Flow" / "CFO" → operating_cash_flow
   - "Capital Expenditures" / "CapEx" / "Purchase of PP&E" → capex
   - "Total Liabilities" or "Long-term Debt" + "Short-term Debt" → total_debt

2. Unit detection:
   - If numbers are in millions, set unit="millions"
   - If numbers are in thousands, set unit="thousands"
   - If numbers are actual values, set unit="actual"
   - ALWAYS note the unit in your response

3. Period detection:
   - Find the reporting period (Q1 2026, FY 2025, etc.)
   - Find the period end date (March 31, 2026 etc.)
   - If multiple periods shown, extract the MOST RECENT one

4. Confidence scoring:
   - 0.9-1.0: Field clearly labeled, no ambiguity
   - 0.7-0.9: Field found but required some interpretation
   - 0.5-0.7: Field inferred from other data or unclear labeling
   - Below 0.5: Do not include — mark as not found

5. Entity name: Find the company name in the document header or title.

6. NEVER fabricate values. If a field is not in the document, return null.

Return ONLY valid JSON in this exact structure:
{
  "entity_name": string or null,
  "document_type": "income_statement" | "balance_sheet" | "cash_flow" | "full_financials" | "unknown",
  "period": string or null,
  "period_end_date": "YYYY-MM-DD" or null,
  "currency": "USD" or detected currency,
  "unit": "actual" | "thousands" | "millions" | "billions",
  "extraction_notes": "brief explanation of what you found and any issues",
  "fields": {
    "revenue": {"value": number or null, "confidence": 0-1, "raw_text": "exact text found", "notes": ""},
    "gross_profit": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "ebitda": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "operating_income": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "net_income": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "free_cash_flow": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "operating_cash_flow": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "capex": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "total_assets": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "total_debt": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "cash_and_equivalents": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "accounts_receivable": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""},
    "accounts_payable": {"value": number or null, "confidence": 0-1, "raw_text": "", "notes": ""}
  }
}
"""


def extract_text_from_file(file_path: str, file_bytes: bytes, filename: str) -> str:
    """Extract raw text from any supported file format."""
    suffix = Path(filename).suffix.lower()

    if suffix == '.pdf':
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except ImportError:
            raise RuntimeError(
                "PDF parsing requires 'pypdf'. Add to requirements.txt."
            )

    elif suffix in ('.xlsx', '.xls'):
        try:
            import openpyxl
            import io
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            lines = []
            for sheet in wb.worksheets:
                lines.append(f"=== Sheet: {sheet.title} ===")
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join(str(c) if c is not None else "" for c in row)
                    if row_text.strip():
                        lines.append(row_text)
            return "\n".join(lines)
        except ImportError:
            raise RuntimeError(
                "Excel parsing requires 'openpyxl'. Add to requirements.txt."
            )

    elif suffix == '.csv':
        return file_bytes.decode('utf-8', errors='replace')

    elif suffix in ('.txt', '.text'):
        return file_bytes.decode('utf-8', errors='replace')

    elif suffix in ('.docx', '.doc'):
        try:
            import docx2txt
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                return docx2txt.process(tmp_path) or ""
            finally:
                os.unlink(tmp_path)
        except ImportError:
            raise RuntimeError(
                "Word parsing requires 'docx2txt'. Add to requirements.txt."
            )

    elif suffix in ('.png', '.jpg', '.jpeg', '.webp'):
        return _extract_text_from_image(file_bytes, suffix)

    else:
        try:
            return file_bytes.decode('utf-8', errors='replace')
        except Exception:
            raise RuntimeError(f"Unsupported file format: {suffix}")


def _extract_text_from_image(image_bytes: bytes, suffix: str) -> str:
    """Use GPT-4o-mini vision to extract text from an image of a financial statement."""
    import base64
    from api.llm.openai_client import get_openai_client

    b64 = base64.b64encode(image_bytes).decode()
    media_type = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.webp': 'image/webp',
    }.get(suffix, 'image/jpeg')

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"}
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this financial document image. "
                        "Preserve numbers, labels, and structure as accurately as possible."
                    )
                }
            ]
        }],
        max_tokens=4000,
    )
    return response.choices[0].message.content or ""


def extract_financials_from_text(
    text: str,
    document_name: str = "document",
    entity_hint: Optional[str] = None,
) -> ExtractionResult:
    """
    Use GPT-4o-mini to extract structured financial data from text.

    Core AI extraction function. Returns a fully structured ExtractionResult
    with confidence scores per field.
    """
    from api.llm.openai_client import get_openai_client

    text_for_ai = text[:8000] if len(text) > 8000 else text

    client = get_openai_client()

    prompt = f"{AXIOM_EXTRACTION_SCHEMA}\n\nDOCUMENT TO ANALYZE:\n{text_for_ai}"
    if entity_hint:
        prompt += f"\n\nHINT: The entity is likely named '{entity_hint}'."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial data extraction expert. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0,
            response_format={"type": "json_object"},
        )

        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        return _parse_extraction_response(data, document_name, text[:500])

    except json.JSONDecodeError as exc:
        logger.error("extraction.json_parse_failed doc=%s err=%s", document_name, exc)
        return ExtractionResult(
            document_name=document_name,
            document_type="unknown",
            period=None,
            period_end_date=None,
            entity_name=entity_hint,
            extraction_notes=f"AI response was not valid JSON: {exc}",
            overall_confidence=0.0,
        )
    except Exception as exc:
        logger.error("extraction.failed doc=%s err=%s", document_name, exc)
        return ExtractionResult(
            document_name=document_name,
            document_type="unknown",
            period=None,
            period_end_date=None,
            entity_name=entity_hint,
            extraction_notes=f"Extraction failed: {exc}",
            overall_confidence=0.0,
        )


def _parse_extraction_response(
    data: Dict[str, Any],
    document_name: str,
    raw_text_sample: str,
) -> ExtractionResult:
    """Parse the AI JSON response into a typed ExtractionResult."""
    UNIT_MULTIPLIERS = {
        'actual': 1.0,
        'thousands': 1_000.0,
        'millions': 1_000_000.0,
        'billions': 1_000_000_000.0,
    }
    unit = data.get('unit', 'actual')
    multiplier = UNIT_MULTIPLIERS.get(unit, 1.0)

    fields_data = data.get('fields', {})
    extracted_fields: List[ExtractedField] = []
    needs_review: List[str] = []
    confidences: List[float] = []

    field_map: Dict[str, Optional[float]] = {}
    for field_name, field_info in fields_data.items():
        if not isinstance(field_info, dict):
            continue
        raw_value = field_info.get('value')
        confidence = float(field_info.get('confidence', 0.0))

        value = None
        if raw_value is not None:
            try:
                value = float(raw_value) * multiplier
            except (TypeError, ValueError):
                value = None
                confidence = 0.0

        ef = ExtractedField(
            field_name=field_name,
            value=value,
            raw_text=str(field_info.get('raw_text', '')),
            confidence=confidence,
            notes=str(field_info.get('notes', '')),
            needs_review=confidence < 0.7 and value is not None,
        )
        extracted_fields.append(ef)
        field_map[field_name] = value

        if value is not None:
            confidences.append(confidence)
        if ef.needs_review:
            needs_review.append(field_name)

    revenue = field_map.get('revenue')
    gross_profit = field_map.get('gross_profit')
    ebitda = field_map.get('ebitda')
    op_income = field_map.get('operating_income')
    net_income = field_map.get('net_income')
    fcf = field_map.get('free_cash_flow')

    def _safe_margin(num: Optional[float], denom: Optional[float]) -> Optional[float]:
        if num is not None and denom and abs(denom) > 0:
            return num / denom
        return None

    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return ExtractionResult(
        document_name=document_name,
        document_type=data.get('document_type', 'unknown'),
        period=data.get('period'),
        period_end_date=data.get('period_end_date'),
        entity_name=data.get('entity_name'),
        currency=data.get('currency', 'USD'),
        unit=unit,
        extraction_notes=data.get('extraction_notes', ''),
        raw_text_sample=raw_text_sample,

        revenue=revenue,
        gross_profit=gross_profit,
        ebitda=ebitda,
        operating_income=op_income,
        net_income=net_income,
        free_cash_flow=fcf,
        operating_cash_flow=field_map.get('operating_cash_flow'),
        capex=field_map.get('capex'),
        total_assets=field_map.get('total_assets'),
        total_debt=field_map.get('total_debt'),
        cash_and_equivalents=field_map.get('cash_and_equivalents'),
        accounts_receivable=field_map.get('accounts_receivable'),
        accounts_payable=field_map.get('accounts_payable'),

        gross_margin=_safe_margin(gross_profit, revenue),
        ebitda_margin=_safe_margin(ebitda, revenue),
        op_margin=_safe_margin(op_income, revenue),
        net_margin=_safe_margin(net_income, revenue),
        fcf_margin=_safe_margin(fcf, revenue),

        fields=extracted_fields,
        overall_confidence=overall_confidence,
        fields_needing_review=needs_review,
    )


def extract_from_file(
    file_bytes: bytes,
    filename: str,
    entity_hint: Optional[str] = None,
) -> ExtractionResult:
    """
    Main entry point: extract financial data from any file.

    Steps:
    1. Extract raw text from file (PDF/Excel/CSV/image)
    2. Send text to GPT-4o-mini for structured extraction
    3. Return ExtractionResult with confidence scores
    """
    try:
        text = extract_text_from_file("", file_bytes, filename)
    except RuntimeError as exc:
        return ExtractionResult(
            document_name=filename,
            document_type="unknown",
            period=None,
            period_end_date=None,
            entity_name=entity_hint,
            extraction_notes=f"Could not extract text: {exc}",
            overall_confidence=0.0,
        )

    if not text.strip():
        return ExtractionResult(
            document_name=filename,
            document_type="unknown",
            period=None,
            period_end_date=None,
            entity_name=entity_hint,
            extraction_notes="Document appears to be empty or unreadable.",
            overall_confidence=0.0,
        )

    return extract_financials_from_text(text, filename, entity_hint)
