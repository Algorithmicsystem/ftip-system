from __future__ import annotations

from typing import Any, Dict

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ExportFormatCapabilities, ExportManifest


PRINT_READY_FORMATS = {"html"}


def export_format_capabilities(export_format: str | None = None) -> Dict[str, Any]:
    normalized = str(export_format or "").strip().lower()
    capabilities = ExportFormatCapabilities(
        html_supported=True,
        markdown_supported=True,
        json_supported=True,
        pdf_ready=normalized == "html",
        docx_ready=False,
        print_ready_html=normalized == "html",
    )
    return capabilities.model_dump(mode="python")


def build_export_layout_metadata(
    manifest_payload: Dict[str, Any] | ExportManifest,
    *,
    export_format: str,
) -> Dict[str, Any]:
    if isinstance(manifest_payload, ExportManifest):
        pack_type = manifest_payload.pack_type
        sections = manifest_payload.ordered_sections or []
    else:
        pack_type = str(manifest_payload.get("pack_type") or "dossier_pack")
        sections = list(manifest_payload.get("ordered_sections") or [])
    ordered_section_keys = [
        str(getattr(section, "section_key", None) or (section.get("section_key") if isinstance(section, dict) else "section"))
        for section in sections
    ]
    normalized = str(export_format or "html").strip().lower()
    return sanitize_payload(
        {
            "pack_type": pack_type,
            "export_format": normalized,
            "cover_block_enabled": True,
            "summary_block_enabled": True,
            "appendix_enabled": True,
            "ordered_section_keys": ordered_section_keys,
            "section_count": len(ordered_section_keys),
            "print_ready": normalized in PRINT_READY_FORMATS,
            "pdf_ready": normalized == "html",
            "docx_ready": False,
            "document_structure": [
                "cover_block",
                "metadata_summary",
                "ordered_sections",
                "appendix_evidence",
            ],
        }
    )


__all__ = ["build_export_layout_metadata", "export_format_capabilities"]
