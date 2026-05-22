from __future__ import annotations

import datetime as dt
import html
import json
import re
import uuid
from typing import Any, Dict

from api.assistant.reports import sanitize_payload
from api.platform.contracts import ExportManifest, RenderedExportResult
from api.platform.export_layouts import (
    build_export_layout_metadata,
    export_format_capabilities,
)
from api.platform.serializers import content_hash, stable_json_dumps


SUPPORTED_EXPORT_FORMATS = ("html", "markdown", "json")


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower())
    return normalized.strip("-") or "platform-export"


def _file_name_hint(manifest: ExportManifest, export_format: str) -> str:
    stem = _slugify(f"{manifest.title}-{manifest.export_id}")
    extension = {"html": "html", "markdown": "md", "json": "json"}.get(
        export_format, "txt"
    )
    return f"{stem}.{extension}"


def _metadata_lines(manifest: ExportManifest) -> list[str]:
    return [
        f"Pack Type: {manifest.pack_type}",
        f"Approval Status: {manifest.approval_status or 'n/a'}",
        f"Framework Version: {manifest.framework_version or 'n/a'}",
        f"Generated At: {manifest.generated_at or 'n/a'}",
        f"Evidence Summary: {manifest.evidence_summary or 'n/a'}",
    ]


def render_export_html(manifest: ExportManifest) -> str:
    section_blocks = []
    for section in manifest.ordered_sections:
        section_blocks.append(
            f"""
            <section class="export-section" data-section-key="{html.escape(section.section_key)}">
              <h2>{html.escape(section.title)}</h2>
              <p>{html.escape(section.content)}</p>
            </section>
            """.strip()
        )
    meta = "".join(
        f"<li>{html.escape(line)}</li>" for line in _metadata_lines(manifest)
    )
    return (
        "<!doctype html>"
        "<html lang=\"en\">"
        "<head>"
        "<meta charset=\"utf-8\" />"
        "<title>"
        + html.escape(manifest.title)
        + "</title>"
        "<style>"
        "@page{margin:0.75in;}"
        "body{font-family:Georgia,serif;background:#f5f1e8;color:#1d1d1d;margin:40px;line-height:1.5;}"
        ".export-shell{max-width:980px;margin:0 auto;background:#fff;border:1px solid #d9d2c1;padding:32px;}"
        ".export-kicker{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#6c6257;}"
        "h1{margin:8px 0 12px;font-size:30px;}h2{margin:0 0 8px;font-size:20px;}"
        ".export-meta{margin:0 0 28px;padding-left:18px;color:#544d45;}"
        ".export-section{border-top:1px solid #ebe2d2;padding-top:18px;margin-top:18px;}"
        ".export-footer{border-top:2px solid #1d1d1d;padding-top:18px;margin-top:26px;font-size:12px;color:#6c6257;}"
        "@media print{body{background:#fff;margin:0;} .export-shell{border:none;box-shadow:none;max-width:none;padding:0;}}"
        "</style>"
        "</head><body><div class=\"export-shell\">"
        f"<div class=\"export-kicker\">AXIOM Institutional Export</div><h1>{html.escape(manifest.title)}</h1>"
        f"<p>{html.escape(manifest.subtitle or '')}</p>"
        f"<ul class=\"export-meta\">{meta}</ul>"
        + "".join(section_blocks)
        + f"<div class=\"export-footer\">Export ID {html.escape(manifest.export_id)} · Checksum-ready payload</div>"
        "</div></body></html>"
    )


def render_export_markdown(manifest: ExportManifest) -> str:
    blocks = [
        f"# {manifest.title}",
        "",
        manifest.subtitle or "",
        "",
        "## Metadata",
        *[f"- {line}" for line in _metadata_lines(manifest)],
    ]
    for section in manifest.ordered_sections:
        blocks.extend(
            [
                "",
                f"## {section.title}",
                "",
                section.content or "No content available.",
            ]
        )
    blocks.extend(
        [
            "",
            "---",
            f"Export ID: {manifest.export_id}",
        ]
    )
    return "\n".join(blocks).strip() + "\n"


def render_export_json(manifest: ExportManifest) -> str:
    return stable_json_dumps(manifest.model_dump(mode="python"))


def render_export_manifest(
    manifest_payload: Dict[str, Any] | ExportManifest,
    *,
    export_format: str,
    metadata: Dict[str, Any] | None = None,
) -> RenderedExportResult:
    manifest = (
        manifest_payload
        if isinstance(manifest_payload, ExportManifest)
        else ExportManifest.model_validate(manifest_payload)
    )
    normalized_format = str(export_format or "html").strip().lower()
    if normalized_format not in SUPPORTED_EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {normalized_format}")
    if normalized_format == "html":
        rendered = render_export_html(manifest)
        content_type = "text/html; charset=utf-8"
    elif normalized_format == "markdown":
        rendered = render_export_markdown(manifest)
        content_type = "text/markdown; charset=utf-8"
    else:
        rendered = render_export_json(manifest)
        content_type = "application/json"
    layout_metadata = build_export_layout_metadata(
        manifest,
        export_format=normalized_format,
    )
    payload = {
        "render_id": str(uuid.uuid4()),
        "export_id": manifest.export_id,
        "export_format": normalized_format,
        "content_type": content_type,
        "rendered_content": rendered,
        "file_name_hint": _file_name_hint(manifest, normalized_format),
        "section_count": len(manifest.ordered_sections or []),
        "checksum": content_hash(
            {
                "export_id": manifest.export_id,
                "export_format": normalized_format,
                "rendered_content": rendered,
            }
        ),
        "generated_at": now_utc(),
        "metadata": sanitize_payload(
            {
                **dict(metadata or {}),
                "format_capabilities": export_format_capabilities(normalized_format),
                "layout_metadata": layout_metadata,
            }
        ),
    }
    return RenderedExportResult.model_validate(payload)
