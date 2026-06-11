/* AXIOM AI Document Upload Component
 * Reusable drag-and-drop zone for PE portco and SMB entity financial uploads.
 */

function createUploadZone(containerId, entityId, entityType, onSuccess) {
  const el = document.getElementById(containerId);
  if (!el) return;

  el.innerHTML = `
    <div class="upload-zone__drop-area" id="${containerId}-drop"
         style="border:2px dashed var(--border-subtle);border-radius:8px;padding:20px;text-align:center;cursor:pointer;background:var(--bg-secondary);transition:border-color .2s;">
      <input type="file" id="${containerId}-input" accept=".pdf,.xlsx,.xls,.csv,.txt,.docx,.png,.jpg,.jpeg"
             style="display:none;">
      <div style="font-size:24px;margin-bottom:8px;">📄</div>
      <div style="font-size:13px;font-weight:600;color:var(--text-primary);margin-bottom:4px;">
        Drop financial statement here
      </div>
      <div style="font-size:11px;color:var(--text-muted);">PDF, Excel, CSV, Word, or Image</div>
      <div style="margin-top:8px;">
        <button onclick="document.getElementById('${containerId}-input').click()"
                style="padding:5px 12px;font-size:11px;background:var(--accent-primary);color:#fff;border:none;border-radius:4px;cursor:pointer;">
          Browse File
        </button>
      </div>
    </div>
    <div class="upload-zone__preview" id="${containerId}-preview" style="display:none;margin-top:10px;"></div>`;

  const dropArea = document.getElementById(`${containerId}-drop`);
  const fileInput = document.getElementById(`${containerId}-input`);

  // Drag-and-drop handlers
  dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.style.borderColor = 'var(--accent-primary)';
  });
  dropArea.addEventListener('dragleave', () => {
    dropArea.style.borderColor = 'var(--border-subtle)';
  });
  dropArea.addEventListener('drop', async e => {
    e.preventDefault();
    dropArea.style.borderColor = 'var(--border-subtle)';
    const file = e.dataTransfer.files[0];
    if (file) await _handleUploadFile(file, containerId, entityId, entityType, onSuccess);
  });
  dropArea.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (file) await _handleUploadFile(file, containerId, entityId, entityType, onSuccess);
  });
}

async function _handleUploadFile(file, containerId, entityId, entityType, onSuccess) {
  const preview = document.getElementById(`${containerId}-preview`);
  if (!preview) return;

  preview.style.display = 'block';
  preview.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;padding:10px;background:var(--bg-secondary);border-radius:6px;">
      <div class="loading-skeleton" style="width:20px;height:20px;border-radius:50%;flex-shrink:0;"></div>
      <div style="font-size:12px;color:var(--text-secondary);">AI is reading your document...</div>
    </div>`;

  try {
    const extraction = await uploadAndExtract(file, entityId, entityType);
    _renderExtractionPreview(preview, extraction, file, containerId, entityId, entityType, onSuccess);
  } catch (err) {
    preview.innerHTML = `<div class="alert-banner warning" style="font-size:11px;">Upload failed: ${err.message}</div>`;
  }
}

async function uploadAndExtract(file, entityId, entityType) {
  const formData = new FormData();
  formData.append('file', file);
  if (entityId) formData.append('entity_hint', entityId);

  const r = await fetch('/extract/preview', {
    method: 'POST',
    headers: { 'X-FTIP-API-Key': localStorage.getItem('ftip_api_key') || '' },
    body: formData,
  });
  if (!r.ok) throw new Error(`Server error: ${r.status}`);
  return r.json();
}

async function applyExtraction(file, entityId, entityType) {
  const formData = new FormData();
  formData.append('file', file);

  const endpoint = entityType === 'pe_portco'
    ? `/extract/pe/portco/${entityId}`
    : `/extract/smb/entity/${entityId}`;

  const r = await fetch(endpoint, {
    method: 'POST',
    headers: { 'X-FTIP-API-Key': localStorage.getItem('ftip_api_key') || '' },
    body: formData,
  });
  if (!r.ok) throw new Error(`Server error: ${r.status}`);
  return r.json();
}

function _renderExtractionPreview(container, extraction, file, containerId, entityId, entityType, onSuccess) {
  const conf = extraction.overall_confidence || 0;
  const confPct = Math.round(conf * 100);
  const confCls = conf >= 0.8 ? 'var(--signal-buy)' : conf >= 0.6 ? 'var(--signal-hold)' : 'var(--signal-sell)';
  const confLabel = conf >= 0.8 ? 'High confidence' : conf >= 0.6 ? 'Review recommended' : 'Manual review required';
  const confBannerCls = conf >= 0.8 ? 'success' : conf >= 0.6 ? 'warning' : 'danger';

  const fmt = v => v != null ? `$${(v / 1e6).toFixed(1)}M` : '—';
  const fmtPct = v => v != null ? `${(v * 100).toFixed(1)}%` : '—';

  const keyFields = [
    { label: 'Revenue', value: fmt(extraction.revenue) },
    { label: 'Gross Margin', value: fmtPct(extraction.gross_margin) },
    { label: 'EBITDA', value: fmt(extraction.ebitda) },
    { label: 'Net Income', value: fmt(extraction.net_income) },
    { label: 'FCF', value: fmt(extraction.free_cash_flow) },
  ].filter(f => f.value !== '—');

  container.innerHTML = `
    <div class="alert-banner ${confBannerCls}" style="font-size:11px;margin-bottom:8px;">
      <strong>${confLabel}</strong> — ${confPct}% confidence · ${extraction.document_type || 'unknown'}
      ${extraction.entity_name ? ` · Entity: ${extraction.entity_name}` : ''}
      ${extraction.period ? ` · Period: ${extraction.period}` : ''}
    </div>
    ${keyFields.length ? `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px;margin-bottom:10px;">
      ${keyFields.map(f => `
        <div class="metric-card" style="padding:7px;text-align:center;">
          <span class="metric-card__label" style="font-size:10px;">${f.label}</span>
          <span class="metric-card__value" style="font-size:14px;">${f.value}</span>
        </div>`).join('')}
    </div>` : '<div class="alert-banner info" style="font-size:11px;margin-bottom:8px;">No financial figures extracted.</div>'}
    ${extraction.fields_needing_review?.length ? `
    <div style="font-size:11px;color:var(--signal-hold);margin-bottom:8px;">
      Review needed: ${extraction.fields_needing_review.join(', ')}
    </div>` : ''}
    ${extraction.extraction_notes ? `
    <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">${extraction.extraction_notes}</div>` : ''}
    <div style="display:flex;gap:8px;flex-wrap:wrap;">
      <button id="${containerId}-apply-btn"
              style="padding:5px 12px;font-size:11px;background:var(--signal-buy);color:#fff;border:none;border-radius:4px;cursor:pointer;">
        Apply to ${entityId || 'entity'}
      </button>
      <button onclick="document.getElementById('${containerId}-input').click()"
              style="padding:5px 12px;font-size:11px;background:var(--bg-tertiary);color:var(--text-secondary);border:1px solid var(--border-subtle);border-radius:4px;cursor:pointer;">
        Upload Different File
      </button>
    </div>`;

  document.getElementById(`${containerId}-apply-btn`)?.addEventListener('click', async () => {
    const btn = document.getElementById(`${containerId}-apply-btn`);
    if (btn) btn.textContent = 'Saving...';
    try {
      const result = await applyExtraction(file, entityId, entityType);
      if (onSuccess) onSuccess(result.extraction, result.intelligence);
      container.innerHTML = `<div class="alert-banner success" style="font-size:11px;">
        Saved and intelligence updated. Confidence: ${confPct}%</div>`;
    } catch (err) {
      container.innerHTML += `<div class="alert-banner warning" style="font-size:11px;margin-top:6px;">Save failed: ${err.message}</div>`;
    }
  });
}
