/**
 * AXIOM Core — shared infrastructure for SIGNAL, CAPITAL, CFO terminals
 * Handles auth, API, clock, charts, tables, panels, search, upload, export
 */

/* ─── Auth ─────────────────────────────────────────────────────────────── */
const AuthManager = {
  _key: null,

  async init() {
    this._key = localStorage.getItem('ftip_api_key') || '';
    if (!this._key) {
      try {
        const d = await fetch('/config/client').then(r => r.json());
        if (d.api_key) {
          this._key = d.api_key;
          localStorage.setItem('ftip_api_key', this._key);
        }
      } catch (_) {}
    }
    return !!this._key;
  },

  getKey() { return this._key || localStorage.getItem('ftip_api_key') || ''; },

  setKey(k) {
    this._key = k;
    localStorage.setItem('ftip_api_key', k);
  },
};

/* ─── API Client ────────────────────────────────────────────────────────── */
async function axiomFetch(path, opts = {}) {
  const headers = {
    'X-FTIP-API-Key': AuthManager.getKey(),
    ...(opts.headers || {}),
  };
  const r = await fetch(path, { ...opts, headers });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

/* ─── Toast ─────────────────────────────────────────────────────────────── */
const Toast = {
  show(msg, type = 'info', ms = 3500) {
    let wrap = document.getElementById('ax-toast-wrap');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.id = 'ax-toast-wrap';
      wrap.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:6px;';
      document.body.appendChild(wrap);
    }
    const colors = { info: '#58a6ff', success: '#3fb950', warning: '#d29922', error: '#f85149' };
    const t = document.createElement('div');
    t.className = 'ax-toast';
    t.style.cssText = `background:var(--ax-surface,#1c2128);border:1px solid ${colors[type] || colors.info};
      border-left:3px solid ${colors[type] || colors.info};border-radius:6px;padding:10px 14px;
      font-size:12px;color:var(--ax-text,#e6edf3);max-width:320px;animation:ax-fadeIn .2s ease;`;
    t.textContent = msg;
    wrap.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; setTimeout(() => t.remove(), 300); }, ms);
  },
};

/* ─── Clock ──────────────────────────────────────────────────────────────── */
function startClock(elId) {
  const el = document.getElementById(elId);
  if (!el) return;
  function tick() {
    const now = new Date();
    const et = new Intl.DateTimeFormat('en-US', {
      timeZone: 'America/New_York',
      weekday: 'short', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }).format(now);
    el.textContent = et + ' ET';
  }
  tick();
  setInterval(tick, 1000);
}

/* ─── Market Status ─────────────────────────────────────────────────────── */
function updateMarketStatus(elId) {
  const el = document.getElementById(elId);
  if (!el) return;
  const now = new Date();
  const et = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const h = et.getHours(), m = et.getMinutes(), day = et.getDay();
  const mins = h * 60 + m;
  const isWeekday = day >= 1 && day <= 5;
  const isOpen = isWeekday && mins >= 570 && mins < 960; // 9:30–16:00
  const isPremarket = isWeekday && mins >= 240 && mins < 570;
  const isAfterHours = isWeekday && mins >= 960 && mins < 1200;
  el.className = 'ax-market-status ' + (isOpen ? 'ax-market-open' : isPremarket ? 'ax-market-pre' : isAfterHours ? 'ax-market-after' : 'ax-market-closed');
  el.textContent = isOpen ? '● MARKET OPEN' : isPremarket ? '● PRE-MARKET' : isAfterHours ? '● AFTER HOURS' : '● MARKET CLOSED';
}

/* ─── Macro Strip ───────────────────────────────────────────────────────── */
const MacroStrip = {
  _data: null,
  async load(elId) {
    const el = document.getElementById(elId);
    if (!el) return;
    try {
      const d = await axiomFetch('/macro/snapshot');
      this._data = d;
      const assets = d.cross_asset || {};
      const items = Object.entries(assets).slice(0, 8).map(([k, v]) => {
        const chg = v.change_pct != null ? v.change_pct : 0;
        const cls = chg >= 0 ? 'ax-up' : 'ax-down';
        const sign = chg >= 0 ? '+' : '';
        return `<span class="ax-strip-item"><span class="ax-strip-label">${k}</span><span class="ax-strip-val ${cls}">${sign}${chg.toFixed(2)}%</span></span>`;
      }).join('');
      el.innerHTML = items || '<span class="ax-strip-item ax-muted">— macro unavailable —</span>';
    } catch (_) {
      el.innerHTML = '<span class="ax-strip-item ax-muted">— macro unavailable —</span>';
    }
  },
};

/* ─── Chart Helpers (Chart.js 4.x) ─────────────────────────────────────── */
function destroyIfExists(canvasId) {
  const existing = Chart.getChart(canvasId);
  if (existing) existing.destroy();
}

function buildLineChart(canvasId, labels, datasets, opts = {}) {
  destroyIfExists(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: datasets.map(ds => ({
        borderWidth: 2, pointRadius: 0, tension: 0.35, fill: ds.fill ?? false,
        borderColor: ds.color || '#58a6ff',
        backgroundColor: ds.fill ? (ds.color || '#58a6ff') + '18' : 'transparent',
        ...ds,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
      plugins: { legend: { display: !!opts.legend, labels: { color: '#8b949e', font: { size: 11 } } }, tooltip: { mode: 'index', intersect: false } },
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 6, font: { size: 10 } }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
      },
      ...opts.scaleOverrides,
    },
  });
}

function buildBarChart(canvasId, labels, data, color = '#58a6ff', opts = {}) {
  destroyIfExists(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data, backgroundColor: color + 'cc', borderColor: color, borderWidth: 1, borderRadius: 3, ...opts.dataset }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
      },
    },
  });
}

function buildDonutChart(canvasId, labels, data, colors, opts = {}) {
  destroyIfExists(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0, hoverOffset: 4 }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '65%',
      plugins: { legend: { position: 'right', labels: { color: '#8b949e', font: { size: 11 }, boxWidth: 10, padding: 8 } } },
      ...opts,
    },
  });
}

function buildSparkline(canvasId, data, color = '#58a6ff') {
  destroyIfExists(canvasId);
  const ctx = document.getElementById(canvasId)?.getContext('2d');
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.map((_, i) => i),
      datasets: [{ data, borderColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: color + '18' }],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
    },
  });
}

/* ─── Formatters ────────────────────────────────────────────────────────── */
const Fmt = {
  currency(v, decimals = 1) {
    if (v == null) return '—';
    const abs = Math.abs(v);
    const sign = v < 0 ? '-' : '';
    if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(decimals)}B`;
    if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(decimals)}M`;
    if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(decimals)}K`;
    return `${sign}$${abs.toFixed(0)}`;
  },
  pct(v, decimals = 1) {
    if (v == null) return '—';
    const num = Math.abs(v) > 1 ? v : v * 100;
    return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}%`;
  },
  score(v) {
    if (v == null) return '—';
    return parseFloat(v).toFixed(2);
  },
  num(v, decimals = 2) {
    if (v == null) return '—';
    return parseFloat(v).toFixed(decimals);
  },
  date(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' });
  },
};

/* ─── Score Renderers ───────────────────────────────────────────────────── */
function renderScoreBar(score, max = 100, color) {
  const pct = Math.min(100, (score / max) * 100);
  const c = color || (pct >= 70 ? '#3fb950' : pct >= 40 ? '#d29922' : '#f85149');
  return `<div class="ax-score-bar-wrap" style="display:flex;align-items:center;gap:6px;">
    <div style="flex:1;height:4px;background:#21262d;border-radius:2px;">
      <div style="width:${pct}%;height:100%;background:${c};border-radius:2px;"></div>
    </div>
    <span style="font-size:11px;color:${c};min-width:28px;text-align:right;">${parseFloat(score).toFixed(1)}</span>
  </div>`;
}

function renderSignalBadge(signal) {
  const map = {
    BUY:      { bg: '#3fb95022', color: '#3fb950', text: 'BUY' },
    SELL:     { bg: '#f8514922', color: '#f85149', text: 'SELL' },
    HOLD:     { bg: '#d2992222', color: '#d29922', text: 'HOLD' },
    STRONG_BUY:  { bg: '#3fb95033', color: '#3fb950', text: 'STRONG BUY' },
    STRONG_SELL: { bg: '#f8514933', color: '#f85149', text: 'STRONG SELL' },
  };
  const s = signal?.toUpperCase().replace(' ', '_');
  const cfg = map[s] || { bg: '#8b949e22', color: '#8b949e', text: signal || '—' };
  return `<span style="padding:2px 7px;border-radius:3px;font-size:10px;font-weight:600;background:${cfg.bg};color:${cfg.color};">${cfg.text}</span>`;
}

function renderRiskBadge(risk) {
  const map = {
    low:      '#3fb950',
    medium:   '#d29922',
    high:     '#f0883e',
    critical: '#f85149',
  };
  const c = map[risk?.toLowerCase()] || '#8b949e';
  return `<span style="padding:2px 7px;border-radius:3px;font-size:10px;font-weight:600;background:${c}22;color:${c};">${(risk || '—').toUpperCase()}</span>`;
}

/* ─── Table Builder ──────────────────────────────────────────────────────── */
function buildSortableTable(containerId, columns, rows, opts = {}) {
  const el = document.getElementById(containerId);
  if (!el) return;

  let sortCol = opts.defaultSort ?? 0;
  let sortDir = opts.defaultDir ?? 'desc';

  function render() {
    const sorted = [...rows].sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol];
      if (av == null) return 1;
      if (bv == null) return -1;
      const cmp = typeof av === 'string' ? av.localeCompare(bv) : av - bv;
      return sortDir === 'asc' ? cmp : -cmp;
    });

    const thead = columns.map((c, i) => {
      const arrow = i === sortCol ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '';
      return `<th style="cursor:pointer;white-space:nowrap;" data-col="${i}">${c.label}${arrow}</th>`;
    }).join('');

    const tbody = sorted.map(row => {
      const cells = columns.map((c, i) => {
        const raw = row[i];
        const rendered = c.render ? c.render(raw, row) : (raw ?? '—');
        return `<td>${rendered}</td>`;
      }).join('');
      return `<tr class="ax-row-hover">${cells}</tr>`;
    }).join('');

    el.innerHTML = `
      <table class="ax-table" style="width:100%;border-collapse:collapse;">
        <thead><tr>${thead}</tr></thead>
        <tbody>${tbody}</tbody>
      </table>`;

    el.querySelectorAll('th[data-col]').forEach(th => {
      th.addEventListener('click', () => {
        const col = parseInt(th.dataset.col);
        if (col === sortCol) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
        else { sortCol = col; sortDir = 'desc'; }
        render();
        if (opts.onSort) opts.onSort(sortCol, sortDir);
      });
    });

    if (opts.onRowClick) {
      el.querySelectorAll('tbody tr').forEach((tr, i) => {
        tr.style.cursor = 'pointer';
        tr.addEventListener('click', () => opts.onRowClick(sorted[i], i));
      });
    }
  }

  render();
}

/* ─── Panel Framework ────────────────────────────────────────────────────── */
class AxiomPanel {
  constructor(id, loader, opts = {}) {
    this.id = id;
    this.loader = loader;
    this.intervalMs = opts.intervalMs || 0;
    this.autoStart = opts.autoStart !== false;
    this._timer = null;
    if (this.autoStart) this.load();
  }

  async load() {
    try {
      await this.loader();
    } catch (e) {
      const el = document.getElementById(this.id);
      if (el) el.innerHTML = `<div class="ax-alert-warn" style="font-size:12px;">Failed to load: ${e.message}</div>`;
    }
  }

  startRefresh(ms) {
    const interval = ms || this.intervalMs;
    if (!interval) return;
    this._timer = setInterval(() => this.load(), interval);
  }

  stop() {
    if (this._timer) clearInterval(this._timer);
  }
}

/* ─── Tab Manager ────────────────────────────────────────────────────────── */
function initTabs(tabGroupSelector, onActivate) {
  const tabs = document.querySelectorAll(tabGroupSelector);
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const group = tab.closest('[data-tab-group]');
      const groupId = group?.dataset.tabGroup;
      document.querySelectorAll(`[data-tab-group="${groupId}"] .ax-tab`).forEach(t => t.classList.remove('ax-tab--active'));
      tab.classList.add('ax-tab--active');
      const panel = tab.dataset.panel;
      document.querySelectorAll(`[data-tab-group="${groupId}"] .ax-panel`).forEach(p => {
        p.style.display = p.id === panel ? '' : 'none';
      });
      if (onActivate) onActivate(panel, tab);
    });
  });

  // Activate first tab in each group
  document.querySelectorAll('[data-tab-group]').forEach(group => {
    const first = group.querySelector('.ax-tab');
    if (first) first.click();
  });
}

/* ─── Search (Universe Autocomplete) ────────────────────────────────────── */
class AxiomSearch {
  constructor(inputId, resultsId, onSelect) {
    this.input = document.getElementById(inputId);
    this.results = document.getElementById(resultsId);
    this.onSelect = onSelect;
    this._universe = [];
    this._load();
    if (this.input) this.input.addEventListener('input', () => this._suggest());
  }

  async _load() {
    try {
      const d = await axiomFetch('/intelligence/universe/scores');
      this._universe = Array.isArray(d) ? d : (d.scores || d.data || []);
    } catch (_) {}
  }

  _suggest() {
    const q = this.input?.value?.trim()?.toUpperCase();
    if (!q || q.length < 1) { if (this.results) this.results.innerHTML = ''; return; }
    const matches = this._universe.filter(s => (s.symbol || s.ticker || '').startsWith(q)).slice(0, 8);
    if (!this.results) return;
    if (!matches.length) { this.results.innerHTML = ''; return; }
    this.results.innerHTML = matches.map(s => {
      const sym = s.symbol || s.ticker;
      const sig = s.signal || '';
      return `<div class="ax-search-item" data-sym="${sym}" style="padding:6px 10px;cursor:pointer;display:flex;justify-content:space-between;font-size:12px;">
        <span>${sym}</span>${sig ? renderSignalBadge(sig) : ''}
      </div>`;
    }).join('');
    this.results.querySelectorAll('.ax-search-item').forEach(item => {
      item.addEventListener('click', () => {
        if (this.input) this.input.value = item.dataset.sym;
        this.results.innerHTML = '';
        if (this.onSelect) this.onSelect(item.dataset.sym);
      });
    });
  }
}

/* ─── Upload Handler ─────────────────────────────────────────────────────── */
class AxiomUpload {
  constructor(containerId, entityId, entityType, onSuccess) {
    this.containerId = containerId;
    this.entityId = entityId;
    this.entityType = entityType;
    this.onSuccess = onSuccess;
    this._init();
  }

  _init() {
    const el = document.getElementById(this.containerId);
    if (!el) return;
    el.innerHTML = `
      <div class="ax-drop-zone" id="${this.containerId}-drop" style="cursor:pointer;">
        <input type="file" id="${this.containerId}-input" accept=".pdf,.xlsx,.xls,.csv,.txt,.docx,.png,.jpg,.jpeg" style="display:none;">
        <div style="font-size:22px;margin-bottom:6px;">📄</div>
        <div style="font-size:12px;font-weight:600;margin-bottom:3px;">Drop financial document here</div>
        <div style="font-size:10px;color:var(--ax-muted);">PDF · Excel · CSV · Word · Image</div>
        <button onclick="document.getElementById('${this.containerId}-input').click()"
                style="margin-top:8px;padding:4px 12px;font-size:11px;background:var(--ax-accent);color:#fff;border:none;border-radius:4px;cursor:pointer;">
          Browse
        </button>
      </div>
      <div id="${this.containerId}-preview" style="display:none;margin-top:8px;"></div>`;
    const drop = document.getElementById(`${this.containerId}-drop`);
    const input = document.getElementById(`${this.containerId}-input`);
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('ax-drop-zone--over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('ax-drop-zone--over'));
    drop.addEventListener('drop', async e => {
      e.preventDefault(); drop.classList.remove('ax-drop-zone--over');
      const f = e.dataTransfer.files[0];
      if (f) await this._handle(f);
    });
    drop.addEventListener('click', e => { if (e.target.tagName !== 'BUTTON') input.click(); });
    input.addEventListener('change', async () => { if (input.files[0]) await this._handle(input.files[0]); });
  }

  async _handle(file) {
    const preview = document.getElementById(`${this.containerId}-preview`);
    if (!preview) return;
    preview.style.display = 'block';
    preview.innerHTML = `<div style="font-size:12px;color:var(--ax-muted);padding:8px;">AI reading document...</div>`;
    try {
      const fd = new FormData();
      fd.append('file', file);
      if (this.entityId) fd.append('entity_hint', this.entityId);
      const r = await fetch('/extract/preview', {
        method: 'POST',
        headers: { 'X-FTIP-API-Key': AuthManager.getKey() },
        body: fd,
      });
      if (!r.ok) throw new Error(`${r.status}`);
      const ext = await r.json();
      this._renderPreview(preview, ext, file);
    } catch (e) {
      preview.innerHTML = `<div class="ax-alert-warn" style="font-size:11px;">Upload failed: ${e.message}</div>`;
    }
  }

  _renderPreview(container, ext, file) {
    const conf = ext.overall_confidence || 0;
    const pct = Math.round(conf * 100);
    const clr = conf >= 0.8 ? '#3fb950' : conf >= 0.6 ? '#d29922' : '#f85149';
    const label = conf >= 0.8 ? 'High confidence' : conf >= 0.6 ? 'Review recommended' : 'Manual review required';
    const fmt = v => v != null ? Fmt.currency(v) : null;
    const fmtP = v => v != null ? Fmt.pct(v) : null;
    const fields = [
      ['Revenue', fmt(ext.revenue)], ['Gross Margin', fmtP(ext.gross_margin)],
      ['EBITDA', fmt(ext.ebitda)], ['Net Income', fmt(ext.net_income)],
      ['FCF', fmt(ext.free_cash_flow)],
    ].filter(([, v]) => v);
    container.innerHTML = `
      <div style="border:1px solid ${clr}33;border-left:3px solid ${clr};border-radius:6px;padding:10px;background:var(--ax-surface2,#161b22);">
        <div style="font-size:11px;color:${clr};font-weight:600;margin-bottom:8px;">${label} — ${pct}% · ${ext.document_type || 'document'}${ext.period ? ' · ' + ext.period : ''}</div>
        ${fields.length ? `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:5px;margin-bottom:8px;">
          ${fields.map(([l, v]) => `<div class="ax-kpi" style="text-align:center;padding:6px;"><div class="ax-kpi__label" style="font-size:9px;">${l}</div><div class="ax-kpi__value" style="font-size:13px;">${v}</div></div>`).join('')}
        </div>` : ''}
        ${ext.fields_needing_review?.length ? `<div style="font-size:10px;color:#d29922;margin-bottom:6px;">Review: ${ext.fields_needing_review.join(', ')}</div>` : ''}
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button id="${this.containerId}-apply-btn" style="padding:4px 12px;font-size:11px;background:#3fb950;color:#fff;border:none;border-radius:4px;cursor:pointer;">
            Apply to ${this.entityId || 'entity'}
          </button>
          <button onclick="document.getElementById('${this.containerId}-input').click()"
                  style="padding:4px 12px;font-size:11px;background:var(--ax-surface,#1c2128);color:var(--ax-text-muted,#8b949e);border:1px solid var(--ax-border,#30363d);border-radius:4px;cursor:pointer;">
            Different File
          </button>
        </div>
      </div>`;
    document.getElementById(`${this.containerId}-apply-btn`)?.addEventListener('click', async () => {
      const btn = document.getElementById(`${this.containerId}-apply-btn`);
      if (btn) btn.textContent = 'Saving...';
      try {
        const endpoint = this.entityType === 'pe_portco'
          ? `/extract/pe/portco/${this.entityId}`
          : `/extract/smb/entity/${this.entityId}`;
        const fd = new FormData(); fd.append('file', file);
        const r = await fetch(endpoint, { method: 'POST', headers: { 'X-FTIP-API-Key': AuthManager.getKey() }, body: fd });
        if (!r.ok) throw new Error(`${r.status}`);
        const result = await r.json();
        if (this.onSuccess) this.onSuccess(result.extraction, result.intelligence);
        container.innerHTML = `<div class="ax-alert-ok" style="font-size:11px;">Saved — ${pct}% confidence.</div>`;
        Toast.show('Document extracted and applied.', 'success');
      } catch (e) {
        container.innerHTML += `<div class="ax-alert-warn" style="font-size:11px;margin-top:4px;">Save failed: ${e.message}</div>`;
      }
    });
  }
}

/* ─── KPI Card Helper ────────────────────────────────────────────────────── */
function setKPI(id, value, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  if (opts.color) el.style.color = opts.color;
  if (opts.cls) el.className = (el.className || '') + ' ' + opts.cls;
  el.textContent = value ?? '—';
}

function renderSkeletons(containerId, count = 3) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = Array(count).fill(`<div class="ax-skel" style="height:60px;border-radius:6px;margin-bottom:8px;"></div>`).join('');
}

/* ─── Export ─────────────────────────────────────────────────────────────── */
const ExportManager = {
  toCSV(data, filename = 'axiom-export.csv') {
    if (!data?.length) return;
    const keys = Object.keys(data[0]);
    const rows = [keys.join(','), ...data.map(r => keys.map(k => JSON.stringify(r[k] ?? '')).join(','))];
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = filename; a.click();
    URL.revokeObjectURL(a.href);
  },
  toJSON(data, filename = 'axiom-export.json') {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = filename; a.click();
    URL.revokeObjectURL(a.href);
  },
};

/* ─── Boot ───────────────────────────────────────────────────────────────── */
async function axiomBoot(opts = {}) {
  await AuthManager.init();
  if (opts.clockEl) startClock(opts.clockEl);
  if (opts.marketEl) {
    updateMarketStatus(opts.marketEl);
    setInterval(() => updateMarketStatus(opts.marketEl), 60000);
  }
  if (opts.macroEl) MacroStrip.load(opts.macroEl);
  if (opts.tabGroup) initTabs(`.${opts.tabGroup} .ax-tab`, opts.onTabActivate);
}
