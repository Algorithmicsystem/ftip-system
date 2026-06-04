/* PE Intelligence Dashboard Panel */

let _currentPEOrg = null;

async function loadPEPortfolio(orgId) {
  if (!orgId) orgId = document.getElementById('pe-org-input')?.value?.trim();
  if (!orgId) return;

  _currentPEOrg = orgId;

  const body = document.getElementById('pe-body');
  if (!body) return;

  body.innerHTML = [1,2,3].map(() =>
    '<div class="loading-skeleton skeleton-line full" style="height:60px;margin-bottom:8px;border-radius:6px;"></div>'
  ).join('');

  try {
    const [overview, pipeline, lpReport] = await Promise.all([
      API.get(`/pe/portfolio/${orgId}/overview`).catch(() => null),
      API.get(`/pe/portfolio/${orgId}/exit-pipeline`).catch(() => null),
      API.get(`/pe/portfolio/${orgId}/lp-report`).catch(() => null),
    ]);

    if (!overview && !pipeline && !lpReport) {
      body.innerHTML = `<div class="alert-banner info">No PE data available for org: ${orgId}</div>`;
      return;
    }

    renderPortfolioHealthGrid(overview?.entities || []);
    renderExitPipeline(pipeline?.pipeline || []);
    renderSchilitAlerts(overview?.entities || []);
    renderLPReportSummary(lpReport);
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load PE portfolio: ${err.message}</div>`;
  }
}

function renderPortfolioHealthGrid(entities) {
  const body = document.getElementById('pe-body');
  if (!body) return;

  const summaryHTML = entities.length > 0 ? `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-bottom:14px;">
      ${entities.slice(0, 12).map(e => {
        const score = e.health_score ?? 50;
        const scoreCls = score >= 70 ? 'var(--signal-buy)' : score >= 40 ? 'var(--signal-hold)' : 'var(--signal-sell)';
        const runway = e.cash_runway_months != null ? `${e.cash_runway_months}mo` : '—';
        return `
          <div class="metric-card" style="padding:8px;cursor:pointer;" onclick="loadPEEntityDetail('${e.entity_id || e.company_name}')">
            <div style="font-size:11px;font-weight:600;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${e.company_name || e.entity_id}</div>
            <div style="font-family:var(--font-mono);font-size:18px;font-weight:700;color:${scoreCls};margin:4px 0;">${score.toFixed(0)}</div>
            <div style="font-size:10px;color:var(--text-muted);">Runway: ${runway}</div>
          </div>`;
      }).join('')}
    </div>` : '<div class="alert-banner info">No portfolio entities found.</div>';

  body.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:8px;">Portfolio Health</div>
    ${summaryHTML}
    <div id="pe-exit-pipeline"></div>
    <div id="pe-schillit-alerts"></div>
    <div id="pe-lp-summary"></div>`;
}

function renderExitPipeline(pipeline) {
  const el = document.getElementById('pe-exit-pipeline');
  if (!el) return;

  if (!pipeline || pipeline.length === 0) {
    el.innerHTML = `<div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Exit Pipeline</div>
      <div class="text-muted text-sm">No exits in pipeline.</div>`;
    return;
  }

  const rows = pipeline.map(p => {
    const readiness = p.exit_readiness_score ?? 0;
    const readCls = readiness >= 70 ? 'var(--signal-buy)' : readiness >= 40 ? 'var(--signal-hold)' : 'var(--signal-sell)';
    const moic = p.projected_moic != null ? `${p.projected_moic.toFixed(2)}x` : '—';
    const route = p.exit_route || 'Unknown';
    return `
      <div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border-subtle);">
        <div style="flex:1;font-size:12px;font-weight:600;color:var(--text-primary);">${p.company_name || p.entity_id}</div>
        <div style="font-size:11px;color:var(--text-muted);">${route}</div>
        <div style="font-family:var(--font-mono);font-size:12px;color:${readCls};">${readiness.toFixed(0)}</div>
        <div style="font-family:var(--font-mono);font-size:12px;color:var(--accent-primary);">${moic}</div>
      </div>`;
  }).join('');

  el.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Exit Pipeline</div>
    <div style="font-size:10px;color:var(--text-muted);display:flex;gap:10px;padding-bottom:4px;border-bottom:1px solid var(--border-subtle);">
      <span style="flex:1;">Company</span><span>Route</span><span>Readiness</span><span>MOIC</span>
    </div>
    ${rows}`;
}

function renderSchilitAlerts(entities) {
  const el = document.getElementById('pe-schillit-alerts');
  if (!el) return;

  const alerts = entities
    .filter(e => (e.schillit_distress_score ?? 0) >= 60 || (e.cash_runway_months ?? 999) < 12)
    .sort((a, b) => (b.schillit_distress_score ?? 0) - (a.schillit_distress_score ?? 0))
    .slice(0, 5);

  if (alerts.length === 0) {
    el.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Distress Alerts</div>
      <div class="alert-banner success" style="font-size:11px;">No distress signals detected across portfolio.</div>`;
    return;
  }

  const alertRows = alerts.map(e => {
    const score = e.schillit_distress_score ?? 0;
    const runway = e.cash_runway_months != null ? `${e.cash_runway_months}mo runway` : '';
    return `
      <div class="alert-banner ${score >= 80 ? 'danger' : 'warning'}" style="font-size:11px;margin-bottom:4px;">
        <strong>${e.company_name || e.entity_id}</strong> — Distress: ${score.toFixed(0)}${runway ? ` · ${runway}` : ''}
      </div>`;
  }).join('');

  el.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Distress Alerts</div>
    ${alertRows}`;
}

function renderLPReportSummary(report) {
  const el = document.getElementById('pe-lp-summary');
  if (!el || !report) return;

  const tvpi = report.tvpi != null ? report.tvpi.toFixed(2) : '—';
  const dpi  = report.dpi  != null ? report.dpi.toFixed(2)  : '—';
  const irr  = report.net_irr != null ? `${(report.net_irr * 100).toFixed(1)}%` : '—';
  const vintage = report.vintage_year || '—';

  el.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:8px;margin-top:14px;">LP Report Summary</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">
      ${[['TVPI', tvpi], ['DPI', dpi], ['Net IRR', irr], ['Vintage', vintage]].map(([label, val]) => `
        <div class="metric-card" style="text-align:center;padding:8px;">
          <span class="metric-card__label">${label}</span>
          <span class="metric-card__value" style="font-size:16px;">${val}</span>
        </div>`).join('')}
    </div>
    ${report.narrative ? `
    <div style="margin-top:10px;padding:8px;background:var(--bg-tertiary);border-radius:6px;font-size:11px;color:var(--text-secondary);line-height:1.6;">
      ${report.narrative}
    </div>` : ''}`;
}

function loadPEEntityDetail(entityId) {
  const input = document.getElementById('pe-org-input');
  if (input) input.value = entityId;
  loadPEPortfolio(entityId);
}
