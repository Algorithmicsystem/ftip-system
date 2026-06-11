/* PE Intelligence Dashboard Panel */

let _currentPEOrg = null;

async function loadPEDemo() {
  const body = document.getElementById('pe-body');
  if (!body) return;
  _currentPEOrg = 'DEAL_FLOW';
  body.innerHTML = [1,2,3].map(() =>
    '<div class="loading-skeleton skeleton-line full" style="height:60px;margin-bottom:8px;border-radius:6px;"></div>'
  ).join('');

  try {
    const dealFlow = await API.get('/pe/deal-flow').catch(() => null);
    const candidates = dealFlow?.candidates || [];

    if (!candidates.length) {
      body.innerHTML = `<div class="alert-banner info">No acquisition candidates found. Run the AXIOM pipeline to populate deal flow.</div>`;
      return;
    }

    body.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:4px;">
        Live Deal Flow — ${dealFlow.candidates_found || candidates.length} candidates · ${dealFlow.universe_screened || 0} screened
      </div>
      <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
        <input id="pe-symbol-search" type="text" placeholder="Symbol (e.g. AAPL)" style="flex:1;padding:5px 8px;font-size:11px;background:var(--bg-secondary);border:1px solid var(--border-subtle);border-radius:4px;color:var(--text-primary);">
        <button onclick="loadPESymbolForensic()" style="padding:5px 10px;font-size:11px;background:var(--accent-primary);color:#fff;border:none;border-radius:4px;cursor:pointer;">Forensic Analysis</button>
      </div>
      <div id="pe-deal-grid"></div>
      <div id="pe-exit-pipeline"></div>
      <div id="pe-schillit-alerts"></div>
      <div id="pe-forensic-result"></div>`;

    const grid = document.getElementById('pe-deal-grid');
    if (grid) {
      grid.innerHTML = `
        <div style="font-size:10px;color:var(--text-muted);display:grid;grid-template-columns:60px 1fr 70px 70px 60px;gap:6px;padding-bottom:4px;border-bottom:1px solid var(--border-subtle);margin-bottom:4px;">
          <span>Symbol</span><span>Rationale</span><span>Acq.</span><span>DAS</span><span>DAU</span>
        </div>
        ${candidates.slice(0, 10).map(c => {
          const acq = c.acquisition_score ?? 0;
          const das = c.das_score ?? acq;
          const dau = c.dau ?? 50;
          const dauCls = dau <= 40 ? 'var(--signal-sell)' : dau >= 65 ? 'var(--signal-buy)' : 'var(--signal-hold)';
          const dasCls = das >= 60 ? 'var(--signal-buy)' : das >= 40 ? 'var(--signal-hold)' : 'var(--signal-sell)';
          const grade = c.das_grade || '—';
          return `
            <div style="display:grid;grid-template-columns:60px 1fr 70px 70px 60px;gap:6px;padding:5px 0;border-bottom:1px solid var(--border-subtle);align-items:start;cursor:pointer;" onclick="document.getElementById('pe-symbol-search').value='${c.symbol}';loadPESymbolForensic();">
              <span style="font-size:12px;font-weight:700;color:var(--text-primary);">${c.symbol}</span>
              <span style="font-size:10px;color:var(--text-secondary);line-height:1.4;">${(c.rationale || '').substring(0, 80)}${(c.rationale||'').length > 80 ? '…' : ''}</span>
              <span style="font-family:var(--font-mono);font-size:12px;color:var(--accent-primary);text-align:right;">${acq.toFixed(0)}</span>
              <span style="font-family:var(--font-mono);font-size:12px;color:${dasCls};text-align:right;">${das.toFixed(0)} <span style="font-size:10px;">${grade}</span></span>
              <span style="font-family:var(--font-mono);font-size:12px;color:${dauCls};text-align:right;">${dau.toFixed(0)}</span>
            </div>`;
        }).join('')}`;
    }

    const exitCandidates = candidates.slice(0, 5).map(c => ({
      company_name: c.symbol,
      entity_id: c.symbol,
      exit_readiness_score: c.das_score ?? c.acquisition_score ?? 50,
      projected_moic: c.das_score ? (1.5 + c.das_score / 100) : null,
      exit_route: 'Acquisition / Strategic',
    }));
    renderExitPipeline(exitCandidates);
    renderSchilitAlerts(candidates.slice(0, 5).map(c => ({
      ...c,
      company_name: c.symbol,
      schillit_distress_score: c.schilit_risk === 'high' ? 75 : c.schilit_risk === 'medium' ? 45 : 15,
    })));
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load deal flow: ${err.message}</div>`;
  }
}

async function loadPESymbolForensic() {
  const sym = (document.getElementById('pe-symbol-search')?.value || '').trim().toUpperCase();
  if (!sym) return;
  const el = document.getElementById('pe-forensic-result');
  if (!el) return;
  el.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:60px;margin-bottom:8px;border-radius:6px;"></div>';
  try {
    const [forensic, das] = await Promise.all([
      API.get(`/pe/forensic/${sym}`).catch(() => null),
      API.get(`/pe/das/${sym}`).catch(() => null),
    ]);
    const risk = forensic?.overall_risk || 'unknown';
    const riskCls = risk === 'low' ? 'var(--signal-buy)' : risk === 'medium' ? 'var(--signal-hold)' : 'var(--signal-sell)';
    const dasScore = das?.das_score ?? das?.total ?? null;
    el.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:8px;margin-top:16px;">${sym} — Forensic + DAS</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:8px;">
        <div class="metric-card" style="text-align:center;padding:8px;">
          <span class="metric-card__label">Forensic Risk</span>
          <span class="metric-card__value" style="font-size:16px;color:${riskCls};text-transform:capitalize;">${risk}</span>
        </div>
        <div class="metric-card" style="text-align:center;padding:8px;">
          <span class="metric-card__label">DAS Score</span>
          <span class="metric-card__value" style="font-size:16px;">${dasScore != null ? dasScore.toFixed(1) : '—'}</span>
        </div>
        <div class="metric-card" style="text-align:center;padding:8px;">
          <span class="metric-card__label">DAS Grade</span>
          <span class="metric-card__value" style="font-size:16px;">${das?.das_grade || '—'}</span>
        </div>
        <div class="metric-card" style="text-align:center;padding:8px;">
          <span class="metric-card__label">EIS Impact</span>
          <span class="metric-card__value" style="font-size:16px;">${forensic?.eis_impact != null ? `-${forensic.eis_impact.toFixed(0)}` : '—'}</span>
        </div>
      </div>
      ${forensic?.forensic_summary ? `<div style="padding:8px;background:var(--bg-tertiary);border-radius:6px;font-size:11px;color:var(--text-secondary);line-height:1.6;margin-bottom:6px;">${forensic.forensic_summary}</div>` : ''}
      ${forensic?.red_flags?.length ? `<div style="font-size:11px;color:var(--signal-sell);margin-bottom:4px;"><strong>Red flags:</strong> ${forensic.red_flags.join(' · ')}</div>` : ''}
      ${forensic?.green_flags?.length ? `<div style="font-size:11px;color:var(--signal-buy);"><strong>Green flags:</strong> ${forensic.green_flags.join(' · ')}</div>` : ''}
      ${das?.investment_thesis ? `<div style="margin-top:8px;padding:8px;background:var(--bg-tertiary);border-radius:6px;font-size:11px;color:var(--text-secondary);">${das.investment_thesis}</div>` : ''}`;
  } catch (err) {
    el.innerHTML = `<div class="alert-banner warning">Forensic analysis failed: ${err.message}</div>`;
  }
}

async function loadPEPortfolio(orgId) {
  if (!orgId) orgId = document.getElementById('pe-org-input')?.value?.trim();
  if (!orgId) {
    // Auto-show demo when no org is selected
    await loadPEDemo();
    const body = document.getElementById('pe-body');
    if (body) {
      const banner = document.createElement('div');
      banner.className = 'alert-banner info';
      banner.style.cssText = 'margin-bottom:10px;font-size:11px;';
      banner.textContent = 'DEMO MODE — enter an org ID above and click Load to analyze a real portfolio';
      body.prepend(banner);
    }
    return;
  }

  _currentPEOrg = orgId;

  const body = document.getElementById('pe-body');
  if (!body) return;

  body.innerHTML = [1,2,3].map(() =>
    '<div class="loading-skeleton skeleton-line full" style="height:60px;margin-bottom:8px;border-radius:6px;"></div>'
  ).join('');

  try {
    const [overview, stressAlerts, lpReport] = await Promise.all([
      API.get(`/pe/portfolio/${orgId}/overview`).catch(() => null),
      API.get(`/pe/portfolio/${orgId}/stress-alerts`).catch(() => null),
      API.get(`/pe/portfolio/${orgId}/lp-report`).catch(() => null),
    ]);
    // stress-alerts is the closest existing route; exit-pipeline data comes from overview
    const pipeline = overview;

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
