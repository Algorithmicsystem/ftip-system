/* Symbol Intelligence Deep-Dive Panel */

let _currentSymbol = null;

async function loadSymbolIntelligence(symbol) {
  if (!symbol) symbol = document.getElementById('symbol-search')?.value?.trim()?.toUpperCase();
  if (!symbol) return;

  _currentSymbol = symbol;

  // Update header
  const badgeEl = document.getElementById('symbol-signal-badge');
  const ratingEl = document.getElementById('symbol-analyst-rating');
  const body = document.getElementById('symbol-body');

  if (badgeEl) { badgeEl.className = 'signal-badge hold'; badgeEl.textContent = 'Loading…'; }
  if (ratingEl) ratingEl.textContent = symbol;

  // Show skeleton in active tab
  const activePane = document.querySelector('.tab-pane.active');
  if (activePane) {
    activePane.innerHTML = `
      <div class="loading-skeleton skeleton-line full" style="margin-bottom:8px;"></div>
      <div class="loading-skeleton skeleton-line medium" style="margin-bottom:8px;"></div>
      <div class="loading-skeleton skeleton-line short"></div>`;
  }

  try {
    const [intel, explain] = await Promise.all([
      API.get(`/intelligence/universal/${symbol}`).catch(() => null),
      API.get(`/explain/${symbol}`).catch(() => null),
    ]);

    renderIntelligenceTab(intel, symbol);
    renderRiskTab(intel, symbol);
    renderExplanationTab(explain, intel);

    // Update header
    if (intel && badgeEl) {
      const sig = (intel.signal_label || 'HOLD').toLowerCase().replace('_', '-').replace(' ', '-');
      badgeEl.className = `signal-badge ${sig}`;
      badgeEl.textContent = intel.signal_label || 'HOLD';
    }
    if (intel && ratingEl) {
      ratingEl.textContent = `${symbol} · ${intel.analyst_rating || '—'} · DAU ${(intel.dau || 0).toFixed(1)}`;
    }
  } catch (err) {
    if (activePane) activePane.innerHTML = `<div class="alert-banner warning">Could not load intelligence for ${symbol}: ${err.message}</div>`;
  }
}

function renderIntelligenceTab(data, symbol) {
  const el = document.getElementById('tab-intelligence');
  if (!el) return;

  if (!data) {
    el.innerHTML = `<div class="alert-banner info">No AXIOM intelligence available for ${symbol}.</div>`;
    return;
  }

  const scores = `
    ${scoreBarHTML('EIS Score', data.eis_score, '#10b981')}
    ${scoreBarHTML('CAPS Score', data.caps_score, '#3b82f6')}
    ${scoreBarHTML('Factor Composite', data.factor_composite_score, '#8b5cf6')}
    ${data.osms_score != null ? scoreBarHTML('OSMS (Alt Data)', data.osms_score, '#f59e0b') : ''}
    ${data.ias_score != null ? scoreBarHTML('IAS (Insider)', data.ias_score, '#f59e0b') : ''}`;

  const evidenceHTML = (data.key_reasons || []).length > 0
    ? (data.key_reasons || []).map(r => `
        <div class="evidence-item supporting">
          <span class="evidence-item__icon">✓</span>
          <span class="evidence-item__text">${r}</span>
        </div>`).join('')
    : `<div class="text-muted text-sm">No evidence items available.</div>`;

  el.innerHTML = `
    <div id="scores-section" style="margin-bottom:14px;">${scores}</div>
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Evidence</div>
    <div>${evidenceHTML}</div>
    ${data.primary_driver ? `
      <div style="margin-top:12px;padding:8px 12px;background:var(--bg-tertiary);border-radius:6px;font-size:12px;">
        <span class="text-muted" style="font-size:10px;text-transform:uppercase;letter-spacing:.06em;">Primary Driver · </span>
        <span style="color:var(--text-primary);">${data.primary_driver}</span>
      </div>` : ''}`;
}

function renderRiskTab(data, symbol) {
  const el = document.getElementById('tab-risk-tab');
  if (!el) return;

  if (!data) {
    el.innerHTML = `<div class="alert-banner info">No risk data for ${symbol}.</div>`;
    return;
  }

  const frag = data.fragility_score ?? 0;
  const scps = data.scps_score ?? 0;
  const bfs  = data.bfs_score  ?? 0;
  const fragCls = frag >= 70 ? 'var(--signal-sell)' : frag >= 50 ? 'var(--signal-hold)' : 'var(--signal-buy)';
  const scpsCls  = scps >= 65 ? 'var(--signal-sell)' : 'var(--accent-primary)';

  el.innerHTML = `
    <div class="risk-gauges">
      <div class="metric-card" style="text-align:center;">
        <span class="metric-card__label">Fragility</span>
        <span class="metric-card__value" style="color:${fragCls};font-size:20px;">${frag.toFixed(0)}</span>
      </div>
      <div class="metric-card" style="text-align:center;">
        <span class="metric-card__label">SCPS</span>
        <span class="metric-card__value" style="color:${scpsCls};font-size:20px;">${scps.toFixed(0)}</span>
      </div>
      <div class="metric-card" style="text-align:center;">
        <span class="metric-card__label">BFS</span>
        <span class="metric-card__value" style="font-size:20px;">${bfs.toFixed(0)}</span>
      </div>
    </div>
    ${data.var_1d_99 != null ? `
      <div class="metric-card" style="flex-direction:row;align-items:center;justify-content:space-between;margin-bottom:10px;">
        <span class="metric-card__label">VaR 1d 99%</span>
        <span class="metric-card__value" style="font-size:16px;color:var(--signal-sell);">${(data.var_1d_99 * 100).toFixed(2)}%</span>
      </div>` : ''}
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Invalidation Conditions</div>
    <div style="font-size:12px;color:var(--text-secondary);line-height:1.7;">
      ${(data.invalidation_conditions || ['Fragility spike above 70', 'Regime shift to HIGH_VOL', 'IC degradation to WEAK']).map(c => `<div>• ${c}</div>`).join('')}
    </div>`;
}

function renderExplanationTab(explain, intel) {
  const el = document.getElementById('tab-explanation');
  if (!el) return;

  if (!explain && !intel) {
    el.innerHTML = `<div class="alert-banner info">No explanation data available.</div>`;
    return;
  }

  const chain = explain?.reasoning_chain?.steps || [];
  const chainHTML = chain.length > 0
    ? chain.map((step, i) => `
        <div class="reasoning-step">
          <span class="reasoning-step__num">${i + 1}</span>
          <div class="reasoning-step__content">
            <div class="reasoning-step__premise">${step.premise || step.factor || '—'}</div>
            <div class="reasoning-step__conclusion">${step.conclusion || step.reasoning || '—'}</div>
          </div>
        </div>`).join('')
    : `<div class="text-muted text-sm">No reasoning chain available.</div>`;

  const counter = explain?.counterfactuals || [];
  const counterHTML = counter.length > 0
    ? counter.map(c => `
        <div class="evidence-item contradicting">
          <span class="evidence-item__icon">↺</span>
          <span class="evidence-item__text">${c.description || c.scenario || JSON.stringify(c)}</span>
        </div>`).join('')
    : `<div class="text-muted text-sm">No counterfactual data available.</div>`;

  el.innerHTML = `
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Reasoning Chain</div>
    ${chainHTML}
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-top:14px;margin-bottom:8px;">What Would Flip This Signal?</div>
    ${counterHTML}`;
}

function handleSymbolSearch() {
  const val = document.getElementById('symbol-search')?.value?.trim()?.toUpperCase();
  if (val) loadSymbolIntelligence(val);
}

function handleGlobalSymbolSearch() {
  const val = document.getElementById('global-symbol-input')?.value?.trim()?.toUpperCase();
  if (val) {
    // Sync to symbol search box
    const symInput = document.getElementById('symbol-search');
    if (symInput) symInput.value = val;
    switchPanel('symbol');
    loadSymbolIntelligence(val);
  }
}
