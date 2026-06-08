/* Morning Briefing Panel */

async function loadMorningBriefing() {
  const body = document.getElementById('briefing-body');
  const dateEl = document.getElementById('briefing-date');
  const chipEl = document.getElementById('briefing-regime-chip');
  if (!body) return;

  const textEl = document.getElementById('briefing-text');
  const sriEl = document.getElementById('briefing-sri');
  const signalsEl = document.getElementById('briefing-signals');
  const oppsEl = document.getElementById('briefing-opps');

  if (textEl) {
    textEl.innerHTML = '<div class="loading-skeleton skeleton-line full"></div><div class="loading-skeleton skeleton-line medium" style="margin-top:8px;"></div><div class="loading-skeleton skeleton-line short" style="margin-top:8px;"></div>';
  }
  if (sriEl) sriEl.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:70px;"></div>';
  if (signalsEl) signalsEl.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:70px;"></div>';
  if (oppsEl) oppsEl.innerHTML = '';
  if (!textEl && !sriEl) {
    body.innerHTML = `
      <div class="loading-skeleton skeleton-line full"></div>
      <div class="loading-skeleton skeleton-line medium" style="margin-top:8px;"></div>
      <div class="loading-skeleton skeleton-line short" style="margin-top:8px;"></div>`;
  }

  try {
    const data = await API.get('/jobs/briefing/morning');
    renderBriefing(data);
    if (dateEl) dateEl.textContent = data.briefing_date || '';
    if (chipEl) {
      const regime = (data.regime_context?.regime_label || 'unknown').toLowerCase();
      chipEl.className = 'regime-chip ' + (
        regime.includes('trend') ? 'trending' :
        regime.includes('chop') ? 'choppy' :
        regime.includes('vol') ? 'high-vol' : 'unknown'
      );
      chipEl.textContent = data.regime_context?.regime_label || 'Unknown';
    }
  } catch (err) {
    const status = err.statusCode || 0;
    const msg = status === 401
      ? 'Configure your API key in the topbar to view the morning briefing.'
      : status === 503 || status === 500
        ? 'Briefing temporarily unavailable — pipeline may be running.'
        : 'Could not load morning briefing. Refresh to retry.';
    body.innerHTML = `<div class="alert-banner warning">${msg}</div>`;
  }
}

function renderBriefing(data) {
  const sri = data.systemic_risk_index ?? 50;
  const sriBarCls = sri >= 85 ? 'critical' : sri >= 70 ? 'warning' : sri >= 40 ? 'elevated' : 'stable';
  const sriLabel = sri >= 85 ? 'Critical' : sri >= 70 ? 'High Alert' : sri >= 50 ? 'Warning' : sri >= 25 ? 'Elevated' : 'Stable';
  const sriColor = sri >= 70 ? 'var(--signal-sell)' : sri >= 40 ? 'var(--signal-hold)' : 'var(--signal-buy)';

  const top = (data.top_opportunities || [])[0];
  const risk = (data.key_risks || [])[0];
  const ca = data.cross_asset_context || {};
  const briefText = data.briefing_text || '';

  const textEl = document.getElementById('briefing-text');
  const sriEl = document.getElementById('briefing-sri');
  const signalsEl = document.getElementById('briefing-signals');
  const oppsEl = document.getElementById('briefing-opps');

  if (textEl) {
    textEl.innerHTML = briefText.split('\n\n').map(p => `<p style="margin-bottom:6px;">${p}</p>`).join('') || '<p class="text-muted">No briefing text available.</p>';
  }
  if (sriEl) {
    sriEl.innerHTML = `
      <div class="metric-card">
        <span class="metric-card__label">Systemic Risk Index</span>
        <span class="metric-card__value" style="color:${sriColor};">${sri.toFixed(1)}</span>
        <div class="sri-bar" style="margin-top:6px;">
          <div class="sri-bar__fill ${sriBarCls}" style="width:${sri.toFixed(1)}%;"></div>
        </div>
        <span class="text-sm text-muted" style="margin-top:4px;">${sriLabel}</span>
      </div>`;
  }
  if (signalsEl) {
    signalsEl.innerHTML = `
      <div class="metric-card">
        <span class="metric-card__label">Cross-Asset Signals</span>
        <div class="ca-row" style="margin-top:6px;">
          <span class="ca-row__label">Fixed Income</span>
          <span class="ca-row__signal ${ca.fixed_income_signal || 'neutral'}">${(ca.fixed_income_signal || '—').replace('_', ' ')}</span>
        </div>
        <div class="ca-row">
          <span class="ca-row__label">Volatility</span>
          <span class="ca-row__signal ${ca.volatility_signal || 'neutral'}">${(ca.volatility_signal || '—').replace('_', ' ')}</span>
        </div>
      </div>`;
  }
  if (oppsEl) {
    oppsEl.innerHTML = `
      <div class="evidence-item supporting">
        <span class="evidence-item__icon">▲</span>
        <span class="evidence-item__text">
          <strong>Top Opportunity:</strong> ${top ? `${top.symbol} (DAU ${(top.dau||0).toFixed(1)})` : 'None identified'}
        </span>
      </div>
      <div class="evidence-item contradicting">
        <span class="evidence-item__icon">▼</span>
        <span class="evidence-item__text">
          <strong>Top Risk:</strong> ${risk ? `${risk.symbol} fragility ${(risk.fragility_score||0).toFixed(0)}` : 'No critical risks'}
        </span>
      </div>`;
  }

  // Fall back to full body replacement if sub-elements not present
  if (!textEl && !sriEl) {
    const body = document.getElementById('briefing-body');
    if (!body) return;
    body.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
        <div class="metric-card">
          <span class="metric-card__label">Systemic Risk Index</span>
          <span class="metric-card__value" style="color:${sriColor};">${sri.toFixed(1)}</span>
          <div class="sri-bar" style="margin-top:6px;"><div class="sri-bar__fill ${sriBarCls}" style="width:${sri.toFixed(1)}%;"></div></div>
          <span class="text-sm text-muted" style="margin-top:4px;">${sriLabel}</span>
        </div>
        <div class="metric-card">
          <span class="metric-card__label">Cross-Asset Signals</span>
          <div class="ca-row" style="margin-top:6px;">
            <span class="ca-row__label">Fixed Income</span>
            <span class="ca-row__signal ${ca.fixed_income_signal || 'neutral'}">${(ca.fixed_income_signal || '—').replace('_', ' ')}</span>
          </div>
          <div class="ca-row">
            <span class="ca-row__label">Volatility</span>
            <span class="ca-row__signal ${ca.volatility_signal || 'neutral'}">${(ca.volatility_signal || '—').replace('_', ' ')}</span>
          </div>
        </div>
      </div>
      <div style="font-size:12px;line-height:1.7;color:var(--text-secondary);margin-bottom:12px;">
        ${briefText.split('\n\n').map(p => `<p style="margin-bottom:6px;">${p}</p>`).join('')}
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div class="evidence-item supporting">
          <span class="evidence-item__icon">▲</span>
          <span class="evidence-item__text"><strong>Top Opportunity:</strong> ${top ? `${top.symbol} (DAU ${(top.dau||0).toFixed(1)})` : 'None identified'}</span>
        </div>
        <div class="evidence-item contradicting">
          <span class="evidence-item__icon">▼</span>
          <span class="evidence-item__text"><strong>Top Risk:</strong> ${risk ? `${risk.symbol} fragility ${(risk.fragility_score||0).toFixed(0)}` : 'No critical risks'}</span>
        </div>
      </div>`;
  }
}
