/* SMB Intelligence Dashboard Panel */

let _currentSMBEntity = null;

async function loadSMBIntelligence(entityId) {
  if (!entityId) entityId = document.getElementById('smb-entity-input')?.value?.trim();
  if (!entityId) return;

  _currentSMBEntity = entityId;

  const body = document.getElementById('smb-body');
  if (!body) return;

  body.innerHTML = [1,2,3].map(() =>
    '<div class="loading-skeleton skeleton-line full" style="height:60px;margin-bottom:8px;border-radius:6px;"></div>'
  ).join('');

  try {
    const data = await API.get(`/smb/entity/${entityId}/intelligence-dashboard`).catch(() => null);

    if (!data) {
      body.innerHTML = `<div class="alert-banner info">No SMB intelligence available for entity: ${entityId}</div>`;
      return;
    }

    renderSMBDashboard(data);
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load SMB intelligence: ${err.message}</div>`;
  }
}

function renderSMBDashboard(data) {
  const body = document.getElementById('smb-body');
  if (!body) return;

  const health  = data.health_score ?? 50;
  const runway  = data.cash_runway_months;
  const revenue = data.revenue_trend || 'stable';
  const entity  = data.entity_name || data.entity_id || '—';

  const healthCls = health >= 70 ? 'var(--signal-buy)' : health >= 40 ? 'var(--signal-hold)' : 'var(--signal-sell)';
  const revCls = revenue === 'growing' ? 'var(--signal-buy)' : revenue === 'declining' ? 'var(--signal-sell)' : 'var(--text-muted)';

  body.innerHTML = `
    <!-- Header metrics -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <div>
        <div class="metric-card__label">Entity</div>
        <div style="font-size:14px;font-weight:600;color:var(--text-primary);margin-top:2px;">${entity}</div>
      </div>
      <div style="text-align:center;">
        <div class="metric-card__label">Health Score</div>
        <div style="font-family:var(--font-mono);font-size:24px;font-weight:700;color:${healthCls};">${health.toFixed(0)}</div>
      </div>
      <div style="text-align:right;">
        <div class="metric-card__label">Revenue Trend</div>
        <div style="font-size:13px;font-weight:600;color:${revCls};margin-top:2px;text-transform:capitalize;">${revenue}</div>
      </div>
    </div>

    <!-- Key metrics row -->
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
      <div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Cash Runway</span>
        <span class="metric-card__value" style="font-size:16px;color:${runway != null && runway < 6 ? 'var(--signal-sell)' : runway != null && runway < 12 ? 'var(--signal-hold)' : 'var(--signal-buy)'};">${runway != null ? `${runway}mo` : '—'}</span>
      </div>
      <div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Burn Rate</span>
        <span class="metric-card__value" style="font-size:16px;">${data.monthly_burn_rate != null ? `$${(data.monthly_burn_rate/1000).toFixed(0)}K` : '—'}</span>
      </div>
      <div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Gross Margin</span>
        <span class="metric-card__value" style="font-size:16px;">${data.gross_margin != null ? `${(data.gross_margin*100).toFixed(1)}%` : '—'}</span>
      </div>
    </div>

    <!-- Sub-sections rendered into placeholders -->
    <div id="smb-cashflow-section"></div>
    <div id="smb-supplier-section"></div>
    <div id="smb-pricing-section"></div>
    <div id="smb-alerts-section"></div>`;

  renderCashFlowForecast(data.cashflow_forecast || data.cash_flow_forecast);
  renderSupplierRisks(data.supplier_risks || []);
  renderPricingIntelligence(data.pricing_intelligence || data.pricing);

  // Alerts
  const alertsEl = document.getElementById('smb-alerts-section');
  if (alertsEl && data.alerts && data.alerts.length > 0) {
    alertsEl.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Active Alerts</div>
      ${data.alerts.map(a => `
        <div class="alert-banner ${a.severity === 'critical' ? 'danger' : a.severity === 'warning' ? 'warning' : 'info'}" style="font-size:11px;margin-bottom:4px;">
          ${a.message || a.description || JSON.stringify(a)}
        </div>`).join('')}`;
  }
}

function renderCashFlowForecast(forecast) {
  const el = document.getElementById('smb-cashflow-section');
  if (!el || !forecast) return;

  const months = Array.isArray(forecast) ? forecast : (forecast.months || []);
  if (months.length === 0) {
    el.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Cash Flow Forecast</div>
      <div class="text-muted text-sm">No forecast data available.</div>`;
    return;
  }

  const values = months.map(m => m.net_cash_flow ?? m.value ?? 0);
  const positiveCount = values.filter(v => v >= 0).length;

  el.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Cash Flow Forecast (${months.length}-Month)</div>
    <div id="smb-cashflow-sparkline" class="sparkline-wrapper" style="margin-bottom:6px;"></div>
    <div style="font-size:11px;color:var(--text-muted);">
      Positive months: <span style="color:${positiveCount >= months.length * 0.6 ? 'var(--signal-buy)' : 'var(--signal-sell)'};">${positiveCount}/${months.length}</span>
    </div>`;

  requestAnimationFrame(() => {
    renderSparkline('smb-cashflow-sparkline', values, positiveCount >= values.length * 0.6 ? '#10b981' : '#ef4444');
  });
}

function renderSupplierRisks(risks) {
  const el = document.getElementById('smb-supplier-section');
  if (!el) return;

  if (!risks || risks.length === 0) {
    el.innerHTML = `
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Supplier Risk</div>
      <div class="alert-banner success" style="font-size:11px;">No elevated supplier risks detected.</div>`;
    return;
  }

  const rows = risks.slice(0, 6).map(r => {
    const riskScore = r.risk_score ?? r.concentration_risk ?? 0;
    const riskCls = riskScore >= 70 ? 'var(--signal-sell)' : riskScore >= 40 ? 'var(--signal-hold)' : 'var(--signal-buy)';
    const pct = r.spend_percentage != null ? `${r.spend_percentage.toFixed(1)}%` : '—';
    return `
      <div style="display:flex;align-items:center;gap:10px;padding:5px 0;border-bottom:1px solid var(--border-subtle);">
        <span style="flex:1;font-size:12px;color:var(--text-primary);">${r.supplier_name || r.name || 'Unknown'}</span>
        <span style="font-size:11px;color:var(--text-muted);">${pct} spend</span>
        <span style="font-family:var(--font-mono);font-size:12px;color:${riskCls};">${riskScore.toFixed(0)}</span>
      </div>`;
  }).join('');

  el.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Supplier Risks</div>
    <div style="font-size:10px;color:var(--text-muted);display:flex;gap:10px;padding-bottom:4px;border-bottom:1px solid var(--border-subtle);">
      <span style="flex:1;">Supplier</span><span>Spend</span><span>Risk</span>
    </div>
    ${rows}`;
}

function renderPricingIntelligence(pricing) {
  const el = document.getElementById('smb-pricing-section');
  if (!el || !pricing) return;

  const power   = pricing.pricing_power_score ?? 0;
  const churn   = pricing.churn_risk_score ?? 0;
  const arpu    = pricing.arpu_trend || 'stable';
  const rec     = pricing.recommendation || '';

  const powerCls = power >= 60 ? 'var(--signal-buy)' : power >= 40 ? 'var(--signal-hold)' : 'var(--signal-sell)';
  const churnCls = churn >= 60 ? 'var(--signal-sell)' : churn >= 30 ? 'var(--signal-hold)' : 'var(--signal-buy)';

  el.innerHTML = `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:8px;margin-top:14px;">Pricing Intelligence</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:8px;">
      <div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Pricing Power</span>
        <span class="metric-card__value" style="font-size:16px;color:${powerCls};">${power.toFixed(0)}</span>
      </div>
      <div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Churn Risk</span>
        <span class="metric-card__value" style="font-size:16px;color:${churnCls};">${churn.toFixed(0)}</span>
      </div>
      <div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">ARPU Trend</span>
        <span class="metric-card__value" style="font-size:13px;text-transform:capitalize;">${arpu}</span>
      </div>
    </div>
    ${rec ? `<div style="padding:8px;background:var(--bg-tertiary);border-radius:6px;font-size:11px;color:var(--text-secondary);line-height:1.6;">${rec}</div>` : ''}`;
}
