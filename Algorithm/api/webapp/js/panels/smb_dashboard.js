/* SMB Intelligence Dashboard Panel */

let _currentSMBEntity = null;

async function loadSMBDemo() {
  // Auto-load COST as the default ticker example
  await loadSMBIntelligence('COST');
}

async function loadSMBIntelligence(entityId) {
  if (!entityId) entityId = document.getElementById('smb-entity-input')?.value?.trim();
  if (!entityId) {
    // Auto-show demo when no entity is selected
    await loadSMBDemo();
    const body = document.getElementById('smb-body');
    if (body) {
      const banner = document.createElement('div');
      banner.className = 'alert-banner info';
      banner.style.cssText = 'margin-bottom:10px;font-size:11px;';
      banner.textContent = 'DEMO MODE — enter an entity ID above and click Load to analyze a real business';
      body.prepend(banner);
    }
    return;
  }

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
    // Upload zone for SMB financials (appended after dashboard renders)
    requestAnimationFrame(() => {
      const body2 = document.getElementById('smb-body');
      if (body2 && typeof createUploadZone === 'function') {
        const uploadSection = document.createElement('div');
        uploadSection.innerHTML = `
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:18px;">Connect Your Financials</div>
          <div id="smb-upload-zone"></div>
          <div style="text-align:center;font-size:11px;color:var(--text-muted);margin:8px 0;">— OR —</div>
          <div style="display:flex;align-items:center;gap:8px;">
            <span style="font-size:11px;color:var(--text-secondary);">QuickBooks</span>
            <button onclick="alert('QuickBooks integration coming soon')"
                    style="padding:4px 10px;font-size:11px;background:var(--bg-tertiary);color:var(--text-secondary);border:1px solid var(--border-subtle);border-radius:4px;cursor:pointer;">
              Connect
            </button>
          </div>`;
        body2.appendChild(uploadSection);
        createUploadZone('smb-upload-zone', entityId, 'smb_entity', (extraction, intel) => {
          if (intel && typeof renderSMBDashboard === 'function') renderSMBDashboard(intel);
        });
      }
    });
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load SMB intelligence: ${err.message}</div>`;
  }
}

function renderSMBDashboard(data) {
  const body = document.getElementById('smb-body');
  if (!body) return;

  // Support both private SMB entities and public ticker fallback
  const isPublicTicker = data.dscr != null || data.credit_score != null;
  const health  = data.health_score ?? data.credit_score ?? data.overall_health_score ?? 50;
  const runway  = data.cash_runway_months;
  const revenue = data.revenue_trend || (data.revenue_growth_pct > 3 ? 'growing' : data.revenue_growth_pct < -1 ? 'declining' : 'stable');
  const entity  = data.entity_name || data.symbol || data.entity_id || '—';

  // Credit metrics: prefer top-level from public_intelligence, else modules.credit
  const dscr = data.dscr ?? data.modules?.credit?.dscr ?? null;
  const creditScore = data.credit_score ?? null;
  const maxDebt = data.max_additional_debt_usd ?? null;
  const grossMarginPct = data.gross_margin_pct ?? (data.gross_margin != null ? data.gross_margin * 100 : null);
  const revGrowthPct = data.revenue_growth_pct ?? null;
  const pricingPower = data.pricing_power_score ?? data.modules?.pricing?.score ?? null;
  const axiomDau = data.axiom_dau;
  const axiomSig = data.axiom_signal;

  const healthCls = health >= 70 ? 'var(--signal-buy)' : health >= 40 ? 'var(--signal-hold)' : 'var(--signal-sell)';
  const revCls = revenue === 'growing' ? 'var(--signal-buy)' : revenue === 'declining' ? 'var(--signal-sell)' : 'var(--text-muted)';
  const dscrCls = dscr != null ? (dscr >= 1.5 ? 'var(--signal-buy)' : dscr >= 1.0 ? 'var(--signal-hold)' : 'var(--signal-sell)') : 'var(--text-muted)';

  body.innerHTML = `
    <!-- Header -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;flex-wrap:wrap;gap:8px;">
      <div>
        <div class="metric-card__label">Entity / Symbol</div>
        <div style="font-size:14px;font-weight:600;color:var(--text-primary);margin-top:2px;">${entity}${isPublicTicker && data.sector ? ` <span style="font-size:10px;color:var(--text-muted);font-weight:400;">${data.sector}</span>` : ''}</div>
      </div>
      ${axiomSig ? `<div style="text-align:center;"><div class="metric-card__label">AXIOM Signal</div><div style="font-size:14px;font-weight:700;color:${axiomSig==='BUY'?'var(--signal-buy)':axiomSig==='SELL'?'var(--signal-sell)':'var(--signal-hold)'};">${axiomSig} ${axiomDau!=null?`· ${axiomDau.toFixed(1)}`:''}` + `</div></div>` : ''}
      <div style="text-align:right;">
        <div class="metric-card__label">Revenue Trend</div>
        <div style="font-size:13px;font-weight:600;color:${revCls};margin-top:2px;text-transform:capitalize;">${revenue}${revGrowthPct!=null?` (${revGrowthPct>0?'+':''}${revGrowthPct.toFixed(1)}%)`:''}` + `</div>
      </div>
    </div>

    <!-- Credit & pricing row -->
    <div style="display:grid;grid-template-columns:repeat(${isPublicTicker?4:3},1fr);gap:8px;margin-bottom:14px;">
      ${dscr != null ? `<div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">DSCR</span>
        <span class="metric-card__value" style="font-size:16px;color:${dscrCls};">${dscr.toFixed(2)}x</span>
      </div>` : ''}
      ${grossMarginPct != null ? `<div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Gross Margin</span>
        <span class="metric-card__value" style="font-size:16px;">${grossMarginPct.toFixed(1)}%</span>
      </div>` : ''}
      ${pricingPower != null ? `<div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Pricing Power</span>
        <span class="metric-card__value" style="font-size:16px;color:${pricingPower>=65?'var(--signal-buy)':pricingPower>=45?'var(--signal-hold)':'var(--signal-sell)'};">${pricingPower.toFixed(0)}</span>
      </div>` : ''}
      ${maxDebt != null ? `<div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Max Add. Debt</span>
        <span class="metric-card__value" style="font-size:${maxDebt>1e9?'13':'16'}px;">${maxDebt>=1e9?`$${(maxDebt/1e9).toFixed(1)}B`:maxDebt>=1e6?`$${(maxDebt/1e6).toFixed(0)}M`:`$${(maxDebt/1000).toFixed(0)}K`}</span>
      </div>` : (runway!=null ? `<div class="metric-card" style="text-align:center;padding:8px;">
        <span class="metric-card__label">Cash Runway</span>
        <span class="metric-card__value" style="font-size:16px;color:${runway<6?'var(--signal-sell)':runway<12?'var(--signal-hold)':'var(--signal-buy)'};">${runway}mo</span>
      </div>` : '')}
    </div>

    <!-- Sub-sections -->
    <div id="smb-cashflow-section"></div>
    <div id="smb-supplier-section"></div>
    <div id="smb-pricing-section"></div>
    <div id="smb-alerts-section"></div>`;

  // Cash flow forecast
  const forecastData = data.cashflow_forecast || data.cash_flow_forecast;
  if (forecastData) {
    const months = Array.isArray(forecastData)
      ? forecastData.map(m => ({ net_cash_flow: m.projected_fcf ?? m.net_cash_flow ?? m.value ?? 0 }))
      : (forecastData.months || []);
    renderCashFlowForecast(months);
  }
  renderSupplierRisks(data.supplier_risks || []);

  // Pricing: support both private SMB format and public_intelligence format
  const pricingRaw = data.pricing_intelligence || data.pricing;
  if (pricingRaw) {
    const pricingNorm = {
      pricing_power_score: pricingRaw.pricing_power_score ?? pricingPower ?? 50,
      churn_risk_score: pricingRaw.churn_risk_score ?? 0,
      arpu_trend: pricingRaw.arpu_trend || (pricingRaw.revenue_growth_pct > 3 ? 'growing' : 'stable'),
      recommendation: pricingRaw.recommendation || pricingRaw.pricing_action || '',
    };
    renderPricingIntelligence(pricingNorm);
  }

  // Alerts
  const alertsEl = document.getElementById('smb-alerts-section');
  if (alertsEl) {
    const alerts = data.alerts || [];
    if (alerts.length > 0) {
      alertsEl.innerHTML = `
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Active Alerts</div>
        ${alerts.map(a => `
          <div class="alert-banner ${a.severity === 'critical' ? 'danger' : a.severity === 'warning' ? 'warning' : 'info'}" style="font-size:11px;margin-bottom:4px;">
            ${a.message || a.description || JSON.stringify(a)}
          </div>`).join('')}`;
    } else if (isPublicTicker && data.recommendation) {
      const recCls = data.recommendation === 'strong_buy_signals' ? 'success' : data.recommendation === 'caution_review_needed' ? 'warning' : 'info';
      alertsEl.innerHTML = `
        <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;margin-top:14px;">Assessment</div>
        <div class="alert-banner ${recCls}" style="font-size:11px;">${data.recommendation.replace(/_/g,' ')}</div>`;
    }
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
