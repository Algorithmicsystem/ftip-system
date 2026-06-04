/* Risk Monitor Panel */

async function loadRiskMonitor() {
  const body = document.getElementById('risk-body');
  if (!body) return;

  body.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:80px;"></div>';

  try {
    const [sri, history] = await Promise.all([
      API.get('/axiom/risk/sri').catch(() => null),
      API.get('/axiom/risk/sri/history', { lookback_days: 30 }).catch(() => null),
    ]);

    renderRiskPanel(sri, history);
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Risk data unavailable.</div>`;
  }
}

function renderRiskPanel(sri, history) {
  const body = document.getElementById('risk-body');
  if (!body) return;

  const sriVal = sri?.sri ?? 50;
  const sriLabel = sri?.sri_label ?? 'stable';
  const sriLabelCap = sriLabel.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  const sriCls = sriVal >= 70 ? 'var(--signal-sell)' : sriVal >= 40 ? 'var(--signal-hold)' : 'var(--signal-buy)';
  const barCls = sriVal >= 85 ? 'critical' : sriVal >= 70 ? 'warning' : sriVal >= 40 ? 'elevated' : 'stable';

  const histValues = (history || []).map(h => h.sri ?? 50);

  body.innerHTML = `
    <!-- SRI Gauge -->
    <div style="text-align:center;margin-bottom:14px;">
      <div style="font-family:var(--font-mono);font-size:36px;font-weight:700;color:${sriCls};">${sriVal.toFixed(1)}</div>
      <div class="sri-bar" style="margin:8px 0;">
        <div class="sri-bar__fill ${barCls}" style="width:${sriVal.toFixed(1)}%;"></div>
      </div>
      <div style="font-size:12px;color:var(--text-muted);">${sriLabelCap} · ${sri?.recommendation || ''}</div>
    </div>

    <!-- Sparkline -->
    ${histValues.length > 1 ? `
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:4px;">30-Day SRI History</div>
      <div id="sri-sparkline" class="sparkline-wrapper"></div>
    </div>` : ''}

    <!-- Components -->
    ${sri?.components ? `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">Components</div>
    ${Object.entries(sri.components).map(([k, v]) => scoreBarHTML(
      k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      v.value ?? 50,
      (v.value ?? 50) >= 70 ? '#ef4444' : (v.value ?? 50) >= 40 ? '#f59e0b' : '#10b981'
    )).join('')}` : ''}

    <!-- Primary driver -->
    ${sri?.primary_driver ? `
    <div style="margin-top:10px;font-size:11px;color:var(--text-muted);">
      Primary driver: <span style="color:var(--accent-primary);">${sri.primary_driver}</span> ·
      Trend: <span style="color:var(--text-secondary);">${sri.trend || 'stable'}</span>
    </div>` : ''}`;

  // Render sparkline after DOM update
  if (histValues.length > 1) {
    requestAnimationFrame(() => renderSparkline('sri-sparkline', histValues));
  }
}
