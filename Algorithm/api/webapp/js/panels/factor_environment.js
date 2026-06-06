/* Factor Environment Panel */

async function loadFactorEnvironment() {
  const body = document.getElementById('factors-body');
  const regimeEl = document.getElementById('factor-regime-label');
  if (!body) return;

  body.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:60px;"></div>';

  try {
    const snapshot = await API.get('/macro/snapshot').catch(() => null);
    // Reshape snapshot into the two-argument form renderFactorPanel expects
    const macro = snapshot ? {
      favored_axiom_factors:   snapshot.macro_intelligence?.favored_factors   || [],
      unfavored_axiom_factors: snapshot.macro_intelligence?.unfavored_factors || [],
      equity_macro_score:      snapshot.macro_intelligence?.equity_macro_score ?? 50,
      macro_regime_label:      snapshot.macro_intelligence?.macro_regime_label || '—',
    } : null;
    const cross = snapshot ? {
      fixed_income_signal: snapshot.cross_asset?.fixed_income_signal,
      currency_signal:     snapshot.cross_asset?.currency_signal,
      commodity_signal:    snapshot.cross_asset?.commodity_signal,
      volatility_signal:   snapshot.cross_asset?.volatility_signal,
      macro_narrative:     snapshot.cross_asset?.macro_narrative,
    } : null;
    renderFactorPanel(macro, cross);
    if (regimeEl && macro) {
      regimeEl.textContent = macro.macro_regime_label || '—';
    }
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Factor data unavailable.</div>`;
  }
}

function renderFactorPanel(macro, cross) {
  const body = document.getElementById('factors-body');
  if (!body) return;

  if (!macro && !cross) {
    body.innerHTML = `<div class="alert-banner info">Macro data not available.</div>`;
    return;
  }

  const favored  = macro?.favored_axiom_factors  || [];
  const unfavored = macro?.unfavored_axiom_factors || [];
  const score    = macro?.equity_macro_score ?? 50;
  const label    = macro?.macro_regime_label || '—';

  const factors = [
    ...favored.map(f  => ({ name: f, value: 0.6, type: 'favored' })),
    ...unfavored.map(f => ({ name: f, value: 0.4, type: 'unfavored' })),
  ].slice(0, 8);

  const caSignals = cross ? [
    { label: 'Fixed Income', signal: cross.fixed_income_signal },
    { label: 'Currency',     signal: cross.currency_signal },
    { label: 'Commodity',    signal: cross.commodity_signal },
    { label: 'Volatility',   signal: cross.volatility_signal },
  ] : [];

  body.innerHTML = `
    <!-- Macro score -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <div>
        <div class="metric-card__label">Equity Macro Score</div>
        <div class="metric-card__value" style="font-size:20px;">${score.toFixed(0)}</div>
      </div>
      <div style="text-align:right;">
        <div class="metric-card__label">Regime</div>
        <div style="font-size:12px;color:var(--text-secondary);margin-top:2px;">${label}</div>
      </div>
    </div>

    <!-- Factor heatmap -->
    ${factors.length > 0 ? `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">Factor Tilts</div>
    <div id="factor-heatmap-el"></div>` : ''}

    <!-- Cross-asset -->
    ${caSignals.length > 0 ? `
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-top:12px;margin-bottom:4px;">Cross-Asset Signals</div>
    ${caSignals.map(s => `
      <div class="ca-row">
        <span class="ca-row__label">${s.label}</span>
        <span class="ca-row__signal ${s.signal || 'neutral'}">${(s.signal || '—').replace('_', ' ')}</span>
      </div>`).join('')}` : ''}

    <!-- Macro narrative -->
    ${cross?.macro_narrative ? `
    <div style="margin-top:10px;padding:8px;background:var(--bg-tertiary);border-radius:6px;font-size:11px;color:var(--text-secondary);line-height:1.6;">
      ${cross.macro_narrative}
    </div>` : ''}`;

  if (factors.length > 0) {
    requestAnimationFrame(() => renderHeatmap('factor-heatmap-el', factors));
  }
}
